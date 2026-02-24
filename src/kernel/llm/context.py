"""LLM context management utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .payload import LLMPayload
from .roles import ROLE

CompressionHook = Callable[[list[list[LLMPayload]], list[LLMPayload]], list[LLMPayload]]
TokenCounter = Callable[[list[LLMPayload]], int]


@dataclass(slots=True)
class LLMContextManager:
    """上下文管理器，负责根据 max_payloads 限制对上下文进行裁剪。
    
    重载 maybe_trim 方法实现裁剪逻辑，默认按照“保留开头的系统/工具消息 + 最近的用户/助手消息”的策略进行裁剪。
    """

    max_payloads: int | None = None
    compression_hook: CompressionHook | None = None

    def maybe_trim(
        self,
        payloads: list[LLMPayload],
        *,
        max_token_budget: int | None = None,
        token_counter: TokenCounter | None = None,
    ) -> list[LLMPayload]:
        """
        根据 max_payloads 和 max_token_budget 对 payloads 进行裁剪。

        裁剪策略：
        1. 保留开头的系统/工具消息（pinned prefix）。
        2. 将剩余消息按用户/助手对话分组，整体裁剪掉较早的对话组。
        3. 如果提供了 compression_hook，则在裁剪掉一批对话组后，调用该 hook 生成压缩后的消息，并将其插入剩余消息的开头。
        4. 如果 max_token_budget 仍然超出，则继续裁剪剩余的对话组，直到满足预算。
        """

        trimmed = payloads

        if self.max_payloads is not None and self.max_payloads > 0 and len(trimmed) > self.max_payloads:
            trimmed = self._trim(trimmed, self.max_payloads)

        if (
            max_token_budget is not None
            and max_token_budget > 0
            and token_counter is not None
            and token_counter(trimmed) > max_token_budget
        ):
            trimmed = self._trim_by_tokens(trimmed, max_token_budget, token_counter)

        return trimmed

    def _trim_by_tokens(
        self,
        payloads: list[LLMPayload],
        token_budget: int,
        token_counter: TokenCounter,
    ) -> list[LLMPayload]:
        """
        根据 token_budget 对 payloads 进行裁剪
        """

        pinned, tail = self._split_pinned_prefix(payloads)
        groups = self._build_qa_groups(tail)
        if not groups:
            return payloads

        kept_groups = list(groups)
        dropped_groups: list[list[LLMPayload]] = []

        while len(kept_groups) > 1:
            candidate = pinned + self._flatten_groups(kept_groups)
            if token_counter(candidate) <= token_budget:
                break
            dropped_groups.append(kept_groups.pop(0))

        remaining_payloads = self._flatten_groups(kept_groups)
        hook_payloads = self._apply_compression_hook(dropped_groups, remaining_payloads)

        if hook_payloads:
            combined = pinned + hook_payloads + remaining_payloads
            while len(kept_groups) > 1 and token_counter(combined) > token_budget:
                kept_groups.pop(0)
                remaining_payloads = self._flatten_groups(kept_groups)
                combined = pinned + hook_payloads + remaining_payloads
            return combined

        return pinned + remaining_payloads

    def _trim(self, payloads: list[LLMPayload], max_payloads: int) -> list[LLMPayload]:
        pinned, tail = self._split_pinned_prefix(payloads)
        groups = self._build_qa_groups(tail)
        if not groups:
            return payloads

        kept_groups = list(groups)
        dropped_groups: list[list[LLMPayload]] = []

        while len(kept_groups) > 1 and self._payload_len(pinned, kept_groups) > max_payloads:
            dropped_groups.append(kept_groups.pop(0))

        remaining_payloads = self._flatten_groups(kept_groups)
        hook_payloads = self._apply_compression_hook(dropped_groups, remaining_payloads)

        if hook_payloads:
            remaining_payloads = self._flatten_groups(kept_groups)

        while len(kept_groups) > 1 and (
            len(pinned) + len(hook_payloads) + len(remaining_payloads) > max_payloads
        ):
            kept_groups.pop(0)
            remaining_payloads = self._flatten_groups(kept_groups)

        return pinned + hook_payloads + remaining_payloads

    def _split_pinned_prefix(self, payloads: list[LLMPayload]) -> tuple[list[LLMPayload], list[LLMPayload]]:
        """将 payloads 拆分为 pinned 消息和对话消息两部分。

        pinned 消息定义为：所有 SYSTEM 和 TOOL 角色的消息，无论其出现在列表的任何位置，
        均视为固定部分，始终被保留，不参与裁剪。
        对话消息为剩余的 USER 和 ASSISTANT 消息，按原始顺序保留。
        """
        pinned_roles = {ROLE.SYSTEM, ROLE.TOOL}
        pinned = [p for p in payloads if p.role in pinned_roles]
        tail = [p for p in payloads if p.role not in pinned_roles]
        return pinned, tail

    def _build_qa_groups(self, payloads: list[LLMPayload]) -> list[list[LLMPayload]]:
        """将消息分组。一个组作为一个不可分割的最小裁剪单位。
        
        分组策略：
        1. 每一个 USER 角色开始一个新组。
        2. 后续的 ASSISTANT 和 TOOL_RESULT 消息紧跟在该 USER 组内。
        3. 如果在第一个 USER 之前有孤立的消息（如历史遗留），它们会各自独立成组。
        """
        groups: list[list[LLMPayload]] = []
        current: list[LLMPayload] = []

        for payload in payloads:
            # 遇到 USER 角色，开启新组
            if payload.role == ROLE.USER:
                if current:
                    groups.append(current)
                current = [payload]
            elif not current:
                # 处理第一个 USER 之前的孤立消息
                groups.append([payload])
            else:
                # 归入当前组（确保 user-assistant-tool_result 连带关系）
                current.append(payload)

        if current:
            groups.append(current)

        return groups

    def _apply_compression_hook(
        self,
        dropped_groups: list[list[LLMPayload]],
        remaining_payloads: list[LLMPayload],
    ) -> list[LLMPayload]:
        if not self.compression_hook or not dropped_groups:
            return []
        return self.compression_hook(dropped_groups, remaining_payloads)

    def _flatten_groups(self, groups: list[list[LLMPayload]]) -> list[LLMPayload]:
        return [payload for group in groups for payload in group]

    def _payload_len(self, pinned: list[LLMPayload], groups: list[list[LLMPayload]]) -> int:
        return len(pinned) + sum(len(group) for group in groups)
