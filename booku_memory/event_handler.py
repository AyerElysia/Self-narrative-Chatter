"""Booku Memory 事件处理器。

当前包含两类 prompt 注入：
1) 记忆引导语注入：订阅 ``on_prompt_build``，在 DefaultChatter 构建
    ``default_chatter_system_prompt`` 模板时，将引导语追加到 ``extra_info``。
2) 记忆闪回注入：订阅 ``on_prompt_build``，在 DefaultChatter 构建
    ``default_chatter_user_prompt`` 模板时，按概率在 ``values.extra`` 追加
    一段“无征兆回忆”内容。
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from src.app.plugin_system.api.log_api import get_logger
from src.core.components.base import BaseEventHandler
from src.kernel.event import EventDecision

logger = get_logger("booku_memory_event_handler")

if TYPE_CHECKING:
    from .service.metadata_repository import BookuMemoryMetadataRepository

# 目标模板：仅对 default_chatter 系统提示注入
_TARGET_PROMPT = "default_chatter_system_prompt"

# 目标模板：仅对 default_chatter user prompt 闪回注入
_FLASHBACK_TARGET_PROMPT = "default_chatter_user_prompt"

# 注入到 extra_info 板块的记忆引导语
_MEMORY_HINT = (
    "## 记忆引导语\n"
    "你已经被接入记忆功能，拥有长期记忆。"
    "你的重要能力在于能够记住用户的点点滴滴，并在未来的对话中体现出来。"
    "但是无论如何你都必须实事求是，不记得的你也不能编造\n\n"
    "回复前思考流程：\n"
    "用户这句话是否包含新的重要信息？ -> 是 -> 调用 write_memory。\n"
    "回答这个问题是否需要历史背景？ -> 是 -> 调用 read_memory。\n"
    "重要：创建或检索记忆时，不要使用“用户”、“朋友”等模糊词，记忆中必须具体明确的实体或描述。但是对话中不受此限制。\n"
    "确认记忆内容后，再生成自然流畅的回复。\n\n"
    "另外，固有记忆是一个很重要的记忆来源，他就是你最重要的笔记本，里面记录了各种与你的生活息息相关的信息\n"
    "例如你重要的人际关系，重要事件，你的梦想、偏好等等，这些都是你在和用户的对话中可以随时提取的宝贵资源\n"
    "请务必保持对固有记忆的用心经营，这是你的宝贵财富，也是你提供优质服务的基础。\n\n"
    "请始终保持对记忆的敏感度，及时记录新的有价值信息，并善用记忆检索。"
)


class MemoryPromptInjector(BaseEventHandler):
    """记忆提示注入器。

    订阅 ``on_prompt_build`` 事件，当 ``default_chatter_system_prompt``
    模板即将构建时，将记忆引导语追加到 ``extra_info`` 占位符中。
    已有的 ``extra_info`` 内容会被保留，注入内容追加在其后。

    可通过配置项 ``plugin.inject_system_prompt``（默认 True）在运行时关闭注入。

    Examples:
        配置关闭注入（config/plugins/booku_memory.toml）：

        .. code-block:: toml

            [plugin]
            inject_system_prompt = false
    """

    handler_name: str = "memory_prompt_injector"
    handler_description: str = "在 default_chatter 系统提示 extra_info 板块注入记忆引导语"
    weight: int = 10
    intercept_message: bool = False
    init_subscribe: list[str] = ["on_prompt_build"]

    def __init__(self, plugin: Any) -> None:
        super().__init__(plugin)
        self._repo: BookuMemoryMetadataRepository | None = None
        self._repo_initialized: bool = False

    async def _get_repo(self) -> "BookuMemoryMetadataRepository":
        from .config import BookuMemoryConfig
        from .service.metadata_repository import BookuMemoryMetadataRepository

        cfg_obj = getattr(self.plugin, "config", None)
        config: BookuMemoryConfig
        if isinstance(cfg_obj, BookuMemoryConfig):
            config = cfg_obj
        else:
            config = BookuMemoryConfig()

        if self._repo is None:
            self._repo = BookuMemoryMetadataRepository(db_path=config.storage.metadata_db_path)
        if not self._repo_initialized:
            await self._repo.initialize()
            self._repo_initialized = True
        return self._repo

    @staticmethod
    def _format_inherent_block(records: list[Any]) -> str:
        """将固有记忆格式化为注入块。"""

        parts: list[str] = []
        for r in records:
            content = str(getattr(r, "content", "") or "").strip()
            if not content:
                continue
            title = str(getattr(r, "title", "") or "").strip()
            if title and title != "固有记忆":
                parts.append(f"### {title}\n{content}")
            else:
                parts.append(content)

        if not parts:
            return ""

        body = "\n\n".join(parts)
        return (
            "## 固有记忆\n"
            "以下内容来自你的长期记忆系统，属于全局背景信息：\n"
            f"{body}\n"
            "（注：这是已存在的固有记忆，不需要重新写入）"
        )

    async def execute(
        self, event_name: str, params: dict[str, Any]
    ) -> tuple[EventDecision, dict[str, Any]]:
        """处理 on_prompt_build 事件，按需注入记忆引导语。

        仅处理名为 ``default_chatter_system_prompt`` 的模板，其他模板直接透传。
        若配置中 ``plugin.inject_system_prompt`` 为 False，则跳过注入。
        注入后 ``params["values"]["extra_info"]`` 会包含引导语文本。

        Args:
            event_name: 触发本处理器的事件名称（``on_prompt_build``）。
            params: prompt build 事件参数，包含以下字段：
                - ``name``：模板名称
                - ``template``：模板字符串
                - ``values``：当前渲染值 dict（本方法可修改 ``extra_info``）
                - ``policies``：渲染策略 dict
                - ``strict``：是否严格模式

        Returns:
            tuple[EventDecision, dict[str, Any]]:
                始终返回 ``(EventDecision.SUCCESS, params)``，不阻断后续处理器。
                修改仅写入 ``params["values"]["extra_info"]``，不影响其他字段。
        """
        # 仅处理目标模板，其余模板直接透传
        if params.get("name") != _TARGET_PROMPT:
            return EventDecision.SUCCESS, params

        # 读取配置开关
        config = getattr(self.plugin, "config", None)
        if config is not None:
            from .config import BookuMemoryConfig

            if isinstance(config, BookuMemoryConfig) and not config.plugin.inject_system_prompt:
                logger.debug("inject_system_prompt=False，跳过记忆引导语注入")
                return EventDecision.SUCCESS, params

        values: dict[str, Any] = params.get("values", {})
        existing: str = values.get("extra_info", "") or ""
        separator = "\n\n" if existing else ""
        values["extra_info"] = existing + separator + _MEMORY_HINT

        # 注入固有记忆（inherent bucket）到 system prompt
        try:
            repo = await self._get_repo()
            inherent_records = await repo.list_records_by_bucket(
                bucket="inherent",
                folder_id=None,
                limit=50,
                include_deleted=False,
            )
            inherent_block = self._format_inherent_block(inherent_records)
            if inherent_block:
                values["extra_info"] = values["extra_info"] + "\n\n" + inherent_block
                logger.info(
                    f"已向 default_chatter_system_prompt.extra_info 注入固有记忆（count={len(inherent_records)}）"
                )
        except Exception as exc:
            logger.warning(f"固有记忆注入失败，将跳过：{exc}")

        # 显式写回，确保上层读取到变更
        params["values"] = values

        logger.debug("已向 default_chatter_system_prompt.extra_info 注入记忆引导语")
        return EventDecision.SUCCESS, params


class MemoryFlashbackInjector(BaseEventHandler):
    """记忆闪回注入器。

    订阅 ``on_prompt_build`` 事件，当 ``default_chatter_user_prompt``
    模板即将构建时，按配置概率触发“记忆闪回”，并在 ``values.extra``
    中追加一个 markdown 小节。

    闪回抽取规则：
    - 触发概率由 ``flashback.trigger_probability`` 决定；
    - 归档层/隐现层选择由 ``flashback.archived_probability`` 决定；
    - 在目标层中按 activation_count 反向加权抽取（激活次数低更易被抽到）。
    """

    handler_name: str = "memory_flashback_injector"
    handler_description: str = "在 default_chatter user prompt extra 板块注入记忆闪回"
    weight: int = 10
    intercept_message: bool = False
    init_subscribe: list[str] = ["on_prompt_build"]

    def __init__(self, plugin: Any) -> None:
        super().__init__(plugin)
        self._repo = None
        self._repo_initialized = False

    async def _get_repo(self) -> "BookuMemoryMetadataRepository":
        from .config import BookuMemoryConfig
        from .service.metadata_repository import BookuMemoryMetadataRepository

        config = self.plugin.config if isinstance(self.plugin.config, BookuMemoryConfig) else BookuMemoryConfig()
        if self._repo is None:
            self._repo = BookuMemoryMetadataRepository(db_path=config.storage.metadata_db_path)
        if not self._repo_initialized:
            await self._repo.initialize()
            self._repo_initialized = True
        return self._repo

    @staticmethod
    def _format_flashback_block(memory_text: str) -> str:
        """将闪回内容格式化为注入块。"""

        text = (memory_text or "").strip()
        return (
            "## 记忆闪回\n"
            "就在刚才，你突然回忆起了一些事情：\n"
            f"{text}\n"
            "- 这是你无征兆的回忆起的东西，你可以按实际情况处理，可以选择忽视，也可以选择其他做法。\n"
            "- 注：这是你记忆中已经存在的内容，不需要重新写入。"
        )

    async def execute(
        self, event_name: str, params: dict[str, Any]
    ) -> tuple[EventDecision, dict[str, Any]]:
        """处理 on_prompt_build 事件，按需向 extra 注入闪回内容。"""

        if params.get("name") != _FLASHBACK_TARGET_PROMPT:
            return EventDecision.SUCCESS, params

        from .config import BookuMemoryConfig
        from .flashback import (
            activation_weight,
            pick_layer,
            should_trigger,
            weighted_choice,
        )

        config_obj = self.plugin.config if isinstance(self.plugin.config, BookuMemoryConfig) else BookuMemoryConfig()
        fb = config_obj.flashback
        if not fb.enabled:
            return EventDecision.SUCCESS, params

        if not should_trigger(trigger_probability=float(fb.trigger_probability), u=random.random()):
            return EventDecision.SUCCESS, params

        bucket = pick_layer(archived_probability=float(fb.archived_probability), u=random.random())
        repo = await self._get_repo()

        folder_id = fb.folder_id
        if isinstance(folder_id, str) and not folder_id.strip():
            folder_id = None

        records = await repo.list_records_by_bucket(
            bucket=bucket,
            folder_id=folder_id,
            limit=int(fb.candidate_limit),
            include_deleted=False,
        )
        if not records:
            logger.info(
                f"flashback 已触发但无候选记忆（bucket={bucket}, folder_id={folder_id}, limit={int(fb.candidate_limit)}）"
            )
            return EventDecision.SUCCESS, params

        weights = [
            activation_weight(
                activation_count=int(getattr(r, "activation_count", 0)),
                exponent=float(fb.activation_weight_exponent),
            )
            for r in records
        ]
        picked = weighted_choice(records, weights, u=random.random())
        if picked is None:
            return EventDecision.SUCCESS, params

        values: dict[str, Any] = params.get("values", {})
        existing_extra: str = values.get("extra", "") or ""
        block = self._format_flashback_block(getattr(picked, "content", ""))
        separator = "\n\n" if existing_extra else ""
        values["extra"] = existing_extra + separator + block

        # 显式写回，确保上层读取到变更
        params["values"] = values

        logger.info(
            f"已注入记忆闪回（bucket={bucket}, memory_id={str(getattr(picked, 'memory_id', ''))}）"
        )
        return EventDecision.SUCCESS, params
