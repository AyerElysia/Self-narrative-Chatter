"""Default Chatter 子代理决策模块。"""

from __future__ import annotations

from typing import Any

import json_repair

from src.core.config import get_core_config
from src.core.models.stream import ChatStream
from src.core.prompt import get_prompt_manager
from src.kernel.llm import LLMPayload, ROLE, Text


async def decide_should_respond(
    chatter: Any,
    logger: Any,
    unreads_text: str,
    chat_stream: ChatStream,
    history_text: str,
    fallback_prompt: str,
) -> dict[str, Any]:
    """执行子代理决策并返回 should_respond 结果。"""
    try:
        request = chatter.create_request("sub_actor", "sub_agent", max_context=5)
    except (ValueError, KeyError):
        return {"should_respond": True, "reason": "未找到 sub_actor 配置，默认响应"}

    nickname = get_core_config().personality.nickname
    tmpl = get_prompt_manager().get_template("default_chatter_sub_agent_prompt")
    if tmpl:
        sub_prompt = await tmpl.set("nickname", nickname).build()
    else:
        sub_prompt = fallback_prompt.format(nickname=nickname)

    request.add_payload(LLMPayload(ROLE.SYSTEM, Text(sub_prompt)))

    if history_text:
        request.add_payload(LLMPayload(ROLE.USER, Text(history_text)))

    request.add_payload(
        LLMPayload(ROLE.USER, Text(f"【新收到待判定消息】\n{unreads_text}"))
    )

    try:
        response = await request.send(stream=False)
        await response

        content = response.message
        if not content or not content.strip():
            logger.warning("Sub-agent 返回了空内容，默认进行响应")
            return {"should_respond": True, "reason": "模型未返回判断内容"}

        try:
            result = json_repair.loads(content)
            if isinstance(result, dict):
                return {
                    "should_respond": bool(result.get("should_respond", True)),
                    "reason": result.get("reason", "未提供理由"),
                }
        except Exception as error:
            logger.debug(f"Sub-agent JSON 解析失败: {error} | 内容: {content[:500]}")

        logger.warning(f"Sub-agent 无法找到有效的 JSON 结构: {content[:200]}...")
        return {"should_respond": True, "reason": "解析 JSON 失败，默认响应"}
    except Exception as error:
        logger.error(f"Sub-agent 决策过程异常: {error}", exc_info=True)
        return {"should_respond": True, "reason": f"执行异常: {error}"}
