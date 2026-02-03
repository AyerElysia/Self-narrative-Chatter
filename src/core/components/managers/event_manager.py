"""事件管理器。

本模块提供事件管理器，负责管理所有事件处理器的注册、订阅和发布。
支持按权重排序执行和消息拦截功能。
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from src.kernel.logger import get_logger

from src.core.components.base.event_handler import BaseEventHandler
from src.core.components.types import EventType



logger = get_logger("event_manager")


class EventManager:
    """事件管理器。

    负责管理所有事件处理器的注册、订阅和发布。
    支持按权重排序执行和消息拦截功能。

    Attributes:
        _subscription_map: 事件订阅映射表，事件类型 -> 处理器列表
        _handler_map: 处理器映射，处理器签名 -> 处理器实例
        _lock: 用于线程安全操作的异步锁

    Examples:
        >>> manager = EventManager()
        >>> manager.build_subscription_map()
        >>> # 发布事件
        >>> manager.publish_event(EventType.ON_MESSAGE_RECEIVED, {"message": "Hello"})
    """

    def __init__(self) -> None:
        """初始化事件管理器。"""
        self._subscription_map: Dict[EventType, List[Tuple[BaseEventHandler, str]]] = defaultdict(list)
        self._handler_map: Dict[str, BaseEventHandler] = {}
        self._lock = asyncio.Lock()

        logger.info("事件管理器初始化完成")

    def build_subscription_map(self) -> None:
        """构建事件订阅映射表。

        遍历所有已注册的事件处理器，根据它们的订阅信息构建映射表。
        处理器按权重降序排序，权重高的优先执行。

        Examples:
            >>> manager.build_subscription_map()
        """
        try:
            # 尝试获取当前事件循环
            loop = asyncio.get_running_loop()
            # 如果有运行的事件循环，创建异步任务
            loop.create_task(self._build_subscription_map())
        except RuntimeError:
            # 没有运行的事件循环，直接同步构建
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._build_subscription_map())
            finally:
                # 清理事件循环
                loop.close()

    async def _build_subscription_map(self) -> None:
        """内部方法：构建事件订阅映射表。"""
        async with self._lock:
            # 清空现有映射表
            self._subscription_map.clear()

            # 从全局注册表获取所有事件处理器组件
            from src.core.components.registry import get_global_registry
            from src.core.components.types import ComponentType
            registry = get_global_registry()

            # 获取所有 EVENT_HANDLER 类型的组件
            event_handler_classes = registry.get_by_type(ComponentType.EVENT_HANDLER)

            # 需要从 plugin manager 获取实例化的插件，然后实例化事件处理器
            from src.core.components.managers.plugin_manager import get_plugin_manager
            plugin_manager = get_plugin_manager()

            for signature, handler_cls in event_handler_classes.items():
                try:
                    # 解析签名获取插件名称
                    from src.core.components.types import parse_signature
                    sig = parse_signature(signature)
                    plugin_name = sig["plugin_name"]

                    # 获取插件实例
                    plugin_instance = plugin_manager.get_plugin(plugin_name)
                    if not plugin_instance:
                        logger.warning(f"未找到插件实例: {plugin_name}")
                        continue

                    # 实例化事件处理器
                    handler = handler_cls(plugin_instance)
                    handler.signature = signature  # 设置签名属性

                    # 添加到处理器映射表
                    self._handler_map[signature] = handler

                    # 获取处理器订阅的事件
                    subscribed_events = handler.get_subscribed_events()

                    # 将处理器添加到每个订阅事件的映射表中
                    for event in subscribed_events:
                        if isinstance(event, EventType):
                            # 按权重降序排序（权重高的在前）
                            self._subscription_map[event].append(
                                (handler, signature)
                            )

                    logger.debug(f"已注册事件处理器: {signature}")

                except Exception as e:
                    logger.error(f"实例化事件处理器 {signature} 失败: {e}")
                    continue

            # 对每个事件的处理器列表按权重排序
            for event in self._subscription_map:
                self._subscription_map[event].sort(
                    key=lambda x: x[0].weight,
                    reverse=True
                )

            logger.info(
                f"订阅映射表构建完成，共处理 {len(self._handler_map)} 个 "
                f"事件处理器，覆盖 {len(self._subscription_map)} 种事件类型"
            )

    async def publish_event(
        self,
        event: EventType | str,
        kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """发布事件给订阅者。

        Args:
            event: 事件类型（EventType 枚举或字符串）
            kwargs: 事件参数字典

        Returns:
            Dict[str, Any]: 发布结果，包含每个处理器的执行状态

        Examples:
            >>> result = await manager.publish_event(
            ...     EventType.ON_MESSAGE_RECEIVED,
            ...     {"message": "Hello", "sender": "user1"}
            ... )
        """
        if kwargs is None:
            kwargs = {}

        # 如果事件是字符串格式，转换为 EventType 枚举
        if isinstance(event, str):
            try:
                event = EventType(event)
            except ValueError:
                logger.warning(f"未知的事件类型: {event}")
                return {"error": f"Unknown event type: {event}"}

        logger.debug(f"发布事件: {event}")

        # 获取该事件的所有订阅处理器
        subscribed_handlers = self._subscription_map.get(event, [])
        results = {}

        # 按权重顺序执行处理器
        for handler, signature in subscribed_handlers:
            try:
                logger.debug(f"执行事件处理器: {signature}")

                # 执行处理器
                success, intercepted, message = await handler.execute(kwargs)

                # 记录结果
                results[signature] = {
                    "success": success,
                    "intercepted": intercepted,
                    "message": message
                }

                # 如果处理器拦截了消息，停止执行后续处理器
                if intercepted:
                    logger.info(f"事件被处理器 {signature} 拦截，停止执行后续处理器")
                    break

            except Exception as e:
                logger.error(f"事件处理器 {signature} 执行失败: {e}")
                results[signature] = {
                    "success": False,
                    "intercepted": False,
                    "message": f"执行失败: {str(e)}"
                }

        logger.debug(f"事件 {event} 发布完成，共执行 {len(results)} 个处理器")
        return results

    def register_handler(self, signature: str, handler: BaseEventHandler) -> None:
        """注册单个事件处理器。

        Args:
            signature: 处理器签名
            handler: 事件处理器实例

        Examples:
            >>> manager.register_handler("my_plugin:event_handler:log", handler)
        """
        async def _register_handler() -> None:
            async with self._lock:
                self._handler_map[signature] = handler
                logger.debug(f"已注册事件处理器: {signature}")

                # 可以在这里重新构建订阅映射表
                await self._rebuild_subscription_map()

        asyncio.create_task(_register_handler())

    def unregister_handler(self, signature: str) -> None:
        """注销单个事件处理器。

        Args:
            signature: 处理器签名

        Examples:
            >>> manager.unregister_handler("my_plugin:event_handler:log")
        """
        async def _unregister_handler() -> None:
            async with self._lock:
                if signature in self._handler_map:
                    handler = self._handler_map.pop(signature)

                    # 从订阅映射表中移除
                    subscribed_events = handler.get_subscribed_events()
                    for event in subscribed_events:
                        if isinstance(event, EventType):
                            self._subscription_map[event] = [
                                (h, s) for h, s in self._subscription_map[event]
                                if s != signature
                            ]

                    logger.debug(f"已注销事件处理器: {signature}")

                    # 重新构建订阅映射表以确保顺序正确
                    await self._rebuild_subscription_map()

        asyncio.create_task(_unregister_handler())

    def get_handlers_for_event(self, event: EventType | str) -> List[Tuple[BaseEventHandler, str]]:
        """获取指定事件的所有处理器。

        Args:
            event: 事件类型

        Returns:
            List[Tuple[BaseEventHandler, str]]: 处理器列表，包含处理器实例和签名

        Examples:
            >>> handlers = manager.get_handlers_for_event(EventType.ON_MESSAGE_RECEIVED)
        """
        if isinstance(event, str):
            try:
                event = EventType(event)
            except ValueError:
                return []

        return self._subscription_map.get(event, [])

    def get_handler(self, signature: str) -> Optional[BaseEventHandler]:
        """获取指定签名的事件处理器。

        Args:
            signature: 处理器签名

        Returns:
            Optional[BaseEventHandler]: 处理器实例，不存在返回 None

        Examples:
            >>> handler = manager.get_handler("my_plugin:event_handler:log")
        """
        return self._handler_map.get(signature)

    def get_all_handlers(self) -> Dict[str, BaseEventHandler]:
        """获取所有事件处理器。

        Returns:
            Dict[str, BaseEventHandler]: 处理器映射表

        Examples:
            >>> handlers = manager.get_all_handlers()
        """
        return self._handler_map.copy()

    async def _rebuild_subscription_map(self) -> None:
        """重新构建订阅映射表（内部使用）。"""
        # 清空现有映射表
        self._subscription_map.clear()

        # 重新填充映射表
        for signature, handler in self._handler_map.items():
            subscribed_events = handler.get_subscribed_events()

            for event in subscribed_events:
                if isinstance(event, EventType):
                    # 按权重降序排序
                    self._subscription_map[event].append(
                        (handler, signature)
                    )

        # 对每个事件的处理器列表按权重排序
        for event in self._subscription_map:
            self._subscription_map[event].sort(
                key=lambda x: x[0].weight,
                reverse=True
            )

        logger.debug("订阅映射表已重新构建")

    def get_event_stats(self) -> Dict[str, int]:
        """获取事件统计信息。

        Returns:
            Dict[str, int]: 统计信息，包含处理器数量和事件类型数量

        Examples:
            >>> stats = manager.get_event_stats()
            >>> print(stats["handler_count"])  # 处理器总数
            >>> print(stats["event_type_count"])  # 事件类型总数
        """
        return {
            "handler_count": len(self._handler_map),
            "event_type_count": len(self._subscription_map),
            "total_subscriptions": sum(len(handlers) for handlers in self._subscription_map.values())
        }
