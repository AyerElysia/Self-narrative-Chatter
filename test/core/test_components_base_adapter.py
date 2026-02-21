"""测试 BaseAdapter 类。"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock, MagicMock, patch

import pytest

from src.core.components import BaseAdapter
from src.core.components import BasePlugin


class TestAdapter(BaseAdapter):
    """测试用的适配器实现。"""

    adapter_name = "test_adapter"
    adapter_version = "1.0.0"
    adapter_description = "Test adapter"
    platform = "test_platform"

    async def from_platform_message(self, raw: Any):
        """解析平台消息。"""
        from mofox_wire import MessageEnvelope

        return MessageEnvelope(
            direction="incoming",
            message_info={
                "platform": self.platform,
                "message_id": raw.get("message_id", "test_msg_id"),
                "time": 0.0,
            },
            message_segment=[{"type": "text", "data": raw.get("content", "test content")}],
            raw_message=raw,
        )

    async def _send_platform_message(self, envelope) -> None:
        """发送消息到平台。"""
        # 测试实现
        pass

    # 重写父类方法以避免实际调用
    async def _parent_start(self) -> None:
        """Mock 父类 start。"""
        pass

    async def _parent_stop(self) -> None:
        """Mock 父类 stop。"""
        pass

    def is_connected(self) -> bool:
        """Mock 连接状态。"""
        return True

    async def get_bot_info(self) -> dict:
        """Mock Bot 信息。"""
        return {
            "bot_id": "test_bot",
            "bot_name": "Test Bot",
            "platform": self.platform,
        }


class TestBaseAdapter:
    """测试 BaseAdapter 基类。"""

    def test_adapter_class_attributes(self):
        """测试适配器类属性。"""
        assert TestAdapter.adapter_name == "test_adapter"
        assert TestAdapter.adapter_version == "1.0.0"
        assert TestAdapter.adapter_description == "Test adapter"
        assert TestAdapter.platform == "test_platform"
        assert TestAdapter.dependencies == []

    def test_get_signature_without_plugin_name(self):
        """测试未设置插件名称时获取签名。"""
        signature = TestAdapter.get_signature()
        assert signature is None

    def test_get_signature_with_plugin_name(self):
        """测试设置插件名称后获取签名。"""
        TestAdapter._plugin_ = "test_plugin"
        signature = TestAdapter.get_signature()
        assert signature == "test_plugin:adapter:test_adapter"
        # 重置
        TestAdapter._plugin_ = "unknown_plugin"

    def test_adapter_initialization(self):
        """测试适配器初始化。"""
        mock_sink = MagicMock()
        mock_plugin = MagicMock(spec=BasePlugin)

        adapter = TestAdapter(core_sink=mock_sink, plugin=mock_plugin)

        assert adapter.plugin == mock_plugin
        assert adapter._health_check_task_info is None
        assert adapter._running is False

    @pytest.mark.asyncio
    async def test_adapter_start(self):
        """测试适配器启动。"""
        mock_sink = MagicMock()
        adapter = TestAdapter(core_sink=mock_sink)

        # Mock 父类 start 和 get_task_manager
        with patch("src.kernel.concurrency.task_manager.get_task_manager") as mock_tm:
            mock_task_info = MagicMock()
            mock_task_info.task_id = "test_task_id"
            mock_tm_instance = MagicMock()
            mock_tm_instance.create_task.return_value = mock_task_info
            mock_tm.return_value = mock_tm_instance

            # Mock 父类 start
            with patch("mofox_wire.AdapterBase.start", new_callable=AsyncMock):
                await adapter.start()

                assert adapter._running is True
                assert adapter._health_check_task_info is not None

    @pytest.mark.asyncio
    async def test_adapter_stop(self):
        """测试适配器停止。"""
        mock_sink = MagicMock()
        adapter = TestAdapter(core_sink=mock_sink)
        adapter._running = True
        adapter._health_check_task_info = MagicMock()

        # Mock get_task_manager
        with patch("src.kernel.concurrency.task_manager.get_task_manager") as mock_tm:
            mock_tm_instance = MagicMock()
            mock_tm.return_value = mock_tm_instance

            # Mock 父类 stop
            with patch("mofox_wire.AdapterBase.stop", new_callable=AsyncMock):
                await adapter.stop()

                assert adapter._running is False
                assert adapter._health_check_task_info is None

    @pytest.mark.asyncio
    async def test_on_adapter_loaded_hook(self):
        """测试适配器加载钩子。"""
        mock_sink = MagicMock()
        adapter = TestAdapter(core_sink=mock_sink)

        # 默认实现应该不抛出异常
        await adapter.on_adapter_loaded()

    @pytest.mark.asyncio
    async def test_on_adapter_unloaded_hook(self):
        """测试适配器卸载钩子。"""
        mock_sink = MagicMock()
        adapter = TestAdapter(core_sink=mock_sink)

        # 默认实现应该不抛出异常
        await adapter.on_adapter_unloaded()

    @pytest.mark.asyncio
    async def test_health_check_default(self):
        """测试默认健康检查。"""
        mock_sink = MagicMock()
        adapter = TestAdapter(core_sink=mock_sink)

        # Mock is_connected 方法
        adapter.is_connected = Mock(return_value=True)

        result = await adapter.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_loop(self):
        """测试健康检查循环。"""
        mock_sink = MagicMock()
        adapter = TestAdapter(core_sink=mock_sink)
        adapter._running = True

        call_count = [0]

        async def mock_sleep(interval):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise asyncio.CancelledError()

        with patch("asyncio.sleep", mock_sleep):
            # 由于 TestAdapter 重写了 is_connected 返回 True
            # health_check 应该返回 True，不会触发 reconnect
            await adapter._health_check_loop()

        # 测试通过，没有异常

    @pytest.mark.asyncio
    async def test_health_check_loop_triggers_reconnect(self):
        """测试健康检查失败时触发重连。"""
        mock_sink = MagicMock()
        adapter = TestAdapter(core_sink=mock_sink)
        adapter._running = True

        call_count = [0]

        async def mock_sleep(interval):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise asyncio.CancelledError()

        # Mock health_check 返回 False
        with patch("asyncio.sleep", mock_sleep):
            with patch.object(adapter, "health_check", new_callable=AsyncMock, return_value=False):
                with patch.object(adapter, "reconnect", new_callable=AsyncMock) as mock_reconnect:
                    await adapter._health_check_loop()

                    # 确保重连被调用至少一次
                    assert mock_reconnect.call_count >= 1

    @pytest.mark.asyncio
    async def test_reconnect_default(self):
        """测试默认重连逻辑。"""
        mock_sink = MagicMock()
        adapter = TestAdapter(core_sink=mock_sink)

        # Mock stop 和 start 方法
        adapter.stop = AsyncMock()
        adapter.start = AsyncMock()

        await adapter.reconnect()

        adapter.stop.assert_called_once()
        adapter.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_platform_message_not_implemented(self):
        """测试未实现发送消息方法时抛出异常。"""
        mock_sink = MagicMock()

        # 创建一个没有实现 _send_platform_message 的适配器
        class IncompleteAdapter(BaseAdapter):
            adapter_name = "incomplete"
            platform = "test"

            async def from_platform_message(self, raw):
                pass

        adapter = IncompleteAdapter(core_sink=mock_sink)
        mock_envelope = MagicMock()

        with pytest.raises(NotImplementedError):
            await adapter._send_platform_message(mock_envelope)

    @pytest.mark.asyncio
    async def test_send_platform_message_with_transport_config(self):
        """测试有传输配置时发送消息。"""
        mock_sink = MagicMock()
        adapter = TestAdapter(core_sink=mock_sink)

        # 设置传输配置（使用 type: ignore 绕过严格类型检查）
        adapter._transport_config = {"test": "config"}  # type: ignore[assignment]

        # 由于 TestAdapter 实现了 _send_platform_message，这里不会抛出异常
        mock_envelope = MagicMock()
        await adapter._send_platform_message(mock_envelope)
        # 测试通过，没有抛出异常


class CustomAdapterWithHooks(TestAdapter):
    """带有自定义钩子的测试适配器。"""

    loaded_called = False
    unloaded_called = False

    async def on_adapter_loaded(self) -> None:
        """自定义加载钩子。"""
        CustomAdapterWithHooks.loaded_called = True
        await super().on_adapter_loaded()

    async def on_adapter_unloaded(self) -> None:
        """自定义卸载钩子。"""
        CustomAdapterWithHooks.unloaded_called = True
        await super().on_adapter_unloaded()


class TestAdapterHooks:
    """测试适配器生命周期钩子。"""

    @pytest.mark.asyncio
    async def test_custom_on_adapter_loaded(self):
        """测试自定义加载钩子被调用。"""
        mock_sink = MagicMock()
        adapter = CustomAdapterWithHooks(core_sink=mock_sink)

        CustomAdapterWithHooks.loaded_called = False

        await adapter.on_adapter_loaded()

        assert CustomAdapterWithHooks.loaded_called is True

    @pytest.mark.asyncio
    async def test_custom_on_adapter_unloaded(self):
        """测试自定义卸载钩子被调用。"""
        mock_sink = MagicMock()
        adapter = CustomAdapterWithHooks(core_sink=mock_sink)

        CustomAdapterWithHooks.unloaded_called = False

        await adapter.on_adapter_unloaded()

        assert CustomAdapterWithHooks.unloaded_called is True


class CustomAdapterWithHealthCheck(TestAdapter):
    """带有自定义健康检查的测试适配器。"""

    async def health_check(self) -> bool:
        """自定义健康检查。"""
        # 模拟检查连接状态
        return True


class TestAdapterHealthCheck:
    """测试适配器健康检查功能。"""

    @pytest.mark.asyncio
    async def test_custom_health_check(self):
        """测试自定义健康检查方法。"""
        mock_sink = MagicMock()
        adapter = CustomAdapterWithHealthCheck(core_sink=mock_sink)

        result = await adapter.health_check()
        assert result is True


class CustomAdapterWithReconnect(TestAdapter):
    """带有自定义重连逻辑的测试适配器。"""

    reconnect_called = False

    async def reconnect(self) -> None:
        """自定义重连逻辑。"""
        CustomAdapterWithReconnect.reconnect_called = True
        await super().reconnect()


class TestAdapterReconnect:
    """测试适配器重连功能。"""

    @pytest.mark.asyncio
    async def test_custom_reconnect(self):
        """测试自定义重连方法。"""
        mock_sink = MagicMock()
        adapter = CustomAdapterWithReconnect(core_sink=mock_sink)

        # Mock stop 和 start
        adapter.stop = AsyncMock()
        adapter.start = AsyncMock()

        CustomAdapterWithReconnect.reconnect_called = False

        await adapter.reconnect()

        assert CustomAdapterWithReconnect.reconnect_called is True
        adapter.stop.assert_called_once()
        adapter.start.assert_called_once()


# ---------------------------------------------------------------------------
# 适配器命令 (send_adapter_command / get_bot_info) 相关测试
# ---------------------------------------------------------------------------


class CustomAdapterWithCommand(TestAdapter):
    """带有自定义命令处理的测试适配器。"""

    async def send_adapter_command(
        self, command_name: str, command_data: dict
    ) -> dict:
        """自定义命令处理。"""
        if command_name == "echo":
            return {"status": "ok", "data": command_data, "message": "echo success"}
        if command_name == "fail":
            return {"status": "failed", "message": "command failed", "data": None}
        # 其他命令交由基类处理
        return await super().send_adapter_command(command_name, command_data)


class TestAdapterCommand:
    """测试适配器命令功能（send_adapter_command / get_bot_info）。"""

    @pytest.mark.asyncio
    async def test_send_adapter_command_default_returns_error(self):
        """测试基类默认实现对未知命令返回 error 状态。"""
        mock_sink = MagicMock()
        adapter = TestAdapter(core_sink=mock_sink)

        result = await adapter.send_adapter_command("unknown_cmd", {})

        assert result["status"] == "error"
        assert "unknown_cmd" in result["message"]

    @pytest.mark.asyncio
    async def test_send_adapter_command_returns_dict(self):
        """测试 send_adapter_command 始终返回字典。"""
        mock_sink = MagicMock()
        adapter = TestAdapter(core_sink=mock_sink)

        result = await adapter.send_adapter_command("any_cmd", {"key": "value"})

        assert isinstance(result, dict)
        assert "status" in result
        assert "message" in result

    @pytest.mark.asyncio
    async def test_send_adapter_command_custom_echo(self):
        """测试自定义命令处理器——echo 命令。"""
        mock_sink = MagicMock()
        adapter = CustomAdapterWithCommand(core_sink=mock_sink)

        payload = {"text": "hello"}
        result = await adapter.send_adapter_command("echo", payload)

        assert result["status"] == "ok"
        assert result["data"] == payload

    @pytest.mark.asyncio
    async def test_send_adapter_command_custom_fail(self):
        """测试自定义命令处理器——fail 命令返回 failed 状态。"""
        mock_sink = MagicMock()
        adapter = CustomAdapterWithCommand(core_sink=mock_sink)

        result = await adapter.send_adapter_command("fail", {})

        assert result["status"] == "failed"
        assert result["data"] is None

    @pytest.mark.asyncio
    async def test_send_adapter_command_custom_falls_back_to_base(self):
        """测试自定义命令处理器中未知命令回退到基类。"""
        mock_sink = MagicMock()
        adapter = CustomAdapterWithCommand(core_sink=mock_sink)

        result = await adapter.send_adapter_command("not_handled", {})

        assert result["status"] == "error"
        assert "not_handled" in result["message"]

    @pytest.mark.asyncio
    async def test_get_bot_info_returns_dict(self):
        """测试 get_bot_info 返回包含必要字段的字典。"""
        mock_sink = MagicMock()
        adapter = TestAdapter(core_sink=mock_sink)

        result = await adapter.get_bot_info()

        assert isinstance(result, dict)
        assert "bot_id" in result
        assert "bot_name" in result
        assert "platform" in result

    @pytest.mark.asyncio
    async def test_get_bot_info_platform_matches(self):
        """测试 get_bot_info 返回的平台与适配器平台一致。"""
        mock_sink = MagicMock()
        adapter = TestAdapter(core_sink=mock_sink)

        result = await adapter.get_bot_info()

        assert result["platform"] == TestAdapter.platform

    @pytest.mark.asyncio
    async def test_send_adapter_command_with_empty_data(self):
        """测试发送空参数的命令。"""
        mock_sink = MagicMock()
        adapter = TestAdapter(core_sink=mock_sink)

        result = await adapter.send_adapter_command("some_command", {})

        assert isinstance(result, dict)
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_send_adapter_command_with_complex_data(self):
        """测试发送复杂参数的命令。"""
        mock_sink = MagicMock()
        adapter = CustomAdapterWithCommand(core_sink=mock_sink)

        payload = {"nested": {"a": 1, "b": [1, 2, 3]}, "flag": True}
        result = await adapter.send_adapter_command("echo", payload)

        assert result["status"] == "ok"
        assert result["data"] == payload
