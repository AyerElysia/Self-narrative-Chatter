"""测试 src.core.components.base.collection 模块。"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.components.base.collection import BaseCollection
from src.core.components.types import ChatType


class ConcreteCollection(BaseCollection):
    """具体的 Collection 实现用于测试。"""

    collection_name = "test_collection"
    collection_description = "Test collection"
    associated_platforms = []
    chatter_allow = []
    chat_type = ChatType.ALL

    async def get_contents(self) -> list[str]:
        """获取集合内容。"""
        return [
            "plugin1:action:action1",
            "plugin1:tool:tool1",
            "plugin1:collection:collection1",
        ]


class TestBaseCollection:
    """测试 BaseCollection 类。"""

    @pytest.fixture(autouse=True)
    def reset_class_attributes(self):
        """在每个测试前重置类属性。"""
        # 备份原始值
        original_plugin_name = ConcreteCollection.plugin_name
        yield
        # 恢复原始值
        ConcreteCollection.plugin_name = original_plugin_name

    def test_collection_initialization(self, mock_plugin):
        """测试 Collection 初始化。"""
        collection = ConcreteCollection(mock_plugin)
        assert collection.plugin == mock_plugin
        assert collection.collection_name == "test_collection"
        assert collection.collection_description == "Test collection"

    def test_get_signature(self, mock_plugin):
        """测试获取签名。"""
        collection = ConcreteCollection(mock_plugin)
        assert collection.get_signature() is None

        ConcreteCollection.plugin_name = "my_plugin"
        collection2 = ConcreteCollection(mock_plugin)
        assert collection2.get_signature() == "my_plugin:collection:test_collection"

    def test_get_contents(self, mock_plugin):
        """测试获取内容。"""
        import asyncio

        collection = ConcreteCollection(mock_plugin)
        contents = asyncio.run(collection.get_contents())

        assert len(contents) == 3
        assert "plugin1:action:action1" in contents
        assert "plugin1:tool:tool1" in contents
        assert "plugin1:collection:collection1" in contents

    def test_to_schema(self, mock_plugin):
        """测试生成 schema。"""
        collection = ConcreteCollection(mock_plugin)
        schema = collection.to_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test_collection"
        assert schema["function"]["description"] == "Test collection"
        assert schema["function"]["parameters"]["type"] == "object"
        assert schema["function"]["parameters"]["properties"] == {}
        assert schema["function"]["parameters"]["required"] == []

    @patch("src.core.managers.collection_manager.get_collection_manager")
    def test_execute_with_signature(self, mock_get_manager, mock_plugin):
        """测试执行 Collection（有签名）。"""
        import asyncio

        # 设置 mock
        mock_manager = MagicMock()
        mock_manager.unpack_collection = AsyncMock(return_value=[
            MagicMock(__signature__="plugin1:action:action1"),
            MagicMock(__signature__="plugin1:tool:tool1"),
        ])
        mock_get_manager.return_value = mock_manager

        ConcreteCollection.plugin_name = "my_plugin"
        collection = ConcreteCollection(mock_plugin)

        success, result = asyncio.run(collection.execute("stream_123"))

        assert success is True
        assert result["collection"] == "my_plugin:collection:test_collection"
        assert result["stream_id"] == "stream_123"
        assert result["components_count"] == 2
        assert "plugin1:action:action1" in result["components"]
        assert "plugin1:tool:tool1" in result["components"]

    @patch("src.core.managers.collection_manager.get_collection_manager")
    def test_execute_recursive(self, mock_get_manager, mock_plugin):
        """测试执行 Collection（递归）。"""
        import asyncio

        mock_manager = MagicMock()
        mock_manager.unpack_collection = AsyncMock(return_value=[
            MagicMock(__signature__="plugin1:action:action1"),
        ])
        mock_get_manager.return_value = mock_manager

        ConcreteCollection.plugin_name = "my_plugin"
        collection = ConcreteCollection(mock_plugin)

        success, result = asyncio.run(collection.execute("stream_123"))

        # 检查是否使用了 recursive=True
        call_kwargs = mock_manager.unpack_collection.call_args[1]
        assert call_kwargs.get("recursive") is True

    def test_execute_no_signature(self, mock_plugin):
        """测试执行 Collection（无签名）。"""
        import asyncio
        from unittest.mock import MagicMock

        class NoSigCollection(BaseCollection):
            collection_name = "no_sig"
            plugin_name = "unknown_plugin"

            async def get_contents(self) -> list[str]:
                return []

        # Create a mock plugin without plugin_name attribute to test signature failure
        mock_plugin_no_name = MagicMock()
        # Deliberately don't set plugin_name, or set it to empty string
        mock_plugin_no_name.plugin_name = ""  # Empty string to trigger signature check failure

        collection = NoSigCollection(mock_plugin_no_name)

        success, result = asyncio.run(collection.execute("stream_123"))

        assert success is False
        assert "error" in result


class TestCollectionAttributes:
    """测试 Collection 类属性。"""

    def test_collection_with_all_attributes(self, mock_plugin):
        """测试带有所有属性的集合。"""
        from src.core.components.types import ChatType

        class FullCollection(BaseCollection):
            collection_name = "full_collection"
            collection_description = "Full collection description"
            associated_platforms = ["telegram", "discord"]
            chatter_allow = ["chatter1"]
            chat_type = ChatType.PRIVATE
            dependencies = ["other_plugin:tool:database"]

            async def get_contents(self) -> list[str]:
                return []

        collection = FullCollection(mock_plugin)
        assert collection.collection_name == "full_collection"
        assert collection.collection_description == "Full collection description"
        assert collection.associated_platforms == ["telegram", "discord"]
        assert collection.chatter_allow == ["chatter1"]
        assert collection.chat_type == ChatType.PRIVATE
        assert collection.dependencies == ["other_plugin:tool:database"]


class TestCollectionWithNestedComponents:
    """测试嵌套组件的 Collection。"""

    def test_collection_with_nested_collection(self, mock_plugin):
        """测试包含嵌套 Collection 的集合。"""
        import asyncio

        class NestedCollection(BaseCollection):
            collection_name = "nested_collection"
            plugin_name = "my_plugin"

            async def get_contents(self) -> list[str]:
                return [
                    "my_plugin:action:action1",
                    "my_plugin:collection:inner_collection",
                ]

        collection = NestedCollection(mock_plugin)
        contents = asyncio.run(collection.get_contents())

        assert len(contents) == 2
        assert any("collection" in item for item in contents)

    def test_empty_collection(self, mock_plugin):
        """测试空集合。"""
        import asyncio

        class EmptyCollection(BaseCollection):
            collection_name = "empty_collection"

            async def get_contents(self) -> list[str]:
                return []

        collection = EmptyCollection(mock_plugin)
        contents = asyncio.run(collection.get_contents())

        assert len(contents) == 0
        assert contents == []

    def test_large_collection(self, mock_plugin):
        """测试大型集合。"""
        import asyncio

        # 创建包含大量组件的集合
        components = [f"plugin{i}:action:action{i}" for i in range(100)]

        class LargeCollection(BaseCollection):
            collection_name = "large_collection"

            async def get_contents(self) -> list[str]:
                return components

        collection = LargeCollection(mock_plugin)
        contents = asyncio.run(collection.get_contents())

        assert len(contents) == 100
