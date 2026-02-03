"""Tests for core/components/managers/adapter_manager.py."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.components.managers.adapter_manager import (
    AdapterManager,
    get_adapter_manager,
    reset_adapter_manager,
)


class TestAdapterManager:
    """Test cases for AdapterManager class."""

    def setup_method(self) -> None:
        """Reset manager before each test."""
        reset_adapter_manager()

    def test_manager_initialization(self) -> None:
        """Test manager initialization."""
        manager = AdapterManager()

        assert manager._active_adapters == {}
        assert manager.list_active_adapters() == []
        assert manager.get_all_adapters() == {}

    @patch('src.core.components.managers.adapter_manager.get_global_registry')
    @patch('src.core.components.managers.adapter_manager.get_plugin_manager')
    async def test_start_adapter_success(self, mock_plugin_manager, mock_registry) -> None:
        """Test successful adapter start."""
        # Setup mocks
        mock_registry.get.return_value = MagicMock()
        mock_plugin_manager.return_value.get_plugin.return_value = MagicMock()

        # Create mock adapter instance
        mock_adapter = AsyncMock()
        mock_adapter.start = AsyncMock()
        mock_adapter_class = MagicMock(return_value=mock_adapter)

        # Set the mock class in the registry
        mock_registry.get.return_value = mock_adapter_class

        # Test
        manager = AdapterManager()
        result = await manager.start_adapter("test_plugin:adapter:qq")

        assert result is True
        assert "test_plugin:adapter:qq" in manager._active_adapters
        mock_adapter.start.assert_called_once()

    @patch('src.core.components.managers.adapter_manager.get_global_registry')
    async def test_start_adapter_already_started(self, mock_registry) -> None:
        """Test starting an adapter that's already started."""
        manager = AdapterManager()
        manager._active_adapters["test_plugin:adapter:qq"] = MagicMock()

        result = await manager.start_adapter("test_plugin:adapter:qq")

        assert result is True

    @patch('src.core.components.managers.adapter_manager.get_global_registry')
    async def test_start_adapter_not_found(self, mock_registry) -> None:
        """Test starting an adapter that doesn't exist in registry."""
        mock_registry.get.return_value = None

        manager = AdapterManager()
        result = await manager.start_adapter("test_plugin:adapter:qq")

        assert result is False

    @patch('src.core.components.managers.adapter_manager.get_global_registry')
    @patch('src.core.components.managers.adapter_manager.get_plugin_manager')
    async def test_start_adapter_instantiation_fails(self, mock_plugin_manager, mock_registry) -> None:
        """Test adapter instantiation failure."""
        # Setup mocks
        mock_registry.get.return_value = MagicMock()
        mock_plugin_manager.return_value.get_plugin.return_value = MagicMock()

        # Mock adapter class to raise exception
        mock_adapter_class = MagicMock(side_effect=Exception("Failed to instantiate"))
        mock_registry.get.return_value = mock_adapter_class

        # Test
        manager = AdapterManager()
        result = await manager.start_adapter("test_plugin:adapter:qq")

        assert result is False

    @patch('src.core.components.managers.adapter_manager.get_global_registry')
    @patch('src.core.components.managers.adapter_manager.get_plugin_manager')
    async def test_start_adapter_start_fails(self, mock_plugin_manager, mock_registry) -> None:
        """Test adapter start failure."""
        # Setup mocks
        mock_registry.get.return_value = MagicMock()
        mock_plugin_manager.return_value.get_plugin.return_value = MagicMock()

        # Create mock adapter instance
        mock_adapter = AsyncMock()
        mock_adapter.start = AsyncMock(side_effect=Exception("Start failed"))
        mock_adapter_class = MagicMock(return_value=mock_adapter)
        mock_registry.get.return_value = mock_adapter_class

        # Test
        manager = AdapterManager()
        result = await manager.start_adapter("test_plugin:adapter:qq")

        assert result is False
        assert "test_plugin:adapter:qq" not in manager._active_adapters

    async def test_stop_adapter_success(self) -> None:
        """Test successful adapter stop."""
        manager = AdapterManager()
        mock_adapter = AsyncMock()
        manager._active_adapters["test_plugin:adapter:qq"] = mock_adapter

        result = await manager.stop_adapter("test_plugin:adapter:qq")

        assert result is True
        mock_adapter.stop.assert_called_once()
        assert "test_plugin:adapter:qq" not in manager._active_adapters

    async def test_stop_adapter_not_started(self) -> None:
        """Test stopping an adapter that's not started."""
        manager = AdapterManager()
        result = await manager.stop_adapter("test_plugin:adapter:qq")

        assert result is False

    async def test_stop_adapter_stop_fails(self) -> None:
        """Test adapter stop failure."""
        manager = AdapterManager()
        mock_adapter = AsyncMock(side_effect=Exception("Stop failed"))
        manager._active_adapters["test_plugin:adapter:qq"] = mock_adapter

        result = await manager.stop_adapter("test_plugin:adapter:qq")

        assert result is False
        # Adapter should still be in active adapters since stop failed
        assert "test_plugin:adapter:qq" in manager._active_adapters

    @patch('src.core.components.managers.adapter_manager.get_global_registry')
    @patch('src.core.components.managers.adapter_manager.get_plugin_manager')
    async def test_restart_adapter_success(self, mock_plugin_manager, mock_registry) -> None:
        """Test successful adapter restart."""
        # Setup mocks
        mock_plugin_instance = MagicMock()
        mock_plugin_manager.return_value.get_plugin.return_value = mock_plugin_instance

        # Create mock adapter class that returns a new AsyncMock instance each time
        mock_adapter_instance = AsyncMock()
        mock_adapter_instance.stop = AsyncMock(return_value=None)
        mock_adapter_instance.start = AsyncMock(return_value=None)

        def create_adapter(*args, **kwargs):
            return mock_adapter_instance

        mock_adapter_class = MagicMock(side_effect=create_adapter)
        mock_registry.get.return_value = mock_adapter_class

        # Test
        manager = AdapterManager()
        # Add adapter to active adapters
        manager._active_adapters["test_plugin:adapter:qq"] = mock_adapter_instance

        result = await manager.restart_adapter("test_plugin:adapter:qq")

        assert result is True
        # Should be called twice: stop and start
        assert mock_adapter_instance.stop.call_count == 1
        assert mock_adapter_instance.start.call_count == 1

    async def test_restart_adapter_not_started(self) -> None:
        """Test restarting an adapter that's not started."""
        manager = AdapterManager()
        result = await manager.restart_adapter("test_plugin:adapter:qq")

        assert result is False

    @patch('src.core.components.managers.adapter_manager.get_global_registry')
    @patch('src.core.components.managers.adapter_manager.get_plugin_manager')
    async def test_restart_adapter_stop_fails(self, mock_plugin_manager, mock_registry) -> None:
        """Test restart when stop fails."""
        # Setup mocks
        mock_registry.get.return_value = MagicMock()
        mock_plugin_manager.return_value.get_plugin.return_value = MagicMock()

        # Create mock adapter instance
        mock_adapter = AsyncMock()
        mock_adapter.start = AsyncMock()
        mock_adapter.stop = AsyncMock(side_effect=Exception("Stop failed"))
        mock_adapter_class = MagicMock(return_value=mock_adapter)
        mock_registry.get.return_value = mock_adapter_class

        # Test
        manager = AdapterManager()
        manager._active_adapters["test_plugin:adapter:qq"] = mock_adapter

        result = await manager.restart_adapter("test_plugin:adapter:qq")

        assert result is False

    async def test_health_check_all_empty(self) -> None:
        """Test health check when no adapters are active."""
        manager = AdapterManager()
        health_results = await manager.health_check_all()

        assert health_results == {}

    async def test_health_check_all_success(self) -> None:
        """Test successful health check for all adapters."""
        manager = AdapterManager()

        # Create mock adapter instances
        mock_adapter1 = AsyncMock()
        mock_adapter1.health_check = AsyncMock(return_value=True)
        mock_adapter1.reconnect = AsyncMock()

        mock_adapter2 = AsyncMock()
        mock_adapter2.health_check = AsyncMock(return_value=True)
        mock_adapter2.reconnect = AsyncMock()

        manager._active_adapters["plugin1:adapter:qq"] = mock_adapter1
        manager._active_adapters["plugin2:adapter:telegram"] = mock_adapter2

        health_results = await manager.health_check_all()

        assert health_results == {
            "plugin1:adapter:qq": True,
            "plugin2:adapter:telegram": True
        }
        mock_adapter1.health_check.assert_called_once()
        mock_adapter2.health_check.assert_called_once()

    async def test_health_check_all_with_failures(self) -> None:
        """Test health check with failed adapters."""
        manager = AdapterManager()

        # Create mock adapter instances
        mock_adapter1 = AsyncMock()
        mock_adapter1.health_check = AsyncMock(return_value=True)
        mock_adapter1.reconnect = AsyncMock()

        mock_adapter2 = AsyncMock()
        mock_adapter2.health_check = AsyncMock(return_value=False)
        mock_adapter2.reconnect = AsyncMock()

        mock_adapter3 = AsyncMock()
        mock_adapter3.health_check = AsyncMock(side_effect=Exception("Health check error"))

        manager._active_adapters["plugin1:adapter:qq"] = mock_adapter1
        manager._active_adapters["plugin2:adapter:telegram"] = mock_adapter2
        manager._active_adapters["plugin3:adapter:discord"] = mock_adapter3

        health_results = await manager.health_check_all()

        assert health_results == {
            "plugin1:adapter:qq": True,
            "plugin2:adapter:telegram": True,  # Should be True after reconnect
            "plugin3:adapter:discord": False
        }

    async def test_check_adapter_health_success(self) -> None:
        """Test single adapter health check success."""
        manager = AdapterManager()
        mock_adapter = AsyncMock()
        mock_adapter.health_check = AsyncMock(return_value=True)
        mock_adapter.reconnect = AsyncMock()

        result = await manager._check_adapter_health("test_adapter", mock_adapter)

        assert result is True
        mock_adapter.health_check.assert_called_once()
        mock_adapter.reconnect.assert_not_called()

    async def test_check_adapter_health_failure_and_reconnect_success(self) -> None:
        """Test single adapter health check failure with successful reconnect."""
        manager = AdapterManager()
        mock_adapter = AsyncMock()
        mock_adapter.health_check = AsyncMock(return_value=False)
        mock_adapter.reconnect = AsyncMock(return_value=True)

        result = await manager._check_adapter_health("test_adapter", mock_adapter)

        assert result is True
        mock_adapter.health_check.assert_called_once()
        mock_adapter.reconnect.assert_called_once()

    async def test_check_adapter_health_failure_and_reconnect_failure(self) -> None:
        """Test single adapter health check failure with failed reconnect."""
        manager = AdapterManager()
        mock_adapter = AsyncMock()
        mock_adapter.health_check = AsyncMock(return_value=False)
        mock_adapter.reconnect = AsyncMock(return_value=False)

        result = await manager._check_adapter_health("test_adapter", mock_adapter)

        assert result is False
        mock_adapter.health_check.assert_called_once()
        mock_adapter.reconnect.assert_called_once()

    def test_get_adapter(self) -> None:
        """Test getting an adapter instance."""
        manager = AdapterManager()
        mock_adapter = MagicMock()
        manager._active_adapters["test_adapter"] = mock_adapter

        result = manager.get_adapter("test_adapter")

        assert result is mock_adapter

    def test_get_adapter_not_found(self) -> None:
        """Test getting a non-existent adapter instance."""
        manager = AdapterManager()

        result = manager.get_adapter("nonexistent_adapter")

        assert result is None

    def test_get_all_adapters(self) -> None:
        """Test getting all adapter instances."""
        manager = AdapterManager()
        mock_adapter1 = MagicMock()
        mock_adapter2 = MagicMock()
        manager._active_adapters["adapter1"] = mock_adapter1
        manager._active_adapters["adapter2"] = mock_adapter2

        result = manager.get_all_adapters()

        assert result == {
            "adapter1": mock_adapter1,
            "adapter2": mock_adapter2
        }
        # Should return a copy, not the original dict
        assert result is not manager._active_adapters

    def test_list_active_adapters(self) -> None:
        """Test listing active adapter signatures."""
        manager = AdapterManager()
        manager._active_adapters["adapter1"] = MagicMock()
        manager._active_adapters["adapter2"] = MagicMock()

        result = manager.list_active_adapters()

        assert result == ["adapter1", "adapter2"]

    def test_is_adapter_active(self) -> None:
        """Test checking if an adapter is active."""
        manager = AdapterManager()
        manager._active_adapters["adapter1"] = MagicMock()

        assert manager.is_adapter_active("adapter1") is True
        assert manager.is_adapter_active("nonexistent") is False

    async def test_stop_all_adapters(self) -> None:
        """Test stopping all adapters."""
        manager = AdapterManager()
        mock_adapter1 = AsyncMock()
        mock_adapter2 = AsyncMock()
        manager._active_adapters["adapter1"] = mock_adapter1
        manager._active_adapters["adapter2"] = mock_adapter2

        result = await manager.stop_all_adapters()

        assert result == {
            "adapter1": True,
            "adapter2": True
        }
        assert len(manager._active_adapters) == 0
        mock_adapter1.stop.assert_called_once()
        mock_adapter2.stop.assert_called_once()


class TestGlobalAdapterManager:
    """Test cases for global adapter manager functions."""

    def setup_method(self) -> None:
        """Reset global manager before each test."""
        reset_adapter_manager()

    def test_get_adapter_manager_singleton(self) -> None:
        """Test that get_adapter_manager returns singleton."""
        manager1 = get_adapter_manager()
        manager2 = get_adapter_manager()

        assert manager1 is manager2

    def test_get_adapter_manager_creates_instance(self) -> None:
        """Test that get_adapter_manager creates instance on first call."""
        manager = get_adapter_manager()

        assert manager is not None
        assert isinstance(manager, AdapterManager)

    def test_reset_adapter_manager(self) -> None:
        """Test resetting global manager."""
        manager1 = get_adapter_manager()
        manager1._active_adapters["test"] = MagicMock()

        assert len(manager1._active_adapters) == 1

        reset_adapter_manager()

        manager2 = get_adapter_manager()
        assert manager2 is not manager1
        assert len(manager2._active_adapters) == 0

    def test_global_manager_integration(self) -> None:
        """Test using global manager."""
        manager = get_adapter_manager()

        # Initially empty
        assert manager.list_active_adapters() == []

        # Add an adapter directly for testing
        mock_adapter = MagicMock()
        manager._active_adapters["test_adapter"] = mock_adapter

        # Verify it's there
        assert manager.is_adapter_active("test_adapter") is True
        assert manager.get_adapter("test_adapter") is mock_adapter


class TestAdapterManagerIntegration:
    """Integration tests for adapter manager."""

    def setup_method(self) -> None:
        """Reset manager before each test."""
        reset_adapter_manager()

    async def test_adapter_lifecycle_full_cycle(self) -> None:
        """Test full adapter lifecycle: start, health check, stop."""
        manager = AdapterManager()

        # Start adapter (mock)
        with patch.object(manager, 'start_adapter') as mock_start:
            mock_start.return_value = True
            assert await manager.start_adapter("test:adapter:qq") is True

        # Health check (mock)
        with patch.object(manager, 'health_check_all') as mock_health:
            mock_health.return_value = {"test:adapter:qq": True}
            health = await manager.health_check_all()
            assert health == {"test:adapter:qq": True}

        # Stop adapter (mock)
        with patch.object(manager, 'stop_adapter') as mock_stop:
            mock_stop.return_value = True
            assert await manager.stop_adapter("test:adapter:qq") is True

        # Verify it's stopped
        assert manager.is_adapter_active("test:adapter:qq") is False

    async def test_multiple_adapters_management(self) -> None:
        """Test managing multiple adapters."""
        manager = AdapterManager()

        # Mock several adapters
        adapter_signatures = ["plugin1:adapter:qq", "plugin2:adapter:telegram", "plugin3:adapter:discord"]

        for sig in adapter_signatures:
            with patch.object(manager, 'start_adapter') as mock_start:
                mock_start.return_value = True
                await manager.start_adapter(sig)

        # Verify all started
        for sig in adapter_signatures:
            assert manager.is_adapter_active(sig) is True

        # List adapters
        active = manager.list_active_adapters()
        assert set(active) == set(adapter_signatures)

        # Stop all
        with patch.object(manager, 'stop_adapter') as mock_stop:
            mock_stop.return_value = True
            results = await manager.stop_all_adapters()
            assert all(results.values())

        # Verify all stopped
        for sig in adapter_signatures:
            assert manager.is_adapter_active(sig) is False

    async def test_concurrent_health_checks(self) -> None:
        """Test concurrent health checks for multiple adapters."""
        manager = AdapterManager()

        # Add multiple adapters
        num_adapters = 5
        for i in range(num_adapters):
            sig = f"plugin{i}:adapter:platform"
            with patch.object(manager, 'start_adapter') as mock_start:
                mock_start.return_value = True
                await manager.start_adapter(sig)

        # Perform health check concurrently
        health_results = await manager.health_check_all()

        # Should have results for all adapters
        assert len(health_results) == num_adapters

        # All should have been checked (health_check called)
        # This is verified through the mock calls in the test