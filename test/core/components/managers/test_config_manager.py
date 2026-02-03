"""Tests for core/components/managers/config_manager.py."""

from __future__ import annotations

from unittest.mock import patch


from src.core.components.base.config import BaseConfig
from src.kernel.config import config_section, Field, SectionBase
from src.core.components.managers.config_manager import ConfigManager


class TestConfig(BaseConfig):
    """Test configuration class."""

    config_name: str = "test_config"
    config_description: str = "Test configuration"

    @config_section("general")
    class GeneralSection(SectionBase):
        enabled: bool = Field(default=True, description="Enable feature")
        version: str = Field(default="1.0.0", description="Version")

    general: GeneralSection = Field(default_factory=GeneralSection)


class TestConfigManager:
    """Test cases for ConfigManager class."""

    def setup_method(self) -> None:
        """Reset manager before each test."""
        self.manager = ConfigManager()

    def test_manager_initialization(self) -> None:
        """Test manager initialization."""
        manager = ConfigManager()
        assert manager._configs == {}
        assert isinstance(manager._configs, dict)

    def test_load_config_new_plugin(self) -> None:
        """Test loading configuration for a new plugin."""
        # Create a mock config instance
        mock_config = TestConfig()
        mock_config.general.enabled = False

        # Mock the load_for_plugin method
        with patch.object(TestConfig, 'load_for_plugin') as mock_load_for_plugin:
            mock_load_for_plugin.return_value = mock_config

            # Load config
            config = self.manager.load_config("test_plugin", TestConfig)

            # Verify
            mock_load_for_plugin.assert_called_once_with("test_plugin", auto_generate=True, auto_update=True)
            assert config is mock_config
            assert config.general.enabled is False
            assert "test_plugin" in self.manager._configs
            assert self.manager._configs["test_plugin"] is config

    def test_load_config_cached(self) -> None:
        """Test loading configuration when already cached."""
        # Setup cached config
        cached_config = TestConfig()
        cached_config.general.enabled = True
        self.manager._configs["test_plugin"] = cached_config

        # Load config again
        config = self.manager.load_config("test_plugin", TestConfig)

        # Should return cached instance
        assert config is cached_config
        assert config.general.enabled is True

    def test_load_config_with_auto_generate(self) -> None:
        """Test loading configuration with auto_generate=True."""
        # Create a mock config instance
        mock_config = TestConfig()

        # Mock the load_for_plugin method
        with patch.object(TestConfig, 'load_for_plugin') as mock_load_for_plugin:
            mock_load_for_plugin.return_value = mock_config

            # Load config with auto_generate=True
            config = self.manager.load_config("test_plugin", TestConfig, auto_generate=True)

            # Verify
            mock_load_for_plugin.assert_called_once_with("test_plugin", auto_generate=True, auto_update=True)
            assert config is mock_config

    def test_load_config_without_auto_generate(self) -> None:
        """Test loading configuration with auto_generate=False."""
        # Create a mock config instance
        mock_config = TestConfig()

        # Mock the load_for_plugin method
        with patch.object(TestConfig, 'load_for_plugin') as mock_load_for_plugin:
            mock_load_for_plugin.return_value = mock_config

            # Load config with auto_generate=False
            config = self.manager.load_config("test_plugin", TestConfig, auto_generate=False)

            # Verify
            mock_load_for_plugin.assert_called_once_with("test_plugin", auto_generate=False, auto_update=True)
            assert config is mock_config

    def test_reload_config(self) -> None:
        """Test reloading configuration."""
        # Setup initial config
        initial_config = TestConfig()
        initial_config.general.enabled = True
        self.manager._configs["test_plugin"] = initial_config

        # Create a reloaded config instance
        reloaded_config = TestConfig()
        reloaded_config.general.enabled = False

        # Mock the reload method
        with patch.object(TestConfig, 'reload') as mock_reload:
            mock_reload.return_value = reloaded_config

            # Reload config
            config = self.manager.reload_config("test_plugin", TestConfig)

            # Verify
            mock_reload.assert_called_once()
            assert config is reloaded_config
            assert config.general.enabled is False
            assert "test_plugin" in self.manager._configs
            assert self.manager._configs["test_plugin"] is reloaded_config
            # Original config should be removed
            assert initial_config not in self.manager._configs.values()

    def test_get_config_exists(self) -> None:
        """Test getting existing configuration."""
        config = TestConfig()
        config.general.enabled = True
        self.manager._configs["test_plugin"] = config

        # Get config
        retrieved = self.manager.get_config("test_plugin")

        assert retrieved is config
        assert retrieved.general.enabled is True

    def test_get_config_not_exists(self) -> None:
        """Test getting non-existing configuration."""
        # Get non-existent config
        retrieved = self.manager.get_config("nonexistent_plugin")

        assert retrieved is None

    def test_remove_config_exists(self) -> None:
        """Test removing existing configuration."""
        config = TestConfig()
        self.manager._configs["test_plugin"] = config

        # Remove config
        result = self.manager.remove_config("test_plugin")

        assert result is True
        assert "test_plugin" not in self.manager._configs

    def test_remove_config_not_exists(self) -> None:
        """Test removing non-existing configuration."""
        # Remove non-existent config
        result = self.manager.remove_config("nonexistent_plugin")

        assert result is False
        assert self.manager._configs == {}

    def test_get_loaded_plugins_empty(self) -> None:
        """Test getting loaded plugins when none are loaded."""
        plugins = self.manager.get_loaded_plugins()

        assert plugins == []

    def test_get_loaded_plugins_multiple(self) -> None:
        """Test getting loaded plugins when multiple are loaded."""
        config1 = TestConfig()
        config2 = TestConfig()

        self.manager._configs["plugin1"] = config1
        self.manager._configs["plugin2"] = config2

        plugins = self.manager.get_loaded_plugins()

        assert set(plugins) == {"plugin1", "plugin2"}

    def test_load_config_integration_with_baseconfig(self) -> None:
        """Test integration with BaseConfig functionality."""
        # Mock the load_for_plugin method to return a real config
        mock_config = TestConfig()
        with patch.object(TestConfig, 'load_for_plugin') as mock_load_for_plugin:
            mock_load_for_plugin.return_value = mock_config

            # Load config
            config = self.manager.load_config("integration_test", TestConfig)

            # Verify config is properly initialized
            assert isinstance(config, TestConfig)
            assert config.config_name == "test_config"
            assert config.config_description == "Test configuration"
            assert isinstance(config.general, TestConfig.GeneralSection)
            assert config.general.enabled is True  # default value
            assert config.general.version == "1.0.0"  # default value

    def test_load_config_with_custom_values(self) -> None:
        """Test loading configuration with custom field values."""
        # Create a mock config with custom values
        mock_config = TestConfig()
        mock_config.general.enabled = False
        mock_config.general.version = "2.0.0"

        # Mock the load_for_plugin method
        with patch.object(TestConfig, 'load_for_plugin') as mock_load_for_plugin:
            mock_load_for_plugin.return_value = mock_config

            # Load config
            config = self.manager.load_config("custom_plugin", TestConfig)

            # Verify custom values
            assert config.general.enabled is False
            assert config.general.version == "2.0.0"