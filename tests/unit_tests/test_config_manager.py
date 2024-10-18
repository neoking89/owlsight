import pytest
import sys
from unittest.mock import patch, mock_open

sys.path.append("src")
from owlsight.configurations.config_manager import ConfigManager, DottedDict, _prepare_toggle_choices


@pytest.fixture
def config_manager():
    """Fixture to return a new instance of ConfigManager."""
    return ConfigManager()


def test_singleton_behavior():
    """Ensure ConfigManager follows the singleton pattern."""
    instance1 = ConfigManager()
    instance2 = ConfigManager()
    assert instance1 is instance2, "ConfigManager is not a singleton!"


def test_get_existing_key(config_manager):
    """Test the retrieval of an existing config key."""
    value = config_manager.get("main.max_retries_on_error")
    assert value == 3, f"Expected 3, got {value}"


def test_get_non_existing_key(config_manager):
    """Test getting a non-existent key."""
    value = config_manager.get("non.existing.key", default="default_value")
    assert value == "default_value", f"Expected 'default_value', got {value}"


def test_set_new_key(config_manager):
    """Test setting a new config key."""
    config_manager.set("new.key", "new_value")
    assert config_manager.get("new.key") == "new_value", "Failed to set new key!"


def test_set_existing_key(config_manager):
    """Test setting an existing config key."""
    config_manager.set("main.max_retries_on_error", 5)
    assert config_manager.get("main.max_retries_on_error") == 5, "Failed to update existing key!"


@patch("builtins.open", new_callable=mock_open)
def test_save_config(mock_file, config_manager):
    """Test saving configuration to a file."""
    with patch("os.path.exists", return_value=True):
        config_manager.save("test_config.json")
        mock_file.assert_called_once_with("test_config.json", "w")
        mock_file().write.assert_called()


@patch("builtins.open", new_callable=mock_open, read_data='{"main": {"max_retries_on_error": 5}}')
def test_load_config(mock_file, config_manager):
    """Test loading configuration from a file."""
    with patch("os.path.exists", return_value=True):
        config_manager.load("test_config.json")
        mock_file.assert_called_once_with("test_config.json", "r")
        assert config_manager.get("main.max_retries_on_error") == 5, "Failed to load config!"


@patch("os.path.exists", return_value=False)
@patch("owlsight.configurations.config_manager.logger")
def test_load_non_existing_file(mock_logger, mock_exists, config_manager):
    """Test loading a non-existent config file."""
    # Call the method that triggers the logging error
    config_manager.load("non_existent_config.json")

    # Assert that logger.error was called with the correct message
    mock_logger.error.assert_called_with(
        "Cannot load config. Configuration file does not exist: 'non_existent_config.json'"
    )

def test_dotted_dict():
    """Test DottedDict functionality."""
    dotted = DottedDict({"key": "value", "nested": {"inner_key": "inner_value"}})
    assert dotted.key == "value", "DottedDict failed to retrieve a top-level key"
    assert dotted.nested.inner_key == "inner_value", "DottedDict failed to retrieve a nested key"


def test_config_choices(config_manager):
    """Test the config choices are generated correctly."""
    choices = config_manager.config_choices
    assert "main" in choices, "Main config missing from config choices"
    assert "max_retries_on_error" in choices["main"], "Max retries on error choice missing"
    assert choices["main"]["max_retries_on_error"] == _prepare_toggle_choices(
        choices["main"]["max_retries_on_error"][0], list(range(0, 10))
    ), "Invalid toggle choices for max retries on error"


if __name__ == "__main__":
    pytest.main([__file__])
