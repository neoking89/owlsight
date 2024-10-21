import pytest
import sys
import re
import os
import json
from unittest.mock import patch, mock_open, MagicMock

sys.path.append("src")
from owlsight.configurations.config_manager import ConfigManager, DottedDict, _prepare_toggle_choices
from owlsight.utils.constants import DEFAULTS, CHOICES

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
        result = config_manager.save("test_config.json")
        assert result, "Save operation should return True on success"
        mock_file.assert_called_once_with("test_config.json", "w")
        mock_file().write.assert_called()

@patch("builtins.open", new_callable=mock_open, read_data=json.dumps(DEFAULTS))
def test_load_config(mock_file, config_manager):
    """Test loading configuration from a file."""
    with patch("os.path.exists", return_value=True):
        result = config_manager.load("test_config.json")
        assert result, "Load operation should return True on success"
        mock_file.assert_called_once_with("test_config.json", "r")
        assert config_manager.get("main.max_retries_on_error") == 3, "Failed to load correct config value"

@patch("os.path.exists", return_value=False)
@patch("owlsight.configurations.config_manager.logger")
def test_load_non_existing_file(mock_logger, mock_exists, config_manager):
    """Test loading a non-existent config file."""
    result = config_manager.load("non_existent_config.json")
    assert not result, "Load operation should return False on failure"
    mock_logger.error.assert_called_with(
        "Cannot load config. Configuration file does not exist: 'non_existent_config.json'"
    )

@patch("os.path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data='{"invalid": "json"')
def test_load_invalid_json(mock_file, mock_exists, config_manager):
    """Test loading an invalid JSON file."""
    result = config_manager.load("invalid_config.json")
    assert not result, "Load operation should return False on invalid JSON"

@patch("os.path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data='{"invalid_key": "value"}')
def test_load_invalid_config(mock_file, mock_exists, config_manager):
    """Test loading a config with invalid keys."""
    result = config_manager.load("invalid_config.json")
    assert not result, "Load operation should return False on invalid config"

def test_validate_config_missing_sections(config_manager):
    """Test config validation with missing keys."""
    invalid_config = {"main": {}}
    expected_missing_sections = set(DEFAULTS.keys()) - set(invalid_config.keys())
    expected_error_message = f"Config misses the following sections: {expected_missing_sections}"
    
    with pytest.raises(KeyError, match=re.escape(expected_error_message)):
        config_manager.validate_config(invalid_config)

def test_validate_config_invalid_keys(config_manager):
    """Test config validation with invalid sections."""
    invalid_config = {**DEFAULTS, "invalid_section": {}}
    expected_invalid_sections = {"invalid_section"}
    expected_error_message = f"Config has the following sections, which are not valid in owlsight: {expected_invalid_sections}"
    
    with pytest.raises(KeyError, match=re.escape(expected_error_message)):
        config_manager.validate_config(invalid_config)

def test_validate_config_invalid_value_type(config_manager):
    """Test config validation with invalid value types."""
    invalid_config = {**DEFAULTS, "main": {**DEFAULTS["main"], "extra_index_url": 5}}
    with pytest.raises(TypeError, match="Invalid type int for key 'main.extra_index_url'. Expected type: str"):
        config_manager.validate_config(invalid_config)

def test_validate_config_invalid_choice(config_manager):
    """Test config validation with invalid choice value."""
    invalid_config = {**DEFAULTS, "main": {**DEFAULTS["main"], "max_retries_on_error": 100}}
    with pytest.raises(ValueError, match="Invalid value 100 for key 'main.max_retries_on_error'. Possible values:"):
        config_manager.validate_config(invalid_config)

def test_dotted_dict():
    """Test DottedDict functionality."""
    dotted = DottedDict({"key": "value", "nested": {"inner_key": "inner_value"}})
    assert dotted.key == "value", "DottedDict failed to retrieve a top-level key"
    assert dotted.nested.inner_key == "inner_value", "DottedDict failed to retrieve a nested key"
    
    # Test case insensitivity
    assert dotted.KEY == "value", "DottedDict should be case-insensitive"
    
    # Test setting and deleting attributes
    dotted.new_key = "new_value"
    assert dotted.new_key == "new_value", "DottedDict failed to set a new key"
    del dotted.new_key
    assert "new_key" not in dotted, "DottedDict failed to delete a key"

def test_config_choices(config_manager):
    """Test the config choices are generated correctly."""
    choices = config_manager.config_choices
    assert "main" in choices, "Main config missing from config choices"
    assert "max_retries_on_error" in choices["main"], "Max retries on error choice missing"
    assert choices["main"]["max_retries_on_error"] == _prepare_toggle_choices(
        DEFAULTS["main"]["max_retries_on_error"], CHOICES["main"]["max_retries_on_error"]
    ), "Invalid toggle choices for max retries on error"

def test_prepare_toggle_choices():
    """Test the _prepare_toggle_choices function."""
    current_val = 3
    possible_vals = [1, 2, 3, 4, 5]
    result = _prepare_toggle_choices(current_val, possible_vals)
    assert result == [3, 4, 5, 1, 2], "Toggle choices not prepared correctly"

    # Test when current_val is not in possible_vals
    current_val = 6
    result = _prepare_toggle_choices(current_val, possible_vals)
    assert result == possible_vals, "Toggle choices should remain unchanged when current_val is not in possible_vals"

if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])