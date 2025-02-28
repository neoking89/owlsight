import pytest
import re
import json
from copy import deepcopy
from unittest.mock import patch, mock_open

from owlsight.configurations.constants import CONFIG_DEFAULTS
from owlsight.configurations.config_manager import (
    ConfigManager,
    DottedDict,
    _prepare_toggle_choices,
    _remove_keys_from_config,
)
from owlsight.utils.helper_functions import flatten_dict
from owlsight.configurations.schema import Schema
from owlsight.ui.custom_classes import OptionType


@pytest.fixture(scope="function", autouse=True)
def defaults():
    """Fixture to reset CONFIG_DEFAULTS after each test."""
    yield Schema.get_config_defaults()


@pytest.fixture(scope="function", autouse=True)
def choices():
    """Fixture to return the choices."""
    yield Schema.get_config_choices()


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


@patch("builtins.open", new_callable=mock_open, read_data=json.dumps(CONFIG_DEFAULTS))
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
    expected_missing_sections = set(CONFIG_DEFAULTS.keys()) - set(invalid_config.keys())
    expected_error_message = f"Config misses the following sections: {expected_missing_sections}"

    with pytest.raises(KeyError, match=re.escape(expected_error_message)):
        config_manager.validate_config(invalid_config)


def test_validate_config_invalid_keys(config_manager, defaults):
    """Test config validation with invalid sections."""
    invalid_config = {**defaults, "invalid_section": {}}
    expected_invalid_sections = {"invalid_section"}
    expected_error_message = (
        f"Config has the following sections, which are not valid in owlsight: {expected_invalid_sections}"
    )

    with pytest.raises(KeyError, match=re.escape(expected_error_message)):
        config_manager.validate_config(invalid_config)


def test_validate_config_invalid_value_type(config_manager, defaults):
    """Test config validation with invalid value types."""
    invalid_config = {**defaults, "main": {**defaults["main"], "extra_index_url": 5}}
    with pytest.raises(TypeError, match="Invalid type int for key 'main.extra_index_url'. Expected type: str"):
        config_manager.validate_config(invalid_config)


def test_validate_config_invalid_choice(config_manager, defaults):
    """Test config validation with invalid choice value."""
    invalid_config = {**defaults, "main": {**defaults["main"], "max_retries_on_error": 100}}
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


def test_config_choices(config_manager, defaults, choices):
    """Test the config choices are generated correctly."""
    choices = config_manager.config_choices
    assert "main" in choices, "Main config missing from config choices"
    assert "max_retries_on_error" in choices["main"], "Max retries on error choice missing"
    assert choices["main"]["max_retries_on_error"] == _prepare_toggle_choices(
        defaults["main"]["max_retries_on_error"], choices["main"]["max_retries_on_error"]
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


def test_copy_defaults_does_not_modify_original(defaults):
    """Test that the CONFIG_DEFAULTS copy does not modify the original CONFIG_DEFAULTS."""
    assert defaults["main"]["max_retries_on_error"] == 3, "CONFIG_DEFAULTS should have the right default value"
    defaults_copy = deepcopy(defaults)
    defaults_copy["main"]["max_retries_on_error"] = 5
    assert defaults["main"]["max_retries_on_error"] != 5, "CONFIG_DEFAULTS should not be modified"


def test_defaults_variable_should_not_be_modified(config_manager, defaults):
    """Test that the CONFIG_DEFAULTS variable is not modified by the ConfigManager."""
    config_manager.set("main.max_retries_on_error", 5)
    assert defaults["main"]["max_retries_on_error"] != 5, "ConfigManager should not modify CONFIG_DEFAULTS"


def test_config_choices_should_return_right_types(config_manager):
    """Test that the ConfigManager returns the right types."""
    type_choices = (list, str, type(None))
    for section, choices in config_manager.config_choices.items():
        for key, value in choices.items():
            assert type(value) in type_choices, f"In key '{key}', Expected type {type_choices}, got {type(value)}"


def test_remove_action_optiontypes_empty_config():
    """Test removing action types from an empty config."""
    config = {}
    config_types = {"section1": {"key1": OptionType.ACTION}}
    result = _remove_keys_from_config(config, config_types)
    assert result == {"section1": {}}, "Empty config should result in empty sections"


def test_remove_action_optiontypes_no_action_types():
    """Test when there are no action types to remove."""
    config = {"section1": {"key1": "value1", "key2": "value2"}, "section2": {"key3": "value3"}}
    config_types = {
        "section1": {"key1": OptionType.TOGGLE, "key2": OptionType.EDITABLE},
        "section2": {"key3": OptionType.TOGGLE},
    }
    result = _remove_keys_from_config(config, config_types)
    assert result == config, "Config without action types should remain unchanged"


def test_remove_action_optiontypes_with_action_types():
    """Test removing action type items from config."""
    config = {
        "section1": {"key1": "value1", "key2": "value2", "key3": "value3"},
        "section2": {"key4": "value4", "key5": "value5"},
    }
    config_types = {
        "section1": {"key1": OptionType.ACTION, "key2": OptionType.TOGGLE, "key3": OptionType.ACTION},
        "section2": {"key4": OptionType.EDITABLE, "key5": OptionType.ACTION},
    }
    expected = {"section1": {"key2": "value2"}, "section2": {"key4": "value4"}}
    result = _remove_keys_from_config(config, config_types)
    assert result == expected, "Action type items should be removed"


def test_remove_action_optiontypes_missing_sections():
    """Test handling of sections present in config but not in config_types."""
    config = {"section1": {"key1": "value1"}, "section2": {"key2": "value2"}, "section3": {"key3": "value3"}}
    config_types = {"section1": {"key1": OptionType.TOGGLE}, "section2": {"key2": OptionType.ACTION}}
    expected = {"section1": {"key1": "value1"}, "section2": {}}
    result = _remove_keys_from_config(config, config_types)
    assert result == expected, "Should only process sections defined in config_types"


def test_remove_action_optiontypes_missing_keys():
    """Test handling of keys present in config but not in config_types."""
    config = {"section1": {"key1": "value1", "key2": "value2", "key3": "value3"}}
    config_types = {"section1": {"key1": OptionType.ACTION, "key2": OptionType.TOGGLE}}
    expected = {"section1": {"key2": "value2"}}
    result = _remove_keys_from_config(config, config_types)
    assert result == expected, "Should only process keys defined in config_types"


def test_remove_action_optiontypes_real_config():
    """Test with actual config structure from Schema."""
    config = Schema.get_config_defaults()
    config_types = Schema.get_config_types()
    result = _remove_keys_from_config(config, config_types)

    # Verify no ACTION type items remain
    for section, items in result.items():
        for key in items.keys():
            assert config_types[section][key] != OptionType.ACTION, f"Found ACTION type in {section}.{key}"

    # Verify structure is maintained
    assert isinstance(result, dict), "Result should be a dictionary"
    assert all(isinstance(v, dict) for v in result.values()), "All sections should be dictionaries"


def test_excluded_keys():
    """Test that excluded keys are properly handled during save and load operations."""
    config_manager = ConfigManager()

    # Set up test data
    test_config = {
        "main": {"default_config_on_startup": "test", "other_setting": "value"},
        "other": {"setting": "value"},
    }

    # Apply test config
    config_manager._config = test_config

    # flatten config as is the case in validate_config function
    test_config = flatten_dict(test_config)

    # Test removing excluded keys
    filtered = config_manager._remove_excluded_keys(deepcopy(test_config))

    # Verify excluded key is removed
    assert "main.default_config_on_startup" not in filtered
    # Verify other keys remain
    assert filtered["main.other_setting"] == "value"
    assert filtered["other.setting"] == "value"

    # Verify original config is unchanged
    assert "main.default_config_on_startup" in test_config
