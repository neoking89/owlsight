import pytest
from unittest.mock import MagicMock
from owlsight.processors.text_generation_manager import TextGenerationManager
from owlsight.configurations.config_manager import ConfigManager


@pytest.fixture
def config_manager():
    return ConfigManager()


@pytest.fixture(autouse=True)
def reset_text_generation_manager():
    TextGenerationManager._reset_instance()
    yield


def test_singleton_behavior(config_manager):
    """Test that TextGenerationManager enforces singleton behavior."""
    manager1 = TextGenerationManager(config_manager)
    assert isinstance(manager1, TextGenerationManager)

    with pytest.raises(RuntimeError) as exc_info:
        TextGenerationManager(config_manager)

    assert str(exc_info.value) == "Only one instance of TextGenerationManager can exist at the same time."


def test_initialization(config_manager):
    """Test proper initialization with config manager"""
    manager = TextGenerationManager(config_manager)
    assert manager.config_manager == config_manager
    assert manager.processor is None


def test_processor_management(config_manager):
    """Test core processor lifecycle management"""
    mock_processor = MagicMock()

    # Test initial state
    manager = TextGenerationManager(config_manager)
    assert manager.get_processor() is None

    # Test setting processor
    manager.processor = mock_processor
    assert manager.get_processor() == mock_processor

    # Test clearing processor
    manager.processor = None
    assert manager.get_processor() is None


def test_tool_history_empty(config_manager):
    """Test that tool_history returns an empty list when no tools have been used"""
    manager = TextGenerationManager(config_manager)
    assert manager.tool_history == []


def test_tool_history_with_valid_entries(config_manager):
    """Test tool_history with valid tool entries"""
    manager = TextGenerationManager(config_manager)

    # Add a tool history entry
    test_entry = {"name": "owl_scrape", "arguments": {"urls": ["https://example1.com", "https://example2.com"]}}
    manager._update_tool_history(test_entry["name"], test_entry["arguments"])

    # Verify the entry was added correctly
    history = manager.tool_history
    assert len(history) == 1
    assert history[0] == test_entry


def test_tool_history_with_invalid_entries(config_manager, caplog):
    """Test tool_history handles invalid entries gracefully"""
    manager = TextGenerationManager(config_manager)

    # Add an invalid entry directly to _tool_history
    invalid_entry = "not a valid dict string"
    manager._tool_history.add(invalid_entry)

    # Check that the invalid entry is handled
    history = manager.tool_history
    assert history == []  # Should return empty list for invalid entries


def test_update_tool_history(config_manager):
    """Test _update_tool_history method adds entries correctly"""
    manager = TextGenerationManager(config_manager)

    # Test adding multiple entries
    entries = [("tool1", {"arg1": "value1"}), ("tool2", {"arg2": "value2"})]

    for func_name, arguments in entries:
        manager._update_tool_history(func_name, arguments)

    # Verify all entries were added
    history = manager.tool_history
    assert len(history) == 2
    assert {"name": "tool1", "arguments": {"arg1": "value1"}} in history
    assert {"name": "tool2", "arguments": {"arg2": "value2"}} in history
