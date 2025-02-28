import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from owlsight.app.run_app import _extract_params_chain_tag, CommandResult, clear_history


@pytest.fixture(autouse=True)
def mock_logger():
    with patch("owlsight.app.run_app.logger") as mock:
        yield mock


@pytest.fixture
def mock_code_executor():
    class MockCodeExecutor:
        def __init__(self):
            self.globals_dict = {"owl_var": 42, "regular_var": "test", "another_var": [1, 2, 3]}

    return MockCodeExecutor()


@pytest.fixture
def mock_text_generation_manager():
    class MockProcessor:
        def __init__(self):
            self.chat_history = ["message1", "message2"]

    class MockManager:
        def __init__(self):
            self.processor = MockProcessor()
            self._tool_history = set(["tool1", "tool2"])

    return MockManager()


def test_extract_params_chain_tag_valid(mock_logger):
    """Test _extract_params_chain_tag with valid input."""
    # Test basic case
    key, value = _extract_params_chain_tag("model=gpt4")
    assert key == "model"
    assert value == "gpt4"
    mock_logger.error.assert_not_called()

    # Test with spaces
    key, value = _extract_params_chain_tag("  temperature = 0.7  ")
    assert key == "temperature"
    assert value == "0.7"
    mock_logger.error.assert_not_called()

    # Test with special characters
    key, value = _extract_params_chain_tag("path=/usr/local/bin")
    assert key == "path"
    assert value == "/usr/local/bin"
    mock_logger.error.assert_not_called()


def test_extract_params_chain_tag_invalid(mock_logger):
    """Test _extract_params_chain_tag with invalid input."""
    # Test missing equals sign
    key, value = _extract_params_chain_tag("invalid_param")
    assert key == ""
    assert value == ""
    mock_logger.error.assert_called_once()
    mock_logger.error.reset_mock()

    # Test empty string
    key, value = _extract_params_chain_tag("")
    assert key == ""
    assert value == ""
    mock_logger.error.assert_called_once()
    mock_logger.error.reset_mock()

    # Test multiple equals signs (should only split on first one)
    key, value = _extract_params_chain_tag("key=value=extra")
    assert key == ""
    assert value == ""
    mock_logger.error.assert_called_once()
    mock_logger.error.reset_mock()


def test_command_result_enum():
    """Test CommandResult enum values."""
    # Test that all expected values exist
    assert hasattr(CommandResult, "CONTINUE")
    assert hasattr(CommandResult, "BREAK")
    assert hasattr(CommandResult, "PROCEED")

    # Test that values are unique
    values = [member.value for member in CommandResult]
    assert len(values) == len(set(values)), "CommandResult values must be unique"

    # Test enum behavior
    assert CommandResult.CONTINUE != CommandResult.BREAK
    assert CommandResult.BREAK != CommandResult.PROCEED
    assert CommandResult.PROCEED != CommandResult.CONTINUE


@patch("owlsight.app.run_app.get_cache_dir")
@patch("owlsight.app.run_app.get_default_config_on_startup_path")
@patch("owlsight.app.run_app.os")
@patch("owlsight.app.run_app.Path")
@patch("owlsight.app.run_app.get_pickle_cache")
@patch("owlsight.app.run_app.get_prompt_cache")
@patch("owlsight.app.run_app.get_py_cache")
def test_clear_history(
    mock_py_cache,
    mock_prompt_cache,
    mock_pickle_cache,
    mock_path_class,
    mock_os,
    mock_default_config,
    mock_cache_dir,
    mock_code_executor,
    mock_text_generation_manager,
    mock_logger,
):
    """Test clear_history function."""
    # Setup
    cache_dir = "C:/cache/dir"
    mock_cache_dir.return_value = cache_dir
    mock_default_config.return_value = "C:/cache/dir/default_config.pkl"
    mock_os.listdir.return_value = ["file1.pkl", "file2.pkl", "default_config.pkl"]

    # Mock Path instances
    def create_mock_path(path_str):
        mock_path = MagicMock(spec=Path)
        mock_path.__str__.return_value = str(path_str)
        mock_path.unlink = MagicMock()
        mock_path.__eq__.side_effect = lambda other: str(mock_path) == str(other)
        mock_path.__truediv__ = lambda self, other: create_mock_path(f"{str(self)}/{other}")
        return mock_path

    mock_path_class.side_effect = create_mock_path

    # Run the function
    clear_history(mock_code_executor, mock_text_generation_manager)

    # Assert globals dict only contains owl_ variables
    assert len(mock_code_executor.globals_dict) == 1
    assert "owl_var" in mock_code_executor.globals_dict
    assert mock_code_executor.globals_dict["owl_var"] == 42

    # Assert chat history and used tools are cleared
    assert len(mock_text_generation_manager.processor.chat_history) == 0
    assert len(mock_text_generation_manager._tool_history) == 0

    # Assert cache functions were called
    mock_pickle_cache.assert_called_once()
    mock_prompt_cache.assert_called_once()
    mock_py_cache.assert_called_once()

    # Assert logging
    mock_logger.info.assert_called_once()
