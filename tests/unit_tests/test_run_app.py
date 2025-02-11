import pytest
from unittest.mock import patch
from owlsight.app.run_app import _extract_params_chain_tag, CommandResult
from owlsight.utils.logger import logger


@pytest.fixture(autouse=True)
def mock_logger():
    with patch('owlsight.app.run_app.logger') as mock:
        yield mock


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
