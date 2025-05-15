import pytest
from unittest.mock import patch, MagicMock, call

from owlsight.app.run_app import _extract_params_chain_tag, CommandResult
from owlsight.app.run_app import process_user_request
from owlsight.processors.text_generation_manager import TextGenerationManager
from owlsight.utils.code_execution import CodeExecutor
from owlsight.agentic.core import AgentOrchestrator
from owlsight.configurations.config_manager import ConfigManager

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


def test_process_user_request_without_agentic(mock_logger):
    """Test process_user_request when agentic.active is False."""
    # Create a properly typed mock for TextGenerationManager
    mock_manager = MagicMock(spec=TextGenerationManager)
    
    # Use side_effect to control return values based on input parameters
    def get_config_key_side_effect(key, default=None):
        # Return appropriate values for each key
        if key == 'agentic.active':
            return False
        elif key == 'main.dynamic_system_prompt':
            return False
        elif key == 'rag.active':
            return False
        elif key == 'rag.target_library':
            return ""
        else:
            return default
            
    mock_manager.get_config_key.side_effect = get_config_key_side_effect
    mock_manager.generate.return_value = "Direct Generate Response"
    
    # Set up the remaining mocks
    mock_executor = MagicMock(spec=CodeExecutor)
    mock_config_manager = MagicMock(spec=ConfigManager)
    mock_manager.config_manager = mock_config_manager
    mock_executor.globals_dict = {}
    
    user_input = "Tell me a joke"
    with patch("owlsight.app.run_app.AgentOrchestrator") as mock_agent_orchestrator_class:
        result = process_user_request(user_input, mock_executor, mock_manager)
    
    # Verify individual calls instead of comparing entire lists
    assert mock_manager.get_config_key.call_args_list[0] == call("agentic.active", False)
    assert mock_manager.get_config_key.call_args_list[1] == call("main.dynamic_system_prompt", False)
    assert mock_manager.get_config_key.call_args_list[2] == call("rag.active", False)
    assert mock_manager.get_config_key.call_args_list[3] == call("rag.target_library", "")
    assert mock_manager.get_config_key.call_count == 4
    
    mock_manager.generate.assert_called_once_with(user_input, media_objects={})
    mock_agent_orchestrator_class.assert_not_called()
    assert result == "Direct Generate Response"
    mock_logger.error.assert_not_called()

def test_process_user_request_with_agentic(mock_logger):
    """Test process_user_request when agentic.active is True."""
    mock_manager = MagicMock(spec=TextGenerationManager)
    mock_manager.get_config_key.return_value = True
    mock_executor = MagicMock(spec=CodeExecutor)
    user_input = "Refactor this code: ..."
    mock_orchestrator_instance = MagicMock(spec=AgentOrchestrator)
    mock_orchestrator_instance.process_user_request.return_value = "Agentic Response"
    with patch(
        "owlsight.app.run_app.AgentOrchestrator", return_value=mock_orchestrator_instance
    ) as mock_agent_orchestrator_class:
        result = process_user_request(user_input, mock_executor, mock_manager)
    mock_manager.get_config_key.assert_called()
    mock_manager.generate.assert_not_called()
    mock_agent_orchestrator_class.assert_called_once_with(mock_executor, mock_manager)
    mock_orchestrator_instance.process_user_request.assert_called_once_with(user_input)
    assert result == "Agentic Response"
    mock_logger.error.assert_not_called()
