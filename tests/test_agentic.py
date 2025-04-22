import pytest
from unittest.mock import MagicMock, patch

from owlsight.app.agentic import ToolCreationAgent, StepResult, AgentContext, BaseAgent
from owlsight.utils.code_execution import CodeExecutor
from owlsight.utils.custom_classes import GlobalPythonVarsDict

# Sample data for testing
SAMPLE_FUNCTION_MARKDOWN = """
Some text before the code block.
```python
import math

def add_numbers(x, y):
    \"\"\"Adds two numbers together.\"\"\"
    # Example comment
    result = x + y
    return result
```
Some text after the code block.
"""

# Sample markdown with incorrect formatting or language
INVALID_MARKDOWN_NO_CODE = "This is just text."
INVALID_MARKDOWN_WRONG_LANG = """
```javascript
function subtract(a, b) {
  return a - b;
}
```
"""
INVALID_MARKDOWN_SYNTAX_ERROR = """
```python
def multiply(a, b)
    return a * b
```
"""

MALFORMED_CODE_BLOCK_NO_FUNCTION = """
```python
print("Hello")
x = 10
```
"""

MALFORMED_CODE_BLOCK_SYNTAX_ERROR = """
```python
def invalid_func(a, b)
    return a + b
```
"""


@pytest.fixture
def mock_code_executor():
    """Fixture for a mock CodeExecutor (unused by ToolCreationAgent tests now)."""
    executor = MagicMock(spec=CodeExecutor)
    executor.globals_dict = {}
    return executor


@pytest.fixture
def tool_creation_agent(monkeypatch, tmp_path):
    """Fixture for ToolCreationAgent using a real CodeExecutor but clearing its globals."""
    # Create necessary mocks/dummies for CodeExecutor dependencies
    mock_manager = MagicMock()
    dummy_pyenv_path = "/dummy/pyenv"
    dummy_pip_path = "/dummy/pip"
    temp_directory = str(tmp_path) # Use pytest's tmp_path fixture

    # Use a real CodeExecutor with mocked/dummy dependencies
    real_executor = CodeExecutor(
        manager=mock_manager,
        pyenv_path=dummy_pyenv_path,
        pip_path=dummy_pip_path,
        temp_dir=temp_directory
    )

    # Clear the singleton dict *before* the test runs
    GlobalPythonVarsDict().clear() # Clear the singleton instance

    monkeypatch.setattr(BaseAgent, "code_executor", real_executor) # Patch BaseAgent
    monkeypatch.setattr(BaseAgent, "manager", mock_manager) # Also patch manager on BaseAgent

    agent = ToolCreationAgent()
    agent.llm_call = MagicMock() # Mock LLM call

    yield agent # Use yield for potential cleanup

    # Clear again after test for hygiene
    GlobalPythonVarsDict().clear()


# --- Tests for _extract ---


def test_extract_valid_markdown(tool_creation_agent):
    """Test _extract with valid Python markdown."""
    extracted_data = tool_creation_agent._extract(SAMPLE_FUNCTION_MARKDOWN)
    assert "code_blocks" in extracted_data
    assert len(extracted_data["code_blocks"]) == 1
    lang, code = extracted_data["code_blocks"][0]
    assert lang == "python"
    assert "def add_numbers(x, y):" in code
    assert '"""Adds two numbers together."""' in code
    assert "return result" in code


def test_extract_invalid_markdown_no_code(tool_creation_agent):
    """Test _extract with markdown containing no code blocks."""
    extracted_data = tool_creation_agent._extract(INVALID_MARKDOWN_NO_CODE)
    assert extracted_data == {}


def test_extract_invalid_markdown_wrong_language(tool_creation_agent):
    """Test _extract with markdown containing code blocks of a non-Python language."""
    extracted_data = tool_creation_agent._extract(INVALID_MARKDOWN_WRONG_LANG)
    assert extracted_data == {}


# --- Tests for _register_dynamic_tool ---


def test_register_dynamic_tool_success(tool_creation_agent):
    """Test _register_dynamic_tool successfully registers a function."""
    data_to_register = {
        "code_blocks": [("python", SAMPLE_FUNCTION_MARKDOWN.split("```")[1].strip())]
    }
    registered_names = tool_creation_agent._register_dynamic_tool(data_to_register)

    assert registered_names == ["add_numbers"]
    assert "add_numbers" in GlobalPythonVarsDict()
    func = GlobalPythonVarsDict()["add_numbers"]
    assert callable(func)
    assert func(5, 3) == 8


@patch('owlsight.app.agentic.logger.exception')
def test_register_dynamic_tool_syntax_error(mock_exception, tool_creation_agent):
    """Test _register_dynamic_tool handles syntax errors in the code block."""
    data_to_register = {
        "code_blocks": [("python", MALFORMED_CODE_BLOCK_SYNTAX_ERROR.split("```")[1].strip())]
    }

    registered_names = tool_creation_agent._register_dynamic_tool(data_to_register)

    assert registered_names == []
    mock_exception.assert_called_once()
    assert "SyntaxError" in str(mock_exception.call_args)
    assert "Could not register generated tool" in mock_exception.call_args[0][0]
    assert "add_numbers" not in GlobalPythonVarsDict()


@patch('owlsight.app.agentic.logger.warning')
def test_register_dynamic_tool_no_function_def(mock_warning, tool_creation_agent):
    """Test _register_dynamic_tool handles code blocks without a function definition."""
    data_to_register = {
        "code_blocks": [("python", MALFORMED_CODE_BLOCK_NO_FUNCTION.split("```")[1].strip())]
    }
    registered_names = tool_creation_agent._register_dynamic_tool(data_to_register)

    assert registered_names == []
    mock_warning.assert_called_once()
    assert "Could not identify function name in code block" in mock_warning.call_args[0][0]
    assert "add_numbers" not in GlobalPythonVarsDict()


# --- Test for execute (simplified) ---


def test_execute_integration(tool_creation_agent):
    """Simplified integration test for the execute method."""
    # fill vars_dict to prevent ValueError that occurs when vars_dict is empty
    vars_dict = GlobalPythonVarsDict()
    vars_dict["a"] = 1
    context = AgentContext(user_question="Create a function")
    tool_creation_agent.llm_call.return_value = SAMPLE_FUNCTION_MARKDOWN

    result = tool_creation_agent.execute(context)

    assert "add_numbers" in vars_dict
    assert result.execution_result == ['add_numbers']
