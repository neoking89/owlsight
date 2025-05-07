import pytest
from unittest.mock import MagicMock, patch
from owlsight.agentic.helper_functions import parse_tool_response


from owlsight.agentic.core import ToolCreationAgent, BaseAgent
from owlsight.agentic.models import AgentContext, PlanStep, ExecutionPlan
from owlsight.utils.code_execution import CodeExecutor
from owlsight.utils.custom_classes import GlobalPythonVarsDict

# Sample data for testing
SAMPLE_FUNCTION_MARKDOWN = '''
```python
def example_tool(query: str, max_results: int):
    return {"result": f"Searched {query} with {max_results} results"}
```
'''

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


# def test_define_and_register_tool_success(tool_creation_agent):
#     """Test _define_and_register_tool successfully registers a function."""
#     response = SAMPLE_FUNCTION_MARKDOWN
#     registered_names, err_msg = tool_creation_agent._define_and_register_tool("", response)

#     assert registered_names == {"example_tool"}
#     assert "example_tool" in GlobalPythonVarsDict()
#     func = GlobalPythonVarsDict()["example_tool"]
#     assert callable(func)
#     assert func("test", 5) == {"result": "Searched test with 5 results"}


@patch('owlsight.agentic.core.logger.exception')
def test_define_and_register_tool_error(mock_exception, tool_creation_agent):
    """Test _define_and_register_tool handles syntax errors in the code block."""
    data_to_register = MALFORMED_CODE_BLOCK_SYNTAX_ERROR.split("```")[1].strip()

    registered_names, err_msg = tool_creation_agent._define_and_register_tool("", data_to_register)

    assert registered_names == set()
    assert "No results returned" in err_msg
    assert "example_tool" not in GlobalPythonVarsDict()


@patch('owlsight.agentic.core.logger.error')
def test_define_and_register_tool_no_function_def(mock_error, tool_creation_agent):
    """Test _define_and_register_tool handles code blocks without a function definition."""
    response = MALFORMED_CODE_BLOCK_NO_FUNCTION.split("```")[1].strip()
    registered_names, err_msg = tool_creation_agent._define_and_register_tool("", response)

    assert registered_names == set()
    mock_error.assert_called_once()
    assert "No results returned" in err_msg
    assert "example_tool" not in GlobalPythonVarsDict()


# def test_execute_integration(tool_creation_agent):
#     """Simplified integration test for the execute method."""
#     # fill vars_dict to prevent ValueError that occurs when vars_dict is empty
#     vars_dict = GlobalPythonVarsDict()
#     vars_dict["a"] = 1
#     execution_plan = ExecutionPlan([
#         PlanStep(description="Create a function", agent_name="ToolCreationAgent", reason="Test reason")
#     ])
#     context = AgentContext(user_request="Create a function", execution_plan=execution_plan)
#     tool_creation_agent.llm_call.return_value = SAMPLE_FUNCTION_MARKDOWN

#     result = tool_creation_agent.execute(context)

#     assert result.success, f"Failed with message: {result.execution_result}"
#     assert "example_tool" in vars_dict
#     assert result.execution_result == ['example_tool']

# --- Test Cases for parse_tool_response ---

# Valid JSON Inputs
VALID_JSON_BASIC = '{"tool_name": "test_tool", "parameters": {}, "reason": "Basic test"}'
VALID_JSON_WITH_PARAMS = '{"tool_name": "another_tool", "parameters": {"query": "test query", "max_results": 5}, "reason": "Test with various parameters"}'

# Inputs with Markdown Fences
JSON_WITH_FENCES = """
```json
{
  "tool_name": "fenced_json_tool",
  "parameters": {"id": "abc"},
  "reason": "JSON inside markdown fences"
}
```
"""

# Input requiring Heuristic JSON extraction
HEURISTIC_JSON = """
Okay, I will use the search tool. Here is the selection:
{
  "tool_name": "heuristic_tool",
  "parameters": {"query": "find me stuff"},
  "reason": "Extracted from surrounding text"
}
"""

# Invalid Inputs
INVALID_JSON_STRING = ' "just a string" '
INVALID_JSON_MALFORMED = ' {"tool_name": "bad", parameters: {} '
NEITHER_JSON_NOR_XML = ' Just some plain text explanation. '

# --- Pytest Functions ---
def test_parse_valid_json_basic():
    expected = {"tool_name": "test_tool", "parameters": {}, "reason": "Basic test"}
    assert parse_tool_response(VALID_JSON_BASIC) == expected

def test_parse_valid_json_with_params():
    expected = {
        "tool_name": "another_tool",
        "parameters": {
            "query": "test query",
            "max_results": 5
        },
        "reason": "Test with various parameters"
    }
    assert parse_tool_response(VALID_JSON_WITH_PARAMS) == expected

def test_parse_json_with_fences():
    expected = {"tool_name": "fenced_json_tool", "parameters": {"id": "abc"}, "reason": "JSON inside markdown fences"}
    assert parse_tool_response(JSON_WITH_FENCES) == expected

def test_parse_heuristic_json():
    expected = {"tool_name": "heuristic_tool", "parameters": {"query": "find me stuff"}, "reason": "Extracted from surrounding text"}
    result = parse_tool_response(HEURISTIC_JSON)
    assert result["tool_name"] == expected["tool_name"]
    assert result["parameters"] == expected["parameters"]
    assert result["reason"] in [expected["reason"], ""]

@pytest.mark.parametrize("invalid_input", [
    INVALID_JSON_STRING,
    INVALID_JSON_MALFORMED,
    NEITHER_JSON_NOR_XML,
])
def test_parse_invalid_formats(invalid_input):
    with pytest.raises(ValueError):
        parse_tool_response(invalid_input)
