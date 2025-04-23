import pytest
from unittest.mock import MagicMock, patch
import re

from owlsight.app.agentic.helpers import parse_tool_response
from owlsight.app.agentic.agents.tool_creation import ToolCreationAgent
from owlsight.app.agentic.models import AgentContext
from owlsight.app.agentic.agents.base import BaseAgent
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
    # Extract only the code part from the markdown
    match = re.search(r"```python\s*(.*?)```", SAMPLE_FUNCTION_MARKDOWN, re.DOTALL)
    code_to_register = match.group(1).strip() if match else ""

    data_to_register = {
        "code_blocks": [("python", code_to_register)]
    }
    registered_names = tool_creation_agent._register_dynamic_tool(data_to_register)

    assert registered_names == ["add_numbers"]
    assert "add_numbers" in GlobalPythonVarsDict()
    func = GlobalPythonVarsDict()["add_numbers"]
    assert callable(func)
    assert func(5, 3) == 8


@patch('owlsight.app.agentic.agents.tool_creation.logger.exception')
def test_register_dynamic_tool_syntax_error(mock_exception, tool_creation_agent):
    """Test _register_dynamic_tool handles syntax errors in the code block."""
    # Extract only the code part from the markdown
    match = re.search(r"```python\s*(.*?)```", MALFORMED_CODE_BLOCK_SYNTAX_ERROR, re.DOTALL)
    code_to_register = match.group(1).strip() if match else ""

    data_to_register = {
        "code_blocks": [("python", code_to_register)]
    }

    registered_names = tool_creation_agent._register_dynamic_tool(data_to_register)

    assert registered_names == []
    mock_exception.assert_called_once()
    # Check the actual message passed to logger.exception
    assert "syntax" in mock_exception.call_args[0][0]


def test_register_dynamic_tool_no_function_def(tool_creation_agent):
    """Test _register_dynamic_tool handles code blocks without a function definition."""
    data_to_register = {
        "code_blocks": [("python", MALFORMED_CODE_BLOCK_NO_FUNCTION.split("```")[1].strip())]
    }
    registered_names = tool_creation_agent._register_dynamic_tool(data_to_register)

    assert registered_names == []
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

# --- Test Cases for parse_tool_response ---

# Valid JSON Inputs
VALID_JSON_BASIC = '{"tool_name": "test_tool", "parameters": {}, "reason": "Basic test"}'
VALID_JSON_WITH_PARAMS = '{"tool_name": "another_tool", "parameters": {"param1": "value1", "param2": 123, "param3": true, "param4": ["a", "b"], "param5": {"key": "val"}}, "reason": "Test with various parameters"}'

# Valid XML Inputs
VALID_XML_BASIC = """
<selection>
  <tool_name>xml_tool</tool_name>
  <parameters></parameters>
  <reason>Basic XML test</reason>
</selection>
"""

VALID_XML_WITH_PARAMS = """
<selection>
  <tool_name>xml_params_tool</tool_name>
  <parameters>
    <parameter>
      <name>query</name>
      <value>search query</value>
    </parameter>
    <parameter>
      <name>max_results</name>
      <value>10</value>
    </parameter>
    <parameter>
      <name>enabled</name>
      <value>true</value>
    </parameter>
  </parameters>
  <reason>XML with simple params</reason>
</selection>
"""

# XML with stringified list/dict values (as seen in logs)
VALID_XML_WITH_STRING_LIST = """
<selection>
  <tool_name>owl_scrape</tool_name>
  <parameters>
    <parameter>
      <name>urls</name>
      <value>
        [
          "https://example.com/page1",
          "https://example.com/page2"
        ]
      </value>
    </parameter>
    <parameter>
      <name>max_concurrent</name>
      <value>5</value>
    </parameter>
  </parameters>
  <reason>Test parsing stringified list in XML</reason>
</selection>
"""

VALID_XML_WITH_STRING_DICT = """
<selection>
  <tool_name>config_update</tool_name>
  <parameters>
    <parameter>
      <name>settings</name>
      <value>
        {
          "timeout": 30,
          "retries": 3
        }
      </value>
    </parameter>
  </parameters>
  <reason>Test parsing stringified dict in XML</reason>
</selection>
"""

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

XML_WITH_FENCES = """
```xml
<selection>
  <tool_name>fenced_xml_tool</tool_name>
  <parameters>
      <parameter><name>path</name><value>/data</value></parameter>
  </parameters>
  <reason>XML inside markdown fences</reason>
</selection>
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
INVALID_XML_MALFORMED = ' <selection><tool_name>incomplete </selection '
INVALID_XML_NESTED = ' <selection><tool_name>outer</tool_name><selection><tool_name>inner</tool_name></selection></selection> '
INVALID_XML_MULTIPLE_ROOTS = ' <selection></selection><selection></selection> '
INVALID_XML_MISSING_TOOLNAME = ' <selection><parameters></parameters></selection> '
XML_WITH_BAD_JSON_VALUE = """
<selection>
  <tool_name>bad_json_val</tool_name>
  <parameters>
    <parameter><name>bad_list</name><value>[1, 2</value></parameter>
  </parameters>
  <reason>Bad JSON in value</reason>
</selection>
"""
NEITHER_JSON_NOR_XML = ' Just some plain text explanation. '

# --- Pytest Functions ---
def test_parse_valid_json_basic():
    expected = {"tool_name": "test_tool", "parameters": {}, "reason": "Basic test"}
    assert parse_tool_response(VALID_JSON_BASIC) == expected

def test_parse_valid_json_with_params():
    expected = {
        "tool_name": "another_tool",
        "parameters": {
            "param1": "value1",
            "param2": 123,
            "param3": True,
            "param4": ["a", "b"],
            "param5": {"key": "val"}
        },
        "reason": "Test with various parameters"
    }
    assert parse_tool_response(VALID_JSON_WITH_PARAMS) == expected

def test_parse_valid_xml_basic():
    expected = {"tool_name": "xml_tool", "parameters": {}, "reason": "Basic XML test"}
    assert parse_tool_response(VALID_XML_BASIC) == expected

def test_parse_valid_xml_with_params():
    expected = {
        "tool_name": "xml_params_tool",
        "parameters": {
            "query": "search query",
            "max_results": "10",
            "enabled": "true"
        },
        "reason": "XML with simple params"
    }
    assert parse_tool_response(VALID_XML_WITH_PARAMS) == expected

def test_parse_valid_xml_with_string_list():
    result = parse_tool_response(VALID_XML_WITH_STRING_LIST)
    assert result["tool_name"] == "owl_scrape"
    assert result["reason"] == "Test parsing stringified list in XML"
    assert result["parameters"]["max_concurrent"] == "5"
    assert isinstance(result["parameters"]["urls"], list)
    assert result["parameters"]["urls"] == [
        "https://example.com/page1",
        "https://example.com/page2"
    ]

def test_parse_valid_xml_with_string_dict():
    result = parse_tool_response(VALID_XML_WITH_STRING_DICT)
    assert result["tool_name"] == "config_update"
    assert result["reason"] == "Test parsing stringified dict in XML"
    assert isinstance(result["parameters"]["settings"], dict)
    assert result["parameters"]["settings"] == {
        "timeout": 30,
        "retries": 3
    }

def test_parse_json_with_fences():
    expected = {"tool_name": "fenced_json_tool", "parameters": {"id": "abc"}, "reason": "JSON inside markdown fences"}
    assert parse_tool_response(JSON_WITH_FENCES) == expected

def test_parse_xml_with_fences():
    expected = {"tool_name": "fenced_xml_tool", "parameters": {"path": "/data"}, "reason": "XML inside markdown fences"}
    assert parse_tool_response(XML_WITH_FENCES) == expected

def test_parse_heuristic_json():
    expected = {"tool_name": "heuristic_tool", "parameters": {"query": "find me stuff"}, "reason": "Extracted from surrounding text"}
    result = parse_tool_response(HEURISTIC_JSON)
    assert result["tool_name"] == expected["tool_name"]
    assert result["parameters"] == expected["parameters"]
    assert result["reason"] in [expected["reason"], ""]

@pytest.mark.parametrize("invalid_input", [
    INVALID_JSON_STRING,
    INVALID_JSON_MALFORMED,
    INVALID_XML_MALFORMED,
    INVALID_XML_NESTED,
    INVALID_XML_MULTIPLE_ROOTS,
    INVALID_XML_MISSING_TOOLNAME,
    NEITHER_JSON_NOR_XML,
])
def test_parse_invalid_formats(invalid_input):
    with pytest.raises(ValueError):
        parse_tool_response(invalid_input)

def test_parse_xml_with_bad_json_value():
    result = parse_tool_response(XML_WITH_BAD_JSON_VALUE)
    assert result["tool_name"] == "bad_json_val"
    assert isinstance(result["parameters"]["bad_list"], str)
    assert result["parameters"]["bad_list"].strip() == "[1, 2"
