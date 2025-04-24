import pytest
import ast
import sys
import os
import logging
from unittest.mock import patch

# Add src directory to sys.path to allow importing owlsight
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from typing import Any, Dict, List, Optional

from owlsight.agentic.helper_functions import execute_tool
from owlsight.agentic.models import ToolResult

# --- Mock Tools ---
def mock_tool_basic(name: str, age: int) -> str:
    """A simple mock tool with basic types."""
    return f"Hello {name}, you are {age} years old."

def mock_tool_bool(is_active: bool) -> str:
    """Mock tool for boolean testing."""
    return f"Active status: {is_active}"

def mock_tool_list(items: list) -> int:
    """Mock tool for list testing."""
    return len(items)

def mock_tool_dict(data: dict) -> List[str]:
    """Mock tool for dict testing."""
    return sorted(list(data.keys()))

def mock_tool_no_params() -> str:
    """Mock tool that takes no parameters."""
    return "No parameters needed."

def mock_tool_no_hints(value) -> Any:
    """Mock tool with a parameter but no type hint."""
    return type(value).__name__

def mock_tool_optional(name: Optional[str] = None) -> str:
    """Mock tool with an optional parameter."""
    return f"Name provided: {name is not None}"

def mock_tool_raises_error(x: int) -> None:
    """Mock tool that always raises an error."""
    if x > 0:
        raise ValueError("Value must be non-positive")
    return None

# --- Test Setup (using pytest fixture) ---
@pytest.fixture
def mock_globals_dict() -> Dict[str, Any]:
    """Provides the globals_dict for tests."""
    return {
        "mock_tool_basic": mock_tool_basic,
        "mock_tool_bool": mock_tool_bool,
        "mock_tool_list": mock_tool_list,
        "mock_tool_dict": mock_tool_dict,
        "mock_tool_no_params": mock_tool_no_params,
        "mock_tool_no_hints": mock_tool_no_hints,
        "mock_tool_optional": mock_tool_optional,
        "mock_tool_raises_error": mock_tool_raises_error,
    }

# --- Test Functions (using pytest conventions) ---

def test_successful_execution_basic_types(mock_globals_dict):
    """Test successful execution with basic type casting."""
    tool_data = {
        "tool_name": "mock_tool_basic",
        "parameters": {"name": "Alice", "age": "30"}
    }
    expected_result = "Hello Alice, you are 30 years old."
    result = execute_tool(mock_globals_dict, tool_data)
    assert result.success is True
    assert result.result == expected_result

def test_successful_execution_float(mock_globals_dict):
    """Test successful execution with float casting."""
    def mock_tool_float(value: float) -> float:
        return value * 2
    mock_globals_dict["mock_tool_float"] = mock_tool_float
    tool_data = {
        "tool_name": "mock_tool_float",
        "parameters": {"value": "2.5"}
    }
    result = execute_tool(mock_globals_dict, tool_data)
    assert result.success is True
    assert result.result == 5.0

def test_successful_execution_bool_true(mock_globals_dict):
    """Test successful execution with boolean casting ('true')."""
    tool_data = {
        "tool_name": "mock_tool_bool",
        "parameters": {"is_active": "True"}
    }
    result = execute_tool(mock_globals_dict, tool_data)
    assert result.success is True
    assert result.result == "Active status: True"

def test_successful_execution_bool_false(mock_globals_dict):
    """Test successful execution with boolean casting ('0')."""
    tool_data = {
        "tool_name": "mock_tool_bool",
        "parameters": {"is_active": "0"}
    }
    result = execute_tool(mock_globals_dict, tool_data)
    assert result.success is True
    assert result.result == "Active status: False"

def test_successful_execution_list_casting(mock_globals_dict):
    """Test successful execution with list casting from string."""
    tool_data = {
        "tool_name": "mock_tool_list",
        "parameters": {"items": "['a', 'b', 'c']"}
    }
    result = execute_tool(mock_globals_dict, tool_data)
    assert result.success is True
    assert result.result == 3

def test_successful_execution_list_casting_single_item(mock_globals_dict):
    """Test successful execution with list casting from a non-list string."""
    tool_data = {
        "tool_name": "mock_tool_list",
        "parameters": {"items": "'a'"} # String that evals to 'a', not a list
    }
    result = execute_tool(mock_globals_dict, tool_data)
    assert result.success is True
    assert result.result == 1 # Should become ['a']
    assert isinstance(result.result, int) # Function returns length

def test_successful_execution_dict_casting(mock_globals_dict):
    """Test successful execution with dict casting from string."""
    tool_data = {
        "tool_name": "mock_tool_dict",
        "parameters": {"data": "{'key1': 'val1', 'key2': 'val2'}"}
    }
    result = execute_tool(mock_globals_dict, tool_data)
    assert result.success is True
    assert result.result == ["key1", "key2"] # Returns sorted keys

def test_successful_execution_dict_casting_invalid_string(mock_globals_dict):
    """Test dict casting from a string not evaluatable to a dict."""
    tool_data = {
        "tool_name": "mock_tool_dict",
        "parameters": {"data": "not_a_dict_string"}
    }
    result = execute_tool(mock_globals_dict, tool_data)
    assert result.success is True
    assert result.result == ["value"] # Should become {"value": "not_a_dict..."}

def test_successful_execution_no_params(mock_globals_dict):
    """Test successful execution of a tool with no parameters."""
    tool_data = {"tool_name": "mock_tool_no_params", "parameters": {}}
    result = execute_tool(mock_globals_dict, tool_data)
    assert result.success is True
    assert result.result == "No parameters needed."

def test_successful_execution_no_hints(mock_globals_dict):
    """Test execution with parameters having no type hints."""
    tool_data = {
        "tool_name": "mock_tool_no_hints",
        "parameters": {"value": 123}
    }
    result = execute_tool(mock_globals_dict, tool_data)
    assert result.success is True
    assert result.result == "int" # Tool returns type name

def test_successful_execution_optional_param_provided(mock_globals_dict):
    """Test execution with an optional parameter provided."""
    tool_data = {
        "tool_name": "mock_tool_optional",
        "parameters": {"name": "Bob"}
    }
    result = execute_tool(mock_globals_dict, tool_data)
    assert result.success is True
    assert result.result == "Name provided: True"

def test_successful_execution_optional_param_omitted(mock_globals_dict):
    """Test execution with an optional parameter omitted."""
    tool_data = {"tool_name": "mock_tool_optional", "parameters": {}}
    result = execute_tool(mock_globals_dict, tool_data)
    assert result.success is True
    assert result.result == "Name provided: False"

def test_failure_tool_not_found(mock_globals_dict):
    """Test failure when the tool name is not in globals_dict."""
    tool_data = {"tool_name": "non_existent_tool", "parameters": {}}
    result = execute_tool(mock_globals_dict, tool_data)
    assert result.success is False
    assert "Tool 'non_existent_tool' not found" in result.result

def test_failure_missing_required_parameter(mock_globals_dict):
    """Test failure when a required parameter is missing."""
    tool_data = {
        "tool_name": "mock_tool_basic",
        "parameters": {"name": "Alice"} # Missing 'age'
    }
    result = execute_tool(mock_globals_dict, tool_data)
    assert result.success is False
    assert isinstance(result.result, str)
    assert "missing" in result.result.lower() and "age" in result.result.lower()

def test_failure_extra_parameter(mock_globals_dict):
    """Test failure when an extra parameter is provided to a function without **kwargs."""
    tool_data = {
        "tool_name": "mock_tool_basic",
        "parameters": {"name": "Alice", "age": "30", "extra": "param"}
    }
    result = execute_tool(mock_globals_dict, tool_data)
    assert result.success is False
    assert isinstance(result.result, str)
    assert "got an unexpected keyword argument 'extra'" in result.result

def test_failure_tool_raises_exception(mock_globals_dict):
    """Test failure when the executed tool itself raises an exception."""
    tool_data = {
        "tool_name": "mock_tool_raises_error",
        "parameters": {"x": "5"} # Casts to int 5, which triggers the error
    }
    result = execute_tool(mock_globals_dict, tool_data)
    assert result.success is False
    assert "Value must be non-positive" in result.result

@patch('owlsight.agentic.helper_functions.logger.warning')
def test_warning_casting_failure(mock_warning, mock_globals_dict): 
    """Test casting failure calls logger.warning, but execution succeeds."""
    tool_data = {
        "tool_name": "mock_tool_basic",
        "parameters": {"name": "Alice", "age": "thirty"} # Cannot cast 'thirty' to int
    }

    result = execute_tool(mock_globals_dict, tool_data)

    # Check that logger.warning was called with the expected message format
    mock_warning.assert_called_once()
    # Check the content of the call arguments
    args, _ = mock_warning.call_args
    assert len(args) > 0
    assert "Failed to cast parameter 'age' to <class 'int'>" in args[0]

    # Check that execution succeeded because the mock tool handles the string 'thirty'
    assert result.success is True
    assert result.result == "Hello Alice, you are thirty years old."
