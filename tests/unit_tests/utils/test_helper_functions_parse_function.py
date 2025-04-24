from owlsight.utils.helper_functions import parse_function_call_to_python_code

def test_valid_json_pattern():
    """
    Test that a valid JSON tool call string is correctly transformed into a
    Python markdown-formatted code block.
    """
    input_str = '{"name": "owl_search", "arguments": {"query": "Some query", "max_results": 1, "max_retries": 3}}'
    expected_code_line = "final_result = owl_search(query='Some query', max_results=1, max_retries=3)"
    expected_output = f"```python\n{expected_code_line}\n```"
    result = parse_function_call_to_python_code(input_str)
    assert result == expected_output

def test_no_json_pattern():
    """
    Test that if the input string does not contain a JSON tool call pattern,
    the original string is returned unchanged.
    """
    input_str = "This is a simple string without any JSON pattern."
    result = parse_function_call_to_python_code(input_str)
    assert result == input_str

def test_invalid_json_arguments():
    """
    Test that if the JSON in the tool call is invalid,
    the function returns the original input string.
    """
    input_str = '{"name": "invalid_func", "arguments": {bad_json}}'
    result = parse_function_call_to_python_code(input_str)
    assert result == input_str

def test_extra_whitespace_in_pattern():
    """
    Test that extra whitespace around keys and values does not affect the parsing.
    """
    input_str = '{ "name" : "whitespace_func", "arguments" : { "x" : 10, "y" : "test" } }'
    expected_code_line = "final_result = whitespace_func(x=10, y='test')"
    expected_output = f"```python\n{expected_code_line}\n```"
    result = parse_function_call_to_python_code(input_str)
    assert result == expected_output

def test_mixed_argument_types():
    """
    Test that arguments with mixed types (integer, boolean, null, float)
    are correctly converted into their Python equivalents.
    """
    input_str = '{"name": "complex_func", "arguments": {"a": 1, "b": true, "c": null, "d": 3.14}}'
    expected_code_line = "final_result = complex_func(a=1, b=True, c=None, d=3.14)"
    expected_output = f"```python\n{expected_code_line}\n```"
    result = parse_function_call_to_python_code(input_str)
    assert result == expected_output
