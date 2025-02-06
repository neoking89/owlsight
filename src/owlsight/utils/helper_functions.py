from typing import List, Tuple, Dict, Any, Union
import ast
import os
import shutil
import re
import traceback
import inspect
from datetime import datetime, timedelta
import json
from pathlib import Path

from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style

from owlsight.utils.custom_classes import MediaObject, DoubleBracketsTag, _AVAILBLE_DB_TAGS
from owlsight.utils.logger import logger


def parse_markdown(md_string: str) -> List[Tuple[str, str]]:
    """
    Parses language and code blocks from a markdown string.
    """
    pattern = r"```(\w+)([\s\S]*?)```"
    return [(match[0].strip(), match[1].strip()) for match in re.findall(pattern, md_string)]


def parse_python_placeholders(text: str, var_dict: Dict[str, Any]) -> Any:
    """
    Evaluates expressions inside {{...}} in the given text and replaces them with the result.
    Correctly handles expressions containing braces or other special characters.

    Parameters
    ----------
    text : str
        The input string containing placeholders in the form of `{{...}}`.
    var_dict : dict
        A dictionary where keys correspond to variables used in placeholders.

    Returns
    -------
    Any
        The evaluated object if the entire string is a single placeholder,
        otherwise the string with placeholders replaced.
    """

    def evaluate_expression(expr: str) -> Any:
        try:
            safe_globals = {
                "__builtins__": {
                    "abs": abs,
                    "all": all,
                    "any": any,
                    "bin": bin,
                    "bool": bool,
                    "chr": chr,
                    "divmod": divmod,
                    "enumerate": enumerate,
                    "filter": filter,
                    "float": float,
                    "format": format,
                    "hash": hash,
                    "hex": hex,
                    "int": int,
                    "isinstance": isinstance,
                    "issubclass": issubclass,
                    "iter": iter,
                    "len": len,
                    "list": list,
                    "map": map,
                    "max": max,
                    "min": min,
                    "next": next,
                    "oct": oct,
                    "ord": ord,
                    "pow": pow,
                    "range": range,
                    "repr": repr,
                    "reversed": reversed,
                    "round": round,
                    "sorted": sorted,
                    "str": str,
                    "sum": sum,
                    "tuple": tuple,
                    "zip": zip,
                    "dict": dict,
                    "set": set,
                    "frozenset": frozenset,
                    "datetime": datetime,
                    "timedelta": timedelta,
                }
            }
            safe_globals.update(var_dict)
            return eval(expr, safe_globals, {})
        except Exception as e:
            error_message = f"Error evaluating '{expr}': {str(e)}"
            raise type(e)(error_message) from None

    # Pattern to match balanced double braces
    pattern = r"""
        \{\{            # Opening double braces {{
        (?P<expr>       # Start of named group 'expr'
            [^\{\}]*    # Match any characters except { or }
            (?:         # Non-capturing group
                \{[^\{\}]*\} # Match balanced braces
                [^\{\}]*    # Match any characters except { or }
            )*          # Zero or more times
        )               # End of named group 'expr'
        \}\}            # Closing double braces }}
    """
    regex = re.compile(pattern, re.VERBOSE)

    # Function to replace each placeholder
    def replace_match(m):
        expr = m.group("expr")
        evaluated = evaluate_expression(expr.strip())
        return str(evaluated)

    # Check if the entire text is a single placeholder
    if regex.fullmatch(text):
        expr = regex.fullmatch(text).group("expr")
        return evaluate_expression(expr.strip())
    else:
        return regex.sub(replace_match, text)


def editable_input(prompt_text: str, default_value: str, color: str = "ansicyan") -> str:
    """
    Displays a prompt with a pre-filled editable string and custom color for the default value.

    Parameters
    ----------
    prompt_text : str
        The prompt message shown before the editable string.
    default_value : str
        The string that will be pre-filled and editable by the user.
    color : str, optional
        The color to apply to the default value in the prompt message, default is 'ansicyan'.

    Examples
    --------
    >>> editable_input("Enter your name: ", "John")
    Enter your name: "John" -> Enter your name: "Johnny"
    'Johnny'

    Returns
    -------
    str
        The string edited by the user.
    """
    style = Style.from_dict({"prompt_text": color})

    # Prepare the prompt text with custom color using HTML
    formatted_prompt = HTML(f"<ansicyan>{prompt_text}</ansicyan>")

    # Get the result from the prompt (default value is shown but not styled)
    result = prompt(formatted_prompt, default=default_value, style=style)

    return result.strip()


def force_delete(temp_dir: Union[str, Path]) -> None:
    """
    Forcefully deletes a directory if it exists.

    Parameters
    ----------
    temp_dir : Union[str, Path]
        Path to the directory to delete
    """
    temp_dir = Path(temp_dir)
    if temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            logger.error(f"Error deleting directory {temp_dir}:\n{traceback.format_exc()}")


def remove_temp_directories(lib_path: Union[str, Path]) -> None:
    """
    Removes lingering temporary directories in the virtual environment's library path.

    Parameters
    ----------
    lib_path : Union[str, Path]
        Path to the library directory to clean
    """
    lib_path = Path(lib_path)
    if not lib_path.exists():
        logger.warning(f"Library path does not exist: {lib_path}")
        return

    for d in lib_path.iterdir():
        if d.name.startswith("tmp"):
            logger.info(f"Removing temporary directory: {d}")
            force_delete(d)


def format_error_message(e: Exception) -> str:
    """
    Format an error message to be displayed to the user.

    Parameters
    ----------
    error : Exception
        The exception that occurred.

    Returns
    -------
    str
        The formatted error message.
    """
    return "{e.__class__.__name__}: {e}".format(e=e)


def convert_to_real_type(value):
    """
    Convert a string to its real type if possible (e.g., 'True' -> True, '3.14' -> 3.14).
    """
    if not isinstance(value, str):
        return value

    # Try to evaluate the string and return the result only if it's not a string
    try:
        evaluated_value = ast.literal_eval(value)
        # Only return the evaluated value if it is not a string
        if not isinstance(evaluated_value, str):
            return evaluated_value
    except (ValueError, SyntaxError):
        pass  # Return original string if evaluation fails

    return value  # Return the original string if it's not evaluable


def os_is_windows():
    return os.name == "nt"


def check_invalid_input_parameters(func: callable, kwargs: dict):
    """
    Validate the keyword arguments passed to a class against the __init__ signature.

    Parameters
    ----------
    func : callable
        The callable of which arguments are being validated.
    kwargs : dict
        A dictionary of keyword arguments to validate.

    Raises
    ------
    ValueError
        If there are invalid parameters.
    """
    # Extract the parameters from the __init__ method of the class
    sig = inspect.signature(func)
    sig_params = sig.parameters

    valid_params = [param_name for param_name in sig_params if param_name != "self"]

    # Check for any extra parameters in kwargs that are not in the __init__ signature
    for key in kwargs:
        if key not in sig_params:
            raise ValueError(
                f"Invalid argument: '{key}' is not a valid parameter for '{func.__name__}'\nValid parameters: {valid_params}"
            )


def flatten_dict(d, parent_key="", sep=".") -> dict:
    """Flatten a nested dictionary."""
    flattened = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            flattened.update(flatten_dict(v, new_key, sep=sep))
        else:
            flattened[new_key] = v
    return flattened


def parse_media_tags(text: str, var_dict: Dict[str, Any]) -> Tuple[str, Dict[str, MediaObject]]:
    """
    Parse media syntax patterns [[tag:path|option1=value1|...]] in text and evaluate any
    Python expressions inside {{...}}. Returns the modified text and a dictionary of
    media objects with their options.

    Parameters
    ----------
    text : str
        The input string containing media placeholders and optional Python expressions
    var_dict : dict
        A dictionary where keys correspond to variables used in Python expressions

    Returns
    -------
    Tuple[str, MediaObjects]
        A tuple containing:
        - The text with media placeholders replaced with unique identifiers
        - A dictionary mapping identifiers to MediaObject instances

    Examples
    --------
    >>> var_dict = {"folder": "images", "filename": "cat.jpg"}
    >>> text = "Analyze this: [[image:{{folder}}/{{filename}}|width=512]]"
    >>> result, media_objects = parse_media_tags(text, var_dict)
    >>> print(result)
    'Analyze this: __MEDIA_0__'
    >>> print(media_objects)
    {
        '__MEDIA_0__': MediaObject(
            tag='image',
            path='images/cat.jpg',
            options={'width': '512'}
        )
    }
    """

    def validate_media_syntax(text: str) -> None:
        # Check for valid media tags
        invalid_tags = re.findall(r"\[\[(\w+):", text)
        valid_tags = _AVAILBLE_DB_TAGS
        for t in invalid_tags:
            if t not in valid_tags:
                raise ValueError(f"Invalid media tag: {t}. Must be one of {valid_tags}")

        # Check for missing paths
        if re.search(r"\[\[\w+:\s*(\||\]\])", text):
            raise ValueError("Media path cannot be empty")

        # Check for invalid option format
        option_pattern = r"\|\|(?!\w+=)[^]|]*(?=[\]|])"
        invalid_options = re.findall(option_pattern, text)
        if invalid_options:
            raise ValueError(f"Invalid option format: {invalid_options[0].strip()}. Must be key=value")

    validate_media_syntax(text)

    pattern = r"""\[\[
        (?P<tag>image|audio|video):  # Media tag
        (?P<path>[^\|\]]+)            # Path (anything until || or ])
        (?:\|\|(?P<options>[^\]]+))?    # Optional options after ||
        \]\]"""

    media_objects: Dict[str, MediaObject] = {}
    replacement_count = 0

    def replace_match(match) -> str:
        nonlocal replacement_count

        media_tag: DoubleBracketsTag = match.group("tag")  # type: ignore
        raw_path = match.group("path")
        options_str = match.group("options") or ""

        # Process the path first - evaluate any Python expressions
        processed_path = parse_python_placeholders(raw_path, var_dict)

        # Process options
        options: Dict[str, str] = {}
        if options_str:
            for option in options_str.split("||"):
                if "=" in option:
                    key, value = option.split("=", 1)
                    # Evaluate Python expressions in option values and convert to string
                    processed_value = str(parse_python_placeholders(value.strip(), var_dict))
                    options[key.strip()] = processed_value

        # Create unique identifier
        identifier = f"__MEDIA_{replacement_count}__"
        replacement_count += 1

        # Store media object information using the MediaObject class
        media_objects[identifier] = MediaObject(tag=media_tag, path=processed_path, options=options)

        return identifier

    # Use verbose flag for multiline regex pattern
    regex = re.compile(pattern, re.VERBOSE)

    # First replace media placeholders
    processed_text = regex.sub(replace_match, text)

    # Then evaluate any remaining Python expressions in the text
    processed_text = parse_python_placeholders(processed_text, var_dict)

    return processed_text, media_objects


def extract_square_bracket_tags(
    text: str, tag: Union[str, List[str]], key: str = "path"
) -> List[Union[str, Dict[str, str]]]:
    """
    Extracts square bracket tags from a string and returns a list of either strings or dictionaries.

    Parameters
    ----------
    text : str
        The input string containing square bracket tags.

    tag : Union[str, List[str]]
        A tag or list of tags to search for. Each tag must be one of _AVAILBLE_DB_TAGS.

    key : str, optional
        The key to use for the tag value, by default "path".

    Returns
    -------
    List[Union[str, Dict[str, str]]]
        A list of strings (input text) and/or dictionaries (tag).
    """
    # Convert single tag to list if necessary
    tags = [tag] if isinstance(tag, str) else tag

    if not all(t in _AVAILBLE_DB_TAGS for t in tags):
        raise ValueError(f"Invalid tag in {tags}. Must be a subset of {_AVAILBLE_DB_TAGS}")

    # Construct regex pattern to capture any of the provided tags
    pattern = r"\[\[(?P<tag>" + "|".join(re.escape(t) for t in tags) + "):" + r"(?P<value>.*?)\]\]"
    matches = list(re.finditer(pattern, text))
    result = []
    prev_end = 0

    for match in matches:
        start, end = match.start(), match.end()
        if start > prev_end:
            part = text[prev_end:start].strip()
            if part:
                result.append(part)
        d = {"tag": match.group("tag"), key: match.group("value")}
        result.append(d)
        prev_end = end

    if prev_end < len(text):
        part = text[prev_end:].strip()
        if part:
            result.append(part)

    return result


def parse_function_call_to_python_code(input_str: str) -> str:
    """
    Parses a string for a JSON-like function call pattern of the form:

    {"name": "function_name", "arguments": { ... }}

    If such a pattern is found, it transforms the information into a Python
    markdown code block that executes the function, like so:

        ```python
        final_result = function_name(arg1=value1, arg2=value2, ...)
        ```

    The JSON "arguments" object is converted into valid Python keyword arguments
    (using repr() to ensure proper quoting for strings, etc).

    If the pattern isn't found or the JSON is invalid, the original input string
    is returned.
    """
    pattern = r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}'
    match = re.search(pattern, input_str)
    if match:
        func_name = match.group(1)
        args_str = match.group(2)
        try:
            arguments = json.loads(args_str)
        except json.JSONDecodeError:
            return input_str

        # Build the keyword arguments string.
        kwargs = ", ".join(f"{key}={repr(value)}" for key, value in arguments.items())
        code_line = f"final_result = {func_name}({kwargs})"
        return f"```python\n{code_line}\n```"
    return input_str
