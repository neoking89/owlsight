import inspect
import json
import re
from typing import Callable


def function_to_json_for_tool_calling(func: Callable) -> str:
    """
    Converts a Python function into a JSON structure suitable for function-calling
    with an LLM. This function inspects the target function's signature and docstring
    (assumed to be in NumPy style) and returns a JSON schema-like definition.

    Parameters
    ----------
    func : Callable
        The Python function to be converted.

    Returns
    -------
    str
        A JSON string describing the function's name, short description, and parameter schema.
    """
    docstring = inspect.getdoc(func) or ""
    sig = inspect.signature(func)

    # Extract the short summary (first block of text before a blank line in the docstring)
    doc_lines = docstring.strip().split("\n")
    short_description_lines = []
    for line in doc_lines:
        if not line.strip():
            break
        short_description_lines.append(line.strip())
    short_description = " ".join(short_description_lines)

    # Parse the Parameters section from a Numpy-style docstring.
    parameters_section = ""
    param_section_found = False
    for i, line in enumerate(doc_lines):
        if re.match(r"^Parameters\s*-*\s*$", line.strip(), re.IGNORECASE):
            # We found the start of the parameters section
            param_section_found = True
            # Skip the "Parameters" line itself
            continue

        if param_section_found:
            # If we see something that looks like the start of "Returns" or "Notes" or another
            # standard section, we stop
            if re.match(r"^(Returns|Yields|Notes|Raises|Examples)\s*-*\s*$", line.strip(), re.IGNORECASE):
                break
            # Otherwise, accumulate lines
            parameters_section += line + "\n"

    # Now we have the block of text for the parameters section
    # We'll try to parse each parameter entry. Typically, it's in the form:
    #
    # param_name : type
    #     Description possibly spanning multiple lines.
    #
    # We'll do a simple regex-based approach to capture lines that look like "name : type"
    # and then gather subsequent indentented lines as the description.

    param_pattern = re.compile(r"^(\w[\w\d_]*)\s*:\s*([^\n]+)$")
    # We store parameters here as {param_name: {"type": ..., "description": ...}}
    param_info = {}

    lines = parameters_section.split("\n")
    current_param = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        match = param_pattern.match(stripped)
        if match:
            # Found a new parameter
            current_param = match.group(1)
            param_type = match.group(2).strip()
            param_info[current_param] = {"type": guess_json_type_from_string(param_type), "description": ""}
        else:
            # If we're in a parameter block, accumulate description lines
            if current_param:
                if param_info[current_param]["description"]:
                    param_info[current_param]["description"] += " " + stripped
                else:
                    param_info[current_param]["description"] = stripped

    # If the user provided type hints on the function signature, we can try to refine them.
    # We'll also collect which parameters are required vs. which have defaults.
    required_params = []
    for name, parameter in sig.parameters.items():
        # If the parameter has no default, it's required
        if parameter.default is inspect.Parameter.empty:
            required_params.append(name)
        # If there's a type annotation, we might refine it
        if parameter.annotation is not inspect.Signature.empty:
            annotation_str = (
                parameter.annotation.__name__
                if hasattr(parameter.annotation, "__name__")
                else str(parameter.annotation)
            )
            if name in param_info:
                param_info[name]["type"] = guess_json_type_from_string(annotation_str)
            else:
                # If the docstring didn't list it, create an entry
                param_info[name] = {"type": guess_json_type_from_string(annotation_str), "description": ""}

    # Remove any items parsed as parameters that aren't actually in the function signature
    param_info = {k: v for k, v in param_info.items() if k in sig.parameters}

    # Build the JSON schema structure
    parameters_schema = {
        "type": "object",
        "properties": {},
    }
    if required_params:
        parameters_schema["required"] = required_params

    for param_name, pinfo in param_info.items():
        parameters_schema["properties"][param_name] = {
            "type": pinfo["type"],
            "description": pinfo["description"].strip(),
        }

    # Final structure for function calling
    function_json_dict = {"name": func.__name__, "description": short_description, "parameters": parameters_schema}
    final_dict = {"type": "function", "function": function_json_dict}

    return json.dumps(final_dict, indent=2)


def guess_json_type_from_string(type_str: str) -> str:
    """
    A simple utility function to guess a JSON type based on a type string.
    It handles a few common Python types and defaults to 'string' otherwise.
    """
    # Lowercase everything for easier matching
    ts = type_str.lower()

    # Common built-ins
    if "int" in ts:
        return "integer"
    if "float" in ts or "double" in ts:
        return "number"
    if "bool" in ts:
        return "boolean"
    if "str" in ts or "string" in ts:
        return "string"
    if "dict" in ts or "mapping" in ts:
        return "object"
    if "list" in ts or "array" in ts or "sequence" in ts:
        return "array"

    # Fallback
    return "string"
