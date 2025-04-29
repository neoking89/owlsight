import ast
import inspect
import json
import re
from typing import Any, Dict, get_type_hints, get_origin, get_args, Optional, Union

from owlsight.agentic.constants import AGENT_INFORMATION
from owlsight.agentic.models import ToolResult
from owlsight.app.default_functions import OwlDefaultFunctions
from owlsight.utils.helper_functions import parse_markdown
from owlsight.utils.logger import logger


def parse_json_markdown(response: str) -> dict:
    """
    Parses a markdown response to extract JSON data.

    Parameters
    ----------
    response: str
        The markdown response to parse.

    Returns
    -------
    dict
        The parsed JSON data. Returns an empty dict if parsing fails.
    """
    try:
        _, response = parse_markdown(response)[-1]
    except IndexError:
        logger.warning("No markdown code block found in plan JSON. Trying to parse as JSON directly.")
    try:
        data = json.loads(response)
        return data
    except json.JSONDecodeError:
        return {}


def get_agent_information_for_prompts() -> str:
    """
    Returns a formatted string of agent information for injection into agent prompts.
    """
    exclude_agents = ["ObservationAgent"]
    return "\n".join(f"- {k}: {v}" for k, v in AGENT_INFORMATION.items() if k not in exclude_agents)


def get_available_tools(globals_dict: Dict[str, Any]) -> str:
    """
    Return tool descriptors already registered in the executor's namespace.

    Parameters
    --------
    globals_dict: dict[str, Any]
        Dict with python variables registered in the executor's namespace.

    Returns
    -------
    str
        A formatted string of available tools.

    """
    logger.debug("Getting available tools...")
    tools = OwlDefaultFunctions(globals_dict).owl_tools(as_json=True)
    logger.debug(f"Available tools: {tools}")
    return "\n".join(str(t) for t in tools)


def execute_tool(globals_dict: dict[str, Any], tool_data: dict[str, Any]):
    """
    Safe wrapper that executes a registered tool with parameters cast according to type hints.

    Parameters
    ----------
    globals_dict: dict[str, Any]
        Dict with python variables registered in the executor's namespace.
    tool_data: dict[str, Any]
        Dict containing tool name and parameters.

    Returns
    -------
    ToolResult
        Result of the tool execution.
    """
    tool_name = tool_data.get("tool_name", "")
    raw_params = tool_data.get("parameters", {})

    try:
        # Get function signature and validate required params
        func = globals_dict.get(tool_name, None)
        if func is None:
            return ToolResult(False, f"Tool '{tool_name}' not found.")
        sig = inspect.signature(func)
        required_params = [p.name for p in sig.parameters.values() if p.default == inspect.Parameter.empty]
        missing = [p for p in required_params if p not in raw_params]
        if missing:
            return ToolResult(False, f"Missing required parameters: {missing}")

        # Get type hints from the function signature
        type_hints = get_type_hints(func)

        # Cast parameters according to type hints
        params = {}
        for param_name, param_value in raw_params.items():
            if param_name in type_hints:
                target_type = type_hints[param_name]
                # Handle basic type casting
                try:
                    if target_type == bool and isinstance(param_value, str):
                        # Special handling for boolean values
                        param_value = param_value.lower() in {"true", "yes", "1", "y"}
                    elif get_origin(target_type) is Union:
                        types = get_args(target_type)
                        for t in types:
                            try:
                                param_value = t(param_value)
                                break
                            except (ValueError, TypeError):
                                continue
                    elif target_type in (int, float, str):
                        # Cast to the target type
                        param_value = target_type(param_value)
                    elif target_type == list and isinstance(param_value, str):
                        # Try to convert string to list using ast.literal_eval
                        try:
                            param_value = ast.literal_eval(param_value)
                            if not isinstance(param_value, list):
                                param_value = [param_value]
                        except (ValueError, SyntaxError) as e:
                            logger.warning(f"List parsing failed for {param_name}: {str(e)}")
                            param_value = [param_value]
                    elif target_type == dict and isinstance(param_value, str):
                        # Try to convert string to dict using ast.literal_eval
                        try:
                            param_value = ast.literal_eval(param_value)
                            if not isinstance(param_value, dict):
                                param_value = {"value": param_value}
                        except (ValueError, SyntaxError):
                            # If parsing fails, create a simple key-value dict
                            param_value = {"value": param_value}
                except (ValueError, TypeError):
                    # If casting fails, use the original value
                    logger.warning(f"Failed to cast parameter '{param_name}' to {target_type}")

            params[param_name] = param_value

        result = func(**params)
        logger.info("Tool '%s' executed successfully.", tool_name)
        return ToolResult(True, result)
    except Exception as exc:
        logger.exception("Error while executing tool '%s'", tool_name)
        err_msg = f"A {type(exc).__name__} occurred while executing tool '{tool_name}': {str(exc)}"
        return ToolResult(False, err_msg)


def parse_tool_response(response: str) -> Dict[str, Any]:
    """
    Accepts a single JSON object.
    Returns dict with 'tool_name', 'parameters', 'reason'.
    Raises ValueError if the input format is invalid or cannot be parsed as JSON.
    """
    # Preprocessing: strip whitespace and markdown code fences
    response = response.strip()
    response = re.sub(r"^```[a-zA-Z0-9]*\s*|```\s*$", "", response, flags=re.MULTILINE).strip()

    logger.debug(f"Processing tool response (length {len(response)}): {response[:200]}...")

    # Step 1: Try heuristic extraction for JSON embedded in text
    heuristic_result = _try_heuristic_json_extraction(response)
    if heuristic_result:
        return heuristic_result

    try:
        return _parse_tool_response_json(response)
    except ValueError as e:
        logger.error(f"JSON parsing failed: {e}")
        raise ValueError("Tool response could not be parsed as JSON.") from e


def _extract_complete_json(response: str, match_text: str, start_idx: int) -> Optional[Dict[str, Any]]:
    """
    Extract a complete, balanced JSON object from the response starting at the given index.
    Uses brace counting to find the proper closing bracket.

    Args:
        response: The full response text
        match_text: The partially matched JSON text (must start with '{')
        start_idx: Starting index of the match in the response

    Returns:
        Dict if extraction and parsing succeeds, None otherwise
    """
    if not match_text.startswith("{"):
        return None

    brace_count = 0
    complete_json = ""

    for i in range(start_idx, len(response)):
        char = response[i]
        complete_json += char

        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0:
                # Found complete balanced JSON object
                break

    # Now try to parse the complete JSON
    try:
        candidate = json.loads(complete_json)
        if isinstance(candidate, dict) and "tool_name" in candidate:
            candidate.setdefault("parameters", {})
            candidate.setdefault("reason", "")
            logger.debug(f"Parsed from complete balanced JSON: {candidate}")
            return candidate
    except json.JSONDecodeError:
        logger.debug("Complete JSON extraction succeeded but parsing failed")

    return None


def _try_heuristic_json_extraction(response: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to heuristically extract a JSON object containing 'tool_name' from text.
    This function handles cases where JSON might be embedded in surrounding text.

    Args:
        response: The text to extract JSON from

    Returns:
        Dict if extraction succeeds, None otherwise
    """
    # Initial regex match to find JSON-like content with a tool_name
    json_block_match = re.search(r"\{[\s\S]*?\"tool_name\"[\s\S]*?\}", response)
    if not json_block_match:
        return None

    try:
        # Get the matched text
        match_text = json_block_match.group(0)
        start_idx = response.find(match_text)

        # Try complete balanced extraction first
        complete_result = _extract_complete_json(response, match_text, start_idx)
        if complete_result:
            return complete_result

        # If balanced extraction failed, try the original match directly
        candidate = json.loads(match_text)
        if isinstance(candidate, dict) and "tool_name" in candidate:
            candidate.setdefault("parameters", {})
            candidate.setdefault("reason", "")
            logger.debug(f"Parsed from partial JSON match: {candidate}")
            return candidate
    except Exception as e:
        logger.debug(f"Heuristic JSON extraction failed: {e}")

    return None


def _parse_tool_response_json(response: str) -> Dict[str, Any]:
    """Parses a JSON tool selection response."""
    try:
        candidate = json.loads(response)
        if isinstance(candidate, dict) and "tool_name" in candidate:
            candidate.setdefault("parameters", {})
            # Ensure parameters is a dict (it should be from json.loads if present)
            if not isinstance(candidate.get("parameters"), dict):
                raise ValueError("JSON 'parameters' field is not an object.")
            candidate.setdefault("reason", "")
            logger.debug(f"Parsed as JSON: {candidate}")
            return candidate
        else:
            raise ValueError("Invalid JSON format for tool selection. Expected a single object with 'tool_name'.")
    except json.JSONDecodeError as e:
        logger.debug("Not valid JSON.")
        raise ValueError("Response is not valid JSON.") from e
    except ValueError as e:
        # Re-raise the specific format error
        logger.error(f"JSON format error: {e}")
        raise e
    except Exception as e:
        logger.warning(f"Unexpected error during JSON processing: {e}")
        # Wrap unexpected errors
        raise ValueError("Unexpected error parsing JSON response.") from e
