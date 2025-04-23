"""
Helper functions for the agentic framework.

This module contains utility functions for parsing tool responses,
executing tools, and other supporting operations.
"""

import json
import re
import traceback
import xml.etree.ElementTree as ET
from typing import Any, Dict, Optional, get_type_hints

from owlsight.app.default_functions import OwlDefaultFunctions
from owlsight.utils.code_execution import CodeExecutor
from owlsight.utils.logger import logger


def get_agent_information() -> str:
    """
    Returns a formatted string of agent information for better representation in prompts.
    """
    from owlsight.app.agentic.prompts import AGENT_INFORMATION
    return "\n".join(f"- {k}: {v}" for k, v in AGENT_INFORMATION.items())


def get_available_tools(code_executor: "CodeExecutor") -> str:
    """
    Return tool descriptors already registered in the executor's namespace.
    """
    logger.debug("Getting available tools...")
    tools = OwlDefaultFunctions(code_executor.globals_dict).owl_tools(as_json=True)
    logger.debug(f"Available tools: {tools}")
    return "\n".join(str(t) for t in tools)


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


def _parse_tool_response_xml(response: str) -> Dict[str, Any]:
    """Parses an XML tool selection response."""
    try:
        # Try to extract content within <selection> tags first to handle potential surrounding text
        match = re.search(r"<selection>(.*?)</selection>", response, re.DOTALL | re.IGNORECASE)
        if not match:
            logger.debug("No <selection> tags found, attempting direct XML parse.")
            xml_content_to_parse = response
            # Basic check: ensure it starts like XML
            if not xml_content_to_parse.startswith("<"):
                raise ValueError("Response does not appear to be XML.")
        else:
            # If <selection> tags found, parse the original string assuming <selection> is the root
            xml_content_to_parse = response
            # Check for nested <selection>
            inner_content = match.group(1).strip()
            if "<selection>" in inner_content.lower():
                raise ValueError("Nested <selection> tags detected. Invalid format.")

        try:
            root = ET.fromstring(xml_content_to_parse)
            # Verify the root tag is indeed 'selection' if we used the original response based on match
            if match and root.tag.lower() != "selection":
                # This case should ideally not happen if the regex matched, but good safety check
                raise ValueError("Expected root element <selection> not found despite regex match.")
            # If no match, we directly parsed, the root could be anything, but we expect 'selection'
            elif not match and root.tag.lower() != "selection":
                raise ValueError(f"Expected root element <selection> but found <{root.tag}>.")

        except ET.ParseError as pe:
            if "junk after document element" in str(pe):
                raise ValueError(
                    f"Invalid XML: Multiple root elements found. Expected a single <selection> element. Content: {response[:200]}..."
                ) from pe
            else:
                raise ValueError(
                    f"Invalid XML format for tool selection. ParseError: {pe}\nContent:\n{response[:200]}..."
                ) from pe

        # Simplified parsing assuming direct children
        tool_name = root.findtext("./tool_name", default="").strip()
        reason = root.findtext("./reason", default="").strip()
        param_dict = {}
        parameters_elem = root.find("./parameters")
        if parameters_elem is not None:
            for param in parameters_elem.findall("./parameter"):
                name = param.findtext("./name", default="").strip()
                value_str = param.findtext("./value", default="").strip()  # Value from XML is initially a string
                if name:
                    # Attempt to parse the value string as JSON if it looks like a list/dict
                    parsed_value = value_str  # Default to original string
                    trimmed_value = value_str.strip()
                    if trimmed_value.startswith(("[", "{")) and trimmed_value.endswith(("]", "}")):
                        try:
                            parsed_value = json.loads(trimmed_value)
                            logger.debug(f"Successfully parsed XML parameter '{name}' value as JSON.")
                        except json.JSONDecodeError:
                            logger.warning(
                                f"XML parameter '{name}' value looked like JSON/list but failed to parse. Keeping as string: {value_str[:100]}..."
                            )
                            # Keep parsed_value = value_str (already default)

                    param_dict[name] = parsed_value  # Assign the potentially parsed value

        if not tool_name:
            raise ValueError("Missing <tool_name> in XML selection.")

        result = {"tool_name": tool_name, "parameters": param_dict, "reason": reason}
        logger.debug(f"Parsed as XML: {result}")
        return result

    except ET.ParseError as e:
        # This specific catch might be redundant due to the inner try-except,
        # but serves as a fallback for unexpected ET errors.
        logger.error(f"XML Parse Error: {e}")
        raise ValueError("Failed to parse XML response.") from e
    except ValueError as e:
        # Re-raise specific format validation errors from within the block
        logger.error(f"XML format error: {e}")
        raise e
    except Exception as e:
        logger.warning(f"Unexpected error during XML processing: {e}")
        # Wrap unexpected errors
        raise ValueError("Unexpected error parsing XML response.") from e


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
    # Try finding JSON-like patterns with "tool_name"
    json_patterns = [
        # Pattern for standard JSON object with "tool_name" as first field
        r'{\s*"tool_name"',
        # More forgiving pattern that might catch JSON with other fields first
        r'{\s*"[^"]+"\s*:.*"tool_name"',
    ]

    for pattern in json_patterns:
        matches = re.finditer(pattern, response)
        for match in matches:
            start_idx = match.start()
            match_text = match.group(0)
            result = _extract_complete_json(response, match_text, start_idx)
            if result:
                logger.info(f"Heuristically extracted JSON: {result}")
                return result

    # Look for JSON-like object embedded within code fences or other formatting
    code_block_matches = re.finditer(r"```(?:json)?\s*({.*?})```", response, re.DOTALL)
    for match in code_block_matches:
        json_text = match.group(1)
        try:
            candidate = json.loads(json_text)
            if isinstance(candidate, dict) and "tool_name" in candidate:
                candidate.setdefault("parameters", {})
                candidate.setdefault("reason", "")
                logger.info(f"Extracted JSON from code block: {candidate}")
                return candidate
        except json.JSONDecodeError:
            continue  # Try next match

    return None


def parse_tool_response(response: str) -> Dict[str, Any]:
    """
    Accepts a single JSON object *or* a single XML <selection> element.
    Returns dict with 'tool_name', 'parameters', 'reason'.
    Raises ValueError if the input format is invalid or cannot be parsed as either JSON or XML.
    """
    # Quick check for empty responses
    if not response or not response.strip():
        raise ValueError("Tool selection response is empty.")
    
    # Strip markdown fences if present
    cleaned_response = response.strip()
    fence_match = re.match(r"^```(?:json|xml)?\s*(.*?)\s*```$", cleaned_response, re.DOTALL | re.IGNORECASE)
    if fence_match:
        cleaned_response = fence_match.group(1).strip()
        logger.debug(f"Stripped markdown fences, parsing: {cleaned_response[:200]}...")
    else:
        logger.debug("No markdown fences detected.")
        
    logger.debug(f"Parsing tool response: {cleaned_response[:500]}..." if len(cleaned_response) > 500 else cleaned_response)
    
    # First try parsing as JSON (most direct case)
    try:
        result = _parse_tool_response_json(cleaned_response)
        return result
    except ValueError as json_error:
        logger.debug(f"JSON parsing failed with: {json_error}")
        # JSON parsing failed, try XML next
    
    # Try structured XML parsing
    try:
        result = _parse_tool_response_xml(cleaned_response)
        return result
    except ValueError as xml_error:
        logger.debug(f"XML parsing failed with: {xml_error}")
        # XML parsing failed, try heuristic extraction
    
    # Attempt heuristic extraction for embedded tool specifications
    result = _try_heuristic_json_extraction(cleaned_response)
    if result:
        return result
    
    # If we get here, all parsing attempts failed
    error_message = (
        f"Failed to parse tool selection response. The response must be either:\n"
        f"1. A valid JSON object containing 'tool_name'.\n"
        f"2. A valid XML document with a <selection> root element.\n\n"
        f"Received response (cleaned, truncated):\n{cleaned_response[:300]}..." if len(cleaned_response) > 300 else cleaned_response
    )
    logger.error(error_message)
    raise ValueError(error_message)


def execute_tool(code_executor: CodeExecutor, tool_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safe wrapper that executes a registered tool with parameters cast according to type hints.
    """
    tool_name = tool_data["tool_name"]
    params = tool_data["parameters"]

    # Get the tool function from the executor's namespace
    if tool_name not in code_executor.globals_dict:
        error_msg = f"Tool '{tool_name}' not found in executor namespace."
        logger.error(error_msg)
        return {"success": False, "result": error_msg}

    tool_func = code_executor.globals_dict[tool_name]
    if not callable(tool_func):
        error_msg = f"'{tool_name}' is not a callable function."
        logger.error(error_msg)
        return {"success": False, "result": error_msg}

    # Get type hints from the function
    try:
        type_hints = get_type_hints(tool_func)
    except TypeError:
        logger.warning(f"Could not get type hints for '{tool_name}', using parameters as-is.")
        type_hints = {}
    except Exception as e:
        logger.warning(f"Unexpected error getting type hints for '{tool_name}': {e}")
        type_hints = {}

    # Cast parameters to their expected types if type hints are available
    processed_params = {}
    for param_name, param_value in params.items():
        target_type = type_hints.get(param_name)
        if target_type and param_value is not None:
            # Skip conversion if parameter is already of the target type
            if not isinstance(param_value, target_type):
                try:
                    # Special case for lists: Convert them based on concrete type hints when available
                    if hasattr(target_type, "__origin__") and target_type.__origin__ is list:
                        logger.debug(f"Converting parameter '{param_name}' to list type")
                        processed_params[param_name] = param_value if isinstance(param_value, list) else [param_value]
                    else:
                        # For primitive types, use standard conversion
                        logger.debug(f"Converting parameter '{param_name}' to {target_type}")
                        processed_params[param_name] = target_type(param_value)
                        continue  # Skip adding the original param_value
                except Exception as e:
                    logger.warning(f"Failed to convert parameter '{param_name}' to {target_type}: {e}")
                    # Fall back to original value

        # If no conversion was done or needed, use the original
        processed_params[param_name] = param_value

    # Execute the tool
    try:
        logger.info(f"Executing tool '{tool_name}' with parameters: {processed_params}")
        result = tool_func(**processed_params)
        logger.debug(f"Tool execution result: {result}")
        return {"success": True, "result": result}
    except Exception as e:
        logger.exception(f"Error executing tool '{tool_name}'")
        error_details = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
        }
        return {"success": False, "result": error_details}
