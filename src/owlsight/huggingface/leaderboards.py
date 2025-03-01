"""
Module for interacting with Hugging Face leaderboards through Gradio clients.
Provides functionality to fetch and parse leaderboard data into pandas DataFrames.
"""

import re
from typing import Dict, Optional, Any, Union
import ast
from functools import lru_cache

import pandas as pd


def extract_leaderboard_data(api_text: str) -> str:
    """
    Extract the leaderboard data dictionary from the API usage text.
    Finds the first occurrence of a dictionary with 'headers' and 'data' keys.

    Args:
        api_text (str): The raw API response text containing the leaderboard data

    Returns:
        str: A string representation of the dictionary containing the leaderboard data

    Raises:
        ValueError: If the start or end of the data structure cannot be found
    """
    start_idx = api_text.find("{'headers':")
    if start_idx == -1:
        raise ValueError("Could not find start of data structure")

    # Need to count braces to find the proper end
    level = 0
    in_quotes = False

    for i in range(start_idx, len(api_text)):
        char = api_text[i]

        # Handle quotes
        if char in "\"'":
            in_quotes = not in_quotes
            continue

        if not in_quotes:
            if char == "{":
                level += 1
            elif char == "}":
                level -= 1
                if level == 0:
                    # Found the matching closing brace
                    return api_text[start_idx : i + 1]

    raise ValueError("Could not find end of data structure")


@lru_cache(maxsize=64)
def get_leaderboard_data(leaderboard_id: str) -> pd.DataFrame:
    """
    Fetch and parse data from a Hugging Face leaderboard.

    Parameters:
    ----------
    leaderboard_id (str): The identifier for the leaderboard, e.g., 'mteb/leaderboard'
                            or 'bigcode/bigcode-models-leaderboard'

    Returns:
    -------
    pd.DataFrame: A pandas DataFrame containing the leaderboard data with appropriate columns

    Raises:
    ------
    ValueError: If the leaderboard data cannot be parsed
    ImportError: If gradio is not installed
    """
    try:
        from gradio_client import Client
    except ImportError:
        raise ImportError(
            "gradio is required for this function. "
            "Please install it with 'pip install owlsight[huggingface]' or 'pip install gradio'"
        )

    client = Client(leaderboard_id)
    data_str = extract_leaderboard_data(str(client))
    data_dict: Dict[str, Any] = ast.literal_eval(data_str)
    return pd.DataFrame(data_dict["data"], columns=data_dict["headers"])


def convert_params_to_number(value_str: str) -> Union[float, None]:
    """
    Convert string numbers with 'B' (billions) or 'M' (millions) to float values.

    Parameters:
    ----------
    value_str (str): String representation of a number (e.g., '7B', '559M')

    Returns:
    -------
    Union[float, None]: Numerical value in the base unit, or None if conversion fails
    """
    try:
        # Handle 'Unknown' case
        if value_str.lower() == "unknown":
            return None

        # Remove any whitespace
        value_str = value_str.strip()

        # Get the multiplier based on the unit
        if value_str.endswith("B"):
            multiplier = 1_000_000_000  # Billion
            value = float(value_str[:-1])
        elif value_str.endswith("M"):
            multiplier = 1_000_000  # Million
            value = float(value_str[:-1])
        else:
            # If no unit, try to convert directly
            return float(value_str)

        return value * multiplier

    except (ValueError, AttributeError):
        return None


def parse_huggingface_repo(input_text: str) -> Dict[str, Optional[str]]:
    """Parse Hugging Face repository references from various formats.

    Parameters
    ----------
        input_text: String in format of markdown link, direct path (org/model),
                   or full HF URL

    Returns:
    -------
        Dict with keys: organization, model_name, full_path, original_input, url

    Example:
    ---------
        >>> parse_huggingface_repo("Lajavaness/bilingual-embedding-large")
        {'organization': 'Lajavaness', 'model_name': 'bilingual-embedding-large', ...}
    """
    # Initialize default return structure
    result = {"organization": None, "model_name": None, "full_path": None, "original_input": input_text, "url": None}

    # Extract URL from markdown link if present
    markdown_match = re.search(r"\[(.*?)\]\((https?:\/\/[^)]+)\)", input_text)
    if markdown_match:
        input_text = markdown_match.group(2)

    # Try to match URL or direct path format
    url_match = re.search(r"(?:https?:\/\/(?:www\.)?huggingface\.co\/)?([^/\s]+)\/([^/\s]+)", input_text)

    if url_match:
        result["organization"] = url_match.group(1)
        result["model_name"] = url_match.group(2)
        result["full_path"] = f"{result['organization']}/{result['model_name']}"

        # Only set URL if it's not already a URL
        if "huggingface.co" in input_text:
            result["url"] = input_text
        else:
            result["url"] = f"https://huggingface.co/{result['full_path']}"

    return result


def get_mteb_leaderboard_data(max_params: Optional[int] = None) -> pd.DataFrame:
    """
    Fetch and parse data from the MTEB leaderboard, focussed on text embedding models.

    Parameters:
    ----------
    max_params : Optional[int], default None
        Maximum number of parameters for filtering

    Returns:
    -------
    pd.DataFrame
        DataFrame containing the MTEB leaderboard data with appropriate columns
    """
    leaderboard = "mteb/leaderboard"
    df = get_leaderboard_data(leaderboard)
    df["Number of Parameters"] = df["Number of Parameters"].apply(convert_params_to_number)
    _ = df["Model"].apply(parse_huggingface_repo)
    df["Model"] = _.apply(lambda x: x["full_path"])
    df["Url"] = _.apply(lambda x: x["url"])
    if max_params is not None:
        df = df.where(df["Number of Parameters"] < max_params)
    return df
