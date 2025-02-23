"""
Module for interacting with Hugging Face leaderboards through Gradio clients.
Provides functionality to fetch and parse leaderboard data into pandas DataFrames.
"""

from typing import Dict, Any
from gradio_client import Client
import ast
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
    """
    client = Client(leaderboard_id)
    data_str = extract_leaderboard_data(str(client))
    data_dict: Dict[str, Any] = ast.literal_eval(data_str)
    
    return pd.DataFrame(data_dict["data"], columns=data_dict["headers"])
