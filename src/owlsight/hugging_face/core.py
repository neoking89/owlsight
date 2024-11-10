from functools import lru_cache
from typing import Dict, List
from .helper_functions import get_model_gen, _get_hf_model_data


def create_separator(width: int = 80, char: str = "=") -> str:
    """Create a separator line with given width and character."""
    return char * width


def format_table_row(columns: List[str], widths: List[int]) -> str:
    """Format a row with proper spacing and alignment."""
    return "│ " + " │ ".join(col.ljust(width) for col, width in zip(columns, widths)) + " │"


@lru_cache(maxsize=128)
def get_model_data(model_search: str, top_n_models: int = 10) -> Dict[str, Dict[str, str]]:
    """
    Get and display the model data from the HuggingFace Hub in a visually appealing format.

    Args:
        model_search: Search term for filtering models
        top_n_models: Number of top models to display

    Returns:
        Dictionary containing model information
    """
    model_gen = get_model_gen(top_n=top_n_models, search=model_search)
    model_dict = {model_info.modelId: _get_hf_model_data(model_info) for model_info in model_gen}
    return model_dict


def calculate_column_widths(model_dict: Dict[str, Dict[str, str]]) -> tuple:
    """Calculate optimal column widths based on content."""
    max_model_id = max(len(model_id) for model_id in model_dict.keys())
    max_detail = max(
        len(f"{key}: {str(value)}") for model_data in model_dict.values() for key, value in model_data.items()
    )

    RANK_WIDTH = 4
    MODEL_ID_WIDTH = max(max_model_id + 2, 30)  # minimum 30 chars
    DETAIL_WIDTH = max(max_detail + 2, 40)  # minimum 40 chars
    TOTAL_WIDTH = RANK_WIDTH + MODEL_ID_WIDTH + DETAIL_WIDTH + 10  # padding and separators

    return RANK_WIDTH, MODEL_ID_WIDTH, DETAIL_WIDTH, TOTAL_WIDTH


def show_and_return_model_data(model_search: str, top_n_models: int = 10) -> Dict[str, Dict[str, str]]:
    """Display model data in a formatted table with dynamic sizing."""
    model_dict = get_model_data(model_search, top_n_models)

    # Return early if no models found
    if not model_dict:
        return model_dict

    # Calculate optimal widths based on content
    RANK_WIDTH, MODEL_ID_WIDTH, DETAIL_WIDTH, WIDTH = calculate_column_widths(model_dict)

    # Print header
    print("\n" + create_separator(WIDTH))
    header = f"Available Models for '{model_search}' (Top {top_n_models})"
    print(f"║{header.center(WIDTH-2)}║")
    print(create_separator(WIDTH, "="))

    # Print column headers
    print(format_table_row(["Rank", "Model ID", "Details"], [RANK_WIDTH, MODEL_ID_WIDTH, DETAIL_WIDTH]))
    print(create_separator(WIDTH, "-"))

    # Print each model's information
    for idx, (model_id, model_data) in enumerate(model_dict.items(), 1):
        # Print first row with rank and model ID
        print(format_table_row([str(idx), model_id, "Details:"], [RANK_WIDTH, MODEL_ID_WIDTH, DETAIL_WIDTH]))

        # Print model details
        for key, value in model_data.items():
            formatted_value = str(value).replace("\n", " ").strip()
            detail_text = f"{key}: {formatted_value}"
            print(format_table_row(["", "", detail_text], [RANK_WIDTH, MODEL_ID_WIDTH, DETAIL_WIDTH]))

        # Add separator between models
        if idx < len(model_dict):
            print(create_separator(WIDTH, "-"))

    # Print footer
    print(create_separator(WIDTH))
    print()  # Add final newline for cleaner output

    return model_dict


def format_key_value(key: str, value: str, indent: int = 0) -> str:
    """Format a key-value pair with proper indentation."""
    indent_str = " " * indent
    return f"{indent_str}{key}: {value}"
