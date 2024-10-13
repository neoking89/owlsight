import os

PROMPT_COLOR = "blue"
CHOICE_COLOR = "green"


COLOR_CODES = {
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "reset": "\033[0m",  # Resets to default color
}

MENU_KEYS = {
    "assistant": "how can I assist you?",
}


def get_prompt_history_path() -> str:
    """Returns the path where all prompt history is stored."""
    return os.path.join(os.path.expanduser("~"), "owlsight" ".prompt_history")
