from typing import List, Tuple
import os
import shutil
import re
import traceback

from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style

from src.utils.logger_manager import LoggerManager

logger = LoggerManager.get_logger(__name__)


def extract_markdown(md_string: str) -> List[Tuple[str, str]]:
    """
    Extract language and code blocks from a markdown string.
    """
    pattern = r"```(\w+)([\s\S]*?)```"
    return [
        (match[0].strip(), match[1].strip()) for match in re.findall(pattern, md_string)
    ]


def editable_input(
    prompt_text: str, default_value: str, color: str = "ansicyan"
) -> str:
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


def force_delete(temp_dir: str) -> None:
    """Forcefully deletes a directory if it exists."""
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            logger.error(
                f"Error deleting directory {temp_dir}:\n{traceback.format_exc()}"
            )


def remove_temp_directories(lib_path: str) -> None:
    """Removes lingering temporary directories in the virtual environment's library path."""
    for d in os.listdir(lib_path):
        if d.startswith("tmp"):
            logger.info(f"Removing temporary directory: {d}")
            force_delete(os.path.join(lib_path, d))


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
