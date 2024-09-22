import code
import os
import re
from typing import Dict, List
import traceback

from src.processors.text_generation_manager import TextGenerationManager
from src.utils.custom_exceptions import ModuleNotFoundInVenvError
from src.utils.subprocess_utils import execute_shell_command
from src.utils.helper_functions import (
    extract_markdown,
    editable_input,
    format_error_message,
)
from src.utils.console import get_user_choice
from src.utils.venv_manager import install_module, get_lib_path, get_python_executable
from src.utils.constants import PROMPT_COLOR

from src.utils.logger_manager import LoggerManager

logger = LoggerManager.get_logger(__name__)


def extract_missing_module(stderr: str) -> str:
    match = re.search(r"No module named '(\w+)'", stderr)
    return match.group(1) if match else None


class CodeExecutor:
    def __init__(
        self,
        manager: TextGenerationManager,
        venv_path: str,
        pip_path: str,
        temp_dir: str,
    ):
        self.manager = manager
        self.temp_dir = temp_dir
        self.globals_dict = {}

        self._init_python_properties(venv_path, pip_path)
        self._reset_retries()

    def execute_and_retry(
        self, lang: str, code_block: str, original_question: str
    ) -> bool:
        """
        Execute code block in the specified language and retry if an error occurs.
        """
        while self.retries_left > 0:
            logger.info(
                f"Executing {lang.capitalize()} code (Attempt {self._get_nth_attempt()}/{self.max_retries})..."
            )
            try:
                self.execute_code_block(lang, code_block)
                logger.info(f"Code executed on attempt {self._get_nth_attempt()}.")
                return True
            except Exception as e:
                self.retries_left -= 1
                if self.retries_left > 0:
                    logger.warning(f"Error on attempt {self._get_nth_attempt()}: {e}")
                    logger.info(
                        f"Retrying... ({self._get_nth_attempt()}/{self.max_retries})"
                    )
                    response_with_fixed_code = self._generate_fixed_code_response(
                        original_question, code_block, format_error_message(e)
                    )
                    code_block = (
                        extract_markdown(response_with_fixed_code)[0][1]
                        if response_with_fixed_code
                        else code_block
                    )
                else:
                    logger.error(
                        f"Failed to execute {lang} code after {self.max_retries} attempts."
                    )

        self._reset_retries()

        return False

    def execute_code_block(self, lang: str, code_block: str) -> None:
        if lang == "python":
            self.execute_python_code(code_block)
        elif lang in ["cmd", "bash", "shell"]:
            if "pip install" in code_block:
                module_to_install = code_block.split("pip install")[1].strip()
                logger.info(
                    f"pip install found in command '{code_block}'. Installing module {module_to_install} to target directory {self.temp_dir}"
                )
                self.pip_install(module_to_install)
            else:
                execute_shell_command(code_block, self.venv_path)
        else:
            logger.warning(f"Unsupported language: {lang}")

    def execute_python_code(self, code_block: str) -> None:
        """Execute Python code block."""
        try:
            exec(code_block, self.globals_dict)
        except ModuleNotFoundError as e:
            logger.error(f"Module not found: {e}")
            missing_module = extract_missing_module(str(e))
            module_is_installed = self.pip_install(missing_module)
            if module_is_installed:
                if missing_module not in os.listdir(self.temp_dir):
                    raise ModuleNotFoundInVenvError(
                        missing_module,
                        self.venv_path,
                        os.listdir(self.temp_dir),
                    )
                logger.info(f"Retrying execution after installing {missing_module}")
                self.execute_python_code(code_block)  # Retry execution
            else:
                logger.error(
                    f"Failed to install {missing_module}. Cannot execute the code."
                )
        except Exception as e:
            logger.error(f"Error executing code: {traceback.format_exc()}")
            raise e

    def init_interactive_py_console(self) -> None:
        """Initialize an interactive Python console."""
        console = code.InteractiveConsole(self.globals_dict)

        try:
            console.interact(
                "Interactive Python interpreter activated. Type 'exit()' to quit the console."
            )
        except SystemExit:
            logger.info("Exiting interactive console and returning to the script.")

    def pip_install(self, module: str) -> bool:
        """Install a Python module using pip."""
        logger.info(f"Attempting to install module: {module}")
        extra_index_url = self.manager.get_config_key("main.extra_index_url")
        if extra_index_url:
            module_is_installed = install_module(
                module,
                self.pip_path,
                self.temp_dir,
                "--extra-index-url",
                extra_index_url,
            )
        else:
            module_is_installed = install_module(module, self.pip_path, self.temp_dir)

        return module_is_installed

    @property
    def max_retries(self) -> int:
        return self.manager.config_manager.get("main.max_retries_on_error")

    def _reset_retries(self) -> None:
        self.retries_left = self.max_retries

    def _get_nth_attempt(self) -> int:
        return self.max_retries - self.retries_left + 1

    def _generate_fixed_code_response(
        self, original_question: str, code_block: str, error: str
    ) -> str:
        new_question = f"""\
# ORIGINAL QUESTION:
{original_question}

# ANSWER WHICH GENERATED THE ERROR:
{code_block}

# ERROR:
{error}

# TASK:
1. Analyze the error message.
2. Step-by-step, determine how to fix the error.
3. Generate updated Python code that resolves the issue.
""".strip()
        return self.manager.generate(new_question)

    def _init_python_properties(self, venv_path: str, pip_path: str):
        self.venv_path = venv_path
        self.lib_path = get_lib_path(venv_path)
        self.python_executable = get_python_executable(venv_path)
        self.pip_path = pip_path


def execute_code_with_feedback(
    response: str,
    original_question: str,
    code_executor: CodeExecutor,
    prompt_code_execution: bool = True,
) -> List[Dict]:
    """
    Extract code blocks from a response and execute them with feedback and retry logic.

    Parameters
    ----------
    response : str
        The response containing the code blocks in markdown format.
    original_question : str
        The original question that prompted the code execution.
    code_executor : CodeExecutor
        An instance of CodeExecutor that handles code execution.
    prompt_code_execution : bool
        If True, prompts the user before executing each code block.
        Acts as a safety measure to prevent accidental execution.

    Returns
    -------
    List[Dict]
        A list of dictionaries with execution results, including success status, language, and code.
    """
    results = []

    # Extract code blocks with their associated language
    code_blocks = extract_markdown(response)
    if not code_blocks:
        logger.info("No code blocks found in the response.")
        return results

    # Iterate over extracted code blocks
    for lang, code_block in code_blocks:
        execute_code = True
        code_is_edited = False
        if prompt_code_execution:
            while True:

                # Use the editable_input function to allow users to edit the code block
                if not code_is_edited:
                    logger.info(f"Code block in {lang.capitalize()}:\n{code_block}")
                    code_block = editable_input(
                        "Edit the code block (press ENTER to confirm):\n",
                        code_block,
                        color=PROMPT_COLOR,
                    )
                    logger.info(f"Edited Code Block:\n{code_block}")
                    code_is_edited = True

                # Provide a menu for the user to choose between "Execute", "Skip", or "Write code to file"
                user_choice = get_user_choice(
                    {
                        "Execute code": None,
                        "Skip code": None,
                        "Write code to file": None,
                    }
                )

                if user_choice == "Execute code":
                    logger.info("Executing code block.")
                    break  # Exit the while loop and execute the code
                elif user_choice == "Skip code":
                    logger.info("Skipping code block.")
                    execute_code = False
                    break  # Exit the while loop and skip execution
                elif user_choice == "Write code to file":
                    # Handle writing to a file or going back
                    _handle_write_code_to_file_choice(code_block)
                    # After handling file, stay in the menu for further selection
                    continue  # Stay in the while loop to allow more choices

        if not execute_code:
            continue  # Skip execution if skip code is selected
        is_success = code_executor.execute_and_retry(
            lang, code_block, original_question
        )
        result = {"success": is_success, "language": lang, "code": code_block}
        results.append(result)

    return results


def _handle_write_code_to_file_choice(code_block: str):
    """
    Handles the process of writing a code block to a file, providing options for entering
    a filename or returning to the main menu.

    Parameters
    ----------
    code_block : str
        The code block to write to the file.
    """
    while True:
        file_choice = get_user_choice(
            {
                "Enter filename": None,
                "Go back": None,
            }
        )

        if file_choice == "Go back":
            logger.info("Returning to the main menu.")
            return  # Return to the main menu

        elif file_choice == "Enter filename":
            file_name = input("Enter the filename: ")
            if file_name:  # If a filename is entered
                try:
                    with open(file_name, "w") as f:
                        f.write(code_block)
                        logger.info(f"Code block written to file: {file_name}")
                    # After writing, return to the main menu without breaking the loop
                    return
                except Exception as e:
                    logger.error(
                        f"Error writing code block to file: {e}. Please try again."
                    )
            else:
                logger.info("No file name entered. Please try again.")
