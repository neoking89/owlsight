from typing import List, Optional
import tempfile
import traceback

from src.processors.text_generation import TextGenerationProcessor
from src.main_logic.handlers import handle_interactive_code_execution
from src.utils.code_execution import CodeExecutor, execute_code_with_feedback
from src.utils.helper_functions import force_delete, remove_temp_directories
from src.utils.venv_manager import get_lib_path, get_pip_path, get_venv_path
from src.utils.console import choose_from_prompt_and_menu, print_colored
from src.utils.constants import PROMPT_COLOR


from src.utils.logger_manager import LoggerManager

logger = LoggerManager.get_logger(__name__)


def run_code_generation_loop(
    code_executor: CodeExecutor,
    processor: TextGenerationProcessor,
    max_new_tokens: int,
    stopwords: Optional[List[str]],
    generation_kwargs: Optional[dict],
    prompt_code_execution: bool,
) -> None:
    """Runs the main loop for code generation and user interaction."""
    while True:
        try:
            print_colored("Make a choice:", color=PROMPT_COLOR)
            # Use choose_from_prompt_and_menu to gather user input with menu options
            prompt = "What can I do for you?"
            initial_input = ""  # Start with an empty string for the initial input
            menu_choices = ["python", "clear history", "quit"]

            question = choose_from_prompt_and_menu(prompt, initial_input, menu_choices)

            # Mapping menu choices to actual commands
            if question == "quit":
                logger.info("Quitting...")
                break
            elif question == "python":
                handle_interactive_code_execution(code_executor)
            elif question == "clear history":
                code_executor.globals_dict.clear()
                processor.history.clear()
                logger.info("State and history cleared.")
            else:
                # Handle free-form input (anything not matching the predefined menu options)
                response = processor.generate(
                    question,
                    max_new_tokens=max_new_tokens,
                    stopwords=stopwords,
                    generation_kwargs=generation_kwargs,
                )
                execute_code_with_feedback(
                    response,
                    question,
                    code_executor,
                    prompt_code_execution=prompt_code_execution,
                )

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Restarting...")
            continue
        except Exception:
            logger.error(f"Unexpected error:\n{traceback.format_exc()}")
            raise


def main(
    processor: TextGenerationProcessor,
    max_retries: int = 3,
    max_new_tokens: int = 1024,
    stopwords: Optional[List[str]] = None,
    generation_kwargs: Optional[dict] = None,
    prompt_code_execution: bool = True,
) -> None:
    """
    Main function to run the interactive loop for code generation and execution

    Parameters
    ----------
    processor : TextGenerationProcessor
        The text generation processor to use for generating code.
    max_retries : int, optional
        The maximum number of retries for code execution, by default 3
    max_new_tokens : int, optional
        The maximum number of new tokens to generate, by default 1024
    stopwords : Optional[List[str]], optional
        List of stopwords to stop generation at, by default None
    generation_kwargs : Optional[dict], optional
        Additional keyword arguments for model generation, by default None
        For example: {"top_k": 50, "top_p": 0.95}
    prompt_code_execution : bool, optional
        Whether to prompt the user before executing the generated code, by default True
    """
    venv_path = get_venv_path()
    lib_path = get_lib_path(venv_path)
    pip_path = get_pip_path(venv_path)

    # Remove lingering temporary directories
    remove_temp_directories(lib_path)

    # Create temporary directory in venv to install packages
    with tempfile.TemporaryDirectory(dir=lib_path) as temp_dir:
        logger.info(f"Temporary directory created at: {temp_dir}")

        code_executor = CodeExecutor(
            processor, venv_path, pip_path, temp_dir, max_retries, max_new_tokens
        )

        run_code_generation_loop(
            code_executor,
            processor,
            max_new_tokens,
            stopwords,
            generation_kwargs,
            prompt_code_execution,
        )

    logger.info(f"Removing temporary directory: {temp_dir}")
    force_delete(temp_dir)
