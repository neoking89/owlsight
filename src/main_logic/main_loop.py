import tempfile
import traceback

from src.processors.text_generation_manager import TextGenerationManager
from src.main_logic.handlers import handle_interactive_code_execution
from src.utils.code_execution import CodeExecutor, execute_code_with_feedback
from src.utils.helper_functions import (
    force_delete,
    remove_temp_directories,
    replace_bracket_placeholders,
)
from src.utils.venv_manager import get_lib_path, get_pip_path, get_venv_path
from src.utils.console import get_user_choice, print_colored
from src.utils.constants import PROMPT_COLOR


from src.utils.logger_manager import LoggerManager

logger = LoggerManager.get_logger(__name__)


def run_code_generation_loop(
    code_executor: CodeExecutor,
    manager: TextGenerationManager,
) -> None:
    """Runs the main loop for code generation and user interaction."""
    while True:
        try:
            print_colored("Make a choice:", color=PROMPT_COLOR)

            user_choice_key = None
            user_choice: str | dict = get_user_choice(
                {
                    "what can I do for you?": "",
                    "shell": "",
                    "python": None,
                    "clear history": None,
                    "config": list(manager.get_config().keys()),
                    "quit": None,
                },
                return_value_only=False,
            )

            if isinstance(user_choice, dict):
                user_choice_key = list(user_choice.keys())[0]
                user_choice: str = user_choice[user_choice_key]

            # here we know user_choice is a string
            if not user_choice:
                logger.error("User choice is empty. Please try again.")
                continue

            if user_choice_key == "shell":
                code_executor.execute_code_block(
                    lang=user_choice_key,
                    code_block=user_choice,
                )
                continue
            elif user_choice_key == "config":
                logger.info("Chosen config: " + user_choice)
                nested_config = manager.get_config_choices()[user_choice]
                config_choice: dict = get_user_choice(
                    nested_config, return_value_only=False
                )
                config_key = f"{user_choice}.{list(config_choice.keys())[0]}"
                v = list(config_choice.values())[0]
                manager.update_config(config_key, v)

                continue
            elif user_choice == "quit":
                logger.info("Quitting...")
                break
            elif user_choice == "python":
                handle_interactive_code_execution(code_executor)
            elif user_choice == "clear history":
                code_executor.globals_dict.clear()
                manager.processor.history.clear()
                logger.info("State and history cleared.")
            else:
                user_choice = replace_bracket_placeholders(
                    user_choice, code_executor.globals_dict
                )
                # user_choice is question
                response = manager.generate(
                    user_choice,
                )
                execute_code_with_feedback(
                    response=response,
                    original_question=user_choice,
                    code_executor=code_executor,
                    prompt_code_execution=manager.config_manager.get(
                        "main.prompt_code_execution"
                    ),
                )

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Restarting...")
            continue
        except Exception:
            logger.error(f"Unexpected error:\n{traceback.format_exc()}")
            raise


def main(
    manager: TextGenerationManager,
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

        code_executor = CodeExecutor(manager, venv_path, pip_path, temp_dir)

        run_code_generation_loop(
            code_executor,
            manager,
        )

    logger.info(f"Removing temporary directory: {temp_dir}")
    force_delete(temp_dir)
