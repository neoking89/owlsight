import tempfile
import traceback
from typing import Dict, Union
from enum import Enum, auto

from processors.text_generation_manager import TextGenerationManager
from app.handlers import handle_interactive_code_execution
from utils.code_execution import CodeExecutor, execute_code_with_feedback
from utils.helper_functions import (
    force_delete,
    remove_temp_directories,
    replace_bracket_placeholders,
)
from utils.venv_manager import get_lib_path, get_pip_path, get_pyenv_path
from utils.console import get_user_choice, print_colored
from utils.constants import PROMPT_COLOR
from utils.deep_learning import free_memory
from utils.logger_manager import LoggerManager

logger = LoggerManager.get_logger(__name__)


class CommandResult(Enum):
    CONTINUE = auto()
    BREAK = auto()
    PROCEED = auto()


def run_code_generation_loop(code_executor: CodeExecutor, manager: TextGenerationManager) -> None:
    """Runs the main loop for code generation and user interaction."""
    while True:
        try:
            print_colored("Make a choice:", color=PROMPT_COLOR)
            user_choice, choice_key = get_user_input(manager)

            if not user_choice:
                logger.error("User choice is empty. Please try again.")
                continue

            command_result = handle_special_commands(choice_key, user_choice, code_executor, manager)
            if command_result == CommandResult.BREAK:
                break
            elif command_result == CommandResult.CONTINUE:
                continue

            if manager.processor is None:
                logger.error("Processor not set. Please load a model first by setting 'model.model_id' in the config!")
                continue
            else:
                process_user_question(user_choice, code_executor, manager)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Restarting...")
        except Exception:
            logger.error(f"Unexpected error:\n{traceback.format_exc()}")
            raise


def get_user_input(manager: TextGenerationManager) -> tuple[str, Union[str, None]]:
    user_choice: Union[str, Dict] = get_user_choice(
        {
            "how can I assist you?": "",
            "shell": "",
            "python": None,
            "config": list(manager.get_config().keys()),
            "save": "",
            "load": "",
            "clear history": None,
            "quit": None,
        },
        return_value_only=False,
    )

    if isinstance(user_choice, dict):
        choice_key = list(user_choice.keys())[0]
        return user_choice[choice_key], choice_key
    return user_choice, None


def handle_special_commands(
    choice_key: Union[str, None],
    user_choice: str,
    code_executor: CodeExecutor,
    manager: TextGenerationManager,
) -> CommandResult:
    if choice_key == "shell":
        code_executor.execute_code_block(lang=choice_key, code_block=user_choice)
        return CommandResult.CONTINUE
    elif choice_key == "config":
        config_key = ""
        while not config_key.endswith("back"):
            config_key = handle_config_update(user_choice, manager)
        return CommandResult.CONTINUE
    elif choice_key == "save":
        manager.save_config(user_choice)
        return CommandResult.CONTINUE
    elif choice_key == "load":
        manager.load_config(user_choice)
        return CommandResult.CONTINUE
    elif user_choice == "python":
        handle_interactive_code_execution(code_executor)
        return CommandResult.CONTINUE
    elif user_choice == "clear history":
        clear_history(code_executor, manager)
        return CommandResult.CONTINUE
    elif user_choice == "quit":
        logger.info("Quitting...")
        return CommandResult.BREAK
    return CommandResult.PROCEED


def handle_config_update(user_choice: str, manager: TextGenerationManager) -> str:
    logger.info(f"Chosen config: {user_choice}")

    # Retrieve nested configuration options
    available_choices = manager.get_config_choices()
    selected_config = available_choices[user_choice]

    # Get user choice for the nested configuration
    user_selected_choice = get_user_choice(selected_config, return_value_only=False)

    if isinstance(user_selected_choice, dict):
        nested_key = next(iter(user_selected_choice))  # Get the first key
        config_value = user_selected_choice[nested_key]  # Get the corresponding value
    else:
        nested_key = user_selected_choice
        config_value = None

    # Construct the config key and update the configuration
    config_key = f"{user_choice}.{nested_key}"
    manager.update_config(config_key, config_value)

    return config_key


def clear_history(code_executor: CodeExecutor, manager: TextGenerationManager) -> None:
    code_executor.globals_dict.clear()
    if manager.processor is not None:
        manager.processor.history.clear()
    logger.info("State and history cleared.")


def process_user_question(user_choice: str, code_executor: CodeExecutor, manager: TextGenerationManager) -> None:
    user_choice = replace_bracket_placeholders(user_choice, code_executor.globals_dict)
    response = manager.generate(user_choice)
    execute_code_with_feedback(
        response=response,
        original_question=user_choice,
        code_executor=code_executor,
        prompt_code_execution=manager.config_manager.get("main.prompt_code_execution"),
    )


def run(manager: TextGenerationManager) -> None:
    """
    Main function to run the interactive loop for code generation and execution

    Parameters
    ----------
    manager : TextGenerationManager
        TextGenerationManager instance to handle the code generation and execution
    """
    pyenv_path = get_pyenv_path()
    lib_path = get_lib_path(pyenv_path)
    pip_path = get_pip_path(pyenv_path)

    # Remove lingering temporary directories
    remove_temp_directories(lib_path)

    # Create temporary directory in venv to install packages, until end of execution lifecycle
    with tempfile.TemporaryDirectory(dir=lib_path) as temp_dir:
        logger.info(f"Temporary directory created at: {temp_dir}")

        code_executor = CodeExecutor(manager, pyenv_path, pip_path, temp_dir)

        run_code_generation_loop(code_executor, manager)

    logger.info(f"Removing temporary directory: {temp_dir}")
    free_memory()
    force_delete(temp_dir)
