import tempfile
import traceback
from typing import Union, Tuple
from enum import Enum, auto
import os
from pathlib import Path

from owlsight.configurations.constants import MAIN_MENU
from owlsight.ui.file_dialogs import save_file_dialog, open_file_dialog
from owlsight.ui.console import get_user_choice, get_user_input
from owlsight.ui.custom_classes import AppDTO
from owlsight.processors.text_generation_manager import TextGenerationManager
from owlsight.app.handlers import handle_interactive_code_execution
from owlsight.utils.code_execution import CodeExecutor
from owlsight.utils.helper_functions import (
    force_delete,
    remove_temp_directories,
    extract_square_bracket_tags,
    os_is_windows,
    parse_python_placeholders,
)
from owlsight.utils.venv_manager import get_lib_path, get_pip_path, get_pyenv_path, get_temp_dir
from owlsight.utils.constants import (
    get_cache_dir,
    get_pickle_cache,
    get_prompt_cache,
    get_py_cache,
    get_default_config_on_startup_path,
)
from owlsight.utils.deep_learning import free_cuda_memory
from owlsight.processors.helper_functions import warn_processor_not_loaded
from owlsight.app.agentic import AgentOrchestrator
from owlsight.utils.logger import logger


class CommandResult(Enum):
    """Enum to represent the result of a command from the main menu."""

    CONTINUE = auto()
    BREAK = auto()
    PROCEED = auto()


# Replace the old process_user_question with the new orchestrated version
def process_user_question(
    user_choice: str,
    code_executor: CodeExecutor,
    manager: TextGenerationManager,
    max_steps: int = 3,
    current_step: int = 0,
) -> str:
    """
    Process the user's choice and generate a response.
    Optionally involves multi-step tool usage and result validation.
    """
    orchestrator = AgentOrchestrator()
    return orchestrator.process_user_question(user_choice, code_executor, manager, max_steps, current_step)


def handle_assistant_prompt(user_choice: str, manager: TextGenerationManager, code_executor: CodeExecutor) -> None:
    user_choice_list = extract_square_bracket_tags(user_choice, tag=["load", "chain"], key="params")
    load_tags_present = any(isinstance(item, dict) and item["tag"] == "load" for item in user_choice_list)

    if manager.processor is None and not load_tags_present:
        warn_processor_not_loaded()
        return

    _load_tag = "[[load:"
    if load_tags_present and not user_choice.startswith(_load_tag):
        logger.error(f"Load tags present, but user choice does not start with '{_load_tag}'. Please correct the input.")
        return

    for choice in user_choice_list:
        if isinstance(choice, dict):
            params = choice["params"]
            if choice["tag"] == "load":
                logger.info(f"load tag detected. Loading {params}...")
                if not manager.load_config(params):
                    logger.error(f"Failed to load configuration from {params}. Stopping...")
                    break
            elif choice["tag"] == "chain":
                logger.info("Chain tag detected. Splitting parameters...")
                for param in params.split("||"):
                    key, value = _extract_params_chain_tag(param)
                    if not key:
                        continue
                    if manager.get_config_key(key, None) is None:
                        logger.error(f"Invalid chain parameter: {param}. Key '{key}' not found in config.")
                        continue
                    manager.update_config(key, value)
        else:
            max_steps = manager.get_config_key("agentic.max_steps", 3)
            _ = process_user_question(choice, code_executor, manager, max_steps=max_steps)


def clear_history(code_executor: CodeExecutor, manager: TextGenerationManager) -> None:
    """
    Clears:
    - Variables in the Python interpreter (except those starting with "owl_")
    - Python interpreter history file
    - Prompt history file
    - Chat history in the processor
    - Pickled cache files
    """
    # keep only "owl_*" variables
    temp_dict = {k: v for k, v in code_executor.globals_dict.items() if k.startswith("owl_")}
    code_executor.globals_dict.clear()
    code_executor.globals_dict.update(temp_dict)

    # remove all files in cache folder except the default config file
    cache_dir = get_cache_dir()
    default_config_on_startup_path = get_default_config_on_startup_path(return_cache_path=True)
    files_in_cache_dir = [Path(cache_dir) / path for path in os.listdir(cache_dir)]
    files_in_cache_dir = [file_path for file_path in files_in_cache_dir if file_path != default_config_on_startup_path]

    for file_path in files_in_cache_dir:
        if file_path.is_dir():
            file_path.rmdir()
        else:
            file_path.unlink()

    # clear manager state
    if manager.processor is not None:
        manager.processor.chat_history.clear()
    manager._tool_history.clear()

    logger.info(f"Cleared files in cachefolder '{get_cache_dir()}' and model chathistory.")

    # initialize empty cache files again
    get_pickle_cache()
    get_prompt_cache()
    get_py_cache()


def run_code_generation_loop(code_executor: CodeExecutor, manager: TextGenerationManager) -> None:
    """Runs the main loop for code generation and user interaction."""
    option = None
    user_choice = None
    while True:
        try:
            _option_or_userchoice: bool = option or user_choice
            if _option_or_userchoice:
                start_index = list(MAIN_MENU.keys()).index(_option_or_userchoice)
            else:
                start_index = 0
            user_choice, option = get_user_input(start_index=start_index)

            if not user_choice and option not in ["config", "save", "load"]:
                logger.error("User choice is empty. Please try again.")
                continue

            command_result = handle_special_commands(option, user_choice, code_executor, manager)
            if command_result == CommandResult.BREAK:
                break
            elif command_result == CommandResult.CONTINUE:
                continue

            user_choice = parse_python_placeholders(user_choice, code_executor.globals_dict)
            if not isinstance(user_choice, str):
                logger.error(
                    f"User choice is not a string, but {type(user_choice).__name__}. "
                    "Please only use curly braces '{{expression}}' if the end result "
                    "from the python expression is a string."
                )
                continue
            handle_assistant_prompt(user_choice, manager, code_executor)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Returning to main menu.")
        except Exception:
            logger.error(f"Unexpected error:\n{traceback.format_exc()}")


def handle_special_commands(
    choice_key: Union[str, None],
    user_choice: str,
    code_executor: CodeExecutor,
    manager: TextGenerationManager,
) -> CommandResult:
    """Handles special commands such as shell, config, save, load, python, clear history, and quit."""
    if choice_key == "shell":
        code_executor.execute_code_block(lang=choice_key, code_block=user_choice)
        return CommandResult.CONTINUE
    elif choice_key == "config":
        config_key = ""
        while not config_key.endswith("back"):
            config_key = handle_config_update(user_choice, manager)
        return CommandResult.CONTINUE
    elif choice_key == "save":
        if not user_choice and os_is_windows():
            file_path = save_file_dialog(initial_dir=os.getcwd(), default_filename="owlsight_config.json")
            if not file_path:
                logger.error("No file selected. Please try again.")
                return CommandResult.CONTINUE
            user_choice = file_path
        manager.save_config(user_choice)
        return CommandResult.CONTINUE
    elif choice_key == "load":
        if not user_choice and os_is_windows():
            file_path = open_file_dialog(initial_dir=os.getcwd())
            if not file_path:
                logger.error("No file selected. Please try again.")
                return CommandResult.CONTINUE
            user_choice = file_path
        manager.load_config(user_choice)
        return CommandResult.CONTINUE
    elif user_choice == "python":
        python_compile_mode = manager.get_config_key("main.python_compile_mode", "single")
        code_executor.python_compile_mode = python_compile_mode
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
    """Handles updating the configuration based on the user's choice."""
    logger.info(f"Chosen config: {user_choice}")

    # Retrieve nested configuration options
    available_choices = manager.get_config_choices()
    selected_config = available_choices[user_choice]

    # Get user choice for the nested configuration
    app_dto = AppDTO(return_value_only=False, last_config_choice=user_choice)
    user_selected_choice = get_user_choice(selected_config, app_dto)

    if isinstance(user_selected_choice, dict):
        nested_key = next(iter(user_selected_choice))  # Get the first key
        config_value = user_selected_choice[nested_key]
    else:
        nested_key = user_selected_choice
        config_value = None

    config_key = f"{user_choice}.{nested_key}"
    manager.update_config(config_key, config_value)

    return config_key


def run(manager: TextGenerationManager) -> None:
    """
    Main function to run the interactive loop for code generation and execution
    """
    pyenv_path = get_pyenv_path()
    lib_path = get_lib_path(pyenv_path)
    pip_path = get_pip_path(pyenv_path)

    remove_temp_directories(lib_path)

    temp_dir_location = get_temp_dir(".owlsight_packages")

    with tempfile.TemporaryDirectory(dir=temp_dir_location) as temp_dir:
        logger.info(f"Temporary directory created at: {temp_dir}")
        code_executor = CodeExecutor(manager, pyenv_path, pip_path, temp_dir)
        on_app_startup(manager)
        run_code_generation_loop(code_executor, manager)

    logger.info(f"Removing temporary directory: {temp_dir}")
    free_cuda_memory()
    force_delete(temp_dir)


def on_app_startup(manager: TextGenerationManager):
    """Functionality to execute when the CLI starts up."""
    default_config_path = get_default_config_on_startup_path(return_cache_path=False)
    if default_config_path:
        manager.load_config(default_config_path)
        logger.info(f"Loaded settings from default config '{default_config_path}'")


def _extract_params_chain_tag(param: str) -> Tuple[str, str]:
    """
    Extracts the key and value from a chain parameter string.

    Args:
        param (str): The chain parameter string in the format "param=value".

    Returns:
        Tuple[str, str]: A tuple containing the key and value extracted from the parameter string.
    """
    try:
        key, value = param.split("=")
    except Exception as e:
        logger.error(f"Invalid chain parameter: {param}. Use 'param=value' format.\nException: {e}")
        return "", ""
    key = key.strip()
    value = value.strip()
    return key, value
