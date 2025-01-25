import tempfile
import traceback
from typing import Union
from enum import Enum, auto
import os

from owlsight.configurations.constants import MAIN_MENU
from owlsight.ui.file_dialogs import save_file_dialog, open_file_dialog
from owlsight.ui.console import get_user_choice, get_user_input
from owlsight.ui.custom_classes import AppDTO
from owlsight.processors.text_generation_manager import TextGenerationManager
from owlsight.app.handlers import handle_interactive_code_execution
from owlsight.utils.code_execution import CodeExecutor, execute_code_with_feedback
from owlsight.utils.helper_functions import (
    force_delete,
    remove_temp_directories,
    parse_media_placeholders,
    os_is_windows,
)
from owlsight.utils.venv_manager import get_lib_path, get_pip_path, get_pyenv_path, get_temp_dir
from owlsight.utils.constants import (
    get_cache_dir,
    get_pickle_cache,
    get_prompt_cache,
    get_py_cache,
)
from owlsight.utils.deep_learning import free_cuda_memory
from owlsight.rag.python_lib_search import PythonLibSearcher
from owlsight.processors.helper_functions import warn_processor_not_loaded
from owlsight.utils.logger import logger


class CommandResult(Enum):
    """Enum to represent the result of a command from the mainmenu."""

    CONTINUE = auto()
    BREAK = auto()
    PROCEED = auto()


def run_code_generation_loop(code_executor: CodeExecutor, manager: TextGenerationManager) -> None:
    """Runs the main loop for code generation and user interaction."""
    option = None
    user_choice = None
    while True:
        try:
            # define startindex of arrow in mainmenu
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

            if manager.processor is None:
                warn_processor_not_loaded()
                continue
            else:
                process_user_question(user_choice, code_executor, manager)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Restarting...")
        except Exception:
            logger.error(f"Unexpected error:\n{traceback.format_exc()}")
            # raise


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
        config_value = user_selected_choice[nested_key]  # Get the corresponding value
    else:
        nested_key = user_selected_choice
        config_value = None

    # Construct the config key and update the configuration
    config_key = f"{user_choice}.{nested_key}"
    manager.update_config(config_key, config_value)

    return config_key


def clear_history(code_executor: CodeExecutor, manager: TextGenerationManager) -> None:
    """Clears the following things:

    - All variables in the Python interpreter state, except those starting with "owl_"
    - Python interpreter history file
    - Prompt history file
    - chat history in the processor
    - pickled cache files
    """
    # clear all variables except those starting with "owl_"
    temp_dict = {k: v for k, v in code_executor.globals_dict.items() if k.startswith("owl_")}
    code_executor.globals_dict.clear()
    code_executor.globals_dict.update(temp_dict)

    force_delete(get_cache_dir())

    if manager.processor is not None:
        manager.processor.chat_history.clear()

    logger.info(f"Cleared cachefolder {get_cache_dir()} and model chathistory.")

    # rebuild empty cache files after clearing
    get_pickle_cache()
    get_prompt_cache()
    get_py_cache()


def process_user_question(user_choice: str, code_executor: CodeExecutor, manager: TextGenerationManager) -> None:
    # Parse media placeholders in the user choice, if present.
    user_question, media_objects = parse_media_placeholders(user_choice, code_executor.globals_dict)
    rag_is_active = manager.get_config_key("rag.active", False)
    library_to_rag = manager.get_config_key("rag.target_library", "")
    if rag_is_active and library_to_rag:
        logger.info(f"RAG search enabled. Adding context of python library '{library_to_rag}' to the question.")
        ctx_to_add = f"""
# CONTEXT:
The following context is documentation from the python library {library_to_rag}.
Use this information to help generate a code snippet that answers the question.
"""
        searcher = PythonLibSearcher()
        context = searcher.search(
            library_to_rag, user_question, manager.get_config_key("top_k", 3), cache_dir=get_pickle_cache()
        )
        ctx_to_add += context
        user_question = f"{user_question}\n\n{ctx_to_add}".strip()
        logger.info(f"Context added to the question with approximate amount of {len(context.split())} words")

    response = manager.generate(user_question, media_objects=media_objects)
    execute_code_with_feedback(
        response=response,
        original_question=user_question,
        code_executor=code_executor,
        prompt_code_execution=manager.config_manager.get("main.prompt_code_execution", True),
        prompt_retry_on_error=manager.config_manager.get("main.prompt_retry_on_error", False),
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

    temp_dir_location = get_temp_dir(".owlsight_packages")

    # Create temporary directory in venv to install packages, until end of execution lifecycle
    with tempfile.TemporaryDirectory(dir=temp_dir_location) as temp_dir:
        logger.info(f"Temporary directory created at: {temp_dir}")

        code_executor = CodeExecutor(manager, pyenv_path, pip_path, temp_dir)

        run_code_generation_loop(code_executor, manager)

    logger.info(f"Removing temporary directory: {temp_dir}")
    free_cuda_memory()
    force_delete(temp_dir)
