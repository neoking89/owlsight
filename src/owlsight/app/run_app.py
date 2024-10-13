import tempfile
import traceback
from typing import Dict, Union, Tuple
from enum import Enum, auto
import os

from owlsight.processors.text_generation_manager import TextGenerationManager
from owlsight.app.handlers import handle_interactive_code_execution
from owlsight.utils.code_execution import CodeExecutor, execute_code_with_feedback
from owlsight.utils.helper_functions import (
    force_delete,
    remove_temp_directories,
    replace_bracket_placeholders,
    os_is_windows,
)
from owlsight.utils.venv_manager import get_lib_path, get_pip_path, get_pyenv_path
from owlsight.utils.console import get_user_choice, print_colored
from owlsight.utils.constants import PROMPT_COLOR, MENU_KEYS
from owlsight.utils.deep_learning import free_memory
from owlsight.ui.file_dialogs import save_file_dialog, open_file_dialog
from owlsight.rag.tfidf_search import get_context_for_library
from owlsight.utils.constants import get_prompt_history_path
from owlsight.utils.logger_manager import LoggerManager

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

            if not user_choice and choice_key not in ["config", "save", "load"]:
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
            # raise


def get_user_input(manager: TextGenerationManager) -> Tuple[str, Union[str, None]]:
    user_choice: Union[str, Dict] = get_user_choice(
        {
            MENU_KEYS["assistant"]: "",
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
    """Clears the following things:
    
    - All variables in the Python interpreter state, except those starting with "owl_"
    - Python interpreter history file
    - Prompt history file
    - chat history in the processor
    """
    # clear all variables except those starting with "owl_"
    code_executor.globals_dict = {
        k: v for k, v in code_executor.globals_dict.items() if k.startswith("owl_")
    }
    py_history_file = code_executor.python_interpreter_history_file
    if os.path.exists(py_history_file):
        os.remove(py_history_file)

    prompt_history_file = get_prompt_history_path()
    if os.path.exists(prompt_history_file):
        os.remove(prompt_history_file)
        
    if manager.processor is not None:
        manager.processor.history.clear()
    logger.info("Cleared: Python interpreter history, prompt history, and model chat history.")


def process_user_question(user_choice: str, code_executor: CodeExecutor, manager: TextGenerationManager) -> None:
    user_question = replace_bracket_placeholders(user_choice, code_executor.globals_dict)
    user_question = f"# QUESTION:\n{user_question}\n\n"
    rag_is_active = manager.get_config_key("rag.active", False)
    library_to_rag = manager.get_config_key("rag.library", "")
    if rag_is_active and library_to_rag:
        ctx_to_add = f"""
# CONTEXT:
The following context is documentation from the python library {library_to_rag}.
Use this information to help generate a code snippet that answers the question.
"""

        context = get_context_for_library(library_to_rag, user_question, manager.get_config_key("top_k", 3))
        ctx_to_add += context
        user_question = f"{user_question}\n\n{ctx_to_add}".strip()

    response = manager.generate(user_question)
    execute_code_with_feedback(
        response=response,
        original_question=user_question,
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
