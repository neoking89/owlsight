import tempfile
import traceback
from typing import Union, Tuple, List, Dict
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
    parse_media_tags,
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
from owlsight.rag.python_lib_search import PythonLibSearcher
from owlsight.processors.helper_functions import warn_processor_not_loaded
from owlsight.prompts.system_prompts import ExpertPrompts
from owlsight.utils.logger import logger


class CommandResult(Enum):
    """Enum to represent the result of a command from the mainmenu."""

    CONTINUE = auto()
    BREAK = auto()
    PROCEED = auto()


# TODO: Have an AppManager which encapsulates the run_code_generation_loop and on_startup functionality and might be better fit for keeping track of state?
# Hierarchy would be: AppManager -> CodeExecutor/ TextGenerationManager -> ConfigManager
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

            user_choice = parse_python_placeholders(user_choice, code_executor.globals_dict)
            if not isinstance(user_choice, str):
                logger.error(
                    f"User choice is not a string, but {type(user_choice).__name__}. Please only use curly braces '{{{{expression}}}}' if the end result from the python expression is a string."
                )
                continue
            handle_assistant_prompt(user_choice, manager, code_executor)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Returning to main menu.")
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


def handle_assistant_prompt(user_choice: str, manager: TextGenerationManager, code_executor: CodeExecutor) -> None:
    """
    Process user input from the 'How can I assist you?' field in the main menu.
    Handles extraction of tags, processor validation, and command processing.

    Parameters
    ----------
    user_choice : str
        The raw user input to process
    manager : TextGenerationManager
        Manager instance for handling configurations
    code_executor : CodeExecutor
        Executor for processing code-related requests
    """
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
    get_default_config_on_startup_path(return_cache_path=True)


def process_user_question(
    user_choice: str,
    code_executor: CodeExecutor,
    manager: TextGenerationManager,
    max_steps: int = 3,
    current_step: int = 0,
) -> str:
    """
    Process the user's choice and generate a response.

    Parameters:
    ----------
        user_choice (str): The user's inputted choice
        code_executor (CodeExecutor): The code executor.
        manager (TextGenerationManager): The text generation manager.
        max_steps (int, optional): Maximum number of tool calling steps. Defaults to 3.
        current_step (int, optional): Current step in the tool calling sequence. Defaults to 0.

    Returns:
    -------
        The response generated by the model.
    """
    _handle_dynamic_system_prompt(user_choice, manager)
    # Parse media tags in the user choice, if present.
    user_question, media_objects = parse_media_tags(user_choice, code_executor.globals_dict)
    user_question = _handle_rag_for_python(user_question, manager)

    apply_tools = manager.config_manager.get("agentic.apply_tools", False)
    if apply_tools:
        # Track tool execution state
        tool_state = {
            "step": current_step,
            "max_steps": max_steps,
            "previous_results": code_executor.globals_dict.get("tool_results", []),
        }
        user_question = _handle_apply_tools(user_question, tool_state, manager)

    response = manager.generate(user_question, media_objects=media_objects)
    results = execute_code_with_feedback(
        response=response,
        original_question=user_question,
        code_executor=code_executor,
        prompt_code_execution=manager.config_manager.get("main.prompt_code_execution", True),
        prompt_retry_on_error=manager.config_manager.get("main.prompt_retry_on_error", False),
    )
    if apply_tools:
        response = _handle_tool_result(
            results=results,
            user_choice=user_choice,
            code_executor=code_executor,
            manager=manager,
            current_step=current_step,
            max_steps=max_steps,
        )

    return response


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
        on_app_startup(manager)
        run_code_generation_loop(code_executor, manager)

    logger.info(f"Removing temporary directory: {temp_dir}")
    free_cuda_memory()
    force_delete(temp_dir)


def on_app_startup(manager: TextGenerationManager):
    """
    Functionality to execute when the CLI starts up.
    """
    default_config_path = get_default_config_on_startup_path(return_cache_path=False)
    if default_config_path:
        manager.load_config(default_config_path)
        logger.info(f"Loaded settings from default config '{default_config_path}'")


def _handle_dynamic_system_prompt(user_question: str, manager: TextGenerationManager) -> None:
    dynamic_system_prompt = manager.get_config_key("main.dynamic_system_prompt", False)
    if dynamic_system_prompt:
        prompt_engineer_prompt = ExpertPrompts.prompt_engineering
        manager.update_config("model.system_prompt", prompt_engineer_prompt)
        logger.info(
            "Dynamic system prompt is active. Model will act as Prompt Engineer to create a new system prompt based on user input."
        )
        new_system_prompt = manager.generate(user_question)
        # TODO: handle some kind of parsing of response here?
        manager.update_config("model.system_prompt", new_system_prompt)
        manager.update_config("main.dynamic_system_prompt", False)


def _extract_params_chain_tag(param: str) -> Tuple[str, str]:
    if "=" not in param:
        logger.error(f"Invalid chain parameter: {param}. Use 'param=value' format. Skipping...")
        return "", ""
    key, value = param.split("=")
    key = key.strip()
    value = value.strip()
    return key, value


def _handle_apply_tools(user_question: str, tool_state: Dict, manager: TextGenerationManager) -> str:
    """
    Enhance the user question with tool calling instructions and context from previous steps.

    Parameters
    ----------
    user_question : str
        The original user question
    tool_state : Dict
        Current state of tool execution including step count and previous results

    Returns
    -------
    str
        Enhanced question with tool calling instructions
    """
    previous_results = tool_state["previous_results"]
    current_step = tool_state["step"] + 1
    max_steps = tool_state["max_steps"]

    # Get the last used tool from chat history if available
    last_tool = None
    if hasattr(manager, "processor") and manager.processor and manager.processor.chat_history:
        for msg in reversed(manager.processor.chat_history):
            if isinstance(msg, dict) and "name" in msg.get("function_call", {}):
                last_tool = msg["function_call"]["name"]
                break

    tool_prompt = f"""
# Current Progress (Step {current_step}/{max_steps})

## Previous Results:
{previous_results if previous_results else "No previous results"}
{f"Last tool used: {last_tool}" if last_tool else ""}

## Critical Instructions:
1. ANALYZE your last tool call:
   - Was the information useful?
   - Did you get what you needed?
   - Is there a better approach?

2. NEXT STEPS:
   If the last tool call ({last_tool if last_tool else "none"}) didn't provide what you need:
   - DO NOT repeat the same tool call
   - YOU MUST choose a different tool or approach
   - Consider what complementary information would help

3. REFLECTION:
   - What specific information are you still missing?
   - Which alternative tool would provide different insights?
   - How can you combine information from multiple sources?

## Response Format:
If you need more information:
{{"name": "tool_name", "arguments": {{...}}}}
NOTE: Your tool choice MUST be different if the last result wasn't satisfactory

If you have enough information:
Provide your complete answer using all gathered data.

Remember: 
- Each final_result contains the output from your last tool call
- Repeating the same tool with similar parameters will likely give similar results
- Different tools provide different types of information
"""
    return f"{user_question}\n\n{tool_prompt}".strip()


def _handle_tool_result(
    results: List[Dict],
    user_choice: str,
    code_executor: CodeExecutor,
    manager: TextGenerationManager,
    current_step: int,
    max_steps: int,
) -> str:
    """
    Handle the result of a tool execution with support for multi-step processing.

    Parameters
    ----------
    results : List[Dict]
        The results from the code executor
    user_choice : str
        The user's inputted choice
    code_executor : CodeExecutor
        The code executor instance
    manager : TextGenerationManager
        The text generation manager instance
    current_step : int
        Current step in the tool calling sequence
    max_steps : int
        Maximum number of tool calling steps

    Returns
    -------
    str
        The response from the model
    """
    if not results or not results[0]["success"]:
        logger.warning(f"Tool execution failed or no results. Results: {results}")
        return ""

    if results[0]["code"].startswith("final_result"):
        final_result = code_executor.globals_dict["final_result"]
        logger.info(f"Tool result (Step {current_step + 1}/{max_steps}): {final_result}")

        # Store result for next steps
        tool_results = code_executor.globals_dict.get("tool_results", [])
        tool_results.append(final_result)
        code_executor.globals_dict["tool_results"] = tool_results

        if current_step + 1 < max_steps:
            # Remove last messages to prevent tool usage loop
            if manager.processor and manager.processor.chat_history:
                del manager.processor.chat_history[-2:]

            # Process next step with accumulated results
            return process_user_question(
                user_choice, code_executor, manager, max_steps=max_steps, current_step=current_step + 1
            )
        else:
            logger.info("Reached maximum steps or completed task. Generating final response.")
            # Format final response context
            ctx_to_add = f"""
Use ALL the following information to provide a COMPLETE answer:
Previous Results: {tool_results}

Important: Synthesize ALL gathered information into a coherent response.
""".strip()
            user_question = f"**User Request**:\n{user_choice}\n\n{ctx_to_add}".strip()

            # Generate final response without tools
            manager.update_config("agentic.apply_tools", False)
            response = manager.generate(user_question)
            manager.update_config("agentic.apply_tools", True)
            return response
    else:
        logger.warning(f"Unexpected tool result format. Results: {results}")
        return ""


def _handle_rag_for_python(user_question: str, manager: TextGenerationManager) -> str:
    """
    Handle RAG (Retrieval Augmented Generation) for Python library documentation.

    Args:
        user_question: The original user question
        manager: TextGenerationManager instance for config access

    Returns:
        Modified user question with added context from library documentation
    """
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
    return user_question
