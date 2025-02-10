import tempfile
import traceback
from typing import Union, Tuple, List, Dict
from enum import Enum, auto
import os
import inspect
import re

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
    format_chat_history_as_string,
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


class AlternativeAgent:
    def __init__(
        self, question: str, new_system_prompt: str, manager: TextGenerationManager, code_executor: CodeExecutor
    ):
        self.manager = manager
        self.question = question
        self.code_executor = code_executor
        self.original_state = {
            "system_prompt": manager.get_config_key("model.system_prompt", ""),
            "chat_history": manager.processor.chat_history.copy(),
        }

        # temporary clean old state
        self.manager.processor.chat_history = []
        self.manager.update_config("model.system_prompt", new_system_prompt)
        self.manager.update_config("agentic.apply_tools", False)

    def __enter__(self):
        # You can add any setup code here if needed
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.manager.update_config("model.system_prompt", self.original_state["system_prompt"])
        self.manager.processor.chat_history = self.original_state["chat_history"] + self.manager.processor.chat_history
        self.manager.update_config("agentic.apply_tools", True)


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
    cache_dir = get_cache_dir()
    default_config_on_startup_path = get_default_config_on_startup_path(return_cache_path=True)
    # reset all cache files except the default config on startup
    files_in_cache_dir = [i for i in os.listdir(cache_dir) if i != default_config_on_startup_path]

    for file_path in files_in_cache_dir:
        # file_path = os.path.join(cache_dir, filename)
        force_delete(file_path)

    if manager.processor is not None:
        manager.processor.chat_history.clear()

    logger.info(f"Cleared files in cachefolder '{get_cache_dir()}' and model chathistory.")

    # rebuild empty cache files after clearing
    get_pickle_cache()
    get_prompt_cache()
    get_py_cache()


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
        # Enhance the user question with tool calling instructions and context from previous steps (if any)
        user_question = _handle_apply_tools(user_question, tool_state, manager)

    response = manager.generate(user_question, media_objects=media_objects)
    if apply_tools:
        python_agent_is_enabled = manager.config_manager.get("agentic.enable_python_agent", False)
        if python_agent_is_enabled:
            response = _handle_python_agent(user_choice, response, manager, code_executor)
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
    # TODO: just keep a list of used tools instead of parsing them from chat history
    last_tool = None
    # if hasattr(manager, "processor") and manager.processor and manager.processor.chat_history:
    #     for msg in reversed(manager.processor.chat_history):
    #         extraxcted_tool = extract_and_parse_json(msg["content"])
    #         # if parse_try:
    #         #     try:
    #         #         parse_try = dict(parse_try)
    #         #     except Exception:
    #         #         logger.error(f"Tried to parse tool name to dict, but failed: {parse_try['name']}. Skipping...")
    #         #         continue
    #             last_tool = parse_try["name"]
    #             break

    if current_step > 1:
        # TODO: smaller prompt
        instruction_prompt = f"""
1. ANALYZE your last tool call:
   - Was the information useful for answering the user request?
   - Did you get what you needed?
   - Is there another tool that might provide a better answer?
   - Always prioritize websearches over scraping tools.

2. NEXT STEPS:
   If the last tool call ({last_tool if last_tool else "none"}) didn't provide what you need:
   - DO NOT repeat the same tool call
   - YOU MUST choose a different tool or approach
   - You MUST return the answer in the format of a JSON object with a "name" and "arguments" field
   - Consider what complementary information would help to answer the user request.

3. REFLECTION:
   - What specific information are you still missing?
   - Which alternative tool would provide different insights?
   - How can you combine information from multiple sources?
""".strip()

    else:
        instruction_prompt = """
1. Think deeply and step-by-step about how to approach the user's request.
- Delve the problem into smaller, atomic steps.
- Reason about the steps in a logical sequence.
- Consider the potential tools and their inputs/outputs.
""".strip()

    tool_prompt = f"""
# Current Progress (Step {current_step}/{max_steps})

## Previous Results:
{previous_results if previous_results else "No previous results"}
{f"Last tool used: {last_tool}" if last_tool else ""}

## Critical Instructions:
{instruction_prompt}

## Response Format:
{{"name": "tool_name", "arguments": {{...}}}}
NOTE: Your tool choice MUST be different if the last result wasn't satisfactory

Remember: 
- Each final_result contains the output from your last tool call
- Repeating the same tool with similar parameters will likely give similar results
- Different tools provide different types of information
"""
    return f"{user_question}\n\n{tool_prompt}".strip()


def _handle_python_agent(
    user_request: str, response: str, manager: TextGenerationManager, code_executor: CodeExecutor
) -> str:
    """
    Handle the response from the Python agent.

    Parameters
    ----------
    user_request : str
        The user's request.
    response : str
        The response from the Python agent.
    manager : TextGenerationManager
        The manager object.
    code_executor : CodeExecutor
        The code executor object.

    Returns
    -------
    str
        The response from the Python agent.
    """
    # Get the last tool used
    used_tool = ""
    possible_tool_names = code_executor.globals_dict.get_public_keys()
    tool_name = next((name for name in possible_tool_names if name in response), None)
    if tool_name:
        bound_tool = code_executor.globals_dict.get(tool_name, None)
        if bound_tool:
            tool_code = inspect.getsource(bound_tool).strip()
            used_tool = f"Used tool: {tool_name}\n{tool_code}".strip()

    new_system_prompt = """
You are a expert in Python. 
Analyze the last response from another agent and write new python code if it is more appropriate to address the user's request. 
If the answer has a deterministic outcome, ALWAYS write Python code.
"""
    appropiate_cue = "The last response is appropriate to address the user's request."

    user_question = f"""
User request:\n{user_request}

Last Response:\n{response}

{used_tool}

# TASK:
Think deeply and step-by-step
1. Analyze the last response from another agent. Figure if this answer is appropiate to address the user's request.
2.
a: If the last response is appropriate to address the user's request and does not have a deterministic outcome, just say: {appropiate_cue}. End your response.
b: If the last response is appropriate to address the user's request, but has a deterministic outcome, write Python code to validate the deterministic outcome. Proceed to step 3.
c: If the last response is not appropriate to address the user's request, think about what information is needed to address the user's request. Proceed to step 3.
3. Write the python code to address the user's request. You can use any 3rd party library if needed. 
4. Always provide the generated Python code in your response in Markdown-format.
5. Write a function with an appropiate name and an appropiate docstring. Use numpy-style for writing the docstring.
6. Use this function to address the user's request. Store the result in a variable named "final_result".


# Response Format:
**Explanation:** Explain in your reasoning what information is needed to address the user's request.
**Judgment:** 
If the last response is satisfactory, say {appropiate_cue} and end your response.
If the last response is not satisfactory, use the following format:

```python
# You can import any third-party libraries as needed.
import ...

# Write the python code to address the user's request in a function with an appropiate name.
def new_tool(param1, param2):
    return ....

# store the result in a variable named "final_result"
final_result = new_tool(param1, param2)
""".strip()

    with AlternativeAgent(user_question, new_system_prompt, manager, code_executor) as python_agent:
        new_response = python_agent.manager.generate(python_agent.question)

        # Check if the response is appropriate
        if appropiate_cue.lower() in new_response.lower():
            logger.info("Last response is appropiate to address the user's request.")
            return response

        # # If not appropriate, update the chat history
        # if python_agent.manager.processor and python_agent.manager.processor.chat_history:
        #     logger.info("Last response is not appropiate to address the user's request.")
        #     python_agent.manager.processor.chat_history[-1] = {"role": "assistant", "content": new_response}

        return new_response


def _handle_answer_validation_agent(
    user_request: str,
    final_result: str,
    manager: TextGenerationManager,
    code_executor: CodeExecutor,
):
    """
    Handle the response from the answer validation agent.
    """
    assistant_context = [d for d in manager.processor.chat_history if d["role"] == "assistant"]
    old_chat_history = format_chat_history_as_string(assistant_context)
    system_prompt = "You are an expert at judging how satisfactory the gathered information is to address a user's request. Pay special attention to text with the word 'final_result'."
    question = _create_validation_prompt(
        user_request=user_request,
        old_chat_history=old_chat_history,
        final_result=final_result,
    )

    with AlternativeAgent(question, system_prompt, manager, code_executor) as judge_agent:
        judgment = judge_agent.manager.generate(judge_agent.question)

        # parse the judgment from the html tags
        try:
            judgment = re.findall(r"<judgment>(.*?)</judgment>", judgment, re.DOTALL)[0].strip()
            logger.info(f"Answer validation judgment: {judgment}")
        except Exception as e:
            logger.error(f"Error parsing judgment: {str(e)}")

        if judgment.lower() == "yes":
            logger.info("Answer 'yes' found in judgment.")
            return True

    logger.info("Did not find 'yes' in judgment.")
    return False


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

    if "final_result" in code_executor.globals_dict:
        final_result = code_executor.globals_dict["final_result"]
        logger.info(f"Tool result (Step {current_step + 1}/{max_steps}): {final_result}")

        answer_is_appropriate = _handle_answer_validation_agent(user_choice, final_result, manager, code_executor)
        if answer_is_appropriate:
            logger.info("✅ Enough information gathered to generate a final answer.")
        else:
            logger.info("❌ There is not enough information to generate a final answer (yet!).")

        # Store result for next steps
        tool_results = code_executor.globals_dict.get("tool_results", [])
        tool_results.append(final_result)
        code_executor.globals_dict["tool_results"] = tool_results

        if current_step + 1 < max_steps and not answer_is_appropriate:
            # Process next step with accumulated results
            return process_user_question(
                user_choice, code_executor, manager, max_steps=max_steps, current_step=current_step + 1
            )
        else:
            logger.info("Reached maximum steps or gathered enough information. Generating final response.")
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
        logger.warning(f"Unexpected tool result format. Could not find 'final_result'.\nResults: {results}")
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


def _create_validation_prompt(user_request: str, old_chat_history: str, final_result: str) -> str:
    return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                    TASK                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
IMPORTANT: Your task is ONLY to validate if enough information has been gathered.
DO NOT calculate or provide the final answer yourself.

Key validation rules:
1. For multi-step problems: ALL steps must be completed for a YES judgment
2. Each piece of information must be explicitly present in the context or 'final_result'
3. Pay special attention to context with the word 'final_result' and surrounding text.
4. Do not make assumptions or infer data that isn't explicitly provided
5. If the context shows a list of required steps, check each one individually
6. Focus ONLY on whether required information is present, not on calculating results

Judgment criteria:
- YES: ONLY if ALL required steps are completed AND ALL needed information is present
- PARTIAL: If any step is incomplete OR any required information is missing
- NO: If the result is incorrect or unsuitable

Remember: Your role is to verify information completeness, NOT to solve the problem.

╔══════════════════════════════════════════════════════════════════════════════╗
║                               EVALUATION STEPS                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
1. Look for any explicitly stated steps or requirements in the context
2. For each step or requirement found:
   - Mark it as COMPLETED only if the exact information is present
   - Mark it as PENDING if the information is missing or incomplete
3. Check that you can cite the source of each piece of information
4. Verify information presence without calculating final results
5. Only proceed to judgment after checking all steps

╔══════════════════════════════════════════════════════════════════════════════╗
║                                  CONTEXT                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ ORIGINAL REQUEST ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
{user_request}

▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ CHAT HISTORY ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
{old_chat_history}

▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ FINAL RESULT ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
{final_result}

╔══════════════════════════════════════════════════════════════════════════════╗
║                            RESPONSE FORMAT                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
<goal>
[STATE THE USER'S ULTIMATE GOAL/QUESTION - DO NOT ANSWER IT]
</goal>

<required_steps>
[LIST ALL REQUIRED STEPS FOUND IN CONTEXT]
</required_steps>

<step_completion_status>
[FOR EACH STEP LISTED ABOVE:
- Status: COMPLETED/PENDING
- Source: Where the information came from (chat history/final result)
- Information: What exact information was found (raw data only, no calculations)]
</step_completion_status>

<judgment>
[YES/NO/PARTIAL]
</judgment>

<explanation>
[EXPLAIN WHICH INFORMATION IS STILL MISSING - DO NOT CALCULATE OR PROVIDE SOLUTIONS]
</explanation>

<next_steps>
[IF PARTIAL/NO: List which specific information needs to be gathered next]
</next_steps>
"""
