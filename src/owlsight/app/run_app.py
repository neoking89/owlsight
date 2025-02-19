import tempfile
import traceback
from typing import Union, List, Dict, Tuple
from enum import Enum, auto
import os
import inspect
import re
from pathlib import Path

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
    parse_html_tags,
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


class AgenticRole:
    """
    A context manager that temporarily replaces the system prompt and (optionally) disables
    tool usage. It captures any changes to the chat history and system prompt, restoring
    them when the context closes.
    """

    def __init__(
        self,
        question: str,
        new_system_prompt: str,
        manager: TextGenerationManager,
        code_executor: CodeExecutor,
        disable_tools: bool = True,
    ):
        self.manager = manager
        self.question = question
        self.code_executor = code_executor

        # Save original prompts & chat history
        self.original_state = {
            "system_prompt": manager.get_config_key("model.system_prompt", ""),
            "chat_history": manager.processor.chat_history.copy(),
        }
        self.disable_tools = disable_tools

        # Temporary clean old state
        self.manager.processor.chat_history = []
        self.manager.update_config("model.system_prompt", new_system_prompt)

        if self.disable_tools:
            self.manager.update_config("agentic.apply_tools", False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the original agentic.apply_tools setting
        if self.disable_tools:
            self.manager.update_config("agentic.apply_tools", True)

        # Restore original system prompt & chat history
        self.manager.update_config("model.system_prompt", self.original_state["system_prompt"])
        self.manager.processor.chat_history = self.original_state["chat_history"] + self.manager.processor.chat_history


class CommandResult(Enum):
    """Enum to represent the result of a command from the main menu."""

    CONTINUE = auto()
    BREAK = auto()
    PROCEED = auto()


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


def handle_assistant_prompt(user_choice: str, manager: TextGenerationManager, code_executor: CodeExecutor) -> None:
    """
    Process user input from the 'How can I assist you?' field in the main menu.
    Handles extraction of tags, processor validation, and command processing.
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
    _handle_dynamic_system_prompt(user_choice, manager)
    user_question, media_objects = parse_media_tags(user_choice, code_executor.globals_dict)
    user_question = _handle_rag_for_python(user_question, manager)

    apply_tools = manager.config_manager.get("agentic.apply_tools", False)
    if not apply_tools:
        response = manager.generate(user_question, media_objects=media_objects)
        _ = execute_code_with_feedback(
            response=response,
            original_question=user_question,
            code_executor=code_executor,
            prompt_code_execution=manager.config_manager.get("main.prompt_code_execution", True),
            prompt_retry_on_error=manager.config_manager.get("main.prompt_retry_on_error", False),
        )
        return response

    # Tool agent phase
    tool_state = {
        "step": current_step,
        "max_steps": max_steps,
        "previous_results": code_executor.globals_dict.get("tool_results", []),
    }
    tool_question = _create_tool_agent_prompt(user_question, tool_state, manager)
    tool_agent_system_prompt = (
        "You are an expert planner, specialized in thinking through the next steps "
        "and choosing the appropriate tools to facilitate them. Always use one of the "
        "available tools to answer the user's question."
    )

    # Get tool call from Tool Agent
    with AgenticRole(
        tool_question, tool_agent_system_prompt, manager, code_executor, disable_tools=False
    ) as tool_agent:
        tool_response = tool_agent.manager.generate(tool_question)

    # Execute the tool call and get results
    code_execution_results = execute_code_with_feedback(
        response=tool_response,
        original_question=tool_question,
        code_executor=code_executor,
        prompt_code_execution=False,  # Always execute tool calls
        prompt_retry_on_error=False,
    )

    # Extract the tool information and results
    last_used_tool = _get_last_used_tool(code_executor, tool_response)
    tool_result = code_executor.globals_dict.get("final_result", [])
    python_agent_is_enabled = manager.config_manager.get("agentic.enable_python_agent", False)

    # Pass both the tool call and its results to Python agent
    if python_agent_is_enabled and not tool_result:
        response = _handle_python_agent(user_choice, manager, code_executor, last_used_tool)
    else:
        response = tool_response

    if apply_tools:
        return _handle_tool_result(
            results=code_execution_results,
            tool_results=tool_result,
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


def _handle_dynamic_system_prompt(user_question: str, manager: TextGenerationManager) -> None:
    """
    If 'main.dynamic_system_prompt' is enabled, ask the model to create a new system prompt
    based on the user's input, then switch to that prompt for subsequent calls.
    """
    dynamic_system_prompt = manager.get_config_key("main.dynamic_system_prompt", False)
    if dynamic_system_prompt:
        prompt_engineer_prompt = ExpertPrompts.prompt_engineering
        manager.update_config("model.system_prompt", prompt_engineer_prompt)
        logger.info("Dynamic system prompt is active. Model will act as Prompt Engineer to create a new system prompt.")
        new_system_prompt = manager.generate(user_question)
        manager.update_config("model.system_prompt", new_system_prompt)
        manager.update_config("main.dynamic_system_prompt", False)


def _create_tool_agent_prompt(user_question: str, tool_state: Dict, manager: TextGenerationManager) -> str:
    """
    Enhance the user question with tool-calling instructions and context from previous steps,
    guiding the LLM to produce the next-step plan in JSON format.
    """
    previous_results = tool_state["previous_results"]
    current_step = tool_state["step"] + 1
    max_steps = tool_state["max_steps"]
    last_tools = manager.tool_history if manager.tool_history else None
    if current_step > 1 or last_tools:
        logger.info(f"Current used tools found: {last_tools}")

        # parse important steps from judge-agent response:
        last_response = parse_html_tags(manager.processor.chat_history[-1]["content"])
        required_steps = last_response.get("required_steps", "")
        step_completion_status = last_response.get("step_completion_status", "")
        next_steps = last_response.get("next_steps", "")

        # Build progress sections if they exist
        progress_sections = []
        if required_steps:
            progress_sections.append(f"## Required Steps:\n{required_steps}")
        if step_completion_status:
            progress_sections.append(f"## Step Status:\n{step_completion_status}")
        if next_steps:
            progress_sections.append(f"## Next Steps:\n{next_steps}")

        progress_content = "\n\n".join(progress_sections)

        instruction_prompt = f"""
## TASK:
1. Examine your previous tool calls:
   - Was the information useful for answering the user's request?
   - Did you get what you needed?

2. Decide your next steps carefully:
   - Think step-by-step about what else is required.
   - Look closely at **Last tools used:** (if any). Do NOT repeat any of them with the same arguments.
   - If you must use another tool, respond with a valid JSON object:
       {{"name": "<tool_name>", "arguments": {{...}}}}
   - Make sure you ONLY respond with that JSON object, nothing else.
   - **AGAIN**: DO NOT repeat any of the tools used with the same arguments!

{progress_content}
""".strip()
    else:
        instruction_prompt = """
## TASK:
1. Think step-by-step about how to approach the user's request.
2. If you need a tool, respond ONLY with a JSON object:
   {"name": "<tool_name>", "arguments": {...}}
3. Do not provide any additional text beyond that JSON.
4. Use descriptive and functional argument names for clarity. Do not use placeholder names, like "/path/to/file.txt" or "insert api key here".
""".strip()

    additional_info = manager.config_manager.get("agentic.additional_information", "")
    tool_prompt = f"""
# Current Progress (Step {current_step}/{max_steps})

## Previous Results:
{previous_results if previous_results else "No previous results"}
{f"**Last tools used:** {last_tools}" if last_tools else ""}

## Additional Information:
{additional_info}

## CRITICAL INSTRUCTIONS:
{instruction_prompt}

## TOOL GUIDELINES:
- If any information is given in ## Additional Information, use this instead of below instructions.
- Use `owl_search` if you need general information.
- Use `owl_scrape` for scraping a known URL.
- Use `owl_read` to read a local file or directory.
- Use `owl_write` to write to a local file.
- Use `owl_import` to import a Python file.
- Other tools may be used for specialized tasks.

## REQUIRED RESPONSE FORMAT:
{{"name": "tool_name", "arguments": {{...}}}}
"""
    return f"# User Request:\n{user_question}\n\n{tool_prompt}".strip()


def _get_last_used_tool(code_executor: CodeExecutor, response: str) -> Dict[str, str]:
    """
    Parse the last used tool from the response, along with its function body.
    If none is found, returns an empty dict.
    """
    tool_code = ""
    possible_tool_names = code_executor.globals_dict.get_public_keys()
    tool_name = next((name for name in possible_tool_names if name in response), None)
    if tool_name:
        bound_tool = code_executor.globals_dict.get(tool_name, None)
        if bound_tool:
            tool_code = inspect.getsource(bound_tool).strip()

    return {tool_name: tool_code} if tool_name else {}


def _handle_python_agent(
    user_request: str,
    manager: TextGenerationManager,
    code_executor: CodeExecutor,
    tool_name: Dict[str, str],
) -> str:
    """
    Expert Python agent for code validation and refinement with enhanced security
    and prompt engineering features. Implements input validation, secure coding
    practices, and structured prompting.
    """
    if not all(isinstance(arg, (str, dict)) for arg in (user_request, tool_name)):
        raise ValueError("Invalid input types for Python agent handling")

    validation_checks = {
        "def": "missing def",
        ":": "missing colon",
        "(": "missing paren",
        ")": "missing paren",
        "    ": "missing indent",
        "return": "missing return",
    }

    system_prompt = """
# Role
You are an expert Python developer.

# Task
Write Python code based on a user request.

```python
def solution_<descriptive_name>(...) -> <return_type>:
    '''Write a docstring explaining the functionality of the function.'''
    # Implementation
    # Verification logic if needed

# define the "final_result" variable with the created function
final_result = solution(...)
```

## Code Requirements
- Function with clear, declarative name and type hints
- Concise docstring in Numpy-style format
- Error handling
- Secure defaults
- Markdown format with ```
- Testable verification code for deterministic solutions
- The variable name "final_result" is defined with the created function

## Forbidden Patterns
- eval/exec
- Unsafe deserialization
- Bare except clauses
- Fabricated information (written code should be factual and accurate)
"""

    validation_rules = "\n".join([f"- {desc} check" for desc in validation_checks.values()])

    user_prompt = f"""
**User Request**: {user_request}

## Validation Checklist
{validation_rules}
""".strip()
    with AgenticRole(user_prompt, system_prompt, manager, code_executor) as agent:
        new_response = agent.manager.generate(agent.question)

        if all(keyword in new_response for keyword in validation_checks):
            return new_response

        logger.warning("Code validation failed, returning empty string.")
        return ""


def _handle_answer_validation_agent(
    user_request: str,
    final_result: str,
    manager: TextGenerationManager,
    code_executor: CodeExecutor,
) -> bool:
    """
    Engages a specialized 'validation agent' to confirm whether all necessary info
    has been gathered to finalize the user's request.

    Returns a boolean indicating whether the answer is appropriate.
    """
    response = ""
    assistant_context = [d for d in manager.processor.chat_history if d["role"] == "assistant"]
    old_chat_history = format_chat_history_as_string(assistant_context)
    system_prompt = (
        "You are an expert at verifying completeness. Focus on whether enough data is present, "
        "especially around 'final_result'. Do NOT solve the problem yourself."
    )
    question = _create_validation_agent_prompt(
        user_request=user_request,
        old_chat_history=old_chat_history,
        final_result=final_result,
    )

    with AgenticRole(question, system_prompt, manager, code_executor) as judge_agent:
        response = judge_agent.manager.generate(judge_agent.question)

        try:
            judgment_str = re.findall(r"<judgment>(.*?)</judgment>", response, re.DOTALL)[0].strip()
            logger.info(f"Answer validation judgment: {judgment_str}")
            if judgment_str.lower() == "yes":
                logger.info("Answer 'yes' found in judgment.")
                return True
        except Exception as e:
            logger.error(f"Error parsing judgment: {str(e)}")

    logger.info("Did not find 'yes' in judgment.")
    return False


def _handle_tool_result(
    results: List[Dict],
    tool_results: List[Dict],
    user_choice: str,
    code_executor: CodeExecutor,
    manager: TextGenerationManager,
    current_step: int,
    max_steps: int,
) -> str:
    """
    Handle the result of a tool execution with support for multi-step processing,
    final validation, and building the ultimate response.
    """
    if not results or not any(r["success"] for r in results):
        logger.warning(f"Tool execution failed or no results. Results: {results}")
        return ""

    final_result = code_executor.globals_dict.get("final_result", None)
    if final_result is None:
        logger.warning(f"No 'final_result' found in globals after tool execution.\nResults: {results}")
        return ""

    logger.info(f"Tool result (Step {current_step + 1}/{max_steps}): {final_result}")

    answer_is_appropriate = _handle_answer_validation_agent(user_choice, final_result, manager, code_executor)
    if answer_is_appropriate:
        logger.info("Enough information gathered to generate a final answer.")
    else:
        logger.info("More information needed to generate a final answer.")

    tool_results = code_executor.globals_dict.get("tool_results", [])
    tool_results.append(final_result)
    code_executor.globals_dict["tool_results"] = tool_results

    if current_step + 1 < max_steps and not answer_is_appropriate:
        return process_user_question(
            user_choice, code_executor, manager, max_steps=max_steps, current_step=current_step + 1
        )

    logger.info("Reached maximum steps or decided enough info is present. Generating final response.")
    ctx_to_add = f"""
Use ALL the following gathered data:
Previous Results: {tool_results}

Synthesize everything into one coherent final answer.
""".strip()
    user_question = f"**User Request**:\n{user_choice}\n\n{ctx_to_add}".strip()

    manager.update_config("agentic.apply_tools", False)
    response = manager.generate(user_question)
    manager.update_config("agentic.apply_tools", True)

    return response


def _handle_rag_for_python(user_question: str, manager: TextGenerationManager) -> str:
    """
    If Retrieval-Augmented Generation (RAG) is enabled, add relevant Python library docstrings
    to the user question.
    """
    rag_is_active = manager.get_config_key("rag.active", False)
    library_to_rag = manager.get_config_key("rag.target_library", "")
    if rag_is_active and library_to_rag:
        logger.info(f"RAG search enabled. Adding docs from python library '{library_to_rag}'.")
        ctx_to_add = f"""
# CONTEXT:
Below is documentation from the Python library '{library_to_rag}'.
Use it to assist in answering the user's question.
"""
        searcher = PythonLibSearcher()
        context = searcher.search(
            library_to_rag, user_question, manager.get_config_key("top_k", 3), cache_dir=get_pickle_cache()
        )
        ctx_to_add += context
        user_question = f"{user_question}\n\n{ctx_to_add}".strip()
        logger.info(f"Context added (~{len(context.split())} words).")
    return user_question


def _create_validation_agent_prompt(user_request: str, old_chat_history: str, final_result: str) -> str:
    """
    Builds a prompt for a specialized validation agent that checks if all
    required steps/data are present to fulfill the user's request.
    """
    return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                    TASK                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
IMPORTANT: Your task is ONLY to validate if enough information has been gathered.
Do NOT calculate or provide the final answer yourself.

Validation rules:
1. Multi-step: ALL steps must be addressed to respond "YES".
2. Do not guess or infer data not explicitly present.
3. If any vital data is missing, do NOT say "YES".

Possible judgments:
- YES: If all necessary data is present
- PARTIAL: Data is partially present, or some steps incomplete
- NO: Data is incorrect, missing critical parts, or entirely irrelevant

╔══════════════════════════════════════════════════════════════════════════════╗
║                                 CONTEXT                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
▓▓▓ ORIGINAL REQUEST ▓▓▓
{user_request}

▓▓▓ CHAT HISTORY ▓▓▓
{old_chat_history}

▓▓▓ FINAL RESULT ▓▓▓
{final_result}

╔══════════════════════════════════════════════════════════════════════════════╗
║                          RESPONSE FORMAT (REQUIRED)                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
<goal>
[Restate the user's ultimate goal or question; do not answer it]
</goal>

<required_steps>
[List the steps found in context]
</required_steps>

<step_completion_status>
[For each step, show COMPLETED/PENDING, plus source (chat/final_result)]
</step_completion_status>

<judgment>
[YES/NO/PARTIAL]
</judgment>

<explanation>
[If PARTIAL/NO, explain what info is missing. Do NOT solve the problem.]
</explanation>

<next_steps>
[If PARTIAL/NO, specify what additional data is needed next]
</next_steps>
"""
