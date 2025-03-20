"""
This module contains all agentic logic for the OwlSight application.
It includes agent implementations, context management, and orchestration.
"""

import inspect
import re
from typing import Any, Dict, List, Optional, Protocol, Type, TypedDict

from owlsight.processors.text_generation_manager import TextGenerationManager
from owlsight.utils.code_execution import CodeExecutor, execute_code_with_feedback
from owlsight.rag.python_lib_search import PythonLibSearcher
from owlsight.utils.helper_functions import (
    parse_media_tags,
    parse_html_tags,
    format_chat_history_as_string,
)
from owlsight.utils.constants import get_pickle_cache
from owlsight.prompts.system_prompts import ExpertPrompts
from owlsight.app.default_functions import OwlDefaultFunctions
from owlsight.utils.custom_classes import GlobalPythonVarsDict
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


class AgentContext(TypedDict, total=False):
    """TypedDict representing the context for agent operations."""

    # Core context fields used by AgentOrchestrator
    step: int  # Current processing step
    max_steps: int  # Maximum allowed steps
    previous_results: List[str]  # Results from previous agents
    media_objects: Optional[Dict[str, str]]  # Media objects associated with the query
    should_continue: bool  # Whether to continue to the next agent or cycle

    # Fields introduced by ToolSelectionAgent
    tool_response: str  # Response from the tool agent
    code_execution_results: Dict[str, Any]  # Results from code execution
    last_used_tool: Dict[str, str]  # Information about the last used tool
    tool_result: Any  # Result from the tool execution

    # Fields introduced by PythonAgent
    python_response: str  # Response from the Python agent
    refined_code: str  # Refined code from the Python agent

    # Fields introduced by ValidationAgent
    final_result: str  # Final synthesized result
    answer_is_appropriate: bool  # Whether the answer is appropriate/complete


class Agent(Protocol):
    """Protocol defining the interface for all agents in the system."""

    def process(
        self,
        user_question: str,
        code_executor: "CodeExecutor",
        manager: "TextGenerationManager",
        context: Optional[AgentContext] = None,
    ) -> Dict[str, Any]:
        """
        Process a user question and return a result dict.

        Parameters:
        ----------
            user_question: The question or request from the user
            code_executor: The code executor instance
            manager: The text generation manager instance
            context: Additional context from previous agent executions

        Returns:
        ---------
            Dict containing at least:
                - 'response': str - The agent's response
                - 'should_continue': bool - Whether to continue to next agent
                - 'context': AgentContext - Updated context for next agent
        """
        ...


class ToolSelectionAgent:
    """Agent responsible for creating plans and selecting tools."""

    def process(
        self,
        user_question: str,
        code_executor: "CodeExecutor",
        manager: "TextGenerationManager",
        context: Optional[AgentContext] = None,
    ) -> Dict[str, Any]:
        """Process the user question using the planning agent."""
        # Initialize context if not provided
        context = context or {}

        # Create the tool agent prompt with current state information
        tool_state = {
            "step": context.get("step", 0),
            "max_steps": context.get("max_steps", 3),
            "previous_results": context.get("previous_results", code_executor.globals_dict.get("tool_results", [])),
        }
        tool_question = self._create_tool_agent_prompt(user_question, tool_state, manager)

        # Define the system prompt for the planning agent
        tool_agent_system_prompt = (
            "You are an expert planner, specialized in thinking through the next steps "
            "and choosing the appropriate tools to facilitate them. Always use one of the "
            "available tools to answer the user's question."
        )

        # Step 1: Execute the tool agent to get a plan with tool selection
        with AgenticRole(
            tool_question, tool_agent_system_prompt, manager, code_executor, disable_tools=False
        ) as tool_agent:
            tool_response = tool_agent.manager.generate(tool_agent.question)

        # Step 2: Execute the selected tool and capture results
        code_execution_results = execute_code_with_feedback(
            response=tool_response,
            original_question=tool_question,
            code_executor=code_executor,
            prompt_code_execution=False,  # Always execute tool calls
            prompt_retry_on_error=False,
        )

        # Step 3: Extract tool information for use by subsequent agents
        last_used_tool = get_last_used_tool(code_executor, tool_response)
        tool_result = code_executor.globals_dict.get("final_result", [])

        # Update the context directly with new information
        context["tool_response"] = tool_response
        context["code_execution_results"] = code_execution_results
        context["last_used_tool"] = last_used_tool
        context["tool_result"] = tool_result
        context["should_continue"] = True

        return {"response": tool_response, "should_continue": True, "context": context}

    @staticmethod
    def _create_tool_agent_prompt(user_question: str, context: AgentContext, manager: TextGenerationManager) -> str:
        """
        Enhance the user question with tool-calling instructions and context from previous steps,
        guiding the LLM to produce the next-step plan in JSON format.
        """
        previous_results = context.get("previous_results", [])
        current_step = context.get("step", 0) + 1
        max_steps = context.get("max_steps", 3)
        last_tools = manager.tool_history if manager.tool_history else None

        if current_step > 1 or last_tools:
            logger.info(f"Current used tools found: {last_tools}")

            # parse important steps from validation agent response:
            if manager.processor.chat_history:
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
            else:
                progress_content = ""

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


class PythonAgent:
    """Agent responsible for Python code validation and refinement."""

    def process(
        self,
        user_question: str,
        code_executor: "CodeExecutor",
        manager: "TextGenerationManager",
        context: Optional[AgentContext] = None,
    ) -> Dict[str, Any]:
        """Process Python code validation and refinement."""
        context = context or {}

        last_used_tool = context.get("last_used_tool", {})

        # Skip Python agent if not enabled
        python_agent_is_enabled = manager.config_manager.get("agentic.enable_python_agent", False)
        if not python_agent_is_enabled:
            return {"response": context.get("tool_response", ""), "should_continue": True, "context": context}

        # Process with Python agent
        python_response = self._handle_python_agent(user_question, manager, code_executor, last_used_tool)

        # Update the context directly
        context["python_response"] = python_response
        context["should_continue"] = True

        return {
            "response": python_response or context.get("tool_response", ""),
            "should_continue": True,
            "context": context,
        }

    def _handle_python_agent(
        self,
        user_request: str,
        manager: TextGenerationManager,
        code_executor: "CodeExecutor",
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


class ValidationAgent:
    """Agent responsible for validating if enough information has been gathered."""

    def process(
        self,
        user_question: str,
        code_executor: "CodeExecutor",
        manager: "TextGenerationManager",
        context: Optional[AgentContext] = None,
    ) -> Dict[str, Any]:
        """Validate if enough information has been gathered."""
        context = context or {}

        # Get results from previous steps
        code_execution_results = context.get("code_execution_results", [])
        current_step = context.get("step", 0)
        max_steps = context.get("max_steps", 3)

        # Extract final result
        if not code_execution_results or not any(r["success"] for r in code_execution_results):
            logger.warning(f"Tool execution failed or no results. Results: {code_execution_results}")
            final_result = ""
        else:
            final_result = code_executor.globals_dict.get("final_result", None)
            if final_result is None:
                logger.warning("No 'final_result' found in globals after tool execution.")
                final_result = ""

        logger.info(f"Tool result (Step {current_step + 1}/{max_steps}): {final_result}")

        # Check if answer is appropriate
        answer_is_appropriate = self._handle_answer_validation(user_question, final_result, manager, code_executor)

        if answer_is_appropriate:
            logger.info("Enough information gathered to generate a final answer.")
        else:
            logger.info("More information needed to generate a final answer.")

        # Update tool results in globals
        tool_results = code_executor.globals_dict.get("tool_results", [])
        tool_results.append(final_result)
        code_executor.globals_dict["tool_results"] = tool_results

        # Determine if we should continue to another cycle
        if current_step + 1 >= max_steps:
            # If at max steps, we want to go to ResponseSynthesisAgent regardless
            should_continue = True
        else:
            # If not at max steps and answer is not appropriate, continue to next step
            should_continue = not answer_is_appropriate

        # Update context directly
        context["final_result"] = final_result
        context["answer_is_appropriate"] = answer_is_appropriate
        context["tool_results"] = tool_results
        context["should_continue"] = should_continue
        context["step"] = current_step + 1

        return {"response": final_result, "should_continue": should_continue, "context": context}

    @staticmethod
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

    def _handle_answer_validation(
        self,
        user_request: str,
        final_result: str,
        manager: TextGenerationManager,
        code_executor: "CodeExecutor",
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
        question = self._create_validation_agent_prompt(
            user_request=user_request,
            old_chat_history=old_chat_history,
            final_result=final_result,
        )

        with AgenticRole(question, system_prompt, manager, code_executor) as judge_agent:
            response = judge_agent.manager.generate(judge_agent.question)

            try:
                judgment_str = re.findall(r"<judgment>(.*?)</judgment>", response, re.DOTALL)[0].strip().lower()
                logger.info(f"Answer validation judgment: {judgment_str}")

                if judgment_str == "yes":
                    logger.info("Answer 'yes' found in judgment.")
                    return True
                elif judgment_str == "partial":
                    logger.info("Answer 'partial' found in judgment. More information needed.")
                    return False
                elif judgment_str == "no":
                    logger.info("Answer 'no' found in judgment. Information is incorrect or missing.")
                    return False
                else:
                    logger.warning(f"Unknown judgment value: {judgment_str}. Treating as not appropriate.")
                    return False
            except Exception as e:
                logger.error(f"Error parsing judgment: {str(e)}")

        logger.info("No valid judgment found. Treating as not appropriate.")
        return False


class ResponseSynthesisAgent:
    """Agent responsible for synthesizing the final response."""

    def process(
        self,
        user_question: str,
        code_executor: "CodeExecutor",
        manager: "TextGenerationManager",
        context: Optional[AgentContext] = None,
    ) -> Dict[str, Any]:
        """Synthesize a final response."""
        context = context or {}

        # Get tool results
        tool_results = context.get("tool_results", code_executor.globals_dict.get("tool_results", []))

        # Create synthetic prompt
        ctx_to_add = f"""
Use ALL the following gathered data:
Previous Results: {tool_results}

Synthesize everything into one coherent final answer.
""".strip()
        user_prompt = f"**User Request**:\n{user_question}\n\n{ctx_to_add}".strip()

        # Disable tool application for final response
        original_tools_setting = manager.config_manager.get("agentic.apply_tools", True)
        manager.update_config("agentic.apply_tools", False)

        # Generate final response
        response = manager.generate(user_prompt)

        # Restore original setting
        manager.update_config("agentic.apply_tools", original_tools_setting)

        # Format the response
        formatted_response = f"""
┌────────────────────────────────────────┐
│             FINAL RESPONSE             │
└────────────────────────────────────────┘
{response}
─────────────────────────────────────────
""".strip()
        print(formatted_response)

        return {"response": formatted_response, "should_continue": False, "context": context}


class AgentOrchestrator:
    """Orchestrates the execution of multiple agents in sequence."""

    def __init__(self, agents: List[Type[Agent]] = None):
        # Default agent pipeline - each agent is responsible for a specific aspect of processing
        self.agents = agents or [
            ToolSelectionAgent,  # Selects and executes appropriate tools
            PythonAgent,  # Refines Python code (if enabled)
            ValidationAgent,  # Determines if enough information has been gathered
            ResponseSynthesisAgent,  # Synthesizes the final response
        ]

    def process_user_question(
        self,
        user_choice: str,
        code_executor: "CodeExecutor",
        manager: "TextGenerationManager",
        max_steps: int = 3,
        current_step: int = 0,
    ) -> str:
        """
        Process the user's choice through a chain of agents.

        Args:
            user_choice: The user's question or request
            code_executor: The code executor instance
            manager: The text generation manager instance
            max_steps: Maximum number of processing cycles
            current_step: Current processing cycle

        Returns:
            The final response
        """
        # Preprocess the user question
        _handle_dynamic_system_prompt(user_choice, manager)
        user_question, media_objects = parse_media_tags(user_choice, code_executor.globals_dict)
        user_question = _handle_rag_for_python(user_question, manager)

        # Check if tools are disabled
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

        # Initialize context
        context: AgentContext = {
            "step": current_step,
            "max_steps": max_steps,
            "previous_results": code_executor.globals_dict.get("tool_results", []),
            "media_objects": media_objects,
        }

        # Process through agents
        response = ""
        skip_to_response_synthesis = False
        last_agent_class = None

        logger.info(f"Starting agent processing for user request: {user_question}")
        available_tools = [
            getattr(obj, "__name__", None)
            for obj in OwlDefaultFunctions(GlobalPythonVarsDict()).owl_tools(as_json=False)
        ]
        logger.info(f"Available tools: {available_tools}")
        for agent_class in self.agents:
            # Skip ResponseSynthesisAgent unless we've got a "yes" judgment or reached max steps
            if agent_class == ResponseSynthesisAgent and not skip_to_response_synthesis:
                # Skip ResponseSynthesisAgent if judgment wasn't "yes" and we haven't reached max_steps
                current_step = context.get("step", 0)
                if (
                    last_agent_class == ValidationAgent
                    and not context.get("answer_is_appropriate", False)
                    and current_step < max_steps
                ):
                    logger.info(f"Skipping ResponseSynthesisAgent - step {current_step}/{max_steps}")
                    continue

            logger.info(f"Using {agent_class.__name__}, step {context['step'] + 1}/{context['max_steps']}")
            agent = agent_class()
            result = agent.process(user_question, code_executor, manager, context)

            # Update context with agent results
            context = result["context"]
            response = result["response"]
            last_agent_class = agent_class

            # Check if we should continue to the next agent
            if not result["should_continue"]:
                break

            # Set flag for ResponseSynthesisAgent if ValidationAgent returned a "yes" judgment or max_steps reached
            if agent_class == ValidationAgent:
                # Check if ValidationAgent returned "yes" or if we've reached max_steps
                current_step = context.get("step", 0)
                if context.get("answer_is_appropriate", False) or current_step >= max_steps:
                    logger.info(f"Setting skip_to_response_synthesis to True - step {current_step}/{max_steps}")
                    skip_to_response_synthesis = True

        # Check if we need to restart the process for another iteration
        if context.get("should_continue", False) and (
            last_agent_class != ValidationAgent
            or (
                last_agent_class == ValidationAgent
                and not context.get("answer_is_appropriate", False)
                and context.get("step", 0) < max_steps
            )
        ):
            return self.process_user_question(
                user_choice,
                code_executor,
                manager,
                max_steps=max_steps,
                current_step=context.get("step", current_step + 1),
            )

        # Ensure ResponseSynthesisAgent is called if ValidationAgent yields "yes" judgment
        if last_agent_class == ValidationAgent and context.get("answer_is_appropriate", False):
            logger.info("ValidationAgent yielded 'yes' judgment, proceeding to ResponseSynthesisAgent")
            response_agent = ResponseSynthesisAgent()
            result = response_agent.process(user_question, code_executor, manager, context)
            response = result["response"]

        return response


def get_last_used_tool(code_executor: "CodeExecutor", response: str) -> Dict[str, str]:
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
