"""
Revised and optimized OwlSight agentic logic, context management, and orchestration.
"""

import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Type, Tuple

from owlsight.processors.text_generation_manager import TextGenerationManager
from owlsight.utils.code_execution import CodeExecutor, execute_code_with_feedback
from owlsight.rag.python_lib_search import PythonLibSearcher
from owlsight.utils.helper_functions import (
    parse_media_tags,
    parse_xml_tags_to_dict,
    parse_xml,
    format_chat_history_as_string,
)
from owlsight.utils.constants import get_pickle_cache
from owlsight.prompts.system_prompts import ExpertPrompts
from owlsight.app.default_functions import OwlDefaultFunctions
from owlsight.utils.custom_classes import GlobalPythonVarsDict
from owlsight.utils.logger import logger


AVAILABLE_AGENTS = {
    "ToolSelectionAgent": "Use when external data retrieval, API calls, or specialized tool usage is required.",
    "PythonAgent": "Use ONLY for writing deterministic Python code (purely computational).",
    "TextAnalysisAgent": "Use for analyzing text, summarizing, extracting info, or generating strategies.",
}


class AgenticRole:
    """
    A context manager that temporarily replaces the system prompt and (optionally) disables
    tool usage. It captures any changes to chat history and system prompt, restoring them afterward.
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
        # Save original state
        self.original_state = {
            "system_prompt": manager.get_config_key("model.system_prompt", ""),
            "chat_history": manager.processor.chat_history.copy(),
        }
        self.disable_tools = disable_tools

        # Clear old history & set new prompt
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
        old_history = self.original_state["chat_history"]
        # Retain new messages added while in context
        self.manager.processor.chat_history = old_history + self.manager.processor.chat_history


@dataclass
class AgentContext:
    """Holds context for agent operations."""

    # Core context fields
    step: int = 0
    max_steps: int = 3
    previous_results: List[str] = field(default_factory=list)
    media_objects: Optional[Dict[str, str]] = field(default_factory=dict)
    should_continue: bool = True
    final_results: List[Dict[str, Any]] = field(default_factory=list)

    # Planning
    planning: Dict[str, Any] = field(default_factory=dict)
    current_plan_index: int = 0

    # Tool usage
    last_used_tool: Dict[str, str] = field(default_factory=dict)

    # Validation
    answer_is_appropriate: bool = False
    completed_steps: Dict[str, Dict[str, str]] = field(default_factory=dict)
    
    # Error tracking
    step_errors: Dict[int, List[str]] = field(default_factory=dict)
    step_attempts: Dict[int, int] = field(default_factory=dict)
    current_error: Optional[str] = None


class Agent(Protocol):
    """
    Protocol defining the interface for all agents.
    """

    def process(self, user_question: str, context: AgentContext) -> Dict[str, Any]:
        """
        Process a user question and return a dict containing:
            - 'response': str
            - 'should_continue': bool
            - 'context': AgentContext
        """


class RouterPlanningAgent:
    """
    Analyzes the user request, determines sub-tasks, and routes them to appropriate agents.
    """

    def __init__(self, code_executor: CodeExecutor, manager: TextGenerationManager):
        self.code_executor = code_executor
        self.manager = manager

    def process(self, user_question: str, context: AgentContext) -> Dict[str, Any]:
        """
        Creates a plan by analyzing the user question and then storing the plan in `context`.
        """
        router_prompt = self._create_router_agent_prompt(user_question)
        agent_list = "".join(f"- {k}: {v}\n" for k, v in AVAILABLE_AGENTS.items())

        # CHANGE: Shortened system prompt to reduce tokens
        router_system_prompt = f"""
You are an expert planner and router. Analyze the user request:
1. Break it into several subtasks if needed. Try to make the steps as atomic as possible.
2. Assign each subtask to the most suitable agent:
   - ToolSelectionAgent for external data or tool usage
   - PythonAgent ONLY for purely computational tasks
   - TextAnalysisAgent for analysis, summarization, or strategy
3. Return a structured plan.

AVAILABLE AGENTS:
{agent_list}
""".strip()

        with AgenticRole(router_prompt, router_system_prompt, self.manager, self.code_executor, disable_tools=True):
            router_response = self.manager.generate(router_prompt)

        planning = self._extract_planning_from_response(router_response)
        context.planning = planning
        context.should_continue = True

        logger.info(f"Planning result: {planning}")
        return {"response": router_response, "should_continue": True, "context": context}

    @staticmethod
    def _create_router_agent_prompt(user_question: str) -> str:
        """
        Gathers available tools and prepares a prompt for planning.
        """
        sep = "#" * 50
        available_tools = "\n".join(
            str(obj) for obj in OwlDefaultFunctions(GlobalPythonVarsDict()).owl_tools(as_json=True)
        )
        # Simplify tool listing
        return f"""
User Question:
{user_question}

Your task: Create a plan (several steps) with an agent for each subtask.

AVAILABLE TOOLS:
{sep}
{available_tools}
{sep}

Response Format:
<plan>
Step 1: ...
Agent: [{" | ".join(AVAILABLE_AGENTS.keys())}]
Reason: ...
Step 2: ...
Agent: ...
Reason: ...
</plan>

<reasoning>
...
</reasoning>
""".strip()

    @staticmethod
    def _extract_planning_from_response(response: str) -> Dict[str, Any]:
        """
        Parse <plan> and <reasoning> from the router agent's output.
        """
        plan_match = parse_xml(response, "plan")
        reasoning_match = parse_xml(response, "reasoning")

        if not plan_match:
            logger.warning("No plan found in router response.")
            return {"steps": [], "reasoning": ""}

        plan_text = plan_match.strip()
        reasoning = reasoning_match.strip() if reasoning_match else ""
        steps = []
        current_step = {}

        for line in plan_text.split("\n"):
            line = line.strip()
            if not line:
                continue

            if line.lower().startswith("step "):
                if current_step and "description" in current_step:
                    steps.append(current_step)
                current_step = {"description": line}
            elif line.lower().startswith("agent:"):
                current_step["agent"] = line[len("Agent:") :].strip()
            elif line.lower().startswith("reason:"):
                current_step["reason"] = line[len("Reason:") :].strip()

        if current_step and "description" in current_step:
            steps.append(current_step)

        return {"steps": steps, "reasoning": reasoning}


class ToolSelectionAgent:
    """
    Uses or calls external tools if needed. Expects final step to have
    a JSON {"name": "tool_name", "arguments": {...}} if a tool is invoked.
    """

    def __init__(self, code_executor: CodeExecutor, manager: TextGenerationManager):
        self.code_executor = code_executor
        self.manager = manager

    def process(self, user_question: str, context: AgentContext) -> Dict[str, Any]:
        tool_question = self._create_tool_agent_prompt(user_question, context, self.manager)

        # CHANGE: Shortened system prompt
        system_prompt = (
            "You are an expert in tool selection. If you need a tool, respond ONLY with a JSON object:\n"
            '{"name": "<tool_name>", "arguments": {...}}\nNo extra text.\n'
        )
        with AgenticRole(tool_question, system_prompt, self.manager, self.code_executor, disable_tools=False):
            tool_response = self.manager.generate(tool_question)

        # CHANGE: Wrap the final tool result in a dict to keep final_results consistent
        final_result = _get_final_result_from_python_code(tool_response, user_question, self.code_executor)
        if not isinstance(final_result, dict):
            final_result = {f"tool_result_for_{user_question}": final_result}
        context.final_results.append(final_result)

        last_used_tool = get_last_used_tool(self.code_executor, tool_response)
        context.last_used_tool = last_used_tool
        context.should_continue = True

        return {"response": tool_response, "should_continue": True, "context": context}

    @staticmethod
    def _create_tool_agent_prompt(user_question: str, context: AgentContext, manager: TextGenerationManager) -> str:
        """
        Incorporates prior results and tool usage instructions into a prompt for tool calls.
        """
        previous_results = context.previous_results
        final_results = context.final_results
        current_step = context.step + 1
        max_steps = context.max_steps

        # If manager has a tool_history attribute, consider it, else fallback
        last_tools = getattr(manager, "tool_history", None)

        progress_content = ""
        if manager.processor.chat_history:
            last_response_dict = parse_xml_tags_to_dict(manager.processor.chat_history[-1]["content"])
            required_steps = last_response_dict.get("required_steps", "")
            step_status = last_response_dict.get("step_completion_status", "")
            next_steps = last_response_dict.get("next_steps", "")

            sections = []
            if required_steps:
                sections.append(f"Required Steps:\n{required_steps}")
            if step_status:
                sections.append(f"Step Status:\n{step_status}")
            if next_steps:
                sections.append(f"Next Steps:\n{next_steps}")

            progress_content = "\n\n".join(sections)

        instruction_prompt = (
            "1. Check if previous tool calls gave needed info.\n"
            "2. Decide next steps carefully. If you must use another tool, return only valid JSON.\n"
            "3. Do NOT repeat the same tool call with same arguments.\n"
            f"{progress_content}"
        )

        additional_info = manager.config_manager.get("agentic.additional_information", "")
        tool_prompt = f"""
User Request:
{user_question}

Step {current_step}/{max_steps}

Previous Results: {previous_results if previous_results else "None"}
Final Results from Previous Steps: {list_of_dicts_to_llm_context(final_results) if final_results else "None"}
{f"Last Tools Used: {last_tools}" if last_tools else ""}
Additional Info: {additional_info}

Instructions:
{instruction_prompt}

Possible Tools:
- owl_search, owl_scrape, owl_read, owl_write, owl_import
Return Format:
{{"name": "tool_name", "arguments": {{...}}}}
""".strip()

        return tool_prompt


class TextAnalysisAgent:
    """
    Handles advanced text analysis, summarization, sentiment, and insights.
    """

    def __init__(self, code_executor: CodeExecutor, manager: TextGenerationManager):
        self.code_executor = code_executor
        self.manager = manager

    def process(self, user_question: str, context: AgentContext) -> Dict[str, Any]:
        # CHANGE: Shorter system prompt
        system_prompt = """
You are a text analysis expert. Summarize, extract, and analyze text accurately. 
Focus on the given sub-task and prior data only.
"""

        current_step_index = getattr(context, "current_plan_index", 0)
        steps = context.planning.get("steps", [])
        current_step_description = ""
        filtered_results = []

        if steps and 0 <= current_step_index < len(steps):
            current_step = steps[current_step_index]
            current_step_description = current_step.get("description", "")
            filtered_results = context.final_results[:current_step_index]
        else:
            filtered_results = context.final_results

        additional_info = list_of_dicts_to_llm_context(filtered_results)
        analysis_prompt = f"""
**User Request**: {user_question}

Current Sub-Task: {current_step_description}

Context from Previous Steps:
{additional_info}
""".strip()

        with AgenticRole(analysis_prompt, system_prompt, self.manager, self.code_executor) as agent:
            analysis_response = agent.manager.generate(agent.question)

            # Attempt to parse structured data
            try:
                structured_data = parse_xml_tags_to_dict(analysis_response)
                if not structured_data:
                    final_result = {"text_analysis_result": analysis_response}
                else:
                    final_result = structured_data
            except Exception as e:
                logger.warning(f"Error parsing structured data: {e}")
                final_result = {"text_analysis_result": analysis_response}

            context.final_results.append(final_result)

        context.should_continue = True
        return {"response": analysis_response, "should_continue": True, "context": context}


class PythonAgent:
    """
    Writes or refines Python code with improved context awareness.
    """

    def __init__(self, code_executor: CodeExecutor, manager: TextGenerationManager):
        self.code_executor = code_executor
        self.manager = manager

    def process(self, user_question: str, context: AgentContext) -> Dict[str, Any]:
        last_used_tool = context.last_used_tool
        python_response = self._handle_python_agent(user_question, last_used_tool, context)

        final_result = _get_final_result_from_python_code(python_response, user_question, self.code_executor)
        if not isinstance(final_result, dict):
            final_result = {f"result of user request '{user_question}'": final_result}

        context.final_results.append(final_result)
        context.should_continue = True

        return {
            "response": python_response,
            "should_continue": True,
            "context": context,
        }

    def _handle_python_agent(
        self,
        user_request: str,
        tool_name: Dict[str, str],
        context: AgentContext,
    ) -> str:
        """
        Generates Python code with enhanced context awareness.
        Uses data from previous steps including tool results and planning context.
        """
        # Get current planning step information if available
        current_step_info = ""
        if (hasattr(context, "planning") and context.planning and 
            "steps" in context.planning and 
            context.current_plan_index < len(context.planning["steps"])):
            step = context.planning["steps"][context.current_plan_index]
            current_step_info = f"""
Current Planning Step:
- Description: {step.get('description', 'N/A')}
- Agent: {step.get('agent', 'N/A')}
- Reason: {step.get('reason', 'N/A')}
"""

        # Extract tool information 
        tool_info = ""
        if tool_name:
            name = next(iter(tool_name.keys()), "")
            code = tool_name.get(name, "")
            tool_info = f"""
Last Used Tool: {name}
Tool Code: 
```python
{code}
```
"""

        # Get previous execution results from context
        previous_results = list_of_dicts_to_llm_context(context.final_results)
        
        # Check if any relevant data exists in the globals dict
        globals_data = ""
        globals_dict = self.code_executor.globals_dict
        if globals_dict and len(globals_dict) > 0:
            # Exclude built-in and private variables
            relevant_vars = {
                k: v for k, v in globals_dict.items() 
                if not k.startswith("_") and k not in ("__builtins__")
            }
            if relevant_vars:
                globals_data = "Available Variables in Global Context:\n"
                for var_name, var_value in relevant_vars.items():
                    # Only include short string representation of values
                    var_repr = str(var_value)
                    if len(var_repr) > 100:
                        var_repr = var_repr[:100] + "..."
                    globals_data += f"- {var_name}: {type(var_value).__name__} = {var_repr}\n"

        # Enhanced system prompt
        system_prompt = """
You are an expert Python developer. Your task is to write clean, functional Python code 
that builds upon previous steps and uses REAL data from context.

REQUIREMENTS:
1. Use ACTUAL data from previous steps - NEVER use placeholder values
2. Write complete, executable Python functions
3. Always assign the final result to a variable named 'final_result'
4. Include error handling with try-except blocks where appropriate
5. Document your code with clear comments

If no relevant data is available from previous steps, clearly indicate this in your 
code comments and provide appropriate fallback behavior.
"""

        # Validation requirements - keep the existing structure but enhance
        validation_checks = {
            "def": "missing function definition",
            ":": "missing colon",
            "(": "missing paren",
            ")": "missing paren",
            "    ": "missing indent",
            "return": "missing return",
            "final_result": "missing final_result assignment",
        }

        # Create a structured user prompt
        user_prompt = f"""
User Request: {user_request}

{current_step_info}
{tool_info}
{globals_data}

Previous Results:
{previous_results}

Write Python code that processes this REAL data to solve the user's request.
Your code MUST include: functions, proper indentation, return statements, and 
assignment to a 'final_result' variable.

Focus on using the ACTUAL data shown above rather than creating example data.
"""

        with AgenticRole(user_prompt, system_prompt, self.manager, self.code_executor) as agent:
            new_response = agent.manager.generate(agent.question)

            # Perform validation checks
            if all(keyword in new_response for keyword in validation_checks):
                logger.info("Python code validation successful")
                return new_response

            missing_elements = [msg for kw, msg in validation_checks.items() if kw not in new_response]
            logger.warning(f"Code validation failed: {', '.join(missing_elements)}. Returning empty string.")
            return ""


class ValidationAgent:
    """
    Validates whether enough information is present to finalize the user's request.
    """

    def __init__(self, code_executor: CodeExecutor, manager: TextGenerationManager):
        self.code_executor = code_executor
        self.manager = manager

    def process(self, user_question: str, context: AgentContext) -> Dict[str, Any]:
        current_step = context.step
        max_steps = context.max_steps
        final_results_text = list_of_dicts_to_llm_context(context.final_results)

        answer_is_appropriate, response = self._handle_answer_validation(user_question, final_results_text)

        if answer_is_appropriate:
            logger.info("Enough info to generate final answer.")
        else:
            logger.info("Further steps may be needed.")

        # Decide if we continue or not
        if current_step + 1 >= max_steps:
            should_continue = True  # Move to final response anyway
        else:
            should_continue = not answer_is_appropriate

        context.answer_is_appropriate = answer_is_appropriate
        context.should_continue = should_continue
        context.step = current_step + 1

        return {"response": response, "should_continue": should_continue, "context": context}

    @staticmethod
    def _create_validation_agent_prompt(user_request: str, old_chat_history: str, final_results: str) -> str:
        # CHANGE: Removed large ASCII boxes and simplified instructions
        return f"""
Your task is only to check if enough information has been gathered to satisfy the user request.

Validation rules:
1. If all critical data is present, respond YES.
2. If partially missing or incomplete, respond PARTIAL.
3. If major data is missing or invalid, respond NO.
Do not solve the user's request.

User Request:
{user_request}

Chat History:
{old_chat_history}

Gathered Final Results:
{final_results}

REQUIRED RESPONSE FORMAT (XML):
<goal>[Restate user's request]</goal>
<required_steps>[List steps if known]</required_steps>
<step_completion_status>[Mention step statuses in some structured form]</step_completion_status>
<judgment>[YES / PARTIAL / NO]</judgment>
<explanation>[If partial/no, what's missing]</explanation>
<next_steps>[If partial/no, specify next step(s)]</next_steps>
""".strip()

    def _handle_answer_validation(self, user_request: str, final_results: str) -> Tuple[bool, str]:
        """
        Calls a specialized 'validation agent' to confirm if all needed info is present.
        Returns (bool, str) -> (answerIsAppropriate, fullResponse).
        """
        assistant_context = [d for d in self.manager.processor.chat_history if d["role"] == "assistant"]
        old_chat_history = format_chat_history_as_string(assistant_context)
        system_prompt = "You verify data completeness. Do NOT solve or elaborate. Only judge completeness."
        question = self._create_validation_agent_prompt(user_request, old_chat_history, final_results)

        with AgenticRole(question, system_prompt, self.manager, self.code_executor) as judge_agent:
            response = self.manager.generate(judge_agent.question)

        judgment = False
        try:
            judgment_str = parse_xml(response, "judgment").strip().lower()
            logger.info(f"Validation judgment: {judgment_str}")
            if judgment_str == "yes":
                judgment = True
            elif judgment_str in ["partial", "no"]:
                judgment = False
            else:
                logger.warning(f"Unknown judgment: {judgment_str}. Assuming incomplete.")
                judgment = False
        except Exception as e:
            logger.error(f"Error parsing judgment: {str(e)}")

        return judgment, response


class ResponseSynthesisAgent:
    """
    Synthesizes all final data into a coherent answer.
    """

    def __init__(self, code_executor: CodeExecutor, manager: TextGenerationManager):
        self.code_executor = code_executor
        self.manager = manager

    def process(self, user_question: str, final_results: str) -> str:
        ctx_to_add = f"Use all data below to form a coherent final answer:\n{final_results}"
        user_prompt = f"User Request:\n{user_question}\n\n{ctx_to_add}"
        system_prompt = "You are an expert in response synthesis. Your task is to form a final answer to the User Request based on the provided data."

        with AgenticRole(user_prompt, system_prompt, self.manager, self.code_executor):
            response = self.manager.generate(user_prompt)
        logger.info("Synthesized final response.")

        return response


class AgentOrchestrator:
    """
    Orchestrates the execution of agents in a pipeline to handle complex user requests.
    """

    def __init__(
        self,
        code_executor: CodeExecutor,
        manager: TextGenerationManager,
        max_steps: int,
        agents: List[Type[Agent]] = None,
    ):
        self.agents = agents or [
            RouterPlanningAgent,
            ToolSelectionAgent,
            PythonAgent,
            ValidationAgent,
            ResponseSynthesisAgent,
        ]
        self.code_executor = code_executor
        self.manager = manager
        self.max_steps = max_steps

    def process_user_question(self, user_choice: str) -> str:
        """
        Coordinates the entire multi-agent pipeline based on user choice.
        """
        _handle_dynamic_system_prompt(user_choice, self.manager)
        user_question, media_objects = parse_media_tags(user_choice, self.code_executor.globals_dict)
        user_question = _handle_rag_for_python(user_question, self.manager)

        apply_tools = self.manager.config_manager.get("agentic.apply_tools", False)
        if not apply_tools:
            # No multi-agent orchestration, direct model call
            response = self.manager.generate(user_question, media_objects=media_objects)
            _ = execute_code_with_feedback(
                response=response,
                original_question=user_question,
                code_executor=self.code_executor,
                prompt_code_execution=self.manager.config_manager.get("main.prompt_code_execution", True),
                prompt_retry_on_error=self.manager.config_manager.get("main.prompt_retry_on_error", True),
            )
            return response

        context = AgentContext(
            step=0,
            max_steps=self.max_steps,
            previous_results=self.code_executor.globals_dict.get("tool_results", []),
            media_objects=media_objects,
            final_results=[],
        )

        logger.info(f"Starting agent pipeline for request: {user_question}")
        available_tools = [
            getattr(obj, "__name__")
            for obj in OwlDefaultFunctions(GlobalPythonVarsDict()).owl_tools(as_json=False)
            if hasattr(obj, "__name__")
        ]
        logger.info(f"Available tools: {available_tools}")

        # Always start with RouterPlanningAgent
        router_agent = RouterPlanningAgent(self.code_executor, self.manager)
        logger.info(f"Using RouterPlanningAgent, iteration {context.step + 1}/{context.max_steps}")
        router_result = router_agent.process(user_question, context)
        context = router_result["context"]

        planning = context.planning
        planning_steps = planning.get("steps", [])
        if not planning_steps:
            logger.info("No planning steps generated!")
            return "No planning steps generated!"

        answer_is_appropriate = False
        while not answer_is_appropriate and context.step < self.max_steps:
            answer_is_appropriate, context = self.process_planning_steps(user_question, context, planning_steps)
            steps_status = [step["status"].lower() for step in context.completed_steps.values()]
            not_completed_idx = next((i for i, status in enumerate(steps_status) if status != "completed"), None)
            if not_completed_idx is not None:
                planning_steps = planning_steps[not_completed_idx:]
            else:
                break

        # Finally, synthesize final response
        logger.info("Running ResponseSynthesisAgent for final answer.")
        response_agent = ResponseSynthesisAgent(self.code_executor, self.manager)
        final_ctx = list_of_dicts_to_llm_context(context.final_results)
        response = response_agent.process(user_question, final_ctx)

        return response

    def process_planning_steps(
        self, user_question: str, context: AgentContext, planning_steps: list
    ) -> Tuple[bool, AgentContext]:
        """
        Executes each planned step and then runs a validation agent.
        Includes retry logic for steps that fail.
        """
        max_attempts_per_step = 3  # Maximum attempts for each step
        
        for idx, step in enumerate(planning_steps):
            # Initialize step tracking if not already done
            if idx not in context.step_attempts:
                context.step_attempts[idx] = 0
                context.step_errors[idx] = []
            
            # Check if we've exceeded max attempts for this step
            if context.step_attempts[idx] >= max_attempts_per_step:
                logger.warning(f"Maximum attempts ({max_attempts_per_step}) reached for step {idx + 1}. Moving to next step.")
                continue
                
            context.current_plan_index = idx
            agent_type = step.get("agent", "")
            step_description = step.get("description", f"Step {idx + 1}")
            
            # Construct the request with error context if there were previous errors
            error_context = ""
            if context.step_attempts[idx] > 0 and context.step_errors[idx]:
                error_context = f"""
Previous attempt(s) failed with errors:
{'. '.join(context.step_errors[idx])}

Please try a different approach to solve this problem.
"""
            
            agent_request = f"""
Analyze:
{user_question}

Perform:
{step_description}

{error_context}

Reason:
{step.get("reason", "")}
""".strip()

            logger.info(f"Plan step {idx + 1}/{len(planning_steps)}: {step_description}")
            logger.info(f"Using {agent_type}, attempt {context.step_attempts[idx] + 1}/{max_attempts_per_step}")
            
            # Increment attempt counter for this step
            context.step_attempts[idx] += 1

            if agent_type == "ToolSelectionAgent":
                agent = ToolSelectionAgent(self.code_executor, self.manager)
            elif agent_type == "PythonAgent":
                agent = PythonAgent(self.code_executor, self.manager)
            elif agent_type == "TextAnalysisAgent":
                agent = TextAnalysisAgent(self.code_executor, self.manager)
            else:
                logger.warning(f"Unknown agent type: {agent_type}. Skipping.")
                continue

            result = agent.process(agent_request, context)
            context = result["context"]
            
            # Check for errors in the result
            step_failed = self._check_for_step_failure(result, context)
            
            if step_failed:
                logger.warning(f"Step {idx + 1} failed on attempt {context.step_attempts[idx]}. Retrying step.")
                # Store the error and retry the same step (idx will not change in next loop iteration)
                if context.current_error:
                    context.step_errors[idx].append(context.current_error)
                # Decrement idx to retry the same step
                idx -= 1
                continue
            
            # If we get here, the step was successful
            logger.info(f"Step {idx + 1} completed successfully.")
            
            if not result["should_continue"]:
                logger.info(f"Agent {agent_type} indicated to stop.")
                break

        # Validation after steps
        validation_agent = ValidationAgent(self.code_executor, self.manager)
        logger.info("Using ValidationAgent to check completeness.")
        validation_result = validation_agent.process(user_question, context)
        context = validation_result["context"]

        answer_is_appropriate = context.answer_is_appropriate
        val_response = validation_result.get("response", "")
        step_completion_status = parse_xml(val_response, "step_completion_status")
        completed_steps_dict = parse_xml_tags_to_dict(step_completion_status)
        completed_steps = {s: parse_xml_tags_to_dict(v) for s, v in completed_steps_dict.items()}
        context.completed_steps = completed_steps

        return answer_is_appropriate, context
        
    def _check_for_step_failure(self, result: Dict[str, Any], context: AgentContext) -> bool:
        """
        Checks if a step has failed by examining the response and context.
        Returns True if the step failed and should be retried.
        """
        response = result.get("response", "")
        
        # Check for error indicators in the response
        error_indicators = [
            "error:",
            "file not found",
            "no such file",
            "could not find",
            "failed to",
            "unable to",
            "permission denied"
        ]
        
        # Check for errors in text
        for indicator in error_indicators:
            if indicator.lower() in response.lower():
                context.current_error = f"Detected '{indicator}' in response"
                return True
        
        # Check for errors in final_results
        if context.final_results:
            latest_result = context.final_results[-1]
            for key, value in latest_result.items():
                if isinstance(value, str) and any(indicator.lower() in value.lower() for indicator in error_indicators):
                    context.current_error = f"Error in result: {value}"
                    return True
                    
        # Check tool results
        tool_result = getattr(context, "tool_result", None)
        if isinstance(tool_result, str) and any(indicator.lower() in tool_result.lower() for indicator in error_indicators):
            context.current_error = f"Error in tool result: {tool_result}"
            return True
            
        # No errors detected
        context.current_error = None
        return False


def get_last_used_tool(code_executor: CodeExecutor, response: str) -> Dict[str, str]:
    """
    Parse the last used tool name from `response` if found. Return {tool_name: code}.
    """
    tool_name = None
    tool_code = ""
    possible_tool_names = code_executor.globals_dict.get_public_keys()
    for name in possible_tool_names:
        if name in response:
            tool_name = name
            break

    if tool_name:
        bound_tool = code_executor.globals_dict.get(tool_name, None)
        if bound_tool:
            tool_code = inspect.getsource(bound_tool).strip()

    return {tool_name: tool_code} if tool_name else {}


def list_of_dicts_to_llm_context(data: List[Dict[str, Any]]) -> str:
    """
    Convert list of dicts into a concise text block for LLM consumption.
    Each dict's key-value pairs become a short segment: "Source: <key>\n<value>".
    """
    context_parts = []
    for idx, entry_dict in enumerate(data, start=1):
        if not isinstance(entry_dict, dict):
            entry_dict = {f"unknown_source_{idx}": str(entry_dict)}
        for source, content in entry_dict.items():
            header = f"Source: {source}"
            entry = f"{header}\n{str(content).strip()}"
            context_parts.append(entry)

    context = "\n---\n".join(context_parts)
    logger.info(f"Generated context (~{len(context.split())} words).")
    return context


def _handle_rag_for_python(user_question: str, manager: TextGenerationManager) -> str:
    """
    If RAG is enabled for a Python library, append docstrings from the library to the user question.
    """
    if manager.config_manager.get("rag.active", False) and manager.config_manager.get("rag.target_library", ""):
        library = manager.config_manager.get("rag.target_library")
        logger.info(f"RAG enabled for library '{library}'. Retrieving docs.")
        ctx_to_add = f"Below is doc info for '{library}', which may assist:\n"
        searcher = PythonLibSearcher()
        context = searcher.search(
            library,
            user_question,
            manager.config_manager.get("top_k", 3),
            cache_dir=get_pickle_cache(),
        )
        user_question = f"{user_question}\n\n{ctx_to_add}{context}"
    return user_question


def _handle_dynamic_system_prompt(user_question: str, manager: TextGenerationManager) -> None:
    """
    If 'main.dynamic_system_prompt' is enabled, generate a system prompt on the fly
    and then set it for subsequent calls.
    """
    if manager.config_manager.get("main.dynamic_system_prompt", False):
        prompt_engineer_prompt = ExpertPrompts.prompt_engineering
        manager.update_config("model.system_prompt", prompt_engineer_prompt)
        logger.info("Dynamic system prompt is active. Creating a new system prompt.")
        new_sys_prompt = manager.generate(user_question)
        manager.update_config("model.system_prompt", new_sys_prompt)
        manager.update_config("main.dynamic_system_prompt", False)


def _get_final_result_from_python_code(response: str, original_question: str, code_executor: CodeExecutor) -> Any:
    """
    Executes code from the model response if present. Returns the contents of 'final_result'
    from the code executor's global dict, or an empty list by default.
    """
    _ = execute_code_with_feedback(
        response=response,
        original_question=original_question,
        code_executor=code_executor,
        prompt_code_execution=False,
        prompt_retry_on_error=False,
    )
    return code_executor.globals_dict.get("final_result", [])
