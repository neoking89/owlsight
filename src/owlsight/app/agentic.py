"""
This module contains all agentic logic for the OwlSight application.
It includes agent implementations, context management, and orchestration.
"""

import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Type, Tuple

from owlsight.processors.text_generation_manager import TextGenerationManager
from owlsight.utils.code_execution import CodeExecutor, execute_code_with_feedback
from owlsight.rag.core import DocumentSearcher
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


@dataclass
class AgentContext:
    """Dataclass representing the context for agent operations."""

    # Core context fields used by AgentOrchestrator
    step: int = 0  # Current processing step
    max_steps: int = 3  # Maximum allowed steps
    previous_results: List[str] = field(default_factory=list)  # Results from previous agents
    media_objects: Optional[Dict[str, str]] = field(default_factory=dict)  # Media objects associated with the query
    should_continue: bool = True  # Whether to continue to the next agent or cycle
    final_results: List[Dict[str, Any]] = field(default_factory=list)  # Final results from all agents per step

    # Fields introduced by RouterPlanningAgent
    planning: Dict[str, Any] = field(default_factory=dict)  # Structured planning result with steps and reasoning
    current_plan_index: int = 0  # Current index in the execution plan

    # Fields introduced by ToolSelectionAgent
    last_used_tool: Dict[str, str] = field(default_factory=dict)  # Information about the last used tool

    answer_is_appropriate: bool = False  # Whether the answer is appropriate/complete
    completed_steps: Dict[str, Dict[str, str]] = field(default_factory=dict)  # Completed steps and their results


class Agent(Protocol):
    """Protocol defining the interface for all agents in the system."""

    def process(
        self,
        user_question: str,
        context: AgentContext,
    ) -> Dict[str, Any]:
        """
        Process a user question and return a result dict.

        Parameters:
        ----------
            user_question: The question or request from the user
            context: Additional context from previous agent executions

        Returns:
        ---------
            Dict containing at least:
                - 'response': str - The agent's response
                - 'should_continue': bool - Whether to continue to next agent
                - 'context': AgentContext - Updated context for next agent
        """


class RouterPlanningAgent:
    """Agent responsible for planning and routing tasks to the appropriate agents."""

    def __init__(self, code_executor: "CodeExecutor", manager: "TextGenerationManager"):
        self.code_executor = code_executor
        self.manager = manager

    def process(
        self,
        user_question: str,
        context: AgentContext,
    ) -> Dict[str, Any]:
        """Process the user question to create a plan and route to appropriate agents."""
        # Create router agent prompt with current state information
        router_prompt = self._create_router_agent_prompt(user_question)

        # Define the system prompt for the router/planning agent
        router_system_prompt = """
# ROLE:
You are an expert planner and router for AI tasks. Your purpose is to analyze complex user questions and:
1. Break them down into smaller, atomic sub-tasks when appropriate
2. Determine the most suitable agent to handle each sub-task
3. Create a clear execution plan

# AVAILABLE AGENTS:
- ToolSelectionAgent: Best for tasks requiring information from external sources, API calls, or data retrieval.
- PythonAgent: Best for computational tasks, code generation, or when all information is already available.

# INSTRUCTIONS:
- If the user's question is simple and direct, you may keep it as a single step
- For complex questions, break them into 2-5 logical sub-tasks
- For each sub-task, assign it to either the ToolSelectionAgent or PythonAgent
- Assign a sub-task to ToolSelectionAgent if external information is needed
- Assign a sub-task to PythonAgent if it's purely computational or all required information is present
- Be thoughtful and precise in your planning
"""

        # Step 1: Execute the router agent to get a plan
        with AgenticRole(
            router_prompt, router_system_prompt, self.manager, self.code_executor, disable_tools=True
        ) as router_agent:
            router_response = router_agent.manager.generate(router_agent.question)

        # Extract plan information from response
        planning = self._extract_planning_from_response(router_response)

        # Update context directly with new information
        context.planning = planning
        context.should_continue = True

        # Log the planning results
        logger.info(f"Planning result: {planning}")

        return {"response": router_response, "should_continue": True, "context": context}

    @staticmethod
    def _create_router_agent_prompt(user_question: str) -> str:
        """
        Create a prompt for the router/planning agent that guides it to
        analyze the user question and create an execution plan.
        """
        # Get available tools to consider in planning
        sep = "#" * 50
        available_tools = f"\n{sep}\n".join(
            str(obj) for obj in OwlDefaultFunctions(GlobalPythonVarsDict()).owl_tools(as_json=True)
        )
        available_tools = f"{sep}\n{available_tools}"

        return f"""
# USER QUESTION:
{user_question}

# TASK:
Your task is to analyze this question and create a plan of execution. If appropriate, break it down into smaller sub-tasks.

# AVAILABLE TOOLS:
{available_tools}

# RESPONSE FORMAT (REQUIRED):
<plan>
Step 1: [Description of first sub-task]
Agent: [ToolSelectionAgent or PythonAgent]
Reason: [Brief justification for agent selection]

Step 2: [Description of second sub-task, if needed]
Agent: [ToolSelectionAgent or PythonAgent]
Reason: [Brief justification for agent selection]

[Additional steps as needed...]
</plan>

<reasoning>
[Your detailed analysis of the user's question and why you chose this plan]
</reasoning>
""".strip()

    @staticmethod
    def _extract_planning_from_response(response: str) -> Dict[str, Any]:
        """
        Extract planning information from the router agent's response.
        """
        # Extract the plan section using regex
        plan_match = parse_xml(response, "plan")
        reasoning_match = parse_xml(response, "reasoning")

        if not plan_match:
            logger.warning("No plan found in router response.")
            return {"steps": [], "reasoning": ""}

        plan_text = plan_match.strip()
        reasoning = reasoning_match.strip() if reasoning_match else ""

        # Parse the steps from the plan
        steps = []
        current_step = {}

        for line in plan_text.split("\n"):
            line = line.strip()
            if not line:
                continue

            if line.startswith("Step "):
                # If we were working on a previous step, save it
                if current_step and "description" in current_step:
                    steps.append(current_step)

                # Start a new step
                current_step = {"description": line}
            elif line.startswith("Agent:"):
                if current_step:
                    current_step["agent"] = line[len("Agent:") :].strip()
            elif line.startswith("Reason:"):
                if current_step:
                    current_step["reason"] = line[len("Reason:") :].strip()

        # Add the last step if it exists
        if current_step and "description" in current_step:
            steps.append(current_step)

        return {"steps": steps, "reasoning": reasoning}


class ToolSelectionAgent:
    """Agent responsible for creating plans and selecting tools."""

    def __init__(self, code_executor: "CodeExecutor", manager: "TextGenerationManager"):
        self.code_executor = code_executor
        self.manager = manager

    def process(
        self,
        user_question: str,
        context: AgentContext,
    ) -> Dict[str, Any]:
        """Process the user question using the planning agent."""
        # Create the tool agent prompt with current state information
        tool_question = self._create_tool_agent_prompt(user_question, context, self.manager)

        # Define the system prompt for the planning agent
        tool_agent_system_prompt = (
            "You are an expert planner, specialized in thinking through the next steps "
            "and choosing the appropriate tools to facilitate them. Always use one of the "
            "available tools to answer the user's question."
        )

        # Step 1: Execute the tool agent to get a plan with tool selection
        with AgenticRole(
            tool_question, tool_agent_system_prompt, self.manager, self.code_executor, disable_tools=False
        ) as tool_agent:
            tool_response = tool_agent.manager.generate(tool_agent.question)

        final_result = _get_final_result_from_python_code(tool_response, user_question, self.code_executor)
        context.final_results.append(final_result)

        # Step 3: Extract tool information for use by subsequent agents
        last_used_tool = get_last_used_tool(self.code_executor, tool_response)

        # Update the context directly with new information
        context.last_used_tool = last_used_tool
        context.should_continue = True

        return {"response": tool_response, "should_continue": True, "context": context}

    @staticmethod
    def _create_tool_agent_prompt(user_question: str, context: AgentContext, manager: TextGenerationManager) -> str:
        """
        Enhance the user question with tool-calling instructions and context from previous steps,
        guiding the LLM to produce the next-step plan in JSON format.
        """
        previous_results = context.previous_results
        current_step = context.step + 1
        max_steps = context.max_steps
        last_tools = manager.tool_history if manager.tool_history else None

        if current_step > 1 or last_tools:
            logger.info(f"Current used tools found: {last_tools}")

            # parse important steps from validation agent response:
            if manager.processor.chat_history:
                last_response = parse_xml_tags_to_dict(manager.processor.chat_history[-1]["content"])
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
# TASK:
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
# TASK:
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

    def __init__(self, code_executor: "CodeExecutor", manager: "TextGenerationManager"):
        self.code_executor = code_executor
        self.manager = manager

    def process(
        self,
        user_question: str,
        context: AgentContext,
    ) -> Dict[str, Any]:
        """Process Python code validation and refinement."""
        last_used_tool = context.last_used_tool

        # Process with Python agent
        python_response = self._handle_python_agent(user_question, last_used_tool, context)
        final_result = _get_final_result_from_python_code(python_response, user_question, self.code_executor)
        # make sure final result is a dictionary
        if not isinstance(final_result, dict):
            final_result = {f"result of user request '{user_question}'": final_result}
        context.final_results.append(final_result)

        # Update the context directly
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
# ROLE
You are an expert Python developer.

# TASK
Write Python code based on a user request.

```python
def solution_<descriptive_name>(...) -> <return_type>:
    '''Write a docstring explaining the functionality of the function.'''
    # Implementation
    # Verification logic if needed

# define the "final_result" variable with the created function
final_result = solution(...)
```

## CODE REQUIREMENTS
- Function with clear, declarative name and type hints
- Concise docstring in Numpy-style format
- Error handling
- Secure defaults
- Markdown format with ```
- Testable verification code for deterministic solutions
- The variable name "final_result" is defined with the created function

## FORBIDDEN PATTERNS
- eval/exec
- Unsafe deserialization
- Bare except clauses
- Fabricated information (written code should be factual and accurate)
- Placeholders
"""

        validation_rules = "\n".join([f"- {desc} check" for desc in validation_checks.values()])
        additional_info = list_of_dicts_to_llm_context(context.final_results)

        user_prompt = f"""
**User Request**: {user_request}

## VALIDATION CHECKLIST
{validation_rules}

## ADDITIONAL INFORMATION
The following information has been gathered so far:
{additional_info}
""".strip()
        with AgenticRole(user_prompt, system_prompt, self.manager, self.code_executor) as agent:
            new_response = agent.manager.generate(agent.question)

            if all(keyword in new_response for keyword in validation_checks):
                return new_response

            logger.warning("Code validation failed, returning empty string.")
            return ""


class ValidationAgent:
    """Agent responsible for validating if enough information has been gathered."""

    def __init__(self, code_executor: "CodeExecutor", manager: "TextGenerationManager"):
        self.code_executor = code_executor
        self.manager = manager

    def process(
        self,
        user_question: str,
        context: AgentContext,
    ) -> Dict[str, Any]:
        """Validate if enough information has been gathered."""
        current_step = context.step
        max_steps = context.max_steps
        final_results = list_of_dicts_to_llm_context(context.final_results)

        # Check if answer is appropriate
        answer_is_appropriate, response = self._handle_answer_validation(user_question, final_results)

        if answer_is_appropriate:
            logger.info("Enough information gathered to generate a final answer.")
        else:
            logger.info("More information needed to generate a final answer.")

        # Determine if we should continue to another cycle
        if current_step + 1 >= max_steps:
            # If at max steps, we want to go to ResponseSynthesisAgent regardless
            should_continue = True
        else:
            # If not at max steps and answer is not appropriate, continue to next step
            should_continue = not answer_is_appropriate

        # Update context directly
        context.answer_is_appropriate = answer_is_appropriate
        context.should_continue = should_continue
        context.step = current_step + 1

        return {"response": response, "should_continue": should_continue, "context": context}

    @staticmethod
    def _create_validation_agent_prompt(user_request: str, old_chat_history: str, final_results: str) -> str:
        """
        Builds a prompt for a specialized validation agent that checks if all
        required info has been gathered to fulfill the user's request.
        """
        return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                    TASK                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
IMPORTANT: Your task is ONLY to validate if enough information has been gathered.
Do NOT solve the problem yourself.

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

▓▓▓ FINAL RESULTS ▓▓▓
{final_results}

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
[For each step, show step description, COMPLETED/PENDING, and source (chat/final_result)
Output format should be XML like this:
<step1>
    <step>step description</step>
    <status>COMPLETED</status>
    <source>chat</source>
</step1>
<step2>
    <step>step description</step>
    <status>PENDING</status>
    <source>final_result</source>
</step2>
]
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
""".strip()

    def _handle_answer_validation(
        self,
        user_request: str,
        final_results: str,
    ) -> bool:
        """
        Engages a specialized 'validation agent' to confirm whether all necessary info
        has been gathered to finalize the user's request.

        Returns a boolean indicating whether the answer is appropriate.
        """
        response = ""
        assistant_context = [d for d in self.manager.processor.chat_history if d["role"] == "assistant"]
        old_chat_history = format_chat_history_as_string(assistant_context)
        system_prompt = (
            "You are an expert at verifying completeness. Focus on whether enough data is present."
            "Do NOT solve the problem yourself."
        )
        question = self._create_validation_agent_prompt(
            user_request=user_request,
            old_chat_history=old_chat_history,
            final_results=final_results,
        )
        judgment = False

        with AgenticRole(question, system_prompt, self.manager, self.code_executor) as judge_agent:
            response = judge_agent.manager.generate(judge_agent.question)

            try:
                judgment_str = parse_xml(response, "judgment").strip().lower()
                logger.info(f"Answer validation judgment: {judgment_str}")

                if judgment_str == "yes":
                    logger.info("Answer 'yes' found in judgment. Enough information present to generate a final answer.")
                    judgment = True
                elif judgment_str == "partial":
                    logger.info("Answer 'partial' found in judgment. More information needed.")
                    judgment = False
                elif judgment_str == "no":
                    logger.info("Answer 'no' found in judgment. Information is incorrect or missing.")
                    judgment = False
                else:
                    logger.warning(f"Unknown judgment value: {judgment_str}. Treating as not appropriate.")
                    judgment = False
            except Exception as e:
                logger.error(f"Error parsing judgment: {str(e)}")

        return judgment, response


class ResponseSynthesisAgent:
    """Agent responsible for synthesizing the final response."""

    def __init__(self, code_executor: "CodeExecutor", manager: "TextGenerationManager"):
        self.code_executor = code_executor
        self.manager = manager

    def process(
        self,
        user_question: str,
        final_results: str,
    ) -> str:
        """Synthesize a final response."""
        ctx_to_add = f"""
Use ALL the following gathered data:
Previous Results: {final_results}

Synthesize everything into one coherent final answer.
""".strip()
        user_prompt = f"**User Request**:\n{user_question}\n\n{ctx_to_add}".strip()

        # Disable tool application for final response
        original_tools_setting = self.manager.config_manager.get("agentic.apply_tools", True)
        self.manager.update_config("agentic.apply_tools", False)

        # Generate final response
        response = self.manager.generate(user_prompt)

        # Restore original setting
        self.manager.update_config("agentic.apply_tools", original_tools_setting)

        # Format the response
        formatted_response = f"""
┌────────────────────────────────────────┐
│             FINAL RESPONSE             │
└────────────────────────────────────────┘
{response}
─────────────────────────────────────────
""".strip()
        print(formatted_response)

        return formatted_response


class AgentOrchestrator:
    """Orchestrates the execution of multiple agents in sequence."""

    def __init__(
        self,
        code_executor: "CodeExecutor",
        manager: "TextGenerationManager",
        max_steps: int,
        agents: List[Type[Agent]] = None,
    ):
        # Default agent pipeline - each agent is responsible for a specific aspect of processing
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

    def process_user_question(
        self,
        user_choice: str,
    ) -> str:
        """
        Process the user's choice through a chain of agents.

        Args:
            user_choice: The user's question or request
            current_step: Current processing cycle

        Returns:
            The final response
        """
        # Preprocess the user question
        _handle_dynamic_system_prompt(user_choice, self.manager)
        user_question, media_objects = parse_media_tags(user_choice, self.code_executor.globals_dict)
        user_question = _handle_rag_for_python(user_question, self.manager)

        # Check if tools are disabled
        apply_tools = self.manager.config_manager.get("agentic.apply_tools", False)
        if not apply_tools:
            response = self.manager.generate(user_question, media_objects=media_objects)
            _ = execute_code_with_feedback(
                response=response,
                original_question=user_question,
                code_executor=self.code_executor,
                prompt_code_execution=self.manager.config_manager.get("main.prompt_code_execution", True),
                prompt_retry_on_error=self.manager.config_manager.get("main.prompt_retry_on_error", True),
            )
            return response

        # Initialize context
        context = AgentContext(
            step=0,
            max_steps=self.max_steps,
            previous_results=self.code_executor.globals_dict.get("tool_results", []),
            media_objects=media_objects,
            final_results=[],
        )

        # Process through agents
        response = ""

        logger.info(f"Starting agent processing for user request: {user_question}")
        available_tools = [
            getattr(obj, "__name__")
            for obj in OwlDefaultFunctions(GlobalPythonVarsDict()).owl_tools(as_json=False)
            if hasattr(obj, "__name__")
        ]
        logger.info(f"Available tools: {available_tools}")

        # First, always run the RouterPlanningAgent
        router_agent = RouterPlanningAgent(self.code_executor, self.manager)
        logger.info(f"Using RouterPlanningAgent, iteration {context.step + 1}/{context.max_steps}")
        router_result = router_agent.process(user_question, context)
        context = router_result["context"]

        # Get the planning result
        planning = context.planning
        planning_steps = planning.get("steps", [])

        if not planning_steps:
            logger.info("No planning steps generated!")
            return "No planning steps generated!"

        answer_is_appropriate = False

        # Process planning steps until final answer is appropriate to answer user question or max steps is reached
        while not answer_is_appropriate and context.step < self.max_steps:
            answer_is_appropriate, context = self.process_planning_steps(
                user_question,
                context,
                planning_steps,
            )
            # get the first value which is not completed and update planning steps
            steps_status = [step["status"].lower() for step in context.completed_steps.values()]
            index_not_completed = next((i for i, status in enumerate(steps_status) if status != "completed"), None)

            if index_not_completed is not None:
                planning_steps = planning_steps[index_not_completed:]
            else:
                break

        logger.info("Running ResponseSynthesisAgent to synthesize final response")
        response_agent = ResponseSynthesisAgent(self.code_executor, self.manager)
        final_results = list_of_dicts_to_llm_context(context.final_results)
        response = response_agent.process(user_question, final_results)

        return response

    def process_planning_steps(
        self,
        user_question: str,
        context: AgentContext,
        planning_steps: list,
    ) -> Tuple[bool, AgentContext]:
        """
        Process each step in the plan.

        Parameters
        ----------
        user_question : str
            The user's question.
        context : AgentContext
            The current context of the agent.
        planning_steps : list
            The list of planning steps.

        Returns
        -------
        Tuple[bool, AgentContext]
            A tuple containing a boolean indicating if the answer is appropriate and the updated context.
        """
        for idx, step in enumerate(planning_steps):
            context.current_plan_index = idx
            agent_type = step.get("agent", "")
            step_description = step.get("description", f"Step {idx + 1}")

            logger.info(f"Processing planstep {idx + 1}/{len(planning_steps)}: {step_description}")
            logger.info(f"Using {agent_type}, iteration {context.step + 1}/{context.max_steps}")

            # Select and run the appropriate agent
            if agent_type == "ToolSelectionAgent":
                agent = ToolSelectionAgent(self.code_executor, self.manager)
                result = agent.process(user_question, context)
                context = result["context"]

                if not result["should_continue"]:
                    logger.info(f"Agent {agent_type} indicated to stop processing.")
                    break

            elif agent_type == "PythonAgent":
                agent = PythonAgent(self.code_executor, self.manager)
                result = agent.process(user_question, context)
                context = result["context"]

                if not result["should_continue"]:
                    logger.info(f"Agent {agent_type} indicated to stop processing.")
                    break

            else:
                logger.warning(f"Unknown agent type: {agent_type}. Skipping.")

        # After processing all steps, run ValidationAgent
        validation_agent = ValidationAgent(self.code_executor, self.manager)
        logger.info("Using ValidationAgent after completing plan steps")
        validation_result = validation_agent.process(user_question, context)
        context = validation_result["context"]

        # Get the validation judgment
        answer_is_appropriate = context.answer_is_appropriate
        validation_response = validation_result.get("response", "")
        step_completion_status = parse_xml(validation_response, "step_completion_status")
        completed_steps = parse_xml_tags_to_dict(step_completion_status)
        completed_steps = {step: parse_xml_tags_to_dict(v) for step, v in completed_steps.items()}
        context.completed_steps = completed_steps

        return answer_is_appropriate, context


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


# TODO: add RAG here to heavily filter information due to long context
def list_of_dicts_to_llm_context(data: List[Dict[str, str]]) -> str:
    """
    Convert a list of dictionaries (each with {filename_or_url: content}) to a formatted string
    optimized for LLM context input.

    For each dictionary in the list, each key-value pair is formatted with a header showing
    the filename/URL, followed by its associated content. A delimiter is used to separate entries.

    Parameters:
    ---------
        data (List[Dict[str, str]]): A list of dictionaries, each containing a filename_or_url and its associated content.

    Returns:
    --------
        str: A formatted string containing the context for the LLM.
    """

    context_parts = []

    for entry_dict in data:
        if not isinstance(entry_dict, dict):
            entry_dict = {"unknown_source": str(entry_dict)}
        for source, content in entry_dict.items():
            content = str(content)
            header = f"---\nSource: {source}\n---"
            entry = f"{header}\n{content.strip()}"
            context_parts.append(entry)

    return "\n".join(context_parts)


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
""".strip()
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


def _get_final_result_from_python_code(response: str, original_question: str, code_executor: CodeExecutor) -> List[str]:
    _ = execute_code_with_feedback(
        response=response,
        original_question=original_question,
        code_executor=code_executor,
        prompt_code_execution=False,  # Always execute tool calls
        prompt_retry_on_error=False,
    )
    final_result = code_executor.globals_dict.get("final_result", [])
    return final_result
