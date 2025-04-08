"""
Agentic Flow Example for OwlSight.

This script demonstrates a more structured agentic flow using dedicated data
classes (`StepResult`, `PlanStep`, `ExecutionPlan`) and an `AgentOrchestrator`
to manage the process.

**Core Flow:**
1.  A user question is received by `AgentOrchestrator.process_user_question`.
2.  The `PlannerAgent` is invoked to analyze the question and create an
    `ExecutionPlan`, which is a list of `PlanStep` objects. Each step
    specifies an agent to run and a description of the task.
3.  The `AgentOrchestrator` iterates through the `PlanStep`s in the
    `ExecutionPlan`.
4.  For each step, the orchestrator retrieves the corresponding agent
    (e.g., `ContextAgent`, `ToolCreationAgent`) and calls its `execute` method.
5.  The executed agent performs its task and updates the `result` attribute
    (a `StepResult` object) within its assigned `PlanStep` in the shared context.
6.  If any step fails (either by raising an exception caught by the
    orchestrator or by setting `step.result.success = False` and
    `stop_on_step_failure=True`), the orchestrator can optionally attempt to
    re-plan by calling the `PlannerAgent` again with error context.
7.  After all steps in the plan are executed (or execution stops), the
    `ValidationAgent` is called to check the overall success based on the
    results in the `ExecutionPlan`.
8.  Finally, the `ResponseSynthesisAgent` is called to compile the results
    from the executed steps and the validation check into a final, coherent
    response for the user.
"""

import sys

sys.path.append("src")

import logging
from typing import Any, List, Optional, ClassVar, Dict
from dataclasses import dataclass, field
import re

from owlsight.processors.text_generation_manager import TextGenerationManager
from owlsight.utils.code_execution import CodeExecutor
from owlsight.app.default_functions import OwlDefaultFunctions
from owlsight.utils.helper_functions import parse_xml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# TODO: at end, remove TEST_MODE entirely from code
TEST_MODE = True


def get_agent_information() -> str:
    return "\n".join(f"- {k}: {v}\n" for k, v in AGENT_INFORMATION.items())


def get_available_tools(code_executor: CodeExecutor):
    """
    Returns a formatted string of available tools in OpenAI function calling format.

    Parameters:
    ----------
    code_executor : CodeExecutor
        The CodeExecutor instance containing the globals dictionary.

    Returns:
    -------
    str
        A formatted string listing available tools.
    """
    if not TEST_MODE:
        return "\n".join(
            f"- {k}: {v}\n" for k, v in OwlDefaultFunctions(code_executor.globals_dict).owl_tools(as_json=True)
        )

    available_tools = [
        {
            "type": "function",
            "function": {
                "name": "owl_read",
                "description": "Read LOCAL FILE CONTENTS with advanced document processing.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_source": {
                            "type": "string",
                            "description": "LOCAL FILE SYSTEM PATHS OR BUFFERS ONLY. Can be: - Single local file path (str or Path) - Single buffer content from a file (bytes) - Directory path (requires recursive=True) - List of local file paths DOES NOT SUPPORT WEB URLS",
                        },
                        "recursive": {
                            "type": "boolean",
                            "description": "Whether to recursively process subdirectories when file_source is a directory",
                        },
                        "ignore_patterns": {
                            "type": "string",
                            "description": "List of gitignore-style patterns to exclude",
                        },
                        "ocr_enabled": {
                            "type": "boolean",
                            "description": "Whether to enable OCR for image files",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds for document processing",
                        },
                    },
                    "required": ["file_source"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "owl_scrape",
                "description": "Scrape web content from URLs (use instead of owl_read for web resources).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "urls": {
                            "type": "array",
                            "description": "VALID HTTP/HTTPS URLS TO PROCESS Does not support local file paths",
                        },
                        "max_concurrent": {
                            "type": "integer",
                            "description": "Simultaneous requests allowed",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Request timeout in seconds",
                        },
                    },
                    "required": ["urls"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "owl_search",
                "description": "Execute web search using DuckDuckGo's API.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search phrase to look up",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return (1-20)",
                        },
                        "max_retries": {
                            "type": "integer",
                            "description": "Number of retry attempts for failed requests",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "owl_search_and_scrape",
                "description": "Combines web search and content scraping into a single operation. First searches for URLs using DuckDuckGo, then scrapes content from the found URLs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search phrase to look up",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return from search (1-20)",
                        },
                        "max_concurrent": {
                            "type": "integer",
                            "description": "Maximum number of simultaneous scraping requests allowed",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Request timeout in seconds for scraping",
                        },
                        "max_retries": {
                            "type": "integer",
                            "description": "Number of retry attempts for failed search requests",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "owl_write",
                "description": "Write text content to filesystem.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Absolute path for output file. Preferably use a descriptive filename.",
                        },
                        "content": {
                            "type": "string",
                            "description": "Text content to write",
                        },
                    },
                    "required": ["file_path", "content"],
                },
            },
        },
    ]
    return available_tools


AGENT_INFORMATION = {
    "ToolSelectionAgent": "Use when external data retrieval, API calls, or specialized tool usage is required. ALWAYS plan ToolSelectionAgent in a planningstep after creating a dynamic tool with ToolCreationAgent.",
    "ToolCreationAgent": "Use ONLY to create dynamic tool functions that can later be used by ToolSelectionAgent. ALWAYS plan ToolSelectionAgent in a planningstep after creating a dynamic tool. NEVER use ToolCreationAgent for direct computation - it ONLY creates reusable tools (functions)",
    "ContextAgent": "Use for analyzing text, summarizing, extracting info, or generating strategies.",
}

AVAILABLE_AGENTS = [" | ".join(AGENT_INFORMATION.keys())]

PLANNER_PROMPT = """
You are an expert planner, specializing in task decomposition and agent assignment. 
Analyze the user request:
1. Break it into several subtasks if needed. Try to make the steps as atomic as possible.
2. Assign each subtask to the most suitable agent.
3. Return a structured plan.

KEY WORKFLOW PATTERN:
- If the task requires computation or custom functionality, first use ToolCreationAgent to create a dynamic tool
- Then use ToolSelectionAgent in the next step to execute/use that tool
- NEVER use ToolCreationAgent for direct computation - it ONLY creates reusable tools (functions)

AGENT INFORMATION:
{agent_information}

User Question:
{user_question}

Your task: Create a plan (several steps) with an agent for each subtask.

AVAILABLE TOOLS:
{available_tools}

Response Format:
<plan>
  <step>
    <description>First step description</description>
    <agent>AgentOne</agent>
    <reason>The reason for the first step</reason>
  </step>
  <step>
    <description>Second step description</description>
    <agent>AgentTwo</agent>
    <reason>The reason for the second step</reason>
  </step>
</plan>
""".strip()

CONTEXT_AGENT_PROMPT = """
You answer questions based on the provided context. Analyze the user query and relevant context to provide a direct answer.

User Request:
{user_question}

Step {current_step}/{max_steps}

Previous Results: {previous_results}
Final Results from Previous Steps: {final_results}
Additional Info: {additional_info}

Instructions:
Provide a direct response to the user's request based on all available context.
Be concise but thorough.
""".strip()

TOOL_CREATION_PROMPT = """
You are an expert Python developer specialized in creating tool functions. Your job is to ONLY create reusable Python tool functions based on the user's request.

Guidelines for creating tools:
1. Create functions starting with "dynamic_tool_" followed by a descriptive name
2. All tools must have helpful docstrings (created in Numpy style) explaining what they do and their parameters
3. Tools should perform a single, specific task.
4. Use proper error handling and input validation.
5. Tools should be designed for reuse by the ToolSelectionAgent
6. Never execute tasks directly - ONLY create tools that can be executed later

These functions will be registered in the global namespace for the ToolSelectionAgent to use.

Create a dynamic tool function to help with this request: 
{user_question}

Additional context from previous steps:
{previous_results}

REQUIREMENTS:
1. Function name must start with "dynamic_tool_"
2. Include comprehensive docstrings (created in Numpy style)
3. Implement proper error handling
4. Return results in a structured format (dict, list, etc.)
5. DO NOT execute tasks directly - ONLY create functions that can be called later

Example structure:
```python
def dynamic_tool_name(param1, param2=None):
    \"\"\"
    Description of what this tool does.
    
    Parameters:
    ----------
        param1: Description of parameter
        param2: Description of optional parameter
    
    Returns:
    --------
        Description of return value
    \"\"\"
    # Implementation
    return result
```

Create ONLY the tool function(s) required. Do not provide examples of usage or explanations outside the function definition.
""".strip()

TOOL_SELECTION_PROMPT = """
You are an expert in tool selection. If you need a tool, respond ONLY with a JSON object:
{{"name": "tool_name", "arguments": {...}}}
No extra text.

User Request:
{user_question}

Step {current_step}/{max_steps}

Previous Results: {previous_results}
Final Results from Previous Steps: {final_results}
Additional Info: {additional_info}

Instructions:
1. Check if previous tool calls gave needed info.
2. Decide next steps carefully. If you must use another tool, return only valid JSON.
3. Do NOT repeat the same tool call with same arguments.

Available Tools:
{available_tools}

Return Format:
{"name": "tool_name", "arguments": {...}}
""".strip()

VALIDATION_PROMPT = """
You are a meticulous quality assurance agent. Your task is to validate the results of the previous execution steps and report any failures.

User Request:
{user_question}

Execution Plan & Results:
{execution_plan}

Validation Instructions:
1. Review all step results for completeness and success
2. Check if the user's original request has been fulfilled
3. Identify any missing information or failures
4. Provide a clear validation result

Return a detailed validation report including:
- Overall pass/fail status
- Specific issues found (if any)
- Suggestions for improvement
""".strip()

RESPONSE_SYNTHESIS_PROMPT = """
You are a response synthesis agent. Your task is to compile the results from the execution plan and validation into a final, coherent answer for the user.

User Request:
{user_question}

Execution Results:
{execution_results}

Validation Result:
{validation_result}

Instructions:
1. Synthesize all the information into a well-structured, concise response
2. Focus on directly answering the user's original question 
3. Include relevant details from the execution steps
4. Acknowledge any limitations or issues found during validation
5. Format your response in a clear, user-friendly manner
""".strip()

@dataclass
class StepResult:
    """
    Container for the outcome of an agent's execution step

    Attributes:
        success (bool): Whether the step was executed successfully
        execution_result (Any): The result of the execution step (can be a message, data, or error details/tracebacks)
    """

    success: bool
    execution_result: Any = None


@dataclass
class PlanStep:
    """
    Represents a single step in the execution plan.

    Attributes:
        description (str): A description of the step.
        agent_name (str): The name of the agent to execute the step.
        reason (str): The reason for the step.
        result (Optional[StepResult]): The result of the step execution. This is filled in by the agent after execution.
        data (Optional[Any]): Any data that the agent might return.
    """

    description: str
    agent_name: str
    reason: str
    result: Optional[StepResult] = None
    data: Optional[Any] = None


@dataclass
class ExecutionPlan:
    """Container for the sequence of steps to be executed."""

    steps: List[PlanStep]

    def get_step(self, index: int) -> Optional[PlanStep]:
        """Safely get a step by index."""
        if 0 <= index < len(self.steps):
            return self.steps[index]
        return None

    def get_data(self) -> List[Optional[Any]]:
        """Get all collected data from each step so far."""
        return [step.data for step in self.steps if step.data is not None]

    def __getitem__(self, index: int) -> PlanStep:
        """
        Make ExecutionPlan indexable (e.g., plan[0] instead of plan.get_step(0)).

        Args:
            index: The index of the step to retrieve

        Returns:
            The PlanStep at the specified index

        Raises:
            IndexError: If the index is out of range
        """
        if 0 <= index < len(self.steps):
            return self.steps[index]

        raise IndexError(f"ExecutionPlan index {index} out of range")

    def __len__(self) -> int:
        """
        Return the number of steps in the plan.

        This allows using len(plan) to get the step count.
        """
        return len(self.steps)

    def __str__(self) -> str:
        """Return the plan as a string."""
        return "\n".join(
            [f"Step {i + 1}: {step.description} (Agent: {step.agent_name})" for i, step in enumerate(self.steps)]
        )


@dataclass
class AgentPrompt:
    """
    A flexible prompt template that can be formatted with various parameters.

    This class allows for dynamic prompt construction by accepting a template with
    placeholders and formatting it with provided parameters. Parameters can be
    provided during initialization or when calling format().

    Attributes:
        template: The prompt template with formatting placeholders
        params: Pre-defined parameters to use when formatting the template
    """

    template: str
    params: dict[str, Any] = field(default_factory=dict)

    def format(self, **kwargs) -> str:
        """
        Format the template with given parameters.

        Args:
            **kwargs: Format parameters to be merged with any pre-defined parameters.
                     Overlapping keys will be overridden by these parameters.

        Returns:
            The formatted prompt string
        """
        # Combine pre-defined params with new params, with new params taking precedence
        format_params = {**self.params, **kwargs}

        try:
            return self.template.format(**format_params)
        except KeyError as e:
            raise KeyError(f"Missing required parameter in prompt template: {e}") from None

    def __str__(self) -> str:
        """Return the template string in its unformatted form."""
        return self.template


@dataclass
class AgentContext:
    """
    Represents the shared state passed between agents.

    Attributes:
        user_question (str): The original user question
        current_step (int): The current step in the execution plan
        execution_plan (Optional[ExecutionPlan]): The current execution plan
        error_context (Optional[str]): Error context for critical errors. Propagate errors to the PlannerAgent.
        final_response (Optional[str]): Final response from ResponseSynthesisAgent
    """

    user_question: str
    current_step: int = 0
    execution_plan: Optional[ExecutionPlan] = None
    error_context: Optional[str] = None
    final_response: Optional[str] = None


class BaseAgent:
    """Base class for all agents."""

    # Class variables shared across all instances
    manager: ClassVar[Optional[TextGenerationManager]] = None
    code_executor: ClassVar[Optional[CodeExecutor]] = None

    def __init__(self, name: str, system_prompt: AgentPrompt):
        self.name = name
        self.system_prompt = system_prompt

    def llm_call(self, formatted_prompt: str) -> str:
        """
        Makes a call to the LLM using the TextGenerationManager.

        Args:
            formatted_prompt: The formatted prompt to send to the LLM

        Returns:
            The response from the LLM or a mock response if manager is not available
        """
        if TEST_MODE:
            if self.name == "PlannerAgent":
                return """
I'll analyze the user's request and create a structured execution plan.

<plan>
  <step>
    <description>Research current weather conditions in Chicago</description>
    <agent>ToolSelectionAgent</agent>
    <reason>We need to retrieve up-to-date weather data which requires an external tool</reason>
  </step>
  <step>
    <description>Create a function to analyze temperature trends</description>
    <agent>ToolCreationAgent</agent>
    <reason>A custom tool is needed to process the raw weather data into meaningful trends</reason>
  </step>
  <step>
    <description>Summarize the weather findings and provide recommendations</description>
    <agent>ContextAgent</agent>
    <reason>After collecting and processing the data, we need to synthesize the information into actionable insights</reason>
  </step>
</plan>

Based on the user's question about weather conditions in Chicago, I've created a three-step plan that will:
1. Get the current weather data using available tools
2. Create a specialized function to analyze temperature patterns
3. Provide a final analysis with recommendations
"""
            else:
                raise ValueError(f"{self.name} is not a valid agent name for TEST_MODE.")

        # actual llm call
        if BaseAgent.manager is not None:
            return BaseAgent.manager.generate(formatted_prompt)
        raise ValueError("TextGenerationManager is not initialized")

    def execute(self, context: AgentContext) -> AgentContext:
        """Executes the agent's logic.

        Modifies the context, specifically by updating the result of the
        current plan step within the execution_plan.
        Should ideally not raise exceptions for plan execution errors,
        but capture them in the StepResult object. Unforeseen errors might still raise.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class PlannerAgent(BaseAgent):
    """
    Agent responsible for creating the execution plan, containing the plan steps.

    Each step is a single step in the plan, executed by a different agent.
    """

    def __init__(self):
        planner_prompt = AgentPrompt(
            template=PLANNER_PROMPT,
        )
        super().__init__("PlannerAgent", planner_prompt)

    def execute(self, context: AgentContext) -> AgentContext:
        # This agent *creates* the plan, doesn't execute a step within it.
        logging.info(f"Executing {self.name}...")
        user_question = context.user_question
        agent_information = get_agent_information()
        available_tools = get_available_tools(BaseAgent.code_executor)

        formatted_prompt = self.system_prompt.format(
            user_question=user_question,
            agent_information=agent_information,
            available_tools=available_tools,
            available_agents=AVAILABLE_AGENTS,
        )

        response = self.llm_call(formatted_prompt)
        extracted_steps = self._extract_planning_from_response(response)
        plan_steps = [PlanStep(**step) for step in extracted_steps]

        execution_plan = ExecutionPlan(steps=plan_steps)
        context.execution_plan = execution_plan
        logging.info(f"Planner generated plan: {execution_plan}")

        return context

    @staticmethod
    def _extract_planning_from_response(response: str) -> List[Dict[str, str]]:
        """
        Parse all plansteps from the PlannerAgent's output.
        """
        plan_match = parse_xml(response, "plan")

        if not plan_match:
            logging.warning("No plan found in PlannerAgent response.")
            return []

        plan_text = plan_match.strip()
        steps = []

        # Extract each step XML block
        step_matches = re.findall(r"<step>(.*?)</step>", plan_text, re.DOTALL)

        for step_content in step_matches:
            description = parse_xml(step_content, "description")
            agent = parse_xml(step_content, "agent")
            reason = parse_xml(step_content, "reason")

            if description:
                step = {
                    "description": description.strip(),
                    "agent_name": agent.strip() if agent else "",
                    "reason": reason.strip() if reason else "",
                }
                steps.append(step)

        return steps


class ContextAgent(BaseAgent):
    """
    Agent responsible for providing context and analysis.
    Its main task is distilling and analysing data to extract a certain context (e.g. a summary of a document, a table of contents, etc.).
    """

    def __init__(self):
        context_prompt = AgentPrompt(template=CONTEXT_AGENT_PROMPT)
        super().__init__("ContextAgent", context_prompt)

    def execute(self, context: AgentContext) -> AgentContext:
        plan = context.execution_plan
        current_index = context.current_step
        step = plan.get_step(current_index) if plan else None

        if step:
            try:
                # Format the prompt with actual values
                formatted_prompt = self.system_prompt.format(
                    user_question=context.user_question,
                    current_step=current_index + 1,
                    max_steps=len(plan.steps) if plan else 1,
                    previous_results="None",
                    final_results="None",
                    additional_info="",
                )

                response = self.llm_call(formatted_prompt)

                # In a real implementation, process the response as needed
                analysis_details = response

                # Fallback to simple mock for testing when no actual LLM is available
                if not response or "Mock LLM response" in response:
                    analysis_details = f"Contextual Answer based on '{context.user_question}'."

                logging.info(f"ContextAgent result: {analysis_details[:50]}...")
                step.result = StepResult(success=True, execution_result=analysis_details)
            except Exception as e:
                logging.error(f"{self.name} failed: {e}", exc_info=True)
                step.result = StepResult(success=False, execution_result=f"Error during context analysis: {e}")
        # Error logging for missing step handled by BaseAgent or orchestrator

        return context


class ToolCreationAgent(BaseAgent):
    """
    Agent responsible for creating new tools in Python, which can be executed by the ToolExecutionAgent.
    """

    def __init__(self):
        tool_creation_prompt = AgentPrompt(template=TOOL_CREATION_PROMPT)
        super().__init__("ToolCreationAgent", tool_creation_prompt)

    def execute(self, context: AgentContext) -> AgentContext:
        plan = context.execution_plan
        current_index = context.current_step
        step = plan.get_step(current_index) if plan else None

        if step:
            try:
                # In a real implementation, this would format the prompt with actual values
                # formatted_prompt = self.system_prompt.format(
                #    user_question=context.get('user_question', ''),
                #    previous_results=""  # Would contain info from previous steps
                # )

                # Mock Python code generation or execution
                python_code = "def new_tool():\n    print('Hello from dynamic ToolCreationAgent!')"  # Example tool code
                logging.info(f"ToolCreationAgent result: {python_code}")

                # Simulate an error sometimes for testing the error loop
                # import random
                # if random.random() < 0.3: # 30% chance of error
                #     error_msg = "Simulated error in ToolCreationAgent"
                #     logging.error(f"ToolCreationAgent encountered a simulated error!")
                #     step.result = StepResult(success=False, execution_result=error_msg)
                # else:
                # Only register if no error
                step.result = StepResult(success=True, execution_result=python_code)
            except Exception as e:
                logging.error(f"{self.name} failed: {e}", exc_info=True)
                step.result = StepResult(success=False, execution_result=f"Error during tool creation: {e}")

        return context


class ToolSelectionAgent(BaseAgent):
    """
    Agent responsible for selecting the appropriate tool available in the Python sandbox environment for the current step.
    """

    def __init__(self):
        tool_selection_prompt = AgentPrompt(template=TOOL_SELECTION_PROMPT)
        super().__init__("ToolSelectionAgent", tool_selection_prompt)

    def execute(self, context: AgentContext) -> AgentContext:
        plan = context.execution_plan
        current_index = context.current_step
        step = plan.steps[current_index] if plan else None

        if step:
            try:
                # In a real implementation, we would format the prompt with actual values
                # formatted_prompt = self.system_prompt.format(
                #    user_question=context.get('user_question', ''),
                #    current_step=current_index + 1,
                #    max_steps=len(plan.steps),
                #    previous_results="None",
                #    final_results="None",
                #    additional_info="",
                #    dynamic_tools=""  # Would populate with dynamic tools from previous steps
                # )

                # Default success status
                success_status = True

                # Logic for checking previous step
                if current_index > 0:
                    # For this example, let's assume ToolSelectionAgent depends on ToolCreationAgent
                    prev_index = current_index - 1
                    prev_step = plan.get_step(prev_index)

                    if prev_step and prev_step.agent_name == "ToolCreationAgent" and prev_step.result:
                        if prev_step.result.success:
                            python_result = prev_step.result.execution_result
                            tool_selection_details = (
                                f"Selected 'MockTool' based on ToolCreationAgent result: {python_result}"
                            )
                        else:
                            # Previous step failed, select fallback but maybe this step still 'succeeds' in selecting fallback
                            tool_selection_details = f"Selected 'FallbackTool' as previous ToolCreationAgent step failed ({prev_step.result.execution_result})."
                            logging.warning(f"{self.name}: Previous step failed, selecting fallback.")
                            # Decide if ToolSelection *itself* failed due to dependency failure.
                            # Let's say it still succeeds in selecting *something* (the fallback).
                            success_status = True
                    else:
                        # Previous step missing or no result (shouldn't happen with ToolCreation->ToolSelection plan)
                        logging.warning(f"{self.name}: Previous step or its result missing.")
                        tool_selection_details = "Selected 'DefaultTool' due to missing previous step info."

                    logging.info(f"ToolSelectionAgent result: {tool_selection_details}")
                    step.result = StepResult(success=success_status, execution_result=tool_selection_details)

                # Handle no dependency scenario
                else:
                    tool_selection_details = "Selected 'IndependentTool' as there are no previous steps to depend on."
                    logging.info(f"{self.name} result: {tool_selection_details}")
                    step.result = StepResult(success=True, execution_result=tool_selection_details)

            except Exception as e:
                logging.error(f"{self.name} failed: {e}", exc_info=True)
                step.result = StepResult(success=False, execution_result=f"Error during tool selection: {e}")

        return context


class ValidationAgent(BaseAgent):
    def __init__(self):
        validation_prompt = AgentPrompt(template=VALIDATION_PROMPT)
        super().__init__("ValidationAgent", validation_prompt)

    def execute(self, context: AgentContext) -> AgentContext:
        # This agent is called *after* the plan execution loop.
        # It inspects the results in the plan and updates context["validation_result"].
        plan = context.execution_plan
        validation_passed = True
        details = "Validation passed."

        # In a real implementation, we would format the prompt with actual values
        # formatted_prompt = self.system_prompt.format(
        #    user_question=context.get('user_question', ''),
        #    execution_plan=str(plan),  # Would need proper formatting for plan representation
        # )

        if plan and plan.steps:
            for i, step in enumerate(plan.steps):
                if not step.result:
                    validation_passed = False
                    details = f"Validation failed: Step {i + 1} '{step.description}' ({step.agent_name}) seems not to have run or produced a result."
                    logging.warning(details)
                    break
                elif not step.result.success:
                    validation_passed = False
                    details = f"Validation failed: Step {i + 1} '{step.description}' ({step.agent_name}) did not succeed. Details: {step.result.execution_result if step.result else 'N/A'}"
                    logging.warning(details)
                    break  # Stop on first failure
        else:
            validation_passed = False
            details = "Validation failed: No execution plan found or plan is empty."
            logging.error(details)

        logging.info(f"ValidationAgent result: {details}")
        context["validation_result"] = details
        # We could also return a StepResult object from this agent if needed elsewhere
        # return StepResult(success=validation_passed, execution_result=details) # If we wanted to standardize
        return context


class ResponseSynthesisAgent(BaseAgent):
    def __init__(self):
        response_synthesis_prompt = AgentPrompt(template=RESPONSE_SYNTHESIS_PROMPT)
        super().__init__("ResponseSynthesisAgent", response_synthesis_prompt)

    def execute(self, context: AgentContext) -> AgentContext:
        # This agent is called *after* the ValidationAgent.
        # It examines the execution results and validation to create a final response.
        # Extract relevant information from context
        user_question = context.user_question
        validation_result = context.validation_result
        plan = context.execution_plan

        # In a real implementation, we would format the prompt with actual values
        # formatted_prompt = self.system_prompt.format(
        #    user_question=user_question,
        #    execution_results=self._format_execution_results(plan) if plan else "No execution results available.",
        #    validation_result=validation_result
        # )

        # Default response if there's an issue with the plan
        if not plan or not plan.steps:
            final_response = f"I couldn't process your request: '{user_question}'. No valid execution plan was created."
            logging.warning("ResponseSynthesisAgent: No execution plan to synthesize.")
        else:
            # Custom response based on validation result
            if "failed" in validation_result.lower():
                final_response = f"I encountered an issue while processing your request: '{validation_result}'. Would you like to try again or modify your request?"
            else:
                # Success path: Combine results from all steps
                successful_steps = []
                for i, step in enumerate(plan.steps):
                    if step.result and step.result.success:
                        successful_steps.append(f"Step {i + 1}: {step.result.execution_result}")

                # Format a nice response
                if successful_steps:
                    steps_summary = "\n".join(successful_steps)
                    final_response = (
                        f"I've processed your request: '{user_question}'.\n\nHere's what I found:\n{steps_summary}"
                    )
                else:
                    final_response = f"I processed your request, but there were no successful steps to report. The validation says: {validation_result}"

        logging.info(f"ResponseSynthesisAgent produced final response")
        context.final_response = final_response
        return context


class AgentOrchestrator:
    def __init__(self):
        # Agent instances are still stored here for lookup by name
        self.agents: dict[str, BaseAgent] = {
            "PlannerAgent": PlannerAgent(),
            "ContextAgent": ContextAgent(),
            "ToolCreationAgent": ToolCreationAgent(),
            "ToolSelectionAgent": ToolSelectionAgent(),
            "ValidationAgent": ValidationAgent(),
            "ResponseSynthesisAgent": ResponseSynthesisAgent(),
        }
        self.max_error_loops = 2  # Prevent infinite loops on persistent errors
        self.stop_on_step_failure = (
            False  # Configuration: Should orchestrator stop if a step returns StepResult(success=False)?
        )

    def _get_agent(self, agent_name: str) -> BaseAgent:
        agent = self.agents.get(agent_name)
        if not agent:
            # This indicates a mismatch between planner and available agents
            raise ValueError(f"Orchestrator Error: Unknown agent '{agent_name}' specified in plan.")
        return agent

    def process_user_question(self, user_question: str) -> str:
        context: AgentContext = AgentContext(user_question=user_question)
        error_loops = 0

        while error_loops <= self.max_error_loops:
            plan_failed_or_incomplete = False
            try:
                # 1. Planning Phase (or Re-planning on error)
                router = self._get_agent("PlannerAgent")
                context = router.execute(context)

                plan = context.execution_plan
                if not plan or not plan.steps:
                    logging.error("Planner failed to generate a valid plan.")
                    context.validation_result = "Planning Failed: Planner did not produce a valid plan."
                    context = self._get_agent("ResponseSynthesisAgent").execute(context)  # Synthesize error response
                    return context.final_response

                # 2. Execution Phase (Process plan steps)
                while context.current_step < len(plan.steps):
                    current_step = plan.steps[context.current_step]

                    # Ensure result is None before execution (important for re-runs)
                    current_step.result = None

                    agent_name = current_step.agent_name
                    logging.info(
                        f"--- Executing Plan Step {context.current_step + 1}/{len(plan.steps)}: Agent '{agent_name}' ---"
                    )
                    logging.info(f"    Description: {current_step.description}")

                    agent_to_execute = self._get_agent(agent_name)  # Raises ValueError if agent not found

                    # Execute the agent - it should update current_step.result internally
                    context = agent_to_execute.execute(context)

                    # --- Check step result AFTER execution ---
                    if not current_step.result:
                        # Agent didn't produce a result - treat as failure
                        current_step.result = StepResult(
                            success=False, execution_result=f"Agent '{agent_name}' did not produce a result object."
                        )
                        logging.error(current_step.result.execution_result)
                        plan_failed_or_incomplete = True
                    elif not current_step.result.success:
                        # Agent reported failure
                        error_msg = f"Agent '{agent_name}' reported failure for step {context.current_step + 1}. Details: {current_step.result.execution_result}"
                        logging.error(error_msg)
                        plan_failed_or_incomplete = True

                    # --- Decide whether to continue based on failure ---
                    if plan_failed_or_incomplete and self.stop_on_step_failure:
                        logging.warning(
                            f"Stopping execution due to failure in step {context.current_step + 1} and stop_on_step_failure=True."
                        )
                        break  # Exit the execution loop

                    context.current_step += 1
                # --- End of Execution Loop ---

                # 3. Validation Phase (Executed after all plan steps attempted or stop_on_failure)
                logging.info("--- Executing Validation Phase ---")
                validator = self._get_agent("ValidationAgent")
                context = validator.execute(context)

                # 4. Response Synthesis Phase
                logging.info("--- Executing Response Synthesis Phase ---")
                synthesizer = self._get_agent("ResponseSynthesisAgent")
                context = synthesizer.execute(context)

                # If orchestration successful (no fatal exceptions), break the error loop
                return context.final_response

            except Exception as e:  # Catch orchestration errors (e.g., unknown agent) or unexpected agent errors
                logging.error(f"Orchestration Error during execution: {e}", exc_info=True)
                error_loops += 1
                if error_loops > self.max_error_loops:
                    logging.critical(f"Maximum error re-planning loops ({self.max_error_loops}) exceeded.")
                    return f"Error: Processing failed after multiple attempts. Last error: {e}"

                # Prepare context for re-planning
                plan = context.execution_plan
                step_info = "N/A"
                if plan:
                    current_index = context.current_step
                    step = plan.get_step(current_index)
                    if step:
                        step_info = f"Agent '{step.agent_name}', Step Index {current_index}"

                context.error_context = f"Orchestration error: {e}. Occurred near: {step_info}"
                context.execution_plan = None  # Clear potentially inconsistent plan
                context.current_step = 0
                logging.warning(f"Attempting to re-plan (Attempt {error_loops}/{self.max_error_loops})...")
                # Loop continues, starting with PlannerAgent again

        # Fallback if loop finishes unexpectedly
        return "Error: Unexpected exit from processing loop."


# Example usage
if __name__ == "__main__":
    orchestrator = AgentOrchestrator()
    # orchestrator.stop_on_step_failure = True # Uncomment to test early stopping

    print("\n--- Example 1: Analyze Question ---")
    question1 = "Analyze the main components of the system."
    response1 = orchestrator.process_user_question(question1)
    print(f"\nUser Question: {question1}")
    print(f"Final Response:\n{response1}")

    # print("\n\n--- Example 2: Python/Tool Question ---")
    # question2 = "Create a tool using python for a simple calculator and select it."
    # response2 = orchestrator.process_user_question(question2)
    # print(f"\nUser Question: {question2}")
    # print(f"Final Response:\n{response2}")

    # print("\n\n--- Example 3: General Question ---")
    # question3 = "What is OwlSight?"
    # response3 = orchestrator.process_user_question(question3)
    # print(f"\nUser Question: {question3}")
    # print(f"Final Response:\n{response3}")

    # # To test error handling, uncomment the error simulation in ToolCreationAgent
    # # and run the Python/Tool question again.
    # # Note how failure is now handled via StepResult(success=False)
    # print("\n\n--- Example 4: (Potential Error Test if uncommented in ToolCreationAgent) ---")
    # question4 = "Give me python code to create a file reading tool."
    # response4 = orchestrator.process_user_question(question4)
    # print(f"\nUser Question: {question4}")
    # print(f"Final Response:\n{response4}")
