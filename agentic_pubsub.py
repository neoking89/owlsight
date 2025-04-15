"""
Agentic Flow Example for OwlSight.

This script demonstrates a more structured agentic flow using dedicated data
classes (`StepResult`, `PlanStep`, `ExecutionPlan`) and an `AgentOrchestrator`
to manage the process. The plan should follow a sequential structure, with each
step depending on the previous one. It should follow the REACT framework (reasoning/action),

**Core Flow:**
1. A user question is received by `AgentOrchestrator.process_user_question`.
2. The `PlannerAgent` is invoked to create an `ExecutionPlan` (a list of `PlanStep` objects).
These represent the "scratchpad" of the agent.
3. The `AgentOrchestrator` iterates through the plan steps.
4. For each step, the orchestrator retrieves the corresponding agent (e.g., `ContextAgent`,
   `ToolCreationAgent`, `ToolSelectionAgent`) and calls its `execute` method.
5. At the end of the ExecutionPlan, `ValidationAgent` is invoked to check if:
    a. all steps were executed successfully from "scratchpad". If not, failed steps are delegated back to `PlannerAgent`.
    b. enough data is present to compile a final response.
6. If validation is successful, `ResponseSynthesisAgent` is invoked to compile the final response.

TODO:
Remove the Observer pattern completely

Always give back complete code
Keep code runnable
"""

import sys
import logging
import re
from typing import Any, List, Optional, ClassVar, Dict
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum

sys.path.append("src")

# (Assumed available in your project)
# These imports are placeholders for local modules you have in your codebase:
try:
    from owlsight.processors.text_generation_manager import TextGenerationManager
    from owlsight.utils.code_execution import CodeExecutor
    from owlsight.app.default_functions import OwlDefaultFunctions
    from owlsight.utils.helper_functions import parse_xml
except ImportError:
    # For direct-run demonstration, define minimal stubs for these imports
    class TextGenerationManager:
        def generate(self, prompt: str) -> str:
            return "Generated text"

    class CodeExecutor:
        def __init__(self):
            self.globals_dict = {}

    def parse_xml(text: str, tag: str) -> str:
        """
        Very naive XML tag parser for demonstration.
        Searches for <tag>...</tag> and returns content if found.
        """
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        return ""

    class OwlDefaultFunctions:
        def __init__(self, globals_dict):
            pass

        def owl_tools(self, as_json=False):
            return {"stub_tool": "Tool description"}


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Toggle TEST_MODE for demonstration or real usage
TEST_MODE = True


# -------------------------
# Helper Functions & Constants
# -------------------------
def get_agent_information() -> str:
    return "\n".join(f"- {k}: {v}\n" for k, v in AGENT_INFORMATION.items())


def get_available_tools(code_executor: "CodeExecutor"):
    """
    Returns a list of available tool descriptors in the OpenAI function calling format.
    """
    if not TEST_MODE:
        return "\n".join(
            f"- {k}: {v}\n" for k, v in OwlDefaultFunctions(code_executor.globals_dict).owl_tools(as_json=True)
        )

    # Demonstration only
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
                            "description": "LOCAL FILE SYSTEM PATHS OR BUFFERS ONLY.",
                        },
                        "recursive": {
                            "type": "boolean",
                            "description": "Recursively process subdirectories if a directory is provided",
                        },
                        "ignore_patterns": {
                            "type": "string",
                            "description": "Gitignore-style patterns to exclude",
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
                "description": "Scrape web content from URLs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "urls": {
                            "type": "array",
                            "description": "HTTP/HTTPS URLs to process",
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
    ]
    return available_tools


class EventType(Enum):
    USER_QUESTION_RECEIVED = "user_question_received"
    PLAN_CREATED = "plan_created"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STEP_ERROR = "step_error"


# Prompts and agent information strings
AGENT_INFORMATION = {
    "ToolSelectionAgent": "Use for external data retrieval or specialized tool usage.",
    "ToolCreationAgent": "Use ONLY to create dynamic tool functions for later use.",
    "ContextAgent": "Use for analyzing text, summarizing, or extracting info.",
}
AVAILABLE_AGENTS = [" | ".join(AGENT_INFORMATION.keys())]

# -------------------------
# Agent Prompts
# -------------------------
PLANNER_PROMPT = """
You are an expert planner, specializing in task decomposition and agent assignment. 
Analyze the user request:
1. Break it into several subtasks if needed.
2. Assign each subtask to the most suitable agent.
3. Return a structured plan.

AGENT INFORMATION:
{agent_information}

User Question:
{user_question}

AVAILABLE TOOLS:
{available_tools}

Response Format:
<plan>
  <step>
    <description>Step description</description>
    <agent>AgentName</agent>
    <reason>Reason for this step</reason>
  </step>
  <!-- Repeat <step> for each step in the plan -->
</plan>
"""

CONTEXT_PROMPT = """
You are an expert in extracting information and summarizing content. Analyze the user question and any available context.

User Question:
{user_question}

Available Context:
{available_context}

Task:
- Summarize relevant information.
- Extract any data that might help answer the user question.

Response Format:
<summary>
  <relevant_info>Relevant information summary</relevant_info>
</summary>
"""

TOOL_CREATION_PROMPT = """
You are an expert in tool creation. Based on the user question and context, define a new tool function to help solve the problem.

User Question:
{user_question}

Context:
{available_context}

Task:
- Define a new tool function with clear parameters and description.

Response Format:
<tool>
  <name>tool_name</name>
  <description>Tool description</description>
  <parameters>
    <parameter>
      <name>param_name</name>
      <type>string|number|boolean|array|object</type>
      <description>Parameter description</description>
      <required>true|false</required>
    </parameter>
    <!-- Repeat for each parameter -->
  </parameters>
</tool>
"""

TOOL_SELECTION_PROMPT = """
You are an expert in selecting the right tool for a task. Based on the user question and context, choose the most appropriate tool from the available options.

User Question:
{user_question}

Context:
{available_context}

AVAILABLE TOOLS:
{available_tools}

Task:
- Select the best tool for the task.
- Provide the tool name and the parameters to use.

Response Format:
<selection>
  <tool_name>selected_tool_name</tool_name>
  <parameters>
    <parameter>
      <name>param_name</name>
      <value>param_value</value>
    </parameter>
    <!-- Repeat for each parameter -->
  </parameters>
  <reason>Reason for selecting this tool</reason>
</selection>
"""

VALIDATION_PROMPT = """
You are an expert in validating results. Review the execution plan and results.

User Question:
{user_question}

Execution Plan and Results:
{execution_results}

Task:
- Check if all steps were successful.
- Check if enough data is present to compile a final response.
- If not successful, identify which steps need rework.

Response Format:
<validation>
  <successful>true|false</successful>
  <enough_data>true|false</enough_data>
  <failed_steps>
    <step>
      <index>Step index</index>
      <reason>Reason for rework</reason>
    </step>
    <!-- Repeat for each failed step -->
  </failed_steps>
  <next_action>respond|replan</next_action>
</validation>
"""

RESPONSE_SYNTHESIS_PROMPT = """
You are an expert in crafting clear, concise responses. Synthesize all the information from the execution plan into a final response for the user.

User Question:
{user_question}

Execution Results:
{execution_results}

Task:
- Craft a clear, concise response that answers the user's question.
- Include relevant data or results.
- Avoid mentioning the internal process or agents.

Response Format:
<response>
  Final response content here
</response>
"""


# -------------------------
# Data Classes for the Agentic Flow
# -------------------------
@dataclass
class StepResult:
    """
    Container for the outcome of an agent's execution step.

    Attributes:
        success (bool): Whether the step was executed successfully.
        execution_result (Any): The result of the execution (message, data, error info).
    """

    success: bool
    execution_result: Any = None


@dataclass
class PlanStep:
    """
    Represents a single step in the execution plan.

    Attributes:
        description (str): Description of the step.
        agent_name (str): The agent assigned for this step.
        reason (str): Reason for the step.
        result (Optional[StepResult]): Result of the step execution.
        data (Optional[Any]): Data returned by the agent.
    """

    description: str
    agent_name: str
    reason: str
    result: Optional[StepResult] = None
    data: Optional[Any] = None


@dataclass
class StepErrorInfo:
    """
    Stores details about a single step failure, including the step index,
    step description, the attempt number, and the traceback or error message.
    """

    step_index: int
    step_description: str
    attempt_number: int
    traceback_str: str


@dataclass
class ErrorContext:
    """
    Collects all error occurrences (tracebacks) for any step that fails.
    Each failed attempt can be tracked in `step_errors`.
    """

    step_errors: List[StepErrorInfo] = field(default_factory=list)

    def add_error(self, step_index: int, step_description: str, attempt_number: int, traceback_str: str):
        """Append a new step error record."""
        error_info = StepErrorInfo(
            step_index=step_index,
            step_description=step_description,
            attempt_number=attempt_number,
            traceback_str=traceback_str,
        )
        self.step_errors.append(error_info)

    def __str__(self):
        output = []
        for err in self.step_errors:
            output.append(
                f"[Error] Step #{err.step_index + 1} ({err.step_description}), "
                f"Attempt {err.attempt_number}: {err.traceback_str}"
            )
        return "\n".join(output)


@dataclass
class ExecutionPlan:
    """Container for the sequence of steps to be executed."""

    steps: List[PlanStep]

    def get_step(self, index: int) -> Optional[PlanStep]:
        if 0 <= index < len(self.steps):
            return self.steps[index]
        return None

    def get_data(self) -> List[Optional[Any]]:
        return [step.data for step in self.steps if step.data is not None]

    def __getitem__(self, index: int) -> PlanStep:
        if 0 <= index < len(self.steps):
            return self.steps[index]
        raise IndexError(f"ExecutionPlan index {index} out of range")

    def __len__(self) -> int:
        return len(self.steps)

    def __str__(self) -> str:
        return "\n".join(
            [f"Step {i + 1}: {step.description} (Agent: {step.agent_name})" for i, step in enumerate(self.steps)]
        )


@dataclass
class AgentPrompt:
    """
    A flexible prompt template that can be formatted with various parameters.
    """

    template: str
    params: dict[str, Any] = field(default_factory=dict)

    def format(self, **kwargs) -> str:
        format_params = {**self.params, **kwargs}
        try:
            return self.template.format(**format_params)
        except KeyError as e:
            raise KeyError(f"Missing parameter in prompt template: {e}") from None

    def __str__(self) -> str:
        return self.template


@dataclass
class AgentContext:
    """
    Represents the shared state (or central memory) passed among agents, including:
     - The user's original question
     - The index of the current step
     - The execution plan
     - An ErrorContext that can contain multiple StepErrorInfo records
     - A final_response (if any)
    """

    user_question: str
    current_step: int = 0
    execution_plan: Optional[ExecutionPlan] = None
    error_context: Optional[ErrorContext] = field(default_factory=ErrorContext)
    final_response: Optional[str] = None


# -------------------------
# Base Agent Classes
# -------------------------
class BaseAgent(ABC):
    """Base class for all agents."""

    manager: ClassVar[Optional[TextGenerationManager]] = None
    code_executor: ClassVar[Optional[CodeExecutor]] = None

    def __init__(self, name: str, system_prompt: AgentPrompt):
        self.name = name
        self.system_prompt = system_prompt

    def llm_call(self, formatted_prompt: str) -> str:
        """
        In real usage, calls a text generation manager. In TEST_MODE, uses a fixed response.
        """
        if TEST_MODE or BaseAgent.manager is None:
            return f"{self.name} response for: {formatted_prompt[:50]}..."
        return BaseAgent.manager.generate(formatted_prompt)

    @abstractmethod
    def execute(self, context: AgentContext) -> StepResult:
        """
        Subclasses must implement this method to perform their specific tasks
        and update the context as needed.
        """
        pass


class PlannerAgent(BaseAgent):
    """Agent responsible for creating the execution plan."""

    def __init__(self):
        planner_prompt = AgentPrompt(template=PLANNER_PROMPT)
        super().__init__("PlannerAgent", planner_prompt)

    def execute(self, context: AgentContext) -> StepResult:
        formatted_prompt = self.system_prompt.format(
            user_question=context.user_question,
            agent_information=get_agent_information(),
            available_tools=get_available_tools(BaseAgent.code_executor),
        )
        response = self.llm_call(formatted_prompt)
        plan_steps = self._extract_planning_from_response(response)

        if not plan_steps:
            return StepResult(success=False, execution_result="Failed to create plan")

        context.execution_plan = ExecutionPlan(steps=plan_steps)
        logging.info(f"Plan created with {len(plan_steps)} steps.")
        return StepResult(success=True, execution_result=response)

    def _extract_planning_from_response(self, response: str) -> List[PlanStep]:
        """
        Extracts structured plan steps from the LLM response's XML-like format.
        """
        plan_steps = []
        plan_content = parse_xml(response, "plan")
        if not plan_content:
            return []

        step_pattern = r"<step>(.*?)</step>"
        steps = re.findall(step_pattern, plan_content, re.DOTALL)
        for step_text in steps:
            description = parse_xml(step_text, "description")
            agent_name = parse_xml(step_text, "agent")
            reason = parse_xml(step_text, "reason")
            if description and agent_name:
                plan_steps.append(PlanStep(description, agent_name, reason or "No reason provided"))
        return plan_steps
