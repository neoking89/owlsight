"""
Revised and optimized OwlSight agentic logic, context management, and orchestration.
This implementation is based on the structured flow from agentic_pubsub.py.
"""

import re
from typing import Any, List, Optional, ClassVar, Dict
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import math

from owlsight.processors.text_generation_manager import TextGenerationManager
from owlsight.utils.code_execution import CodeExecutor
from owlsight.app.default_functions import OwlDefaultFunctions
from owlsight.utils.helper_functions import parse_xml
from owlsight.utils.logger import logger


@dataclass
class ToolResult:
    success: bool
    result: Any


def get_agent_information() -> str:
    return "\n".join(f"- {k}: {v}\n" for k, v in AGENT_INFORMATION.items())


def get_available_tools(code_executor: "CodeExecutor"):
    """
    Returns a list of available tool descriptors in the OpenAI function calling format.
    """
    return "\n".join(str(v) for v in OwlDefaultFunctions(code_executor.globals_dict).owl_tools(as_json=True))


def parse_tool_response(response: str) -> dict:
    """
    Parse the tool response from a string in XML format to a dictionary.
    """
    import xml.etree.ElementTree as ET

    # Clean up the response by removing any extra whitespace or newlines
    response = response.strip()
    # Check if the response is wrapped in selection tags
    if response.startswith("<selection>") and response.endswith("</selection>"):
        root = ET.fromstring(response)
        tool_name = root.find("tool_name").text
        parameters = {}
        for param in root.findall("parameters/parameter"):
            name = param.find("name").text
            value = param.find("value").text
            parameters[name] = value
        reason = root.find("reason").text
        return {"tool_name": tool_name, "parameters": parameters, "reason": reason}
    else:
        # Fallback to a default or error handling if the format is unexpected
        raise ValueError("Unexpected response format: Response must be in XML format with selection tags")


def execute_tool(code_executor: CodeExecutor, tool_data: Dict[str, Any]) -> ToolResult:
    """
    Execute the selected tool with the provided arguments using the CodeExecutor.

    Args:
        code_executor: The CodeExecutor instance to use for tool execution.
        tool_data: Dictionary containing tool information, including name and arguments.

    Returns:
        Dictionary with execution results or error information.
    """
    try:
        tool_name = tool_data.get("tool_name")
        parameters = tool_data.get("parameters", {})
        # Convert string arguments to integers where necessary
        converted_parameters = {}
        for key, value in parameters.items():
            if isinstance(value, str) and value.isdigit():
                converted_parameters[key] = int(value)
            elif isinstance(value, str) and value.lower() in ("true", "false"):
                converted_parameters[key] = value.lower() == "true"
            else:
                converted_parameters[key] = value
        if tool_name in code_executor.globals_dict:
            tool_func = code_executor.globals_dict[tool_name]
            result = tool_func(**converted_parameters)
            logger.info(f"Tool {tool_name} executed successfully")
            return ToolResult(success=True, result=result)
        else:
            msg = f"Tool {tool_name} not found in globals"
            logger.warning(msg)
            return ToolResult(success=False, result=msg)
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {str(e)}")
        return ToolResult(success=False, result=str(e))


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

Additional Information:
{additional_information}

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

OBSERVATION_PROMPT = """
You are an expert in filtering and summarizing tool execution results. Analyze the raw output from the tool execution and extract only the core, relevant information.

Task:
{description}

Tool Results:
{tool_result}

Task:
- Filter out unnecessary details, noise, or irrelevant information from the tool execution result, which is not directly relevant to complete the Task.
- Summarize the core information that is relevant to complete the Task, considering the reasoning behind the tool selection.

Response Format:
<observation>
  Concise summary of relevant information to complete the Task
</observation>
"""


@dataclass
class StepResult:
    """Container for the outcome of an agent's execution step.

    Attributes:
        success (bool): Whether the step was executed successfully.
        execution_result (Any): The result of the execution (message, data, error info).
    """

    success: bool
    execution_result: Any = None


@dataclass
class PlanStep:
    """Represents a single step in the execution plan.

    Attributes:
        description (str): Description of the step.
        agent_name (str): The agent assigned for this step.
        reason (str): Reason for the step.
        result (Optional[StepResult]): Result of the step execution.
    """

    description: str
    agent_name: str
    reason: str
    result: Optional[StepResult] = None


@dataclass
class StepErrorInfo:
    """Stores details about a single step failure, including the step index,
    step description, the attempt number, and the traceback or error message."""

    step_index: int
    step_description: str
    attempt_number: int
    traceback_str: str


@dataclass
class ErrorContext:
    """Collects all error occurrences (tracebacks) for any step that fails.
    Each failed attempt can be tracked in `step_errors`."""

    step_errors: List[StepErrorInfo] = field(default_factory=list)

    def add_error(self, step_index: int, step_description: str, attempt_number: int, traceback_str: str) -> None:
        """Append a new step error record."""
        error_info = StepErrorInfo(step_index, step_description, attempt_number, traceback_str)
        self.step_errors.append(error_info)

    def __str__(self) -> str:
        if not self.step_errors:
            return "No errors"
        error_msgs = []
        for err in self.step_errors:
            error_msgs.append(
                f"Step {err.step_index} ({err.step_description}), Attempt {err.attempt_number}: {err.traceback_str}"
            )
        return "\n".join(error_msgs)


@dataclass
class ExecutionPlan:
    """Container for the sequence of steps to be executed."""

    steps: List[PlanStep]

    def get_step(self, index: int) -> Optional[PlanStep]:
        if 0 <= index < len(self.steps):
            return self.steps[index]
        return None

    def get_data(self) -> List[Any]:
        return [
            step.result.execution_result
            for step in self.steps
            if step.result and step.result.execution_result is not None
        ]

    def __getitem__(self, index: int) -> PlanStep:
        return self.steps[index]

    def __len__(self) -> int:
        return len(self.steps)

    def __str__(self) -> str:
        return "\n".join(f"Step {i + 1}: {step.description} ({step.agent_name})" for i, step in enumerate(self.steps))


@dataclass
class AgentPrompt:
    """A flexible prompt template that can be formatted with various parameters."""

    template: str
    params: dict[str, Any] = field(default_factory=dict)

    def format(self, **kwargs) -> str:
        combined_params = {**self.params, **kwargs}
        formatted_prompt = self.template.format(**combined_params)
        self._get_amt_tokens(formatted_prompt, combined_params)
        return formatted_prompt

    def __str__(self) -> str:
        return self.template

    def _get_amt_tokens(self, formatted_prompt: str, params: dict[str, Any]) -> int:
        """
        Estimate token count by dividing character length by
        the average characters-per-token ratio.
        """
        AVG_CHARS_PER_TOKEN = 4

        prompt_chars = len(formatted_prompt)
        prompt_tokens = math.ceil(prompt_chars / AVG_CHARS_PER_TOKEN)
        logger.info(f"Total tokens in 'formatted_prompt': {prompt_chars} chars -> {prompt_tokens} tokens")

        for param, value in params.items():
            val_str = str(value)
            val_chars = len(val_str)
            val_tokens = math.ceil(val_chars / AVG_CHARS_PER_TOKEN)
            logger.info(f"Parameter '{param}': {val_chars} chars -> {val_tokens} tokens")


@dataclass
class AgentContext:
    """Represents the shared state (or central memory) passed among agents, including:
    - The user's original question
    - The index of the current step
    - The execution plan
    - An ErrorContext that can contain multiple StepErrorInfo records
    - A final_response (if any)
    - Accumulated results from previous steps
    """

    user_question: str
    current_step: int = 0
    execution_plan: Optional[ExecutionPlan] = None
    error_context: Optional[ErrorContext] = field(default_factory=ErrorContext)
    final_response: Optional[str] = None
    accumulated_results: List[Any] = field(default_factory=list)


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
        Calls the text generation manager to generate a response.
        """
        return BaseAgent.manager.generate(formatted_prompt)

    @abstractmethod
    def execute(self, context: AgentContext) -> StepResult:
        """
        Subclasses must implement this method to perform their specific tasks
        and update the context as needed.
        """
        pass

    def get_previous_results(self, context: AgentContext) -> str:
        """
        Format accumulated results from previous steps for inclusion in prompts.
        """
        if not context.accumulated_results:
            return "None"
        return "\n".join(str(result) for result in context.accumulated_results)


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
        logger.info(f"Plan created with {len(plan_steps)} steps.")
        return StepResult(success=True, execution_result=plan_steps)

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


class ContextAgent(BaseAgent):
    """Agent responsible for analyzing text and summarizing content."""

    def __init__(self):
        context_prompt = AgentPrompt(template=CONTEXT_PROMPT)
        super().__init__("ContextAgent", context_prompt)

    def execute(self, context: AgentContext) -> StepResult:
        formatted_prompt = self.system_prompt.format(
            user_question=context.user_question,
            available_context=self.get_previous_results(context),
        )
        response = self.llm_call(formatted_prompt)
        context.accumulated_results.append(response)
        return StepResult(success=True, execution_result=response)


class ToolCreationAgent(BaseAgent):
    """Agent responsible for creating dynamic tool functions."""

    def __init__(self):
        tool_creation_prompt = AgentPrompt(template=TOOL_CREATION_PROMPT)
        super().__init__("ToolCreationAgent", tool_creation_prompt)

    def execute(self, context: AgentContext) -> StepResult:
        formatted_prompt = self.system_prompt.format(
            user_question=context.user_question,
            available_context=self.get_previous_results(context),
        )
        response = self.llm_call(formatted_prompt)
        tool_data = self._extract_tool_data(response)
        if tool_data:
            created_tools = self._register_dynamic_tools(tool_data)
            context.accumulated_results.append({"dynamic_tools_created": created_tools})
            return StepResult(success=True, execution_result=response)
        else:
            context.accumulated_results.append("Failed to create tool")
            return StepResult(success=False, execution_result="Failed to extract tool data from response")

    def _extract_tool_data(self, response: str) -> Dict[str, Any]:
        """
        Extract tool definition data from the response.
        """
        tool_content = parse_xml(response, "tool")
        if not tool_content:
            return {}

        tool_data = {
            "name": parse_xml(tool_content, "name"),
            "description": parse_xml(tool_content, "description"),
            "parameters": [],
        }

        params_content = parse_xml(tool_content, "parameters")
        if params_content:
            param_pattern = r"<parameter>(.*?)</parameter>"
            params = re.findall(param_pattern, params_content, re.DOTALL)
            for param_text in params:
                param_data = {
                    "name": parse_xml(param_text, "name"),
                    "type": parse_xml(param_text, "type"),
                    "description": parse_xml(param_text, "description"),
                    "required": parse_xml(param_text, "required") == "true",
                }
                tool_data["parameters"].append(param_data)

        return tool_data if tool_data["name"] else {}

    def _register_dynamic_tools(self, tool_data: Dict[str, Any]) -> List[str]:
        """
        Register dynamic tools in the global namespace.
        This is a placeholder for actual implementation which would involve creating
        a Python function based on the tool data and registering it with the CodeExecutor.
        """
        # For now, we'll just return the tool name as if it was created
        # In a full implementation, we would generate Python code for this tool
        tool_name = tool_data.get("name", "")
        logger.info(f"Dynamic tool {tool_name} created")
        return [tool_name] if tool_name else []


class ToolSelectionAgent(BaseAgent):
    """Agent responsible for selecting and using tools."""

    def __init__(self):
        tool_selection_prompt = AgentPrompt(template=TOOL_SELECTION_PROMPT)
        super().__init__("ToolSelectionAgent", tool_selection_prompt)

    def execute(self, context: AgentContext) -> StepResult:
        formatted_prompt = self.system_prompt.format(
            user_question=context.user_question,
            available_context=self.get_previous_results(context),
            available_tools=get_available_tools(BaseAgent.code_executor),
            additional_information=BaseAgent.manager.config_manager.get("agentic.additional_information", ""),
        )
        response = self.llm_call(formatted_prompt)
        tool_data = parse_tool_response(response)
        if tool_data:
            tool_result = execute_tool(BaseAgent.code_executor, tool_data)
            step_result = StepResult(success=tool_result.success, execution_result=tool_result)
            if tool_result.success:
                context.accumulated_results.append(tool_result)
                logger.info(f"Tool executed with result: {tool_result.result}")
            return step_result
        else:
            context.accumulated_results.append(f"Failed to select tool in step {context.current_step}")
            logger.warning("Failed to parse tool selection response")
            return StepResult(success=False, execution_result="Failed to parse tool selection response")


class ObservationAgent(BaseAgent):
    """
    Agent responsible for filtering and summarizing tool execution results to keep only relevant information. Only used after a tool execution.
    """

    def __init__(self):
        observation_prompt = AgentPrompt(template=OBSERVATION_PROMPT)
        super().__init__("ObservationAgent", observation_prompt)

    def execute(self, context: AgentContext) -> StepResult:
        logger.info(f"Executing {self.name} for step {context.current_step}")

        # Find the last actual tool result by iterating backwards
        tool_result = "No tool result available"
        for result in reversed(context.accumulated_results):
            if isinstance(result, ToolResult):
                tool_result = result.result
                if result.success:
                    break
                else:
                    return StepResult(success=False, execution_result=tool_result)

        description = context.execution_plan[context.current_step].description

        # Only use the last tool result; do not include any previous results or broader history
        formatted_prompt = self.system_prompt.format(
            tool_result=tool_result,
            description=description,
        )
        response = self.llm_call(formatted_prompt)
        filtered_result = parse_xml(response, "observation").strip()
        return StepResult(success=True, execution_result=filtered_result)


class ValidationAgent(BaseAgent):
    """Agent responsible for validating the execution results."""

    def __init__(self):
        validation_prompt = AgentPrompt(template=VALIDATION_PROMPT)
        super().__init__("ValidationAgent", validation_prompt)

    def execute(self, context: AgentContext) -> StepResult:
        execution_results = ""
        if context.execution_plan:
            execution_results = str(context.execution_plan)
        formatted_prompt = self.system_prompt.format(
            user_question=context.user_question,
            execution_results=execution_results,
        )
        response = self.llm_call(formatted_prompt)
        context.accumulated_results.append(response)
        return StepResult(success=True, execution_result=response)


class ResponseSynthesisAgent(BaseAgent):
    """Agent responsible for synthesizing the final response."""

    def __init__(self):
        response_synthesis_prompt = AgentPrompt(template=RESPONSE_SYNTHESIS_PROMPT)
        super().__init__("ResponseSynthesisAgent", response_synthesis_prompt)

    def execute(self, context: AgentContext) -> StepResult:
        execution_results = ""
        if context.execution_plan:
            execution_results = str(context.execution_plan)
        formatted_prompt = self.system_prompt.format(
            user_question=context.user_question,
            execution_results=execution_results,
        )
        response = self.llm_call(formatted_prompt)
        context.final_response = response
        context.accumulated_results.append(response)
        return StepResult(success=True, execution_result=response)


# -------------------------
# Agent Orchestrator
# -------------------------
class AgentOrchestrator:
    """Orchestrates the execution of agents to handle complex user requests."""

    def __init__(self, code_executor: CodeExecutor, manager: TextGenerationManager, max_retries_per_step: int = 3):
        self.code_executor = code_executor
        self.manager = manager
        self.max_retries_per_step = max_retries_per_step
        self.agents: Dict[str, BaseAgent] = {
            "PlannerAgent": PlannerAgent(),
            "ContextAgent": ContextAgent(),
            "ToolCreationAgent": ToolCreationAgent(),
            "ToolSelectionAgent": ToolSelectionAgent(),
            "ValidationAgent": ValidationAgent(),
            "ResponseSynthesisAgent": ResponseSynthesisAgent(),
            "ObservationAgent": ObservationAgent(),
        }
        # Set class variables for agents
        BaseAgent.manager = manager
        BaseAgent.code_executor = code_executor

    def process_user_question(self, user_question: str) -> str:
        """
        Coordinates the multi-agent pipeline to process a user question and return a final response.
        """
        context = AgentContext(user_question=user_question)
        logger.info(f"Processing user question: {user_question}")

        # Step 1: Create the execution plan
        planner_result = self.agents["PlannerAgent"].execute(context)
        if not planner_result.success:
            return "Failed to create execution plan."
        context.execution_plan = ExecutionPlan(steps=planner_result.execution_result)
        logger.info(f"Plan created with {len(context.execution_plan)} steps")

        # Step 2: Execute each step in the plan
        for step_idx in range(len(context.execution_plan)):
            context.current_step = step_idx
            step = context.execution_plan.get_step(step_idx)
            agent_name = step.agent_name
            max_retries = self.max_retries_per_step

            for attempt in range(max_retries):
                logger.info(
                    f"Executing step {step_idx + 1}/{len(context.execution_plan)}: {step.description} with {agent_name} (Attempt {attempt + 1}/{max_retries})"
                )
                try:
                    agent_result = self.agents[agent_name].execute(context)
                    step.result = agent_result

                    # If the agent executed was ToolSelectionAgent, automatically trigger ObservationAgent
                    if agent_name == "ToolSelectionAgent":
                        logger.info(f"Tool execution completed. Triggering ObservationAgent to filter results.")
                        observation_result = self.agents["ObservationAgent"].execute(context)
                        # Update the last result with the filtered observation instead of appending
                        if context.accumulated_results and observation_result.success:
                            context.accumulated_results[-1] = observation_result.execution_result
                        else:
                            context.accumulated_results.append(observation_result.execution_result)
                        logger.info(f"ObservationAgent filtered tool results for step {step_idx + 1}")

                    logger.info(f"Completed step {step_idx + 1}: {step.description}")
                    break
                except Exception as e:
                    logger.error(f"Error in step {step_idx + 1} (Attempt {attempt + 1}): {str(e)}")
                    context.error_context.add_error(step_idx, step.description, attempt + 1, str(e))
                    if attempt == max_retries - 1:
                        logger.error(f"Failed step {step_idx + 1} after {max_retries} attempts")

            if step.result and not step.result.success:
                logger.error(f"Step {step_idx + 1} failed: {step.description}")

        # Step 3: Synthesize the final response if no final response is set
        if context.final_response is None:
            validation_result = self.agents["ValidationAgent"].execute(context)
            if validation_result.success:
                synthesis_result = self.agents["ResponseSynthesisAgent"].execute(context)
                if synthesis_result.success:
                    context.final_response = synthesis_result.execution_result
                else:
                    context.final_response = "Failed to synthesize response."
            else:
                context.final_response = "Validation failed, unable to provide a response."

        logger.info(f"Final response generated for user question: {user_question}")
        return context.final_response


# # -------------------------
# # Main Function (for testing)
# # -------------------------
# if __name__ == "__main__":
#     orchestrator = AgentOrchestrator(code_executor=CodeExecutor(), manager=TextGenerationManager())
#     test_question = "How can I analyze this dataset?"
#     response = orchestrator.process_user_question(test_question)
#     print(f"User Question: {test_question}")
#     print(f"Response: {response}")
