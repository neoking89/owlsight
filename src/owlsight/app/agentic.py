import re
import ast
import json
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


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def safe_cast(value: Any) -> Any:
    """
    Best‑effort conversion of string inputs into Python primitives.
    Handles booleans, None, ints, floats, lists, dicts.

    Any non‑string or failed conversion returns the value unchanged.
    """
    if not isinstance(value, str):
        return value

    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def get_agent_information() -> str:
    return "\n".join(f"- {k}: {v}" for k, v in AGENT_INFORMATION.items())


def get_available_tools(code_executor: "CodeExecutor") -> str:
    """
    Return tool descriptors already registered in the executor's namespace.
    """
    logger.debug("Getting available tools...") # <<< Added logger.debug
    tools = OwlDefaultFunctions(code_executor.globals_dict).owl_tools(as_json=True)
    logger.debug(f"Available tools: {tools}") # <<< Added logger.debug
    return "\n".join(
        str(t) for t in tools
    )


def parse_tool_response(response: str) -> Dict[str, Any]:
    """
    Accepts JSON *or* the original XML format.
    Returns dict with 'tool_name', 'parameters', 'reason'.
    """
    response = response.strip()

    # ---------- JSON branch -------------------------------------------------
    try:
        candidate = json.loads(response)
        if isinstance(candidate, dict) and "tool_name" in candidate:
            candidate.setdefault("parameters", {})
            candidate["parameters"] = {k: safe_cast(v) for k, v in candidate["parameters"].items()}
            return candidate
    except Exception:
        pass

    # ---------- XML branch --------------------------------------------------
    import xml.etree.ElementTree as ET

    if not response.startswith("<selection>"):
        raise ValueError("Unexpected format for tool selection.")

    root = ET.fromstring(response)
    tool_name = root.findtext("tool_name", "").strip()
    reason = root.findtext("reason", "").strip()
    parameters: Dict[str, Any] = {}
    for p in root.findall("parameters/parameter"):
        name = p.findtext("name", "").strip()
        val = safe_cast(p.findtext("value", "").strip())
        parameters[name] = val

    return {"tool_name": tool_name, "parameters": parameters, "reason": reason}


def execute_tool(code_executor: CodeExecutor, tool_data: Dict[str, Any]):
    """
    Safe wrapper that executes a registered tool with converted parameters.
    """
    tool_name = tool_data.get("tool_name", "")
    params = {k: safe_cast(v) for k, v in tool_data.get("parameters", {}).items()}

    try:
        func = code_executor.globals_dict[tool_name]
    except KeyError:
        msg = f"Tool '{tool_name}' not found."
        logger.warning(msg)
        return ToolResult(False, msg)

    try:
        result = func(**params)
        logger.info("Tool '%s' executed successfully.", tool_name)
        return ToolResult(True, result)
    except Exception as exc:
        logger.exception("Error while executing tool '%s'", tool_name)
        return ToolResult(False, str(exc))


# --------------------------------------------------------------------------- #
# Data containers                                                             #
# --------------------------------------------------------------------------- #
@dataclass
class ToolResult:
    success: bool
    result: Any


class EventType(Enum):
    USER_QUESTION_RECEIVED = "user_question_received"
    PLAN_CREATED = "plan_created"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STEP_ERROR = "step_error"


AGENT_INFORMATION = {
    "ToolSelectionAgent": "Use for external data retrieval or specialized tool usage.",
    "ToolCreationAgent": "Use ONLY to create dynamic tool functions for later use.",
    "FinalAgent": "Use for synthesizing the final response.",
}
AVAILABLE_AGENTS = [" | ".join(AGENT_INFORMATION.keys())]

# --------------------------------------------------------------------------- #
# Prompt templates (refined)                                                  #
# --------------------------------------------------------------------------- #
PLANNER_PROMPT = """
You are an expert planner, specializing in task decomposition and agent assignment.

Task:
Analyze the user request:
1. Break it into logically distinct subtasks if needed.
2. Assign each subtask to the most suitable agent.
3. Reason carefully about which tools are necessary for each step, ensuring the chosen tool matches the subtask's requirements (e.g., use `owl_read` for LOCAL files, `owl_scrape` for specific URLs, `owl_search` or `owl_search_and_scrape` for web searches). DO NOT use `owl_read` to process web content obtained from search/scrape tools.
4. If the query can be answered directly based on the model's training data without external tools or data, assign it directly to FinalAgent.
5. **Avoid redundant steps.** If a tool combines actions (like `owl_search_and_scrape`), do not plan separate follow-up steps for those combined actions (like scraping again).
6. **Be specific.** If the request involves multiple distinct locations, items, or topics (e.g., "New York City" and "Amsterdam, Netherlands"), create SEPARATE plan steps with FOCUSED tool queries for EACH distinct entity. Use precise location names.
7. **Understand context flow.** After a `ToolSelectionAgent` step, the `ObservationAgent` runs AUTOMATICALLY to summarize the tool's output. **NEVER plan an explicit step for `ObservationAgent`.** Subsequent steps work with the summary provided automatically in the context.
8. Return a structured plan.

Agent Information:
- ToolSelectionAgent: Use for external data retrieval or specialized tool usage. Its output is AUTOMATICALLY summarized by ObservationAgent before the next step.
- ToolCreationAgent: Use ONLY to create dynamic tool functions for later use.
- FinalAgent: Use for synthesizing the final response using accumulated context (including automatically generated observations).

User Question:
{user_question}

AVAILABLE TOOLS:
{available_tools}

Additional Information:
{additional_information}

Important:
- Prioritize any guidance or constraints provided in the Additional Information when planning.

Response Format:
<plan>
  <step>
    <description>Step description</description>
    <agent>AgentName</agent>
    <reason>Reason for this step, including potential tool usage, expected inputs (e.g., previous observation), and why this agent is chosen.</reason>
  </step>
  <!-- Repeat <step> for each step in the plan -->
</plan>
"""

TOOL_CREATION_PROMPT = """
You are an expert in tool creation. Based on the user question and context, define a new tool function to help solve the problem.

User Question:
{user_question}

Context:
{available_context}

Additional Information:
{additional_information}

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

OBSERVATION_PROMPT = """
You are an expert in analyzing and summarizing tool execution results. Filter out irrelevant information and keep only what is essential for answering the user question.

Description:
{description}

Tool Execution Result:
{tool_result}

Additional Information:
{additional_information}

Task:
- Summarize the tool execution result.

Response Format:
<observation>Summary of relevant information</observation>
"""

VALIDATION_PROMPT = """
You are an expert in validating results. Check if the accumulated results are sufficient to answer the user question.

User Question:
{user_question}

Accumulated Results:
{accumulated_results}

Additional Information:
{additional_information}

Task:
- Determine if the results answer the question.
- If not, suggest the next step or tool to use.

Response Format:
<validation>
  <is_complete>true|false</is_complete>
  <next_step_if_incomplete>Next step or tool suggestion if incomplete</next_step_if_incomplete>
</validation>
"""

RESPONSE_SYNTHESIS_PROMPT = """
You are an expert in synthesizing information to provide a comprehensive and accurate response to the user.

User Question:
{user_question}

Context and Results from Previous Steps:
{previous_results}

Additional Information:
{additional_information}

Task:
- Analyze all available information.
- Provide a clear, concise, and accurate response that addresses the user's query.

Response Format:
<response>
  <content>Final response content</content>
</response>
"""


# --------------------------------------------------------------------------- #
# Execution‑plan & error tracking data classes                                #
# --------------------------------------------------------------------------- #
@dataclass
class StepResult:
    success: bool
    execution_result: Any = None


@dataclass
class PlanStep:
    description: str
    agent_name: str
    reason: str
    result: Optional[StepResult] = None


@dataclass
class StepErrorInfo:
    step_index: int
    step_description: str
    attempt_number: int
    traceback_str: str


@dataclass
class ErrorContext:
    step_errors: List[StepErrorInfo] = field(default_factory=list)

    def add_error(self, step_index: int, step_description: str, attempt_number: int, traceback_str: str):
        self.step_errors.append(
            StepErrorInfo(step_index, step_description, attempt_number, traceback_str)
        )

    def __str__(self):
        if not self.step_errors:
            return "No errors"
        return "\n".join(
            f"Step {e.step_index} ({e.step_description}), Attempt {e.attempt_number}: {e.traceback_str}"
            for e in self.step_errors
        )


@dataclass
class ExecutionPlan:
    steps: List[PlanStep]

    def get_step(self, index: int) -> Optional[PlanStep]:
        return self.steps[index] if 0 <= index < len(self.steps) else None

    def __getitem__(self, index: int) -> PlanStep:
        return self.steps[index]

    def __len__(self):
        return len(self.steps)

    def __str__(self):
        return "\n".join(f"Step {i + 1}: {s.description} ({s.agent_name})" for i, s in enumerate(self.steps))


@dataclass
class AgentPrompt:
    """A flexible prompt template that can be formatted with various parameters."""

    template: str
    params: dict[str, Any] = field(default_factory=dict)

    def format(self, **kwargs) -> str:
        combined_params = {**self.params, **kwargs}
        formatted_prompt = self.template.format(**combined_params)
        logger.debug(self._get_amt_tokens(formatted_prompt, combined_params))
        return formatted_prompt

    def __str__(self) -> str:
        return self.template

    def _get_amt_tokens(self, formatted_prompt: str, params: dict[str, Any]) -> str:
        """
        Estimate token count by dividing character length by
        the average characters-per-token ratio and return log messages as a string.
        """
        AVG_CHARS_PER_TOKEN = 4
        log_lines = []

        prompt_chars = len(formatted_prompt)
        prompt_tokens = math.ceil(prompt_chars / AVG_CHARS_PER_TOKEN)
        log_lines.append(f"Total tokens in 'formatted_prompt': {prompt_chars} chars -> {prompt_tokens} tokens")

        for param, value in params.items():
            val_str = str(value)
            val_chars = len(val_str)
            val_tokens = math.ceil(val_chars / AVG_CHARS_PER_TOKEN)
            log_lines.append(f"Parameter '{param}': {val_chars} chars -> {val_tokens} tokens")

        return "\n".join(log_lines)


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
    error_context: ErrorContext = field(default_factory=ErrorContext)
    final_response: Optional[str] = None
    accumulated_results: List[Any] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Base agent                                                                  #
# --------------------------------------------------------------------------- #
class BaseAgent(ABC):
    manager: ClassVar[Optional[TextGenerationManager]] = None
    code_executor: ClassVar[Optional[CodeExecutor]] = None

    def __init__(self, name: str, system_prompt: AgentPrompt):
        self.name = name
        self.system_prompt = system_prompt

    def llm_call(self, formatted_prompt: str) -> str:
        """
        Generate a response from the LLM.
        """
        if not self.manager:
            raise ValueError("TextGenerationManager not set for BaseAgent.")

        logger.debug(f"Agent '{self.name}' making LLM call with prompt:\n{formatted_prompt}")
        response = self.manager.generate(formatted_prompt)
        logger.debug(f"Agent '{self.name}' received LLM response:\n{response}")
        return response

    @abstractmethod
    def execute(self, context: AgentContext) -> StepResult:
        """
        Execute the agent's task.
        """
        ...

    def get_previous_results(self, context: AgentContext) -> str:
        """
        Format accumulated results from previous steps for inclusion in prompts.
        """
        if not context.accumulated_results:
            return "No previous results"
        out = []
        for i, r in enumerate(context.accumulated_results):
            tag = "tool" if isinstance(r, ToolResult) else "note"
            out.append(f"Step {i + 1} ({tag}): {r.result if isinstance(r, ToolResult) else r}")
        return "\n".join(out)

    def get_additional_information(self) -> str:
        cmgr = getattr(self.manager, "config_manager", None)
        if cmgr is None:
            return ""
        return cmgr.get("agentic.additional_information", "")


# --------------------------------------------------------------------------- #
# Concrete agents                                                             #
# --------------------------------------------------------------------------- #
class PlannerAgent(BaseAgent):
    def __init__(self):
        super().__init__("PlannerAgent", AgentPrompt(PLANNER_PROMPT))

    def execute(self, context: AgentContext) -> StepResult:
        prompt = self.system_prompt.format(
            user_question=context.user_question,
            agent_information=get_agent_information(),
            available_tools=get_available_tools(BaseAgent.code_executor),
            additional_information=self.get_additional_information(),
        )
        reply = self.llm_call(prompt)
        steps: List[PlanStep] = self._extract(reply)
        if not steps:
            return StepResult(False, "Planning failed")
        context.execution_plan = ExecutionPlan(steps)
        return StepResult(True, steps)

    def _extract(self, xml: str) -> List[PlanStep]:
        plan_xml = parse_xml(xml, "plan")
        if not plan_xml:
            return []
        parsed: List[PlanStep] = []
        for seg in re.findall(r"<step>(.*?)</step>", plan_xml, re.DOTALL):
            desc = parse_xml(seg, "description")
            ag = parse_xml(seg, "agent")
            reason = parse_xml(seg, "reason")
            if desc and ag:
                parsed.append(PlanStep(desc, ag, reason or ""))
        return parsed


class ToolCreationAgent(BaseAgent):
    def __init__(self):
        super().__init__("ToolCreationAgent", AgentPrompt(TOOL_CREATION_PROMPT))

    def execute(self, context: AgentContext) -> StepResult:
        prompt = self.system_prompt.format(
            user_question=context.user_question,
            available_context=self.get_previous_results(context),
            additional_information=self.get_additional_information(),
        )
        reply = self.llm_call(prompt)
        data = self._extract(reply)
        if not data:
            return StepResult(False, "Tool extraction failed")
        registered = self._register_dynamic_tool(data)
        context.accumulated_results.append({"dynamic_tools_created": registered})
        return StepResult(True, registered)

    def _extract(self, xml: str) -> Dict[str, Any]:
        node = parse_xml(xml, "tool")
        if not node:
            return {}
        out = {
            "name": parse_xml(node, "name"),
            "description": parse_xml(node, "description"),
            "parameters": [],
        }
        params_block = parse_xml(node, "parameters")
        if params_block:
            for p in re.findall(r"<parameter>(.*?)</parameter>", params_block, re.DOTALL):
                out["parameters"].append(
                    {
                        "name": parse_xml(p, "name"),
                        "type": parse_xml(p, "type"),
                        "description": parse_xml(p, "description"),
                        "required": parse_xml(p, "required") == "true",
                    }
                )
        return out if out["name"] else {}

    # ---------------------- critical patch here ----------------------------
    def _register_dynamic_tool(self, data: Dict[str, Any]) -> List[str]:
        name = data["name"]
        params = [p["name"] for p in data.get("parameters", [])]
        sig = ", ".join(params)
        # Build a simple echo‑style tool; teams can extend.
        code = [
            f"def {name}({sig}):",
            f'    """{data["description"]}"""',
            "    return {'tool': '" + name + "', 'args': locals()}",
            "",
        ]
        source = "\n".join(code)
        ns: Dict[str, Any] = {}
        try:
            exec(source, BaseAgent.code_executor.globals_dict, ns)
            BaseAgent.code_executor.globals_dict[name] = ns[name]
            logger.info("Dynamic tool '%s' registered.", name)
            return [name]
        except Exception as exc:
            logger.exception("Could not register generated tool '%s'", name)
            return []


class ToolSelectionAgent(BaseAgent):
    def __init__(self):
        super().__init__("ToolSelectionAgent", AgentPrompt(TOOL_SELECTION_PROMPT))

    def execute(self, context: AgentContext) -> StepResult:
        prompt = self.system_prompt.format(
            user_question=context.user_question,
            available_context=self.get_previous_results(context),
            available_tools=get_available_tools(BaseAgent.code_executor),
            additional_information=self.get_additional_information(),
        )
        reply = self.llm_call(prompt)
        call = parse_tool_response(reply)
        tool_result = execute_tool(BaseAgent.code_executor, call)
        context.accumulated_results.append(tool_result)
        return StepResult(tool_result.success, tool_result)


class ObservationAgent(BaseAgent):
    def __init__(self):
        super().__init__("ObservationAgent", AgentPrompt(OBSERVATION_PROMPT))

    def execute(self, context: AgentContext) -> StepResult:
        # use the most recent tool result
        tool_result = next(
            (r for r in reversed(context.accumulated_results) if isinstance(r, ToolResult)), None
        )
        if tool_result is None:
            return StepResult(False, "No tool result to observe.")

        desc = context.execution_plan[context.current_step].description
        prompt = self.system_prompt.format(
            description=desc,
            tool_result=tool_result.result,
            additional_information=self.get_additional_information(),
        )
        summary_xml = self.llm_call(prompt)
        summary = parse_xml(summary_xml, "observation").strip()
        
        # Replace the previous ToolResult with the Observation summary
        if context.accumulated_results and isinstance(context.accumulated_results[-1], ToolResult):
            context.accumulated_results[-1] = summary
        else: # Should not happen if called right after a tool, but handle defensively
            context.accumulated_results.append(summary)
            logger.warning("ObservationAgent appended summary instead of replacing. Last result was not ToolResult.")

        return StepResult(True, summary)


class ValidationAgent(BaseAgent):
    def __init__(self):
        super().__init__("ValidationAgent", AgentPrompt(VALIDATION_PROMPT))

    def execute(self, context: AgentContext) -> StepResult:
        prompt = self.system_prompt.format(
            user_question=context.user_question,
            accumulated_results=self.get_previous_results(context),
            additional_information=self.get_additional_information(),
        )
        result = self.llm_call(prompt)
        context.accumulated_results.append(result)
        return StepResult(True, result)


class FinalAgent(BaseAgent):
    def __init__(self):
        super().__init__("FinalAgent", AgentPrompt(RESPONSE_SYNTHESIS_PROMPT))

    def execute(self, context: AgentContext) -> StepResult:
        prompt = self.system_prompt.format(
            user_question=context.user_question,
            previous_results=self.get_previous_results(context),
            additional_information=self.get_additional_information(),
        )
        reply = self.llm_call(prompt)
        context.final_response = reply
        return StepResult(True, reply)


# --------------------------------------------------------------------------- #
# Orchestrator with stop‑the‑world & re‑planning                              #
# --------------------------------------------------------------------------- #
class AgentOrchestrator:
    def __init__(
        self,
        code_executor: CodeExecutor,
        manager: TextGenerationManager,
        max_retries_per_step: int = 3,
        max_replans: int = 2,
    ):
        self.code_executor = code_executor
        self.manager = manager
        self.max_retries = max_retries_per_step
        self.max_replans = max_replans

        self.agents: Dict[str, BaseAgent] = {
            "PlannerAgent": PlannerAgent(),
            "ToolCreationAgent": ToolCreationAgent(),
            "ToolSelectionAgent": ToolSelectionAgent(),
            "ObservationAgent": ObservationAgent(),
            "ValidationAgent": ValidationAgent(),
            "FinalAgent": FinalAgent(),
        }

        BaseAgent.manager = manager
        BaseAgent.code_executor = code_executor

    # --------------------------------------------------------------------- #
    # public API                                                            #
    # --------------------------------------------------------------------- #
    def process_user_question(self, question: str) -> str:
        context = AgentContext(user_question=question)
        logger.info("Received question: %s", question)

        if not self._plan(context):
            return "Planning failed."

        if not self._execute(context):
            return (
                "Execution aborted due to unrecoverable errors:\n"
                f"{context.error_context}"
            )

        if context.final_response is None:
            self.agents["ValidationAgent"].execute(context)
            self.agents["FinalAgent"].execute(context)

        return context.final_response or "Failed to generate answer."

    # ------------------------------------------------------------------ #
    # internal routines                                                  #
    # ------------------------------------------------------------------ #
    def _plan(self, context: AgentContext) -> bool:
        res = self.agents["PlannerAgent"].execute(context)
        return res.success

    def _execute(self, context: AgentContext) -> bool:
        replan_count = 0
        while True:
            for idx, step in enumerate(context.execution_plan.steps):
                context.current_step = idx
                retries = 0
                while retries < self.max_retries:
                    agent = self.agents[step.agent_name]
                    try:
                        result = agent.execute(context)
                        step.result = result
                        if result.success:
                            # auto‑observe after ToolSelection
                            if step.agent_name == "ToolSelectionAgent":
                                self.agents["ObservationAgent"].execute(context)
                            break  # success
                        raise RuntimeError(str(result.execution_result))
                    except Exception as exc:
                        retries += 1
                        logger.error(
                            "Error in step %d (%s): %s", idx + 1, step.agent_name, exc, exc_info=True
                        )
                        context.error_context.add_error(idx, step.description, retries, str(exc))

                # step failed after retries → stop‑the‑world
                if step.result is None or not step.result.success:
                    if replan_count < self.max_replans:
                        replan_count += 1
                        logger.info("Re‑planning due to failure in step %d (attempt %d).", idx + 1, replan_count)
                        if not self._plan(context):
                            return False
                        break  # restart outer loop with new plan
                    return False
            else:
                # loop finished without break: all steps executed
                return True
