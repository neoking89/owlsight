import ast
import json
import inspect
import math
import re
import traceback
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional, get_type_hints

from owlsight.app.default_functions import OwlDefaultFunctions
from owlsight.processors.text_generation_manager import TextGenerationManager
from owlsight.utils.code_execution import CodeExecutor
from owlsight.utils.helper_functions import parse_markdown, parse_xml
from owlsight.utils.logger import logger


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def get_agent_information() -> str:
    return "\n".join(f"- {k}: {v}" for k, v in AGENT_INFORMATION.items())


def get_available_tools(code_executor: "CodeExecutor") -> str:
    """
    Return tool descriptors already registered in the executor's namespace.
    """
    logger.debug("Getting available tools...")
    tools = OwlDefaultFunctions(code_executor.globals_dict).owl_tools(as_json=True)
    logger.debug(f"Available tools: {tools}")
    return "\n".join(str(t) for t in tools)


def parse_tool_response(response: str) -> Dict[str, Any]:
    """
    Accepts a single JSON object *or* a single XML <selection> element.
    Returns dict with 'tool_name', 'parameters', 'reason'.
    Raises ValueError if the input format is invalid or contains multiple selections.
    """
    response = response.strip()
    logger.debug(f"Attempting to parse tool response: {response}")

    # ---------- JSON branch -------------------------------------------------
    try:
        # Attempt to load JSON strictly first
        candidate = json.loads(response)
        if isinstance(candidate, dict) and "tool_name" in candidate:
            candidate.setdefault("parameters", {})
            candidate["parameters"] = {k: v for k, v in candidate.get("parameters", {}).items()}
            candidate.setdefault("reason", "")
            logger.debug(f"Parsed as JSON: {candidate}")
            return candidate
        else:
            raise ValueError("Invalid JSON format for tool selection. Expected a single object with 'tool_name'.")

    except json.JSONDecodeError:
        logger.debug("Not valid JSON, attempting XML parsing.")
        pass
    except ValueError as e:
        # Re-raise the specific format error
        logger.error(f"JSON format error: {e}")
        raise e
    except Exception as e:
        logger.warning(f"Unexpected error during JSON processing: {e}")
        pass

    # ---------- XML branch --------------------------------------------------
    try:
        # Try to extract content within <selection> tags first to handle potential surrounding text
        match = re.search(r"<selection>(.*?)</selection>", response, re.DOTALL | re.IGNORECASE)
        if not match:
            # If no <selection> tags, try parsing the whole response directly as a fallback
            logger.warning("No <selection> tags found, attempting direct XML parse.")
            xml_content = response
            # Basic check: ensure it starts like XML
            if not xml_content.startswith("<"):
                raise ValueError("Tool response is not valid JSON and does not appear to be XML.")
        else:
            xml_content = match.group(1).strip()

        # Ensure we don't have nested <selection> by mistake
        if "<selection>" in xml_content.lower():
            raise ValueError("Nested <selection> tags detected. Invalid format.")

        # Prepend a root tag for safety if parsing extracted content, needed if original lacks single root
        # Though ET.fromstring expects a single root element already
        # We'll parse the *original* response if match was found, assuming <selection> is the root
        try:
            # Add a temporary root if needed, assuming selection content might be fragmented
            # Simplified: ET.fromstring should handle single <selection> content directly
            # We'll parse the *original* response if match was found, assuming <selection> is the root
            root = ET.fromstring(response if match else xml_content)
            # Verify the root tag is indeed 'selection' if we parsed the original response
            if match and root.tag.lower() != "selection":
                raise ValueError("Expected root element <selection> not found.")

        except ET.ParseError as pe:
            # Check if the error is 'junk after document element', indicating multiple roots
            if "junk after document element" in str(pe):
                raise ValueError(
                    f"Invalid XML: Multiple root elements found. Expected a single <selection> element. Content: {response}"
                ) from pe
            else:
                raise ValueError(
                    f"Invalid XML format for tool selection. ParseError: {pe}\nContent:\n{xml_content}"
                ) from pe

        tool_name = root.findtext("tool_name", "").strip()
        reason = root.findtext("reason", "").strip()
        parameters: Dict[str, Any] = {}
        params_element = root.find("parameters")
        if params_element is not None:
            for p in params_element.findall("parameter"):
                name = p.findtext("name", "").strip()
                # Handle potentially None value_text more explicitly
                value_text = p.findtext("value")
                value_str = value_text.strip() if value_text is not None else ""
                val = value_str
                if name:
                    parameters[name] = val

        if not tool_name:
            raise ValueError("Missing 'tool_name' in XML selection.")

        result = {"tool_name": tool_name, "parameters": parameters, "reason": reason}
        logger.debug(f"Parsed as XML: {result}")
        return result

    except ET.ParseError as e:
        # This specifically catches the 'junk after document element' error for multiple selections
        # Now handled potentially inside the 'try' block as well
        logger.error(f"XML ParseError: {e}. Response: {response}")
        raise ValueError(
            f"Invalid XML format for tool selection. Expected a single <selection> element. ParseError: {e}\nResponse:\n{response}"
        ) from e
    except ValueError as e:
        logger.error(f"XML Value Error: {e}. Response: {response}")
        raise e
    except Exception as e:
        # Catch other potential XML parsing errors
        logger.exception(f"Failed to parse XML tool selection: {e}. Response: {response}")
        raise ValueError(f"Failed to parse tool selection (Unknown XML error): {e}\nResponse:\n{response}") from e


def execute_tool(code_executor: CodeExecutor, tool_data: Dict[str, Any]):
    """
    Safe wrapper that executes a registered tool with parameters cast according to type hints.
    """
    tool_name = tool_data.get("tool_name", "")
    raw_params = tool_data.get("parameters", {})

    try:
        func = code_executor.globals_dict[tool_name]
    except KeyError:
        msg = f"Tool '{tool_name}' not found."
        logger.warning(msg)
        return ToolResult(False, msg)

    try:
        # Get type hints from the function signature
        type_hints = get_type_hints(func)
        
        # Cast parameters according to type hints
        params = {}
        for param_name, param_value in raw_params.items():
            if param_name in type_hints:
                target_type = type_hints[param_name]
                # Handle basic type casting
                try:
                    if target_type == bool and isinstance(param_value, str):
                        # Special handling for boolean values
                        param_value = param_value.lower() in {'true', 'yes', '1', 'y'}
                    elif target_type in (int, float, str):
                        # Cast to the target type
                        param_value = target_type(param_value)
                    elif target_type == list and isinstance(param_value, str):
                        # Try to convert string to list using ast.literal_eval
                        try:
                            param_value = ast.literal_eval(param_value)
                            if not isinstance(param_value, list):
                                param_value = [param_value]
                        except (ValueError, SyntaxError):
                            # If parsing fails, treat it as a single item list
                            param_value = [param_value]
                    elif target_type == dict and isinstance(param_value, str):
                        # Try to convert string to dict using ast.literal_eval
                        try:
                            param_value = ast.literal_eval(param_value)
                            if not isinstance(param_value, dict):
                                param_value = {'value': param_value}
                        except (ValueError, SyntaxError):
                            # If parsing fails, create a simple key-value dict
                            param_value = {'value': param_value}
                except (ValueError, TypeError):
                    # If casting fails, use the original value
                    logger.warning(f"Failed to cast parameter '{param_name}' to {target_type}")
            
            params[param_name] = param_value
            
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

# --------------------------------------------------------------------------- #
# Prompt templates (refined)                                                  #
# --------------------------------------------------------------------------- #
PLANNER_PROMPT = """
You are an expert planner, specializing in task decomposition and agent assignment.

Task:
Analyze the user request:
1. Break it into logically distinct subtasks if needed.
2. Assign each subtask to the most suitable agent.
3. Reason carefully about which tools are necessary for each step, ensuring the chosen tool matches the subtask's requirements.
4. If the query can be answered directly based on the model's training data without external tools or data, assign it directly to FinalAgent.
5. **Avoid redundant steps.** If a tool combines actions (like a function containing 'and' or 'or'), do not plan separate follow-up steps for those combined actions (like scraping again).
6. **Be specific AND FOCUSED.** If the request involves multiple distinct locations, items, or topics (e.g., "weather in New York City" and "weather in Amsterdam"), create SEPARATE plan steps. Each step MUST target ONLY ONE of these distinct entities. For instance, one step for 'Get NYC weather' using ToolSelectionAgent, followed by another step for 'Get Amsterdam weather' using ToolSelectionAgent. DO NOT create a single step trying to execute both steps.
7. **Understand context flow.** After a `ToolSelectionAgent` step, the `ObservationAgent` runs AUTOMATICALLY to summarize the tool's output based on the step description. **NEVER plan an explicit step for `ObservationAgent`.** Subsequent steps work with the summary provided automatically in the context.
8. Return a structured plan.

CRITICAL CONSTRAINTS:
- Each step in the plan MUST correspond to a SINGLE, atomic action.
- If multiple distinct actions or tool uses are needed (e.g., searching for two different topics, reading a file then searching), create SEPARATE steps for each action.
- DO NOT combine multiple tool calls or distinct logical operations into a single step.
- DO NOT assign multiple tools to one step.
- A step involving `ToolSelectionAgent` implies the use of exactly ONE tool for that step from **AVAILABLE TOOLS**.

Agent Information:
- ToolSelectionAgent: Use ONLY for selecting and executing ONE specific tool from **AVAILABLE TOOLS**. Its output is AUTOMATICALLY summarized by ObservationAgent.
- ToolCreationAgent: PRIORITIZE this agent whenever the user explicitly requests to create, write, or implement a function, method, tool, utility, or any other programming construct. This agent specializes in creating Python code that can be dynamically registered as a tool. When a task clearly involves implementing a custom function (e.g., "create a function to calculate...", "write code that...", "implement a method for..."), ToolCreationAgent should be the FIRST agent in your plan, not ToolSelectionAgent.
- FinalAgent: Use ONLY for synthesizing the final response using accumulated context (including automatically generated observations). It does NOT use tools directly.

CRITICAL FUNCTION CREATION GUIDANCE:
When the user request explicitly involves writing, creating, or implementing functions, code, or algorithms:
1. Start with ToolCreationAgent to develop the required function
2. Then, use ToolSelectionAgent to execute the function
3. Only use search tools (via ToolSelectionAgent) if ABSOLUTELY necessary for specialized knowledge

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
    <description>Step description (single, atomic action)</description>
    <agent>AgentName</agent>
    <reason>Reason for this step, including potential tool usage (if ToolSelectionAgent), expected inputs (e.g., previous observation), and why this agent is chosen.</reason>
  </step>
  <!-- Repeat <step> for each step in the plan -->
</plan>
"""

TOOL_CREATION_PROMPT = """
You are an expert Python programmer specialized in creating tools for Large Language Models (LLMs).
Your task is to create a Python function based on the user's request.

User Request:
{user_request}

Available Tools:
{tools_list}

Tool Creation History:
{tool_creation_history}

Previous Tool Creation Attempts:
{previous_attempts}

Instructions:
1. Analyze the user request and determine the required functionality.
2. Write a Python function that implements the required logic.
3. The function must:
   - Have a clear name reflecting its purpose (use snake_case).
   - Include a detailed NumPy-style docstring explaining a clear reasoning how it handles the user request, parameters, and return value.
   - Handle potential errors gracefully (e.g., using try-except blocks).
   - Usage of third-party libraries is allowed.
4. Output ONLY the Python function definition, including the docstring. Function definition MUST BE in Markdown-format (```python...```). Do not include any surrounding text, explanations, or example usage.

Example Output Format:

```python
def example_tool(param1: str, param2: int) -> dict:
    \"\"\"Example tool demonstrating the required format.

    This docstring follows the NumPy style guide.
    It should explain a clear reasoning how it handles the user request, parameters, and return value.

    Parameters
    ----------
    param1 : str
        Description of the first parameter.
    param2 : int
        Description of the second parameter.

    Returns
    -------
    dict
        A dictionary containing the result.
    \"\"\"
    try:
        # Tool logic here
        result = {{'input_param1': param1, 'processed_param2': param2 * 2}}
        return result
    except Exception as e:
        return {{'error': str(e)}}
```

Additional Information:
{additional_information}
"""

TOOL_SELECTION_PROMPT = """
You are an expert in selecting the right tool for a task. Based on the step description and context, choose the most appropriate tool from the available options.

Step Description:
{step_description}

Context:
{available_context}

AVAILABLE TOOLS:
{available_tools}

Additional Information:
{additional_information}

CRITICAL CONSTRAINTS:
- You MUST select EXACTLY ONE tool.
- The selected `<tool_name>` MUST EXACTLY match one of the function 'name' fields listed in the `AVAILABLE TOOLS` section above.
- Your response MUST contain only a single `<selection>` block.

Task:
- Select the ONE best tool for the current task step based on the `AVAILABLE TOOLS`.
- Provide the tool name and the parameters to use.

Response Format:
<selection>
  <tool_name>selected_tool_name_from_available_tools</tool_name>
  <parameters>
    <parameter>
      <name>param_name</name>
      <value>param_value</value>
    </parameter>
    <!-- Repeat for each parameter -->
  </parameters>
  <reason>Reason for selecting this SINGLE tool from the `AVAILABLE TOOLS` list</reason>
</selection>
"""

OBSERVATION_PROMPT = """
You are an expert in analyzing tool execution results **in the context of a specific task**. Your goal is to extract and summarize only the information from the tool's output that is directly relevant to achieving the task described. Filter out irrelevant details.

Task Description:
{description}

Tool Execution Result:
{tool_result}

Additional Information:
{additional_information}

Task:
- Analyze the 'Tool Execution Result'.
- Identify the parts of the result that directly address or contribute to fulfilling the 'Task Description'.
- Summarize **only this relevant information**. Ignore details from the tool result that do not pertain to the specific 'Task Description'.

Response Format:
<observation>Summary of information relevant to the Task Description</observation>
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
    replan_attempts: int = 0

    def add_error(self, step_index: int, step_description: str, attempt_number: int, traceback_str: str):
        self.step_errors.append(StepErrorInfo(step_index, step_description, attempt_number, traceback_str))

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

        # --- Post‑processing & automatic fixes ----------------------------------
        # Guarantee every valid plan ends with a FinalAgent step so the user
        # always receives an answer, even if the LLM forgot to add it.
        if steps and steps[-1].agent_name != "FinalAgent":
            steps.append(
                PlanStep(
                    description="Provide the final answer to the user",
                    agent_name="FinalAgent",
                    reason="Every plan must conclude with a synthesis step",
                )
            )

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
        # Enforce that only valid agents are used in plan steps
        allowed = set(AGENT_INFORMATION.keys())
        invalid = [s.agent_name for s in parsed if s.agent_name not in allowed]
        if invalid:
            logger.error(f"PlannerAgent: Invalid agent(s) in plan steps: {invalid}")
            return []
        return parsed


class ToolCreationAgent(BaseAgent):
    def __init__(self):
        super().__init__("ToolCreationAgent", AgentPrompt(TOOL_CREATION_PROMPT))

    def execute(self, context: AgentContext) -> StepResult:
        prompt = self.system_prompt.format(
            user_request=context.user_question,
            tools_list=get_available_tools(BaseAgent.code_executor),
            tool_creation_history="",
            previous_attempts="",
            additional_information=self.get_additional_information(),
        )
        reply = self.llm_call(prompt)
        data = self._extract(reply)
        if not data:
            return StepResult(False, "Tool extraction failed")
        registered = self._register_dynamic_tool(data)
        context.accumulated_results.append({"dynamic_tools_created": registered})
        return StepResult(True, registered)

    def _extract(self, markdown: str) -> Dict[str, Any]:
        """
        Extract Python function code blocks from markdown.
        Only processes markdown-formatted Python code blocks.
        """     
        code_blocks = parse_markdown(markdown)
        python_blocks = [(lang, code) for lang, code in code_blocks if lang.lower() in ('python', 'py')]
        
        if python_blocks:
            return {
                "code_blocks": python_blocks
            }
        
        return {}

    def _register_dynamic_tool(self, data: Dict[str, Any]) -> List[str]:
        """
        Register Python functions extracted from markdown code blocks as dynamic tools.
        """
        registered_tools = []
        
        for _, code_block in data.get("code_blocks", []):
            # Clean the code block: remove leading/trailing whitespace and the language identifier if present
            code_lines = code_block.strip().splitlines()
            if code_lines and code_lines[0].strip().lower() == "python":
                code_to_execute = "\n".join(code_lines[1:]).strip() # Remove 'python' line and strip again
            else:
                code_to_execute = "\n".join(code_lines).strip()

            if not code_to_execute: # Skip if block is empty after cleaning
                continue

            try:
                # Extract the function name using AST to correctly identify it
                import ast
                tree = ast.parse(code_to_execute) # Use cleaned code
                function_name = None
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        function_name = node.name
                        break
                else:
                    logger.warning("Could not identify function name in code block via AST")
                    continue
            
                if function_name is None: # Should be redundant but safe
                     logger.warning("Function name is None after AST walk")
                     continue

                # Execute the code in an isolated namespace
                exec_globals = {} # Start fresh for execution context
                try:
                    exec(code_to_execute, exec_globals, exec_globals) # Use cleaned code
                except Exception as exec_exc:
                     logger.error(f"Exception during exec: {exec_exc}", exc_info=True) # DEBUG
                     continue # Don't proceed if exec failed

                # Add the function and any other definitions from the code block to the main globals dict
                BaseAgent.code_executor.globals_dict.update(exec_globals)
                
                # Check if the expected function was defined in the isolated execution
                check_result = function_name in exec_globals
                if check_result:
                    logger.info("Dynamic tool '%s' and related definitions registered from markdown code block.", function_name)
                    registered_tools.append(function_name)
                else:
                    # This case should ideally not happen if AST parsing succeeded, but log it.
                    logger.warning(f"Function '{function_name}' parsed by AST but not found in exec_globals.")

            except Exception as exc:
                logger.exception("Could not register generated tool from markdown code block: %s", exc)
        
        return registered_tools


class ToolSelectionAgent(BaseAgent):
    def __init__(self):
        super().__init__("ToolSelectionAgent", AgentPrompt(TOOL_SELECTION_PROMPT))

    def execute(self, context: AgentContext) -> StepResult:
        # Allow the LLM a few chances to output a correct, valid selection
        max_attempts = 3
        attempt = 0
        last_error: str = ""

        # Get the current step description
        current_step = context.execution_plan[context.current_step]
        step_description = current_step.description

        while attempt < max_attempts:
            prompt = self.system_prompt.format(
                step_description=step_description,
                available_context=self.get_previous_results(context),
                available_tools=get_available_tools(BaseAgent.code_executor),
                additional_information=self.get_additional_information(),
            )
            reply = self.llm_call(prompt)

            try:
                call = parse_tool_response(reply)
            except ValueError as ve:
                # Parsing failed – retry with the same prompt (LLM may self‑correct)
                last_error = f"Parse error: {ve}"
                attempt += 1
                continue

            # Validate that the selected tool actually exists
            available_json = OwlDefaultFunctions(BaseAgent.code_executor.globals_dict).owl_tools(as_json=True)
            valid_names = {t["function"]["name"] for t in available_json}
            selected = call.get("tool_name")

            if selected not in valid_names:
                last_error = f"Invalid tool selected: '{selected}'. Must be one of {sorted(valid_names)}"
                attempt += 1
                continue

            # Execute the (now validated) tool
            tool_result = execute_tool(BaseAgent.code_executor, call)
            context.accumulated_results.append(tool_result)
            return StepResult(tool_result.success, tool_result)

        # If we exit the loop, all attempts have failed
        return StepResult(False, last_error or "Tool selection failed after multiple attempts")


class ObservationAgent(BaseAgent):
    def __init__(self):
        super().__init__("ObservationAgent", AgentPrompt(OBSERVATION_PROMPT))

    def execute(self, context: AgentContext) -> StepResult:
        # use the most recent tool result
        tool_result = next((r for r in reversed(context.accumulated_results) if isinstance(r, ToolResult)), None)
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
        else:
            context.accumulated_results.append(summary)
            logger.warning("ObservationAgent appended summary instead of replacing. Last result was not ToolResult.")

        return StepResult(True, summary)

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
        self.max_retries_per_step = max_retries_per_step
        self.max_replans = max_replans

        self.agents: Dict[str, BaseAgent] = {
            "PlannerAgent": PlannerAgent(),
            "ToolCreationAgent": ToolCreationAgent(),
            "ToolSelectionAgent": ToolSelectionAgent(),
            "ObservationAgent": ObservationAgent(),
            "FinalAgent": FinalAgent(),
        }

        BaseAgent.manager = manager
        BaseAgent.code_executor = code_executor

    # --------------------------------------------------------------------- #
    # public API                                                            #
    # --------------------------------------------------------------------- #
    def process_user_question(self, question: str) -> str:
        """
        Main entry point to process a user question through the agentic framework.
        Handles initial planning, execution with retries/replanning, and final response generation.
        """
        context = AgentContext(user_question=question)
        replan_count = 0  # Initial replan count for the overall process

        while replan_count <= self.max_replans:
            try:
                # Initial Plan (or replan if previous attempt failed before execution loop)
                if not context.execution_plan or context.error_context.replan_attempts > replan_count:
                    logger.info(f"Planning attempt {replan_count + 1}/{self.max_replans + 1}...")
                    if not self._plan(context):
                        logger.error("Initial planning failed. Cannot proceed.")
                        # Provide context for failure if possible
                        last_error = (
                            context.error_context.step_errors[-1] if context.error_context.step_errors else None
                        )
                        error_info = f": {last_error.traceback_str}" if last_error else ""
                        return f"I'm sorry, I couldn't create a plan to address your request{error_info}"
                    context.error_context.replan_attempts = replan_count  # Sync replan attempts

                # Execute the current plan
                if self._execute(context):
                    # Execution successful, check for final response
                    logger.info("Orchestration completed successfully.")
                    return context.final_response or "Processing completed, but no final response was generated."
                else:
                    # Execution failed, and replanning is handled within _execute
                    # If _execute returns False, it means it halted after max retries/replans
                    logger.error("Execution halted after exhausting retries or replans.")
                    last_error = context.error_context.step_errors[-1] if context.error_context.step_errors else None
                    error_info = (
                        f" Last error at step {last_error.step_index + 1} ('{last_error.step_description}'): {last_error.traceback_str}"
                        if last_error
                        else ""
                    )
                    return f"I'm sorry, I couldn't complete the task due to errors{error_info}. Please try modifying your request."

            except Exception as e:
                # Catch unexpected errors in the process_user_question loop itself
                logger.critical(f"Critical unexpected error during orchestration: {e}", exc_info=True)
                context.error_context.add_error(
                    step_index=context.current_step,
                    step_description="Overall orchestration loop",
                    attempt_number=replan_count + 1,
                    traceback_str=traceback.format_exc(),
                )
                replan_count += 1  # Increment replan count for the outer loop
                if replan_count > self.max_replans:
                    logger.critical("Max replan attempts reached due to critical error. Aborting.")
                    return f"I encountered a critical internal error and couldn't recover after {self.max_replans} attempts. Please try again later."
                else:
                    logger.warning(f"Attempting replan {replan_count}/{self.max_replans} due to critical error.")
                    context.execution_plan = None  # Force replanning
                    continue  # Go back to the start of the while loop to replan

        # Should ideally not be reached if logic is correct, but as a fallback
        return "I was unable to complete your request after multiple attempts."

    def _plan(self, context: AgentContext) -> bool:
        """
        Invokes the PlannerAgent to create or update the execution plan in the context.
        Returns True if planning was successful, False otherwise.
        """
        logger.info("Invoking PlannerAgent...")
        planner = self.agents.get("PlannerAgent")
        if not planner:
            logger.critical("PlannerAgent not found in orchestrator configuration!")
            context.error_context.add_error(-1, "Planning", 1, "PlannerAgent not found.")
            return False
        try:
            plan_result = planner.execute(context)
            if plan_result.success and context.execution_plan and context.execution_plan.steps:
                logger.info(f"Planning successful. Execution plan created with {len(context.execution_plan)} steps.")
                return True
            else:
                logger.error(f"PlannerAgent failed to produce a valid plan. Result: {plan_result.execution_result}")
                context.error_context.add_error(
                    step_index=-1,  # Indicate planning phase error
                    step_description="Planning",
                    attempt_number=1,
                    traceback_str=f"Planner failed: {plan_result.execution_result}",
                )
                return False
        except Exception as e:
            logger.exception("Exception during planning phase.")
            context.error_context.add_error(
                step_index=-1, step_description="Planning", attempt_number=1, traceback_str=traceback.format_exc()
            )
            return False

    def _execute(self, context: AgentContext) -> bool:
        """
        Executes the plan step-by-step with retry and replan logic.
        Returns True if execution completes successfully, False otherwise.
        """
        if not context.execution_plan:
            logger.error("Execution attempt failed: No execution plan exists.")
            return False  # Cannot execute without a plan

        replan_count = context.error_context.replan_attempts  # Get current replan count
        current_plan_steps = context.execution_plan.steps
        step_index = 0

        while step_index < len(current_plan_steps):
            step = current_plan_steps[step_index]
            context.current_step = step_index  # Ensure context reflects current step index
            retries = 0

            # Retry loop for a single step
            while retries < self.max_retries_per_step:
                attempt_number = retries + 1
                logger.info(
                    f"Executing step {step_index + 1}/{len(current_plan_steps)} (Attempt {attempt_number}/{self.max_retries_per_step}): {step.description} | Agent: {step.agent_name}"
                )
                try:
                    agent = self.agents.get(step.agent_name)
                    if not agent:
                        raise ValueError(f"Configuration Error: Agent '{step.agent_name}' not found in orchestrator.")

                    result = agent.execute(context)
                    step.result = result  # Store result on the step itself

                    if result.success:
                        logger.info(f"Step {step_index + 1} successful.")
                        # Special handling: Auto-run ObservationAgent after successful ToolSelectionAgent
                        if step.agent_name == "ToolSelectionAgent":
                            logger.info("Attempting to run ObservationAgent after successful ToolSelection.")
                            try:
                                observer = self.agents.get("ObservationAgent")
                                if observer:
                                    logger.debug("Found ObservationAgent. Executing...")
                                    obs_result = observer.execute(context)
                                    if obs_result.success:
                                        logger.info("ObservationAgent executed successfully.")
                                    else:
                                        # Log failure but likely continue execution
                                        logger.warning(
                                            f"ObservationAgent reported failure: {obs_result.execution_result}"
                                        )
                                else:
                                    # Changed from WARNING to ERROR as this shouldn't happen if initialized correctly
                                    logger.error(
                                        "ObservationAgent not found in orchestrator agents list. Cannot auto-observe tool result."
                                    )
                            except Exception as obs_exc:
                                # Catch errors specifically from the ObservationAgent execution
                                logger.error(
                                    f"Error during automatic ObservationAgent execution: {obs_exc}", exc_info=True
                                )
                                # Decide if this error should halt execution or just be logged. Logging for now.

                        break  # Break retry loop on success

                    else:
                        # Step reported failure explicitly
                        raise RuntimeError(f"Agent '{step.agent_name}' reported failure: {result.execution_result}")

                except Exception as exc:
                    retries += 1
                    error_type = type(exc).__name__
                    error_message = str(exc)
                    traceback_str = traceback.format_exc()
                    logger.error(
                        f"Error in step {step_index + 1} ('{step.description}') Agent '{step.agent_name}' on attempt {attempt_number}: [{error_type}] {error_message}",
                        exc_info=False,  # Avoid duplicate logging if traceback included below
                    )
                    logger.debug(
                        f"Full traceback for step {step_index + 1} error:\n{traceback_str}"
                    )  # Log full traceback at debug level

                    context.error_context.add_error(
                        step_index=step_index,
                        step_description=step.description,
                        attempt_number=attempt_number,
                        traceback_str=f"[{error_type}] {error_message}\n{traceback_str}",  # Include type and message
                    )

                    # --- Intelligent Error Handling Logic (Basic Example) ---
                    # More sophisticated logic could go here based on error types
                    is_recoverable_by_retry = True  # Default assumption
                    is_planning_error = False  # Example flag

                    if (
                        isinstance(exc, (json.JSONDecodeError, ET.ParseError, ValueError))
                        and agent.name == "ToolSelectionAgent"
                    ):
                        # Parsing errors in tool selection might benefit from retry if LLM is flaky
                        logger.warning(
                            f"Parsing error encountered in ToolSelectionAgent, retrying ({retries}/{self.max_retries_per_step})..."
                        )
                    elif isinstance(exc, KeyError) and agent.name == "ToolSelectionAgent":
                        # Tool not found error - likely a planning issue or tool creation failure
                        logger.error("Tool specified by ToolSelectionAgent not found. This might require replanning.")
                        is_recoverable_by_retry = False
                        is_planning_error = True
                    elif (
                        isinstance(exc, RuntimeError)
                        and "Invalid tool selected" in str(exc)
                        and agent.name == "ToolSelectionAgent"
                    ):
                        # The ToolSelectionAgent chose something that isn't in the available tools list.
                        # This is a planning issue – no point retrying the same prompt over and over.
                        logger.error("Invalid tool chosen by ToolSelectionAgent – triggering immediate replanning.")
                        is_recoverable_by_retry = False
                        is_planning_error = True
                    # Add more specific error checks here (e.g., temporary network errors, API errors)

                    if retries >= self.max_retries_per_step or not is_recoverable_by_retry:
                        logger.error(f"Step {step_index + 1} failed permanently after {retries} attempts.")
                        # Decide whether to replan or halt
                        if replan_count < self.max_replans and (
                            is_planning_error or True
                        ):  # Replan on most permanent errors for now
                            replan_count += 1
                            context.error_context.replan_attempts = replan_count  # Update context
                            logger.warning(
                                f"Maximum retries reached for step {step_index + 1}. Triggering replan attempt {replan_count}/{self.max_replans}."
                            )
                            # Pass error context to planner implicitly via AgentContext
                            if self._plan(context):
                                logger.info("Replanning successful. Restarting execution with the new plan.")
                                # Reset execution state for the new plan
                                current_plan_steps = context.execution_plan.steps  # Get potentially new steps
                                step_index = 0  # Restart execution from the first step of the new plan
                                continue  # Continue the outer while loop to start the new plan

                            else:
                                logger.error("Replanning failed. Halting execution.")
                                return False  # Replanning itself failed
                        else:
                            # Max replans reached or error deemed unrecoverable by replanning
                            logger.error(
                                f"Cannot recover from error in step {step_index + 1}. Max replans ({self.max_replans}) reached or error is fatal. Halting execution."
                            )
                            return False  # Halt execution

                    # else: continue retry loop (implicitly done by loop structure)

            # Check if step execution was successful after the retry loop
            if step.result is None or not step.result.success:
                # This case should now be handled by the exception block leading to replan/halt
                logger.critical(
                    f"Execution flow error: Reached end of step {step_index + 1} processing without success, replan, or halt."
                )
                return False  # Should not happen if logic above is correct

            # Move to the next step if successful
            step_index += 1

        # If the loop completes without returning False, all steps succeeded
        logger.info("Execution plan completed successfully.")
        return True
