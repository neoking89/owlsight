import json
import os
import traceback
import xml.etree.ElementTree as ET
import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from owlsight.agentic.constants import AGENT_INFORMATION, EXCLUDED_AGENTS
from owlsight.agentic.exceptions import GuardrailViolationError
from owlsight.agentic.guardrails import (
    GuardrailManager,
    ToolExecutionFollowsToolCreationGuardrail,
)
from owlsight.agentic.helper_functions import (
    execute_tool,
    get_available_tools,
    parse_tool_response,
    parse_json_markdown,
    create_temp_config_filename,
)
from owlsight.agentic.models import AgentContext, AgentPrompt, ExecutionPlan, PlanStep, StepResult
from owlsight.app.default_functions import OwlDefaultFunctions
from owlsight.processors.text_generation_manager import TextGenerationManager
from owlsight.utils.code_execution import CodeExecutor
from owlsight.configurations.config_manager import ConfigManager
from owlsight.utils.helper_functions import parse_markdown
from owlsight.utils.constants import get_default_config_on_startup_path
from owlsight.utils.logger import logger


class BaseAgent(ABC):
    """ "Base class for all agents in the agentic framework."""

    manager: ClassVar[Optional[TextGenerationManager]] = None
    code_executor: ClassVar[Optional[CodeExecutor]] = None

    # Class variables for configuration management
    temp_config_filename: ClassVar[Optional[str]] = None
    config_per_agent: ClassVar[Optional[Dict[str, str]]] = None



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

    def execute(self, context: AgentContext) -> StepResult:
        """
        Execute the agent's task, ensuring pre-execution steps are run.
        """
        self.pre_execute(context)
        return self._execute_impl(context)

    @abstractmethod
    def _execute_impl(self, context: AgentContext) -> StepResult:
        """
        Core implementation of the agent's task. Subclasses must override this.
        """
        ...

    def pre_execute(self, context: AgentContext) -> None:
        """
        Perform any pre-execution tasks.
        """
        # Load agent-specific config if its name is in AGENT_INFORMATION
        if self.name in AGENT_INFORMATION:
            self.load_config_agent()

    def get_additional_information(self) -> str:
        """Retrieves additional information from the config manager as a string."""
        config_manager: Optional[ConfigManager] = getattr(self.manager, "config_manager", None)
        if config_manager is None:
            logger.warning("ConfigManager not found on manager when getting additional information.")
            return ""  # Return empty string
        return config_manager.get("agentic.additional_information", "").strip()

    def set_additional_information(self, info_to_add: str) -> None:
        """Appends the given string to the additional information string in the config manager.

        Retrieves the current information string, appends the new string (with a newline),
        and stores the updated string.
        """
        # Check if info_to_add is actually a non-empty string
        if not isinstance(info_to_add, str) or not info_to_add:
            logger.warning(
                f"set_additional_information called with invalid input (must be non-empty string): {type(info_to_add)}. Cannot add information."
            )
            return None

        config_manager: Optional[ConfigManager] = getattr(self.manager, "config_manager", None)
        if config_manager is None:
            logger.warning("ConfigManager not found on manager when setting additional information. Cannot save.")
            return None

        current_info_str = self.get_additional_information()
        new_info_str = info_to_add.strip()

        # Check if this exact info already exists
        if new_info_str in current_info_str:
            logger.debug(f"Duplicate additional information detected, skipping: {new_info_str[:100]}...")
            return None

        # Append the new info string to the current string
        if current_info_str:
            updated_info_str = f"{current_info_str}\n{new_info_str}"
        else:
            updated_info_str = new_info_str

        # Save the updated string
        config_manager.set("agentic.additional_information", updated_info_str)
        logger.debug(f"Appended to agentic.additional_information. New content snippet: {new_info_str[:100]}...")

    def load_config_agent(self) -> None:
        """
        Load the specific owlsight configuration for the child-agent.
        This way, the child-agent can have its own owlsight configuration/model.
        """
        config_per_agent = self._get_config_per_agent()
        config_per_agent = BaseAgent._set_classvar_config_per_agent(config_per_agent)
        agent_config_path = config_per_agent.get(self.name, "")
        last_config_is_same = agent_config_path == self.manager._last_loaded_config
        if self._agent_config_path_exists() and not last_config_is_same:
            # if another config is used for the first time, remember the first config so that we can sync agents per config even when different configs are used
            logger.debug(f"Agent '{self.name}' found in config_per_agent. Attempting to load its config.")
            model_succesfully_loaded = self.manager.load_config(agent_config_path)
            if not model_succesfully_loaded:
                logger.warning(f"Failed to load config {agent_config_path} for agent '{self.name}'.")
        else:
            if last_config_is_same:
                logger.debug(f"Configuration file for agent '{self.name}' is the same as the last loaded config. Not loading new config.")
            else:
                logger.debug(f"Configuration file for agent '{self.name}' does not exist: {agent_config_path}")

    @classmethod
    def reset_config_per_agent_classvars(cls) -> None:
        """
        Reset the config-related class variables.
        """
        if cls.temp_config_filename and Path(cls.temp_config_filename).exists() and cls.temp_config_filename != get_default_config_on_startup_path():
            os.remove(cls.temp_config_filename)
        cls.config_per_agent: Optional[Dict[str, str]] = None
        cls.temp_config_filename: Optional[str] = None

    @classmethod
    def _set_classvar_config_per_agent(cls, config_per_agent: Dict[str, str]) -> Dict[str, str]:
        """
        Initialize class variable config_per_agent with temporary config filenames.

        Parameters
        ----------
        config_per_agent: dict[str, str]
            Existing dict with agent names as keys and config file paths as values.
        """
        default_config_on_startup = get_default_config_on_startup_path(return_cache_path=False)
        if cls.temp_config_filename is None:
            # first we check if a default config is present, meaning we already loaded a model through an existing config
            if default_config_on_startup:
                cls.temp_config_filename = default_config_on_startup
            else:
                cls.temp_config_filename = create_temp_config_filename()
            logger.debug(f"Created temporary config filename for keeping state of 'agentic.config_per_agent': {cls.temp_config_filename}")
        
        if cls.config_per_agent is None:
            for agent_name in AGENT_INFORMATION.keys():
                if not config_per_agent.get(agent_name, None):
                    config_per_agent[agent_name] = cls.temp_config_filename
            
            cls.config_per_agent = config_per_agent
            # no need to save config if it was the one loaded on startup
            if not default_config_on_startup:
                cls.manager.save_config(cls.temp_config_filename)

        return cls.config_per_agent


    def _get_config_per_agent(self) -> Dict[str, str]:
        config_manager: ConfigManager = getattr(self.manager, "config_manager", None)
        if config_manager is None:
            logger.warning("ConfigManager not found on manager when getting config per agent. Cannot retrieve.")
            return {}
        config_per_agent = config_manager.get("agentic.config_per_agent", {})
        return config_per_agent

    def _agent_config_path_exists(self) -> bool:
        path_str = self._get_config_per_agent().get(self.name)
        if not path_str:
            return False
        return Path(path_str).exists()

    @staticmethod
    def _form_description(step: PlanStep) -> str:
        return f"""
{step.description}
Reason: {step.reason}
        """.strip()


class PlanAgent(BaseAgent):
    """
    Role: Analyzes the user's high-level request and breaks it down into a detailed, step-by-step execution plan.
    Logic: Generates a sequence of tasks (PlanSteps), assigning an appropriate agent (like ToolSelectionAgent or ToolCreationAgent) to each step.
           Aims to create a complete and logical workflow to achieve the user's objective.
    Inputs: Primarily uses the user's initial request and the overall context provided in AgentContext.
    Outputs: An ExecutionPlan object containing a list of PlanSteps.
    Limitations: Does not execute any tasks itself; strictly focuses on planning. Cannot access external tools or filesystem directly.
                 Relies on PlanValidationAgent for refinement and correctness checks.
    Interaction: Typically the first agent activated by the Orchestrator. Passes the generated plan to the PlanValidationAgent.
    """

    def __init__(self):
        super().__init__("PlanAgent", AgentPrompt(AGENT_INFORMATION["PlanAgent"]))

    def _execute_impl(self, context: AgentContext) -> StepResult:
        prompt = self.system_prompt.format(
            user_request=context.user_request,
            available_tools=get_available_tools(BaseAgent.code_executor.globals_dict),
            additional_information=self.get_additional_information(),
        )
        reply = self.llm_call(prompt)
        steps: List[PlanStep] = self._extract(reply)

        self._ensure_final_step(steps)

        if not steps:
            return StepResult(False, "No plansteps where found. Planning failed.")

        # Create the execution plan
        execution_plan = ExecutionPlan(steps)

        # Store the generated plan in the context
        context.execution_plan = execution_plan

        # Planner's job is done; validation happens implicitly in the orchestrator's _plan method
        return StepResult(success=True, execution_result="Plan generated successfully.")

    def _extract(self, plan_json: str) -> List[PlanStep]:
        data = parse_json_markdown(plan_json)
        plan_list = data.get("plan", [])
        if not isinstance(plan_list, list):
            return []
        parsed: List[PlanStep] = []
        for item in plan_list:
            desc = item.get("description", "")
            ag = item.get("agent", "")
            reason = item.get("reason", "")
            if desc and ag:
                parsed.append(PlanStep(desc, ag, reason or ""))
        allowed = set(agent_name for agent_name in AGENT_INFORMATION.keys() if agent_name not in EXCLUDED_AGENTS)
        invalid = [s.agent_name for s in parsed if s.agent_name not in allowed]
        if invalid:
            logger.error(f"PlanAgent: Invalid agent(s) in plan steps: {invalid}")
            return []
        return parsed

    def _ensure_final_step(self, steps: List[PlanStep]) -> None:
        """Ensure the plan concludes with a FinalAgent step."""
        if not steps or steps[-1].agent_name != "FinalAgent":
            steps.append(
                PlanStep(
                    description="Provide the final answer to the user",
                    agent_name="FinalAgent",
                    reason="Every plan must conclude with a synthesis step",
                )
            )


class PlanValidationAgent(BaseAgent):
    """
    Role: Validates and refines the execution plan generated by PlanAgent.
    Logic: Checks the plan for logical consistency, feasibility, adherence to guardrails (e.g., preventing disallowed tool sequences), and completeness.
           May revise the plan by adding, removing, or modifying steps to improve its quality or safety before execution begins.
           It ensures the plan includes a FinalAgent step at the end.
    Inputs: Receives the ExecutionPlan from PlanAgent and accesses AgentContext.
    Outputs: A validated (potentially modified) ExecutionPlan or a failure result if the plan is fundamentally flawed.
    Limitations: Focuses only on plan structure and validation rules. Does not execute steps or interact with external tools.
    Interaction: Follows the PlanAgent. Passes the validated plan back to the Orchestrator for execution.
    """

    def __init__(self):
        super().__init__("PlanValidationAgent", AgentPrompt(AGENT_INFORMATION["PlanValidationAgent"]))

    def validate_plan_by_guardrails(self, plan: ExecutionPlan) -> StepResult:
        """
        Validates the execution plan against all registered guardrails.

        Parameters
        ----------
        plan : ExecutionPlan
            The execution plan to validate.

        Returns
        -------
        StepResult
            Result of the validation: success/failure and execution_result message.
        """
        # Default instance for the application
        guardrail_manager = GuardrailManager()
        for rails in [ToolExecutionFollowsToolCreationGuardrail()]:
            guardrail_manager.register_guardrail(rails)

        try:
            guardrail_manager.validate_plan(plan)
            return StepResult(True, plan.steps)
        except GuardrailViolationError as e:
            logger.error(f"Plan validation failed due to guardrail violation: {e}")
            return StepResult(False, f"Plan validation failed: {e}")

    def _validate_llm_response(self, result_json: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate the structure and required fields of the LLM response JSON."""
        required_fields = ["validation_result", "validation_notes"]

        # The "plan" field is only required if the plan is revised
        if result_json.get("validation_result") == "revised":
            required_fields.append("plan")

        missing_fields = [field for field in required_fields if field not in result_json]
        if missing_fields:
            err_msg = f"Missing required fields in LLM response: {', '.join(missing_fields)}"
            logger.error(err_msg)
            return False, f"Failed to validate plan: {err_msg}"

        # Additional validation for 'plan' field if it exists
        if "plan" in result_json:
            if not isinstance(result_json["plan"], list):
                err_msg = "Invalid 'plan' field: expected a list."
                logger.error(err_msg)
                return False, f"Failed to validate plan: {err_msg}"
            for step in result_json["plan"]:  # Accessing 'plan' is now safe
                if not all(k in step for k in ("description", "agent")):
                    err_msg = "Invalid step in 'plan': missing 'description' or 'agent'."
                    logger.error(err_msg)
                    return False, f"Failed to validate plan: {err_msg}"

        return True, None

    def _execute_impl(self, context: AgentContext) -> StepResult:
        # First validate the plan against guardrails
        guardrail_validation = self.validate_plan_by_guardrails(context.execution_plan)
        guardrail_error = None
        if not guardrail_validation.success:
            # Instead of returning early, capture the error message to send to the LLM
            guardrail_error = guardrail_validation.execution_result
            logger.info(
                f"Plan validation detected guardrail violation: {guardrail_error}. Requesting LLM to revise the plan."
            )
            context.planner_feedback_from_guardrails = guardrail_error

        # Format the plan for inclusion in the prompt
        plan_json = json.dumps(
            {
                "plan": [
                    {"description": step.description, "agent": step.agent_name, "reason": step.reason}
                    for step in context.execution_plan.steps
                ]
            },
            indent=2,
        )

        # Format the prompt with context information
        prompt_params = {
            "user_request": context.user_request,
            "generated_plan": plan_json,
            "available_tools": get_available_tools(BaseAgent.code_executor.globals_dict),
            "guardrails": guardrail_error if guardrail_error else "",
            "additional_information": self.get_additional_information(),
        }

        logger.debug(f"Plan validation input:\n{json.dumps(prompt_params, indent=2)}")

        # Generate response from the LLM
        formatted_prompt = self.system_prompt.format(**prompt_params)
        response = self.llm_call(formatted_prompt)
        logger.debug(f"Plan validation raw output:\n{response}")

        result_json = parse_json_markdown(response)

        # Explicitly check if parsing failed (indicated by empty dict from helper)
        # Also check original response wasn't just empty/whitespace to avoid false positives
        if not result_json and response and response.strip():
            err_msg = "Failed to parse PlanValidationAgent response as JSON."
            logger.error(err_msg + f" Raw response: {response}")
            # Return the specific error message the test expects
            return StepResult(
                success=False,
                execution_result=err_msg,  # Match test assertion
            )

        # Validate the structure and required fields of the parsed JSON
        is_valid, error_message = self._validate_llm_response(result_json)
        if not is_valid:
            return StepResult(success=False, execution_result=error_message)

        # Process the validated response
        try:
            if result_json["validation_result"] == "revised":
                # Update the plan with the revised version from the validated JSON
                new_steps = []
                for step_data in result_json["plan"]:  # Accessing 'plan' is now safe
                    new_step = PlanStep(
                        description=step_data["description"],
                        agent_name=step_data["agent"],
                        reason=step_data.get("reason", ""),
                    )
                    new_steps.append(new_step)

                # Replace the existing plan with the revised one
                context.execution_plan.steps = new_steps

                return StepResult(
                    success=True,
                    execution_result=f"Plan revised: {result_json['validation_notes']}",  # Accessing 'validation_notes' is safe
                )
            else:  # validation_result is 'valid' or potentially something else handled as valid
                # Plan was validated without changes
                return StepResult(
                    success=True,
                    execution_result=f"Plan validated: {result_json['validation_notes']}",  # Accessing 'validation_notes' is safe
                )
        except KeyError as e:
            # This catch block is now less likely due to _validate_llm_response,
            # but kept as a safeguard against unexpected issues.
            err_msg = f"Unexpected missing key after validation: {e}"
            logger.error(err_msg)
            return StepResult(
                success=False,
                execution_result=f"Failed to process validated plan: {err_msg}",
            )


class ToolCreationAgent(BaseAgent):
    """
    Role: Dynamically creates new Python functions (tools) based on the requirements of a plan step.
    Logic: Analyzes the step description and context to understand the needed functionality.
           Writes Python code for a function that fulfills the requirement.
           Registers the newly created function so it becomes available for immediate use by other agents (typically ToolSelectionAgent).
    Inputs: Receives the current PlanStep description and AgentContext (including previous results which might inform the tool's logic).
    Outputs: Python code defining the new tool. Registers the tool within the Python environment.
    Limitations: Creates tools but does not execute them. Relies on the LLM's coding capabilities. Cannot guarantee the created tool is correct or bug-free without execution and observation.
    Interaction: Invoked when a plan step requires functionality not available in existing tools. Its successful execution makes a new tool available, often intended to be used in the very next step by ToolSelectionAgent.
    """

    def __init__(self):
        super().__init__("ToolCreationAgent", AgentPrompt(AGENT_INFORMATION["ToolCreationAgent"]))

    def _execute_impl(self, context: AgentContext) -> StepResult:
        step = context.get_current_step()
        prompt = self.system_prompt.format(
            step_description=BaseAgent._form_description(step),
            available_context=context.get_previous_results(),
            available_tools=get_available_tools(BaseAgent.code_executor.globals_dict),
            additional_information=self.get_additional_information(),
        )
        reply = self.llm_call(prompt)
        data = self._extract(reply)
        if not data:
            err_msg = "Tool extraction failed. No valid markdown Python code block found in LLM response."
            return StepResult(False, err_msg)
        registered = self._register_dynamic_tool(data)
        context.accumulated_results.append({"dynamic_tools_created": registered})
        return StepResult(True, registered)

    def _extract(self, markdown: str) -> Dict[str, Any]:
        """
        Extract Python function code blocks from markdown.
        Only processes markdown-formatted Python code blocks.
        """
        code_blocks = parse_markdown(markdown)
        python_blocks = [(lang, code) for lang, code in code_blocks if lang.lower() in ("python", "py")]

        if python_blocks:
            return {"code_blocks": python_blocks}

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
                code_to_execute = "\n".join(code_lines[1:]).strip()  # Remove 'python' line and strip again
            else:
                code_to_execute = "\n".join(code_lines).strip()

            if not code_to_execute:  # Skip if block is empty after cleaning
                continue

            try:
                # Extract the function name using AST to correctly identify it
                import ast

                tree = ast.parse(code_to_execute)  # Use cleaned code
                function_name = None
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        function_name = node.name
                        break
                else:
                    logger.warning("Could not identify function name in code block via AST")
                    continue

                if function_name is None:  # Should be redundant but safe
                    logger.warning("Function name is None after AST walk")
                    continue

                # Execute the code using the centralized code executor. Executed code will be available in the globals_dict (which is a singleton).
                try:
                    BaseAgent.code_executor.execute_python_code(code_to_execute)
                except Exception as exec_exc:
                    logger.error(f"Exception during dynamic tool code execution: {exec_exc}", exc_info=True)
                    continue
                check_result = function_name in BaseAgent.code_executor.globals_dict
                if check_result:
                    logger.info(
                        "Dynamic tool '%s' and related definitions registered from markdown code block.", function_name
                    )
                    registered_tools.append(function_name)
                else:
                    # This case should ideally not happen if AST parsing succeeded, but log it.
                    logger.warning(f"Function '{function_name}' parsed by AST but not found in exec_globals.")

            except Exception as exc:
                logger.exception("Could not register generated tool from markdown code block: %s", exc)

        return registered_tools


class ToolSelectionAgent(BaseAgent):
    """
    Role: Selects and executes an *existing* tool appropriate for the current plan step.
    Logic: Analyzes the step description and available tools (including dynamically created ones).
           Determines the best tool to use and constructs the necessary parameters based on the context.
           Invokes the chosen tool via the CodeExecutor and captures its output.
    Inputs: The current PlanStep description, AgentContext (especially previous results for parameter values), and the list of available tools.
    Outputs: The result returned by the executed tool.
    Limitations: Can only select and execute tools that are already defined or have been dynamically created and registered. Cannot create new tools itself.
    Interaction: Executes a specific task within the plan using a defined capability. Often followed by ObservationAgent to summarize the tool's output.
    """

    def __init__(self):
        super().__init__("ToolSelectionAgent", AgentPrompt(AGENT_INFORMATION["ToolSelectionAgent"]))

    def _execute_impl(self, context: AgentContext) -> StepResult:
        # Allow the LLM several chances to self‑correct invalid outputs.
        max_attempts = 4  # increased by one for improved resiliency
        attempt = 0
        error_feedback: str = ""  # passed back to the LLM to aid self‑correction

        step = context.get_current_step()
        while attempt < max_attempts:
            prompt = self.system_prompt.format(
                step_description=BaseAgent._form_description(step),
                available_context=context.get_previous_results(),
                available_tools=get_available_tools(BaseAgent.code_executor.globals_dict),
                additional_information=self.get_additional_information(),
            )
            if error_feedback:
                # Append explicit guidance so the model can fix its previous mistake.
                prompt += (
                    "\nPREVIOUS_ERROR:\n"
                    + error_feedback
                    + "\nPlease fix the issue and output ONLY a valid 'selection' JSON object."
                )
            reply = self.llm_call(prompt)

            try:
                call = parse_tool_response(reply)
            except ValueError as ve:
                # Parsing failed – store feedback and retry.
                error_feedback = f"Parse error: {ve}"
                attempt += 1
                continue

            # Validate that the selected tool actually exists
            available_json = OwlDefaultFunctions(BaseAgent.code_executor.globals_dict).owl_tools(as_json=True)
            valid_names = {t["function"]["name"] for t in available_json}
            selected = call.get("tool_name")

            if selected not in valid_names:
                error_feedback = f"Invalid tool selected: '{selected}'. Must be one of {sorted(valid_names)}"
                attempt += 1
                continue

            # Execute the (now validated) tool
            tool_result = execute_tool(BaseAgent.code_executor.globals_dict, call)
            context.accumulated_results.append(tool_result)
            return StepResult(tool_result.success, tool_result)

        # If we exit the loop, all attempts have failed
        return StepResult(False, error_feedback or "Tool selection failed after multiple attempts")


class ObservationAgent(BaseAgent):
    """
    Role: Summarizes and synthesizes the output of a previous step (typically a tool execution) into a concise and relevant observation.
    Logic: Processes the raw output (e.g., tool result, data fetched) from the preceding step.
           Extracts the key information relevant to the overall goal and the subsequent plan steps.
           Formats this information into a clear, condensed natural language summary.
    Inputs: The result (often voluminous or structured data) from the previous PlanStep in AgentContext.
    Outputs: A concise textual summary added to the AgentContext.
    Limitations: Does not perform actions or make decisions beyond summarizing. Its effectiveness depends on the LLM's ability to distill information accurately.
    Interaction: Typically follows steps that produce complex or lengthy outputs (like ToolSelectionAgent) to compress information before the next agent acts, preventing context overload.
    """

    def __init__(self):
        super().__init__("ObservationAgent", AgentPrompt(AGENT_INFORMATION["ObservationAgent"]))

    def _execute_impl(self, context: AgentContext) -> StepResult:
        most_recent_result = next((r for r in reversed(context.accumulated_results)), None)
        if most_recent_result is None:
            return StepResult(False, "No result to observe.")

        step_description = context.get_current_step().description
        prompt = self.system_prompt.format(
            description=step_description,
            information=most_recent_result,
        )

        summary_json = self.llm_call(prompt)
        try:
            summary_dict = parse_json_markdown(summary_json)
            summary = summary_dict.get("observation", "")
            if summary and isinstance(summary, str):
                summary = summary.strip()

            sources = summary_dict.get("sources", [])
            if sources and isinstance(sources, list):
                sources = [s.strip() for s in sources]
                summary = {"observation": summary, "sources": sources}
        except json.JSONDecodeError as e:
            logger.error(f"Observation JSON parsing failed: {e}")
            return StepResult(False, f"Failed to parse observation JSON: {e}")

        # replace last result with its summary to keep all data concise
        context.accumulated_results[-1] = summary

        return StepResult(True, summary)


class FinalAgent(BaseAgent):
    """
    Role: Generates the final, user-facing response based on the completed execution plan and all accumulated results.
    Logic: Reviews the initial user request and the entire history of execution steps, tool outputs, and observations stored in the AgentContext.
           Synthesizes this information into a coherent and comprehensive final answer that directly addresses the user's original objective.
           Formats the answer appropriately (e.g., text, JSON).
    Inputs: The complete AgentContext, including the original request, the plan, all step results, and observations.
    Outputs: The final response string intended for the user.
    Limitations: Does not perform any further actions or tool calls. Solely focused on summarizing the process outcome.
    Interaction: Always the last agent executed in a successful plan. Its output is the final result of the Orchestrator's process_user_request method.
    """

    def __init__(self):
        super().__init__("FinalAgent", AgentPrompt(AGENT_INFORMATION["FinalAgent"]))

    def _execute_impl(self, context: AgentContext) -> StepResult:
        prompt = self.system_prompt.format(
            user_request=context.user_request,
            previous_results=context.get_previous_results(),
        )
        reply = self.llm_call(prompt)
        
        # Parse the JSON response
        try:
            parsed_data = parse_json_markdown(reply)
            if self._json_is_valid(parsed_data):
                content = parsed_data["answer"]["content"]
                content_format = parsed_data["answer"]["format"]
                
                if content_format != "text":
                    formatted_reply = f"```{content_format}\n{content}\n```"
                else:
                    formatted_reply = content
                
                context.final_response = formatted_reply
                return StepResult(True, formatted_reply)
            else:
                logger.warning("Failed to parse FinalAgent response as JSON or missing required fields")
                context.final_response = reply
                return StepResult(True, reply)
        except Exception as e:
            logger.warning(f"Error parsing FinalAgent response: {str(e)}")
            context.final_response = reply
            return StepResult(True, reply)

    def _json_is_valid(self, parsed_data: Dict[str, Any]) -> bool:
        return parsed_data and "answer" in parsed_data and "format" in parsed_data["answer"] and "content" in parsed_data["answer"]


class AgentOrchestrator:
    """
    The AgentOrchestrator is responsible for managing the execution of agents in a step-by-step manner.
    It handles planning, execution, and replanning if necessary.

    Parameters
    ----------
    max_retries_per_step : int
        Maximum number of retries per PlanStep
    max_replans : int
        Maximum number of total replans
    max_validation_retries : int
        Maximum number of validation attempts per planning phase when guardrail violations occur
    """

    def __init__(
        self,
        code_executor: CodeExecutor,
        manager: TextGenerationManager,
        max_retries_per_step: int = 3,
        max_replans: int = 2,
        max_validation_retries: int = 2,
    ):
        self.max_validation_retries = max_validation_retries
        self.code_executor = code_executor
        self.manager = manager
        self.max_retries_per_step = max_retries_per_step
        self.max_replans = max_replans

        self.agents: Dict[str, BaseAgent] = {
            "PlanAgent": PlanAgent(),
            "PlanValidationAgent": PlanValidationAgent(),
            "ToolCreationAgent": ToolCreationAgent(),
            "ToolSelectionAgent": ToolSelectionAgent(),
            "ObservationAgent": ObservationAgent(),
            "FinalAgent": FinalAgent(),
        }

        BaseAgent.manager = manager
        BaseAgent.code_executor = code_executor

    def process_user_request(self, request: str) -> str:
        """
        Main entry point to process a user request through the agentic framework.
        Handles initial planning, execution with retries/replanning, and final response generation.
        """
        context = AgentContext(user_request=request)
        replan_count = 0  # Initial replan count for the overall process

        while replan_count <= self.max_replans:
            try:
                # Determine if planning/replanning is needed
                plan_needed = not context.execution_plan
                # Note: We don't use context.error_context.replan_attempts here as replan_count tracks the overall attempts

                if plan_needed:
                    logger.info(f"Planning attempt {replan_count + 1}/{self.max_replans + 1}...")
                    plan_successful = self._plan(context)

                    if not plan_successful:
                        # Check if the failure was due to a guardrail violation
                        if context.planner_feedback_from_guardrails:
                            logger.warning("Planning failed due to guardrail violation. Attempting replan.")
                            replan_count += 1  # Consume a replan attempt for guardrail failure
                            if replan_count > self.max_replans:
                                logger.error(
                                    f"Max replan attempts ({self.max_replans}) reached after guardrail violation."
                                )
                                return f"I couldn't create a valid plan satisfying all constraints after {self.max_replans + 1} attempts. Please review the constraints or modify your request."
                            continue  # Go to the next iteration to replan
                        else:
                            # Planning failed for another reason (e.g., LLM error, extraction failed)
                            logger.error("Initial planning failed for reasons other than guardrails. Cannot proceed.")
                            last_error = (
                                context.error_context.step_errors[-1] if context.error_context.step_errors else None
                            )
                            error_info = (
                                f": {last_error.traceback_str}"
                                if last_error
                                else "(No specific error detail available)"
                            )
                            return f"I'm sorry, I couldn't create a plan to address your request{error_info}"
                    # else: planning successful, context.execution_plan is now set

                # --- Execution Phase ---
                # Proceed to execution only if we have a valid plan from the current or previous attempt
                if context.execution_plan:
                    execution_successful = self._execute(context)  # _execute handles its own retries/replans

                    if execution_successful:
                        # Overall process successful
                        logger.info("Orchestration completed successfully.")
                        return context.final_response or "Processing completed, but no final response was generated."
                    else:
                        # Execution failed after exhausting internal retries/replans within _execute
                        logger.error("Execution halted after exhausting retries or replans within the execution phase.")
                        last_error = (
                            context.error_context.step_errors[-1] if context.error_context.step_errors else None
                        )
                        error_info = (
                            f" Last error at step {last_error.step_index + 1} ('{last_error.step_description}'): {last_error.traceback_str}"
                            if last_error
                            else ""
                        )
                        # If _execute initiated a replan that failed, replan_count might increase
                        # We might need to check replan_count here again if _execute modifies it and requests outer loop replan
                        # For now, assume _execute handles its replans internally or returns definitive failure
                        return f"I'm sorry, I couldn't complete the task due to errors during execution{error_info}. Please try modifying your request."
                else:
                    # This should only happen if planning failed (non-guardrail) and returned above,
                    # or if max replans were hit due to guardrail violations.
                    # If we reach here unexpectedly, log it.
                    logger.error("Reached execution phase without a valid plan. This might indicate a logic error.")
                    # The loop should have handled returning an error message already.
                    # Add a fallback just in case.
                    return "An unexpected error occurred during planning."

            except Exception as e:
                # Catch unexpected errors in the process_user_request loop itself
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
            finally:
                BaseAgent.reset_config_per_agent_classvars()
                
        # Should ideally not be reached if logic is correct, but as a fallback
        return "I was unable to complete your request after multiple attempts."

    def _invoke_planner(self, context: AgentContext) -> Optional[StepResult]:
        """Invokes the PlanAgent and returns its result."""
        logger.info("Invoking PlanAgent...")
        planner = self.agents.get("PlanAgent")
        if not planner:
            logger.critical("PlanAgent not found!")
            context.error_context.add_error(-1, "Planning", 1, "PlanAgent not found.")
            return None

        try:
            # Handle feedback
            if context.planner_feedback_from_guardrails:
                logger.info(f"Replanning with feedback: {context.planner_feedback_from_guardrails}")
                planner.set_additional_information(context.planner_feedback_from_guardrails)
                context.planner_feedback_from_guardrails = None  # Clear after use

            plan_result = planner.execute(context)

            if not plan_result.success:
                logger.warning(
                    f"PlanAgent failed. Result: {plan_result.execution_result}. Validation might still catch guardrail issues."
                )
                # Don't add error context here yet; validation or plan check handles final failure state

            return plan_result  # Return result for caller check

        except Exception:
            logger.exception("Exception during PlanAgent invocation.")
            context.error_context.add_error(-1, "Planning Exception", 1, traceback.format_exc())
            return None

    def _validate_plan(self, context: AgentContext) -> bool:
        """Validates the plan in the context using PlanValidationAgent with retries."""
        validator = self.agents.get("PlanValidationAgent")
        if not validator:
            logger.warning("PlanValidationAgent not found, skipping validation.")
            return True  # No validator means trivially valid

        validation_attempts = 0
        while validation_attempts < self.max_validation_retries:
            logger.info(
                f"Starting implicit plan validation (attempt {validation_attempts + 1}/{self.max_validation_retries})..."
            )
            try:
                validation_result = validator.execute(context)

                if validation_result.success:
                    logger.info(f"Implicit plan validation successful after {validation_attempts + 1} attempts.")
                    context.planner_feedback_from_guardrails = None  # Clear feedback on success
                    return True  # Validation succeeded

                # Handle validation failure
                if context.planner_feedback_from_guardrails:
                    validation_attempts += 1
                    logger.warning(
                        f"Guardrail violation in validation attempt {validation_attempts}: {context.planner_feedback_from_guardrails}"
                    )
                    if validation_attempts >= self.max_validation_retries:
                        logger.error(
                            f"Max validation retries ({self.max_validation_retries}) reached. Failing validation."
                        )
                        # Feedback remains set for replan
                        return False  # Exhausted retries
                else:
                    # Non-guardrail failure within validator
                    logger.error(
                        f"Non-guardrail validation failure: {validation_result.execution_result}. Aborting validation."
                    )
                    context.error_context.add_error(
                        -1, "Validation Failure", 1, f"Validation failed: {validation_result.execution_result}"
                    )
                    context.planner_feedback_from_guardrails = None  # Clear feedback
                    return False  # Halt

            except Exception:
                logger.exception("Exception during PlanValidationAgent execution.")
                context.error_context.add_error(-1, "Validation Exception", 1, traceback.format_exc())
                context.planner_feedback_from_guardrails = None  # Clear feedback
                return False  # Halt

        # Loop finished due to max retries
        return False

    def _plan(self, context: AgentContext) -> bool:
        """Creates and validates the execution plan."""
        plan_result = self._invoke_planner(context)

        # Check planner outcome - MUST have a plan to validate
        if not plan_result or not context.execution_plan or not context.execution_plan.steps:
            if not context.planner_feedback_from_guardrails:
                logger.error("Planning failed or produced no steps, and no guardrail feedback was provided.")
                if plan_result and not plan_result.success:
                    context.error_context.add_error(
                        -1, "Planning Failure", 1, f"Planner failed: {plan_result.execution_result}"
                    )
            # Even with feedback, lack of plan steps is a failure for this phase
            return False

        # Ensure FinalAgent is always the last step of the plan
        self._ensure_final_agent_as_last_step(context)

        # Validate the generated plan
        validation_successful = self._validate_plan(context)
        if not validation_successful:
            logger.warning("Plan validation failed after retries (if applicable). Check logs for details.")
            # Feedback (if any) is already set by _validate_plan
            return False

        # Log the final, validated plan
        plan_steps_str = "\n".join(
            [
                f"Step {i + 1}: {step.description} (Agent: {step.agent_name})"
                for i, step in enumerate(context.execution_plan.steps)
            ]
        )
        logger.info(f"Final validated execution plan:\n{plan_steps_str}")

        return True  # Planning and validation successful
        
    def _ensure_final_agent_as_last_step(self, context: AgentContext) -> None:
        """Ensure FinalAgent is always the last step of the execution plan."""
        if not context.execution_plan or not context.execution_plan.steps:
            return
            
        steps = context.execution_plan.steps
        
        # If the last step is already FinalAgent, nothing to do
        if steps[-1].agent_name == "FinalAgent":
            return
            
        # Otherwise, add FinalAgent as the last step
        steps.append(
            PlanStep(
                description="Provide the final answer to the user",
                agent_name="FinalAgent",
                reason="Every plan must conclude with a synthesis step",
            )
        )
        logger.info("Added FinalAgent as the last step of the execution plan")

    def _execute_step(self, context: AgentContext, step: PlanStep, step_index: int) -> bool:
        """Executes a single plan step with retries."""
        retries = 0
        while retries < self.max_retries_per_step:
            attempt_number = retries + 1
            logger.info(
                f"Executing step {step_index + 1}/{len(context.execution_plan.steps)} (Attempt {attempt_number}/{self.max_retries_per_step}): {step.description} | Agent: {step.agent_name}"
            )
            try:
                agent = self.agents.get(step.agent_name)
                if not agent:
                    raise ValueError(f"Configuration Error: Agent '{step.agent_name}' not found.")

                result = agent.execute(context)
                step.result = result  # Store result

                if result.success:
                    logger.info(f"Step {step_index + 1} successful.")
                    # Auto-run ObservationAgent after successful ToolSelectionAgent
                    if step.agent_name == "ToolSelectionAgent":
                        self._run_observation_agent(context)
                    return True  # Step succeeded
                else:
                    # Explicit failure reported by the agent
                    raise RuntimeError(f"Agent '{step.agent_name}' reported failure: {result.execution_result}")

            except Exception as exc:
                retries += 1
                error_type = type(exc).__name__
                error_message = str(exc)
                traceback_str = traceback.format_exc()
                logger.error(
                    f"Error in step {step_index + 1} ('{step.description}') Agent '{step.agent_name}' on attempt {attempt_number}: [{error_type}] {error_message}",
                    exc_info=False,
                )
                logger.debug(f"Full traceback for step {step_index + 1} error:\n{traceback_str}")

                context.error_context.add_error(
                    step_index=step_index,
                    step_description=BaseAgent._form_description(step),
                    attempt_number=attempt_number,
                    traceback_str=f"[{error_type}] {error_message}\n{traceback_str}",
                )

                is_recoverable, is_planning_error = self._analyze_execution_error(
                    exc, agent.name if "agent" in locals() else "UnknownAgent"
                )

                if retries >= self.max_retries_per_step or not is_recoverable:
                    logger.error(f"Step {step_index + 1} failed permanently after {retries} attempts.")
                    return False  # Step failed permanently, signal back to _execute

                # If recoverable, loop continues to retry

        return False  # Retries exhausted

    def _run_observation_agent(self, context: AgentContext):
        """Attempts to run the ObservationAgent after a tool execution."""
        logger.info("Attempting to run ObservationAgent...")
        try:
            observer = self.agents.get("ObservationAgent")
            if observer:
                obs_result = observer.execute(context)
                if obs_result.success:
                    logger.info("ObservationAgent executed successfully.")
                else:
                    logger.warning(f"ObservationAgent reported failure: {obs_result.execution_result}")
            else:
                logger.error("ObservationAgent not found in orchestrator agents list.")
        except Exception as obs_exc:
            logger.error(f"Error during automatic ObservationAgent execution: {obs_exc}", exc_info=True)

    def _analyze_execution_error(self, exc: Exception, agent_name: str) -> tuple[bool, bool]:
        """Analyzes an execution error to determine if it's recoverable or indicates a planning issue.
        Returns: (is_recoverable_by_retry, is_planning_error)
        """
        is_recoverable_by_retry = True
        is_planning_error = False

        if isinstance(exc, (json.JSONDecodeError, ET.ParseError, ValueError)) and agent_name == "ToolSelectionAgent":
            logger.warning("Parsing error in ToolSelectionAgent, retrying...")
        elif isinstance(exc, KeyError) and agent_name == "ToolSelectionAgent":
            logger.error("Tool specified by ToolSelectionAgent not found. Likely requires replanning.")
            is_recoverable_by_retry = False
            is_planning_error = True
        elif (
            isinstance(exc, RuntimeError) and "Invalid tool selected" in str(exc) and agent_name == "ToolSelectionAgent"
        ):
            logger.error("Invalid tool chosen by ToolSelectionAgent. Triggering replan.")
            is_recoverable_by_retry = False
            is_planning_error = True
        # Add more specific error checks here

        return is_recoverable_by_retry, is_planning_error

    def _prepare_for_replan(self, context: AgentContext, failed_step_index: int, failed_step_description: str):
        """Appends error context to planner's additional information for replanning."""
        plan_agent = self.agents.get("PlanAgent")
        if plan_agent and isinstance(plan_agent, PlanAgent):
            last_error = (
                context.error_context.step_errors[-1].traceback_str
                if context.error_context.step_errors
                else "Unknown error"
            )
            current_info = plan_agent.get_additional_information() or ""
            new_info = f"{current_info}\n\nPrevious attempt failed at step {failed_step_index + 1} ('{failed_step_description}') with error: {last_error}\nPlease analyze this error and adjust the plan accordingly."
            plan_agent.set_additional_information(new_info.strip())
            logger.info("Appended error context to planner's additional_information.")
        else:
            logger.warning("Could not find PlanAgent to append error context.")

    def _execute(self, context: AgentContext) -> bool:
        """
        Executes the plan step-by-step, handling failures and triggering replans.

        Returns:
            bool whether execution was succesful
        """
        if not context.execution_plan:
            logger.error("Execution attempt failed: No execution plan exists.")
            return False

        step_index = 0
        current_plan_steps = context.execution_plan.steps

        while step_index < len(current_plan_steps):
            step = current_plan_steps[step_index]
            context.current_step = step_index

            step_successful = self._execute_step(context, step, step_index)

            if step_successful:
                step_index += 1
                continue  # Move to the next step

            # --- Step failed permanently ---
            logger.error(f"Step {step_index + 1} ('{step.description}') failed permanently.")
            replan_count = context.error_context.replan_attempts  # Get current count

            # Analyze error to see if replan might help (using the last error added)
            _, is_planning_error = self._analyze_execution_error(
                Exception(
                    context.error_context.step_errors[-1].traceback_str
                    if context.error_context.step_errors
                    else "Unknown"
                ),
                step.agent_name,
            )

            if replan_count < self.max_replans and (
                is_planning_error or True
            ):  # Replan on most permanent errors for now
                replan_count += 1
                context.error_context.replan_attempts = replan_count
                logger.warning(
                    f"Triggering replan attempt {replan_count} out of{self.max_replans} replans left due to step failure."
                )

                self._prepare_for_replan(context, step_index, step.description)

                if self._plan(context):
                    logger.info("Replanning successful. Restarting execution with the new plan.")
                    current_plan_steps = context.execution_plan.steps  # Get new steps
                    step_index = 0  # Restart from step 0
                    continue  # Continue outer loop with new plan
                else:
                    logger.error("Replanning failed. Halting execution.")
                    return False  # Replanning itself failed
            else:
                # Max replans reached or error deemed unrecoverable
                logger.error(
                    f"Cannot recover from error in step {step_index + 1}. Max replans ({self.max_replans}) reached or error is fatal. Halting execution."
                )
                return False  # Halt execution

        # Loop finished successfully
        logger.info("Execution plan completed successfully.")
        return True
