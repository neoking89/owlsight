import json
import re
import traceback
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional

from owlsight.agentic.constants import AGENT_INFORMATION
from owlsight.agentic.helper_functions import (
    execute_tool,
    get_agent_information,
    get_available_tools,
    parse_tool_response,
)
from owlsight.agentic.models import AgentContext, AgentPrompt, ExecutionPlan, PlanStep, StepResult, ToolResult
from owlsight.agentic.prompts import (
    FINAL_AGENT_PROMPT,
    OBSERVATION_PROMPT,
    PLANNER_PROMPT,
    TOOL_CREATION_PROMPT,
    TOOL_SELECTION_PROMPT,
)
from owlsight.app.default_functions import OwlDefaultFunctions
from owlsight.processors.text_generation_manager import TextGenerationManager
from owlsight.utils.code_execution import CodeExecutor
from owlsight.utils.helper_functions import parse_markdown, parse_xml
from owlsight.utils.logger import logger


class BaseAgent(ABC):
    """ "Base class for all agents in the agentic framework."""

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


class PlannerAgent(BaseAgent):
    """The PlannerAgent is responsible for creating an execution plan based on the user's question."""

    def __init__(self):
        super().__init__("PlannerAgent", AgentPrompt(PLANNER_PROMPT))

    def execute(self, context: AgentContext) -> StepResult:
        prompt = self.system_prompt.format(
            user_question=context.user_question,
            agent_information=get_agent_information(),
            available_tools=get_available_tools(BaseAgent.code_executor.globals_dict),
            additional_information=self.get_additional_information(),
        )
        reply = self.llm_call(prompt)
        steps: List[PlanStep] = self._extract(reply)

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
    """
    The ToolCreationAgent is responsible for creating new tools in python that can be used directly by the ToolCreationAgent.
    Or are available in the Python environment.
    """

    def __init__(self):
        super().__init__("ToolCreationAgent", AgentPrompt(TOOL_CREATION_PROMPT))

    def execute(self, context: AgentContext) -> StepResult:
        prompt = self.system_prompt.format(
            user_request=context.user_question,
            tools_list=get_available_tools(BaseAgent.code_executor.globals_dict),
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

                # Execute the code in an isolated namespace
                exec_globals = {}  # Start fresh for execution context
                try:
                    exec(code_to_execute, exec_globals, exec_globals)  # Use cleaned code
                except Exception as exec_exc:
                    logger.error(f"Exception during exec: {exec_exc}", exc_info=True)  # DEBUG
                    continue  # Don't proceed if exec failed

                # Add the function and any other definitions from the code block to the main globals dict
                BaseAgent.code_executor.globals_dict.update(exec_globals)

                # Check if the expected function was defined in the isolated execution
                check_result = function_name in exec_globals
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
    The ToolSelectionAgent is responsible for selecting the appropriate tool to execute based on the current step description.
    It uses the LLM to determine which tool to use and what parameters to pass.
    """

    def __init__(self):
        super().__init__("ToolSelectionAgent", AgentPrompt(TOOL_SELECTION_PROMPT))

    def execute(self, context: AgentContext) -> StepResult:
        # Allow the LLM several chances to self‑correct invalid outputs.
        max_attempts = 4  # increased by one for improved resiliency
        attempt = 0
        error_feedback: str = ""  # passed back to the LLM to aid self‑correction

        # Get the current step description
        current_step = context.execution_plan[context.current_step]
        step_description = current_step.description

        while attempt < max_attempts:
            prompt = self.system_prompt.format(
                step_description=step_description,
                available_context=self.get_previous_results(context),
                available_tools=get_available_tools(BaseAgent.code_executor.globals_dict),
                additional_information=self.get_additional_information(),
            )
            if error_feedback:
                # Append explicit guidance so the model can fix its previous mistake.
                prompt += (
                    "\nPREVIOUS_ERROR:\n"
                    + error_feedback
                    + "\nPlease fix the issue and output ONLY a valid <selection> XML or JSON object."
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
    The ObservationAgent is responsible for observing the results of the previous step and summarizing them.
    The ObservationAgent always follows after a toolresponse in the execution plan to compress and enrich the output.
    This is to ensure that the toolresponse is as concise and relevant as possible.
    """

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
    """
    FinalAgent is responsible for synthesizing the final response to the user based on the execution plan and previous results.
    It is always the last agent in the chain.
    """

    def __init__(self):
        super().__init__("FinalAgent", AgentPrompt(FINAL_AGENT_PROMPT))

    def execute(self, context: AgentContext) -> StepResult:
        prompt = self.system_prompt.format(
            user_question=context.user_question,
            previous_results=self.get_previous_results(context),
            additional_information=self.get_additional_information(),
        )
        reply = self.llm_call(prompt)
        context.final_response = reply
        return StepResult(True, reply)


class AgentOrchestrator:
    """
    The AgentOrchestrator is responsible for managing the execution of agents in a step-by-step manner.
    It handles planning, execution, and replanning if necessary.
    """

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
