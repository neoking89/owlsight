"""
Orchestrator component for the agentic framework.

This module contains the AgentOrchestrator class which is responsible
for coordinating the execution of agents to solve complex tasks.
"""

import traceback
from typing import Dict

from owlsight.app.agentic.agents.planner import PlannerAgent
from owlsight.app.agentic.agents.tool_creation import ToolCreationAgent
from owlsight.app.agentic.agents.tool_selection import ToolSelectionAgent
from owlsight.app.agentic.agents.observation import ObservationAgent
from owlsight.app.agentic.agents.final import FinalAgent
from owlsight.app.agentic.agents.base import BaseAgent
from owlsight.app.agentic.models import AgentContext
from owlsight.processors.text_generation_manager import TextGenerationManager
from owlsight.utils.code_execution import CodeExecutor
from owlsight.utils.logger import logger


class AgentOrchestrator:
    """
    Coordinates the execution of agents to solve complex tasks.
    
    The AgentOrchestrator is responsible for:
    1. Creating an execution plan using the PlannerAgent
    2. Executing the plan step by step
    3. Handling errors and retries
    4. Implementing replanning when necessary
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
                        isinstance(exc, (ValueError, TypeError, KeyError))
                        and "not found" in str(exc).lower()
                        or "missing" in str(exc).lower()
                        or "invalid" in str(exc).lower()
                    ):
                        # Configuration errors are less likely to resolve with retries
                        is_recoverable_by_retry = False
                        is_planning_error = True
                    
                    # Check for XML parsing, JSON errors that may indicate model response issues
                    if "SyntaxError" in error_type or "ParseError" in error_type or "JSONDecodeError" in error_type:
                        # These may resolve with a retry
                        is_recoverable_by_retry = True
                    
                    # If we've exhausted retries or determined it's not worth retrying
                    if retries >= self.max_retries_per_step or not is_recoverable_by_retry:
                        logger.warning(
                            f"Step {step_index + 1} failed after {retries} attempts. {'Attempting replan' if is_planning_error else 'Marking step as failed and proceeding'}."
                        )
                        
                        if is_planning_error:
                            # If this appears to be a planning-related error, try replanning
                            replan_count += 1
                            context.error_context.replan_attempts = replan_count
                            
                            if replan_count <= self.max_replans:
                                logger.info(f"Initiating replan attempt {replan_count}/{self.max_replans}...")
                                if self._plan(context):
                                    logger.info("Replanning successful. Restarting execution with new plan.")
                                    # Restart execution with the new plan - effectively a recursion
                                    return self._execute(context)
                                else:
                                    logger.error("Replanning failed. Cannot continue execution.")
                                    return False
                            else:
                                logger.error(f"Exceeded maximum replan attempts ({self.max_replans}). Execution failed.")
                                return False
                        
                        # If we've exhausted retries but it's not a planning error, move to the next step
                        # The step's result will be None or a failed result
                        break  # Break the retry loop to move to the next step
                    
                    # Otherwise we'll continue the retry loop
                    logger.info(f"Retrying step {step_index + 1} (attempt {attempt_number + 1}/{self.max_retries_per_step})...")

            # Move to the next step in the plan (if we're still executing)
            step_index += 1

        # If we've made it here, all steps have been attempted (though some may have failed)
        # Check if the last step was the FinalAgent to ensure we have a final response
        if context.execution_plan and context.execution_plan.steps:
            last_step = context.execution_plan.steps[-1]
            if last_step.agent_name == "FinalAgent" and last_step.result and last_step.result.success:
                logger.info("Execution completed successfully with final response.")
                return True
            elif not context.final_response:
                # If we don't have a final response but completed all steps, try to generate one
                logger.warning("Execution completed all steps but no final response was generated.")
                try:
                    final_agent = self.agents.get("FinalAgent")
                    if final_agent:
                        logger.info("Attempting to generate final response with FinalAgent...")
                        final_result = final_agent.execute(context)
                        if final_result.success:
                            logger.info("Successfully generated final response.")
                            return True
                        else:
                            logger.error(f"Failed to generate final response: {final_result.execution_result}")
                    else:
                        logger.error("FinalAgent not found in orchestrator agents list.")
                except Exception as e:
                    logger.exception(f"Error generating final response: {e}")
                    context.error_context.add_error(
                        step_index=len(context.execution_plan.steps),
                        step_description="Generate final response",
                        attempt_number=1,
                        traceback_str=traceback.format_exc(),
                    )
                    return False

        # Check overall success - did we complete all steps without fatal errors?
        for i, step in enumerate(context.execution_plan.steps):
            if not step.result or not step.result.success:
                logger.warning(f"Step {i+1} ({step.description}) did not complete successfully.")
                return False

        return True
