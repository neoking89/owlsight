"""
ObservationAgent implementation for the agentic framework.

This module contains the ObservationAgent class which is responsible
for filtering and summarizing tool execution results.
"""

from owlsight.app.agentic.agents.base import BaseAgent
from owlsight.app.agentic.models import AgentContext, StepResult, AgentPrompt
from owlsight.app.agentic.prompts import OBSERVATION_PROMPT
from owlsight.utils.logger import logger
from owlsight.utils.helper_functions import parse_xml


class ObservationAgent(BaseAgent):
    def __init__(self):
        super().__init__("ObservationAgent", AgentPrompt(OBSERVATION_PROMPT))
    
    def execute(self, context: AgentContext) -> StepResult:
        """
        Filter and summarize tool execution results.
        
        This agent is automatically triggered after every ToolSelectionAgent execution
        to ensure only relevant information is retained.
        
        Args:
            context: The AgentContext containing the execution plan and current step
            
        Returns:
            StepResult with success status and the observation summary
        """
        # Get the current step from the execution plan
        if not context.execution_plan:
            logger.error("No execution plan available in context.")
            return StepResult(success=False, execution_result="No execution plan available.")
        
        current_step = context.execution_plan.get_step(context.current_step)
        if not current_step:
            logger.error(f"Invalid step index: {context.current_step}")
            return StepResult(success=False, execution_result=f"Invalid step index: {context.current_step}")
        
        # Get the most recent tool result
        if not context.accumulated_results:
            logger.warning("No accumulated results available for observation.")
            return StepResult(success=False, execution_result="No tool results available to observe.")
        
        latest_result = context.accumulated_results[-1]
        
        # Format the prompt
        formatted_prompt = self.system_prompt.format(
            description=current_step.description,
            tool_result=latest_result,
            additional_information=self.get_additional_information()
        )
        
        # Get the response from the LLM
        response = self.llm_call(formatted_prompt)
        
        # Extract the observation
        observation = response
        
        # Try to parse the observation from XML if present
        parsed = parse_xml(response, tag="observation")
        if parsed:
            if isinstance(parsed, str):
                observation = parsed
            else:
                logger.warning(f"Unexpected observation format: {parsed}")
        
        # Clean up any remaining tags
        observation = observation.replace("<observation>", "").replace("</observation>", "").strip()
        
        logger.info(f"Generated observation: {observation[:100]}...")
        
        # Update the latest result with the observation instead of adding a new result
        if context.accumulated_results:
            # Add or update observation field in the latest result
            if isinstance(context.accumulated_results[-1], dict):
                context.accumulated_results[-1]["observation"] = observation
            else:
                # If the latest result isn't a dict, wrap both in a new dict
                original_result = context.accumulated_results[-1]
                context.accumulated_results[-1] = {
                    "original_result": original_result,
                    "observation": observation
                }
        
        return StepResult(success=True, execution_result=observation)
