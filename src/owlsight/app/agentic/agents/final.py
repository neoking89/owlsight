"""
FinalAgent implementation for the agentic framework.

This module contains the FinalAgent class which is responsible
for synthesizing the final response to the user's question.
"""

from owlsight.app.agentic.agents.base import BaseAgent
from owlsight.app.agentic.models import AgentContext, StepResult, AgentPrompt
from owlsight.app.agentic.prompts import RESPONSE_SYNTHESIS_PROMPT
from owlsight.utils.logger import logger
from owlsight.utils.helper_functions import parse_xml


class FinalAgent(BaseAgent):
    def __init__(self):
        super().__init__("FinalAgent", AgentPrompt(RESPONSE_SYNTHESIS_PROMPT))
    
    def execute(self, context: AgentContext) -> StepResult:
        """
        Synthesize the final response based on accumulated context.
        
        Args:
            context: The AgentContext containing the user question and accumulated results
            
        Returns:
            StepResult with success status and the final response
        """
        # Get previous results
        previous_results = self.get_previous_results(context)
        
        # Format the prompt
        formatted_prompt = self.system_prompt.format(
            user_question=context.user_question,
            previous_results=previous_results,
            additional_information=self.get_additional_information()
        )
        
        # Get the response from the LLM
        response = self.llm_call(formatted_prompt)
        
        # Extract the content from the response
        final_response = response
        
        # Try to parse the response content from XML
        parsed = parse_xml(response, target_tag="response")
        if isinstance(parsed, dict) and "content" in parsed:
            final_response = parsed["content"]
        elif isinstance(parsed, str):
            final_response = parsed
        
        # Clean up any remaining XML tags
        final_response = (
            final_response
            .replace("<response>", "")
            .replace("</response>", "")
            .replace("<content>", "")
            .replace("</content>", "")
            .strip()
        )
        
        # Set the final response in the context
        context.final_response = final_response
        
        logger.info(f"Generated final response: {final_response[:100]}...")
        
        return StepResult(success=True, execution_result=final_response)
