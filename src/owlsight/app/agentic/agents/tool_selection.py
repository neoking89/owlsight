"""
ToolSelectionAgent implementation for the agentic framework.

This module contains the ToolSelectionAgent class which is responsible
for selecting and executing appropriate tools based on the current step.
"""

from owlsight.app.agentic.agents.base import BaseAgent
from owlsight.app.agentic.helpers import get_available_tools, parse_tool_response, execute_tool
from owlsight.app.agentic.models import AgentContext, StepResult, AgentPrompt
from owlsight.app.agentic.prompts import TOOL_SELECTION_PROMPT
from owlsight.utils.logger import logger


class ToolSelectionAgent(BaseAgent):
    def __init__(self):
        super().__init__("ToolSelectionAgent", AgentPrompt(TOOL_SELECTION_PROMPT))
    
    def execute(self, context: AgentContext) -> StepResult:
        """
        Select and execute a tool based on the current step description.
        
        Args:
            context: The AgentContext containing the execution plan and current step
            
        Returns:
            StepResult with success status and the tool execution result
        """
        # Get the current step from the execution plan
        if not context.execution_plan:
            logger.error("No execution plan available in context.")
            return StepResult(success=False, execution_result="No execution plan available.")
        
        current_step = context.execution_plan.get_step(context.current_step)
        if not current_step:
            logger.error(f"Invalid step index: {context.current_step}")
            return StepResult(success=False, execution_result=f"Invalid step index: {context.current_step}")
        
        # Get available tools
        available_tools = get_available_tools(self.code_executor)
        
        # Get previous results
        previous_results = self.get_previous_results(context)
        
        # Format the prompt
        formatted_prompt = self.system_prompt.format(
            step_description=current_step.description,
            available_context=previous_results,
            available_tools=available_tools,
            additional_information=self.get_additional_information()
        )
        
        # Get the response from the LLM
        response = self.llm_call(formatted_prompt)
        
        # Parse the tool selection from the response
        try:
            tool_data = parse_tool_response(response)
            
            # Execute the selected tool
            logger.info(f"Selected tool: {tool_data['tool_name']} with parameters: {tool_data['parameters']}")
            execution_result = execute_tool(self.code_executor, tool_data)
            
            if execution_result["success"]:
                logger.info(f"Tool execution successful: {execution_result['result']}")
                result_with_metadata = {
                    "tool_name": tool_data["tool_name"],
                    "tool_parameters": tool_data["parameters"],
                    "tool_reason": tool_data["reason"],
                    "result": execution_result["result"]
                }
                
                # Add tool execution result to accumulated results
                context.accumulated_results.append(result_with_metadata)
                
                return StepResult(success=True, execution_result=result_with_metadata)
            else:
                logger.error(f"Tool execution failed: {execution_result['result']}")
                return StepResult(success=False, execution_result=execution_result["result"])
        
        except Exception as e:
            logger.exception(f"Error during tool selection or execution: {e}")
            return StepResult(success=False, execution_result=f"Error during tool selection or execution: {str(e)}")
