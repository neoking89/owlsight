"""
PlannerAgent implementation for the agentic framework.

This module contains the PlannerAgent class which is responsible
for creating execution plans based on user questions.
"""

import xml.etree.ElementTree as ET
from typing import Dict

from owlsight.app.agentic.helpers import get_available_tools
from owlsight.app.agentic.models import AgentContext, ExecutionPlan, PlanStep, StepResult, AgentPrompt
from owlsight.app.agentic.agents.base import BaseAgent
from owlsight.app.agentic.prompts import PLANNER_PROMPT
from owlsight.utils.logger import logger
from owlsight.utils.helper_functions import parse_xml


class PlannerAgent(BaseAgent):
    def __init__(self):
        super().__init__("PlannerAgent", AgentPrompt(PLANNER_PROMPT))
    
    def execute(self, context: AgentContext) -> StepResult:
        """
        Create an execution plan for the user question.
        
        Args:
            context: The AgentContext containing the user question
            
        Returns:
            StepResult with success status and the execution plan
        """
        # Get available tools for inclusion in the prompt
        available_tools = get_available_tools(self.code_executor)
        
        # Format the prompt with the user question and available tools
        formatted_prompt = self.system_prompt.format(
            user_question=context.user_question,
            available_tools=available_tools,
            additional_information=self.get_additional_information()
        )
        
        # Get response from LLM
        response = self.llm_call(formatted_prompt)
        
        # Extract plan from response
        try:
            plan_data = self._extract(response)
            if not plan_data or "steps" not in plan_data or not plan_data["steps"]:
                logger.error("Failed to extract valid plan from response.")
                return StepResult(success=False, execution_result="No valid plan steps found in response.")
            
            # Create plan steps from extracted data
            steps = []
            for step_data in plan_data["steps"]:
                steps.append(
                    PlanStep(
                        description=step_data.get("description", "Missing description"),
                        agent_name=step_data.get("agent", "Unknown"),
                        reason=step_data.get("reason", "No reason provided")
                    )
                )
            
            # Set the execution plan in the context
            context.execution_plan = ExecutionPlan(steps=steps)
            logger.info(f"Created execution plan with {len(steps)} steps.")
            
            return StepResult(success=True, execution_result=context.execution_plan)
        
        except Exception as e:
            logger.exception("Error extracting plan from response")
            return StepResult(success=False, execution_result=f"Error creating plan: {str(e)}")
    
    def _extract(self, xml: str) -> Dict:
        """
        Extract plan data from XML response.
        
        Args:
            xml: The XML response from the LLM
            
        Returns:
            Dictionary containing the extracted plan data
        """
        # First try to parse using helper function
        parsed = parse_xml(xml, tag="plan")
        if parsed:
            return self._process_parsed_xml(parsed)
        
        # If that fails, try direct XML parsing
        try:
            # Extract <plan> element
            plan_match = xml
            if "<plan>" in xml and "</plan>" in xml:
                start = xml.find("<plan>")
                end = xml.find("</plan>") + len("</plan>")
                plan_match = xml[start:end]
            
            root = ET.fromstring(plan_match)
            steps = []
            
            for step_elem in root.findall("./step"):
                step_data = {
                    "description": step_elem.findtext("./description", ""),
                    "agent": step_elem.findtext("./agent", ""),
                    "reason": step_elem.findtext("./reason", "")
                }
                steps.append(step_data)
            
            return {"steps": steps}
        
        except Exception as e:
            logger.error(f"Error parsing XML: {e}")
            return {}
    
    def _process_parsed_xml(self, parsed_data: Dict) -> Dict:
        """
        Process parsed XML data into a standardized plan format.
        
        Args:
            parsed_data: The parsed XML data
            
        Returns:
            Standardized plan data dictionary
        """
        if isinstance(parsed_data, dict) and "steps" in parsed_data:
            # Already in expected format
            return parsed_data
        
        # Handle case where parsed_data is a list of steps
        if isinstance(parsed_data, list):
            steps = []
            for item in parsed_data:
                if isinstance(item, dict):
                    step = {
                        "description": item.get("description", ""),
                        "agent": item.get("agent", ""),
                        "reason": item.get("reason", "")
                    }
                    steps.append(step)
            return {"steps": steps}
        
        # Handle unexpected format
        logger.warning(f"Unexpected format in parsed XML: {parsed_data}")
        return {}
