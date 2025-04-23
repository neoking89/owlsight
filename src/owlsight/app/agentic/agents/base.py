"""
Base agent class for the agentic framework.

This module defines the BaseAgent class that all concrete agents inherit from.
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Optional

from owlsight.app.agentic.models import AgentContext, AgentPrompt, StepResult
from owlsight.processors.text_generation_manager import TextGenerationManager
from owlsight.utils.code_execution import CodeExecutor
from owlsight.utils.logger import logger


class BaseAgent(ABC):
    """
    Base class for all agents in the agentic framework.
    
    All concrete agent implementations should inherit from this class
    and implement the execute method.
    """
    
    # Class variables shared across all agent instances
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
            raise RuntimeError(f"TextGenerationManager not set for {self.name}")
        
        logger.info(f"Making LLM call for {self.name}")
        return self.manager.generate_response(formatted_prompt)
    
    @abstractmethod
    def execute(self, context: AgentContext) -> StepResult:
        """
        Execute the agent's task.
        
        This method must be implemented by all concrete agent classes.
        It should perform the agent's specific function and return a StepResult.
        
        Args:
            context: The current AgentContext containing all execution state
            
        Returns:
            StepResult with success flag and any execution results
        """
        pass
    
    def get_previous_results(self, context: AgentContext) -> str:
        """
        Format accumulated results from previous steps for inclusion in prompts.
        """
        if not context.accumulated_results:
            return "No previous results."
        
        formatted_results = []
        for i, result in enumerate(context.accumulated_results):
            formatted_results.append(f"Result {i+1}: {result}")
        
        return "\n\n".join(formatted_results)
    
    def get_additional_information(self) -> str:
        """
        Placeholder for any additional information to include in prompts.
        This can be overridden by concrete agent implementations.
        """
        return "No additional information available."
