"""
ToolCreationAgent implementation for the agentic framework.

This module contains the ToolCreationAgent class which is responsible
for creating dynamic tool functions based on user requests.
"""

import ast
import inspect
import re
from typing import Dict, List, Any

from owlsight.app.agentic.agents.base import BaseAgent
from owlsight.app.agentic.helpers import get_available_tools
from owlsight.app.agentic.models import AgentContext, StepResult, AgentPrompt
from owlsight.app.agentic.prompts import TOOL_CREATION_PROMPT
from owlsight.utils.helper_functions import parse_markdown
from owlsight.utils.logger import logger


class ToolCreationAgent(BaseAgent):
    def __init__(self):
        super().__init__("ToolCreationAgent", AgentPrompt(TOOL_CREATION_PROMPT))
    
    def execute(self, context: AgentContext) -> StepResult:
        """
        Create a new dynamic tool based on the user's request.
        
        Args:
            context: The AgentContext containing the user question
            
        Returns:
            StepResult with success status and the names of the registered tools
        """
        # Get available tools for the prompt
        available_tools = get_available_tools(self.code_executor)
        
        # Get the execution history for the prompt
        previous_results = self.get_previous_results(context)
        
        # Format the prompt
        formatted_prompt = self.system_prompt.format(
            user_request=context.user_question,
            tools_list=available_tools,
            tool_creation_history=previous_results,
            previous_attempts=previous_results,
            additional_information=self.get_additional_information()
        )
        
        # Get the response from the LLM
        response = self.llm_call(formatted_prompt)
        
        # Extract code blocks from the response
        extracted_data = self._extract(response)
        
        # Register the extracted tools
        if not extracted_data:
            logger.warning("No valid Python code blocks found in response.")
            return StepResult(success=False, execution_result="No valid Python code blocks found in response.")
        
        # Register the tools
        registered_tool_names = self._register_dynamic_tool(extracted_data)
        if not registered_tool_names:
            logger.warning("Failed to register any tools from the response.")
            return StepResult(success=False, execution_result="Failed to register dynamic tools.")
        
        # Add result to accumulated results
        logger.info(f"Successfully registered tools: {registered_tool_names}")
        
        # Return success
        return StepResult(success=True, execution_result=registered_tool_names)
    
    def _extract(self, markdown: str) -> Dict[str, Any]:
        """
        Extract Python function code blocks from markdown.
        Only processes markdown-formatted Python code blocks.
        
        Args:
            markdown: The markdown response from the LLM
            
        Returns:
            Dictionary containing extracted code blocks
        """
        # Parse the markdown to extract code blocks
        parsed_data = parse_markdown(markdown)
        
        # Filter to only include Python code blocks
        if "code_blocks" in parsed_data:
            code_blocks = []
            for lang, code in parsed_data["code_blocks"]:
                if lang.lower() == "python":
                    code_blocks.append((lang, code))
            
            if not code_blocks:
                logger.warning("No Python code blocks found in markdown.")
                return {}
            
            return {"code_blocks": code_blocks}
        
        # If parse_markdown didn't extract code blocks, try manual extraction
        python_blocks = []
        matches = re.finditer(r"```python\s*(.*?)```", markdown, re.DOTALL)
        for match in matches:
            code = match.group(1).strip()
            if code:
                python_blocks.append(("python", code))
        
        if python_blocks:
            return {"code_blocks": python_blocks}
        
        return {}
    
    def _register_dynamic_tool(self, data: Dict[str, Any]) -> List[str]:
        """
        Register Python functions extracted from markdown code blocks as dynamic tools.
        
        Args:
            data: Dictionary containing extracted code blocks
            
        Returns:
            List of registered function names
        """
        if "code_blocks" not in data:
            logger.warning("No code blocks provided for registration.")
            return []
        
        registered_tools = []
        
        for _, code_text in data["code_blocks"]:
            try:
                # Parse the code to extract function definitions
                tree = ast.parse(code_text)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        function_name = node.name
                        
                        # Get the actual function code based on line numbers
                        function_body = '\n'.join(code_text.split('\n')[node.lineno-1:node.end_lineno])
                        
                        # Execute the function definition
                        try:
                            exec(code_text, self.code_executor.globals_dict)
                            registered_tools.append(function_name)
                            logger.info(f"Successfully registered function: {function_name}")
                        except Exception as exec_err:
                            logger.exception(f"Could not register generated tool {function_name}: {exec_err}")
            
            except SyntaxError as e:
                logger.exception(f"Could not register generated tool due to syntax error: {e}")
                continue
            
            # If we couldn't identify any function names, log a warning
            if not registered_tools:
                logger.warning(f"Could not identify function name in code block: {code_text[:100]}...")
        
        return registered_tools
