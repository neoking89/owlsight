"""
Demo of the Hierarchical Agentic Framework using Qwen2.5-Coder model.

This example shows how to:
1. Set up the framework with appropriate model configurations
2. Process user inquiries through the agent hierarchy
3. Handle function calling and tool usage
"""

import json
import sys
from typing import Dict, Any, Optional

sys.path.append("src")

from owlsight.agents.hierarchical_framework import (
    HierarchicalFramework,
    AgentRole,
    TaskContext
)
from owlsight.processors.text_generation_processors import TextGenerationProcessorGGUF
from owlsight.prompts.system_prompts import AgentPrompts
from owlsight.utils.custom_classes import SingletonDict

class EnhancedAgentPrompts(AgentPrompts):
    """Extended AgentPrompts with tool awareness"""
    
    def __init__(self, available_information: str = "", globals_dict: Optional[SingletonDict] = None):
        super().__init__(available_information)
        self.globals_dict = globals_dict or SingletonDict()
        
    @property
    def architect(self) -> str:
        # Get available tools
        tools_info = self.show_available_tools(self.globals_dict)
        
        # Enhance the original architect prompt with tools information
        original_prompt = super().architect
        return f"""
{original_prompt}

# AVAILABLE TOOLS AND FUNCTIONS:
The following tools and functions are available for use in your planning:

{tools_info}

When creating your plan, consider these tools and specify which ones should be used in each step.
"""

def create_qwen_processor() -> TextGenerationProcessorGGUF:
    """Create a Qwen2.5-Coder GGUF processor with specified configuration"""
    config = {
        "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        "gguf__filename": "qwen2.5-coder-7b-instruct-q6_k.gguf",
        "gguf__verbose": True,
        "gguf__n_ctx": 16384,
        "gguf__n_gpu_layers": 0,
        "gguf__n_batch": 8,
        "gguf__n_cpu_threads": 8,
    }
    return TextGenerationProcessorGGUF(**config)

def create_framework(globals_dict: Optional[SingletonDict] = None) -> HierarchicalFramework:
    """Create the hierarchical framework with Qwen model for all agents"""
    
    # Create base processor
    processor = create_qwen_processor()
    
    # Create configurations for each agent role
    model_configs = {
        AgentRole.ARCHITECT: {
            "processor": processor,
            "prompts": EnhancedAgentPrompts(globals_dict=globals_dict)
        },
        AgentRole.EXECUTOR: {
            "processor": processor,
            "prompts": AgentPrompts()  # Standard prompts for executor
        },
        AgentRole.JUDGE: {
            "processor": processor,
            "prompts": AgentPrompts()  # Standard prompts for judge
        }
    }
    
    return HierarchicalFramework(model_configs)

def process_user_inquiry(
    framework: HierarchicalFramework,
    inquiry: str,
    context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Process a user inquiry through the framework
    
    Parameters
    ----------
    framework : HierarchicalFramework
        The initialized framework
    inquiry : str
        User's inquiry or request
    context : Dict[str, Any], optional
        Additional context for the inquiry
        
    Returns
    -------
    Dict[str, Any]
        Processing results
    """
    task_context = TaskContext(
        task_id="user_inquiry",
        description=inquiry,
        metadata=context or {}
    )
    
    return framework.process_task(task_context)

def main():
    """Main demo function"""
    # Initialize global dictionary for tool tracking
    globals_dict = SingletonDict()
    
    # Create the framework
    framework = create_framework(globals_dict)
    
    # Example inquiries to demonstrate different capabilities
    inquiries = [
        # Simple code generation task
        "Write a Python function to calculate the Fibonacci sequence up to n terms",
        
        # Task requiring tool usage
        "Create a bar chart showing the distribution of word lengths in a given text file",
        
        # Complex multi-step task
        """
        Build a simple web API that:
        1. Accepts a JSON payload with a 'text' field
        2. Performs sentiment analysis on the text
        3. Returns the sentiment score in JSON format
        """
    ]
    
    # Process each inquiry
    for i, inquiry in enumerate(inquiries, 1):
        print(f"\n{'='*80}\nProcessing Inquiry #{i}:\n{inquiry}\n{'='*80}\n")
        
        try:
            result = process_user_inquiry(framework, inquiry)
            print(f"Results:\n{json.dumps(result, indent=2)}")
        except Exception as e:
            print(f"Error processing inquiry: {e}")

if __name__ == "__main__":
    # Run the demo
    main()
