"""
Hierarchical Agentic Framework for Complex Task Processing

This module implements a flexible and extensible hierarchical framework for processing
complex tasks using multiple specialized agents. The framework supports:
1. Task decomposition and planning
2. Execution and monitoring
3. Validation and error handling
4. Dynamic context management
5. Multi-modal processing capabilities
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import json
import logging
import traceback

from owlsight.prompts.system_prompts import AgentPrompts
from owlsight.processors.text_generation_processors import (
    TextGenerationProcessor,
    TextGenerationProcessorTransformers,
    TextGenerationProcessorOnnx,
    TextGenerationProcessorGGUF
)
from owlsight.utils.helper_functions import extract_markdown

logger = logging.getLogger(__name__)

class AgentRole(Enum):
    """Defines the different roles agents can take in the hierarchy"""
    ARCHITECT = "architect"
    EXECUTOR = "executor"
    JUDGE = "judge"
    SPECIALIST = "specialist"  # For domain-specific tasks
    COORDINATOR = "coordinator"  # For multi-agent orchestration

@dataclass
class TaskContext:
    """Represents the context and state of a task being processed"""
    task_id: str
    description: str
    status: str = "pending"
    parent_task_id: Optional[str] = None
    subtasks: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseAgent(ABC):
    """Abstract base class for all agents in the framework"""
    
    def __init__(
        self,
        role: AgentRole,
        processor: TextGenerationProcessor,
        prompts: AgentPrompts
    ):
        self.role = role
        self.processor = processor
        self.prompts = prompts
        self.context: Optional[TaskContext] = None
    
    @property
    def system_prompt(self) -> str:
        """Get the appropriate system prompt for this agent's role"""
        return getattr(self.prompts, self.role.value)
    
    @abstractmethod
    def process(self, task_context: TaskContext) -> Dict[str, Any]:
        """Process the given task within its context"""
        pass
    
    def update_context(self, context: TaskContext) -> None:
        """Update the agent's current context"""
        self.context = context

class ArchitectAgent(BaseAgent):
    """
    Responsible for high-level planning and task decomposition.
    Analyzes complex requests and breaks them down into manageable steps.
    """
    
    def __init__(self, processor: TextGenerationProcessor, prompts: AgentPrompts):
        super().__init__(
            role=AgentRole.ARCHITECT,
            processor=processor,
            prompts=prompts
        )
        
    def process(self, task_context: TaskContext) -> Dict[str, Any]:
        """
        Analyze task and create execution plan
        
        Returns a structured plan with:
        - Analysis of the task
        - High-level approach
        - Detailed steps for execution
        """
        # Generate planning response using the processor
        response = self.processor.generate(
            input_data=self._create_planning_prompt(task_context),
            max_new_tokens=2048,
            temperature=0.2
        )
        
        try:
            extracted_response = extract_markdown(response)
            plan = json.loads(extracted_response)
            # Validate plan structure
            self._validate_plan(plan)
            return plan
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse architect response: {e}")
            raise
            
    def _create_planning_prompt(self, task_context: TaskContext) -> str:
        """Create a prompt for the planning phase"""
        return f"""
Task Description: {task_context.description}

Additional Context:
{json.dumps(task_context.metadata, indent=2)}

Please analyze this task and provide a detailed execution plan following the required JSON format.
"""
    
    def _validate_plan(self, plan: Dict[str, Any]) -> None:
        """Validate the structure of the generated plan"""
        required_fields = {"analysis", "planning", "steps"}
        if not all(field in plan for field in required_fields):
            raise ValueError(f"Plan missing required fields: {required_fields}")

class ExecutorAgent(BaseAgent):
    """
    Responsible for executing individual steps of the plan.
    Handles code generation and execution with error handling.
    """
    
    def __init__(self, processor: TextGenerationProcessor, prompts: AgentPrompts):
        super().__init__(
            role=AgentRole.EXECUTOR,
            processor=processor,
            prompts=prompts
        )
        self.retry_limit = 3
        
    def process(self, task_context: TaskContext) -> Dict[str, Any]:
        """
        Execute a specific step from the plan
        
        Returns execution results including:
        - Generated code
        - Execution status
        - Output data or error information
        """
        step = task_context.metadata.get("step")
        if not step:
            raise ValueError("No step provided in task context")
            
        retry_count = 0
        last_error = None
        
        while retry_count < self.retry_limit:
            try:
                response = self.processor.generate(
                    input_data=self._create_execution_prompt(step),
                    max_new_tokens=1024,
                    temperature=0.1
                )
                extracted_response = extract_markdown(response)
                result = json.loads(extracted_response)
                if result["execution_metadata"]["status"] == "Success":
                    return result
                    
                retry_count += 1
                last_error = result.get("execution_metadata", {}).get("errors", ["Unknown error"])[0]
            except Exception as e:
                logger.error(f"Execution failed: {e}")
                retry_count += 1
                last_error = str(e)
                
        return {
            "task": step["description"],
            "execution_metadata": {
                "status": "Error",
                "retry_count": retry_count,
                "errors": [str(last_error)]
            }
        }
        
    def _create_execution_prompt(self, step: Dict[str, Any]) -> str:
        """Create a prompt for the execution phase"""
        return f"""
Step Details:
{json.dumps(step, indent=2)}

Generate and execute Python code to accomplish this step.
Return the results in the required JSON format.
"""

class JudgeAgent(BaseAgent):
    """
    Responsible for validating execution results and making decisions
    about retrying steps or replanning.
    """
    
    def __init__(self, processor: TextGenerationProcessor, prompts: AgentPrompts):
        super().__init__(
            role=AgentRole.JUDGE,
            processor=processor,
            prompts=prompts
        )
        
    def process(self, task_context: TaskContext) -> Dict[str, Any]:
        """
        Evaluate execution results and make decisions
        
        Returns assessment including:
        - Validation results
        - Decision (proceed, retry, or replan)
        - Feedback for improvement
        """
        execution_result = task_context.metadata.get("execution_result")
        success_criteria = task_context.metadata.get("success_criteria")
        
        if not execution_result or not success_criteria:
            raise ValueError("Missing execution result or success criteria")
            
        response = self.processor.generate(
            input_data=self._create_evaluation_prompt(execution_result, success_criteria),
            max_new_tokens=512,
            temperature=0.1
        )
        extracted_response = extract_markdown(response)
        return json.loads(extracted_response)
        
    def _create_evaluation_prompt(
        self,
        execution_result: Dict[str, Any],
        success_criteria: str
    ) -> str:
        """Create a prompt for the evaluation phase"""
        return f"""
Execution Result:
{json.dumps(execution_result, indent=2)}

Success Criteria:
{success_criteria}

Evaluate the execution result against the success criteria and provide your assessment
in the required JSON format.
"""

class AgentFactory:
    """Factory class for creating different types of agents"""
    
    @staticmethod
    def create_agent(
        role: AgentRole,
        model_config: Dict[str, Any]
    ) -> BaseAgent:
        """Create an agent of the specified role with the given model configuration"""
        
        # Select appropriate processor based on model type
        processor_type = model_config.get("type", "transformers")
        processor_class = {
            "transformers": TextGenerationProcessorTransformers,
            "onnx": TextGenerationProcessorOnnx,
            "gguf": TextGenerationProcessorGGUF
        }.get(processor_type)
        
        if not processor_class:
            raise ValueError(f"Unsupported processor type: {processor_type}")
            
        processor = processor_class(**model_config)
        
        # Create appropriate agent type
        agent_class = {
            AgentRole.ARCHITECT: ArchitectAgent,
            AgentRole.EXECUTOR: ExecutorAgent,
            AgentRole.JUDGE: JudgeAgent
        }.get(role)
        
        if not agent_class:
            raise ValueError(f"Unsupported agent role: {role}")
            
        return agent_class(processor=processor, prompts=model_config.get("prompts", AgentPrompts()))

class HierarchicalFramework:
    """
    Main class for orchestrating the hierarchical framework.
    Manages the interaction between different agents and handles the overall task flow.
    """
    
    def __init__(self, model_configs: Dict[AgentRole, Dict[str, Any]]):
        """
        Initialize the framework with configurations for each agent type
        
        Parameters
        ----------
        model_configs : Dict[AgentRole, Dict[str, Any]]
            Configuration for each agent's model and prompts, keyed by agent role.
            Each config should contain:
            - processor: TextGenerationProcessor instance
            - prompts: AgentPrompts instance (or subclass)
        """
        self.agents = {}
        for role, config in model_configs.items():
            processor = config["processor"]
            prompts = config["prompts"]
            
            agent_class = {
                AgentRole.ARCHITECT: ArchitectAgent,
                AgentRole.EXECUTOR: ExecutorAgent,
                AgentRole.JUDGE: JudgeAgent
            }.get(role)
            
            if not agent_class:
                raise ValueError(f"Unsupported agent role: {role}")
                
            self.agents[role] = agent_class(processor=processor, prompts=prompts)
        
    def process_task(self, task_context: TaskContext) -> Dict[str, Any]:
        """
        Process a task through the entire framework
        
        Parameters
        ----------
        task_context : TaskContext
            Context of the task to be processed
            
        Returns
        -------
        Dict[str, Any]
            Final results of task processing
        """
        try:
            # 1. Planning Phase
            plan = self._planning_phase(task_context)
            
            # 2. Execution Phase
            results = self._execution_phase(task_context, plan)
            
            # 3. Integration Phase
            final_result = self._integrate_results(results)
            
            return final_result
            
        except Exception as e:
            logger.error(f"Task processing failed: {traceback.format_exc()}")
            raise
            
    def _planning_phase(self, task_context: TaskContext) -> Dict[str, Any]:
        """Handle the planning phase using the Architect agent"""
        architect = self.agents[AgentRole.ARCHITECT]
        return architect.process(task_context)
        
    def _execution_phase(
        self,
        task_context: TaskContext,
        plan: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute each step in the plan with validation"""
        executor = self.agents[AgentRole.EXECUTOR]
        judge = self.agents[AgentRole.JUDGE]
        results = []
        
        for step in plan["steps"]:
            step_context = TaskContext(
                task_id=f"{task_context.task_id}_{step['step_number']}",
                description=step["description"],
                parent_task_id=task_context.task_id,
                metadata={"step": step}
            )
            
            # Execute step
            execution_result = executor.process(step_context)
            
            # Validate result
            step_context.metadata["execution_result"] = execution_result
            step_context.metadata["success_criteria"] = step["success_criteria"]
            validation = judge.process(step_context)
            
            results.append({
                "step": step,
                "execution": execution_result,
                "validation": validation
            })
            
        return results
        
    def _integrate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Integrate results from all steps into a final output"""
        return {
            "status": "completed",
            "steps_completed": len(results),
            "step_results": results,
            "summary": self._create_summary(results)
        }
        
    def _create_summary(self, results: List[Dict[str, Any]]) -> str:
        """Create a summary of the processing results"""
        successful_steps = sum(
            1 for r in results
            if r["execution"]["execution_metadata"]["status"] == "Success"
        )
        return f"Completed {successful_steps}/{len(results)} steps successfully"
