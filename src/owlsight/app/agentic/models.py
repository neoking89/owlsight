"""
Data models for the agentic framework.

This module contains the class definitions for representing the state
and data structures used in the agentic framework.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional

from owlsight.utils.logger import logger


@dataclass
class ToolResult:
    success: bool
    result: Any


class EventType(Enum):
    USER_QUESTION_RECEIVED = "user_question_received"
    PLAN_CREATED = "plan_created"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STEP_ERROR = "step_error"


@dataclass
class StepResult:
    success: bool
    execution_result: Any = None


@dataclass
class PlanStep:
    description: str
    agent_name: str
    reason: str
    result: Optional[StepResult] = None


@dataclass
class StepErrorInfo:
    step_index: int
    step_description: str
    attempt_number: int
    traceback_str: str


@dataclass
class ErrorContext:
    step_errors: List[StepErrorInfo] = field(default_factory=list)
    replan_attempts: int = 0

    def add_error(self, step_index: int, step_description: str, attempt_number: int, traceback_str: str):
        error_info = StepErrorInfo(
            step_index=step_index,
            step_description=step_description,
            attempt_number=attempt_number,
            traceback_str=traceback_str
        )
        self.step_errors.append(error_info)

    def __str__(self) -> str:
        parts = [f"ErrorContext(replan_attempts={self.replan_attempts})"]
        for i, error in enumerate(self.step_errors):
            parts.append(f"  Error {i+1}: Step {error.step_index} ({error.step_description}), "
                         f"Attempt {error.attempt_number}")
        return "\n".join(parts)


@dataclass
class ExecutionPlan:
    steps: List[PlanStep]

    def get_step(self, index: int) -> Optional[PlanStep]:
        if 0 <= index < len(self.steps):
            return self.steps[index]
        return None

    def __getitem__(self, index: int) -> PlanStep:
        return self.steps[index]

    def __len__(self) -> int:
        return len(self.steps)

    def __str__(self) -> str:
        parts = [f"ExecutionPlan({len(self.steps)} steps)"]
        for i, step in enumerate(self.steps):
            parts.append(f"  Step {i+1}: {step.description} (Agent: {step.agent_name})")
        return "\n".join(parts)


@dataclass
class AgentPrompt:
    """A flexible prompt template that can be formatted with various parameters."""
    template: str
    params: Dict[str, Any] = field(default_factory=dict)

    def format(self, **kwargs) -> str:
        # Combine default params with call-specific kwargs
        combined_params = {**self.params, **kwargs}
        try:
            formatted = self.template.format(**combined_params)
            debug_info = self._get_amt_tokens(formatted, combined_params)
            logger.debug(f"Formatted prompt: {debug_info}")
            return formatted
        except KeyError as e:
            logger.error(f"Missing required parameter in prompt format: {e}")
            # Provide more context in the error
            raise KeyError(f"Missing required parameter '{e.args[0]}' in prompt format. Available params: {list(combined_params.keys())}") from e

    def __str__(self) -> str:
        # For debugging/representation purposes
        param_str = ", ".join(f"{k}=..." for k in self.params.keys())
        template_preview = self.template[:50] + "..." if len(self.template) > 50 else self.template
        return f"AgentPrompt({template_preview}, {param_str})"

    def _get_amt_tokens(self, formatted_prompt: str, params: Dict[str, Any]) -> str:
        """
        Estimate token count by dividing character length by
        the average characters-per-token ratio and return log messages as a string.
        """
        # Characters per token is ~4 for English text
        chars_per_token = 4
        est_tokens = len(formatted_prompt) / chars_per_token
        
        # Extract param sizes for debugging
        param_sizes = {
            key: len(str(value)) if isinstance(value, str) else f"{len(str(value))} chars"
            for key, value in params.items()
        }
        
        return (
            f"~{est_tokens:.0f} tokens, {len(formatted_prompt)} chars. "
            f"Param sizes: {str(param_sizes)}"
        )


@dataclass
class AgentContext:
    """Represents the shared state (or central memory) passed among agents, including:
    - The user's original question
    - The index of the current step
    - The execution plan
    - An ErrorContext that can contain multiple StepErrorInfo records
    - A final_response (if any)
    - Accumulated results from previous steps
    """
    user_question: str
    current_step: int = 0
    execution_plan: Optional[ExecutionPlan] = None
    error_context: ErrorContext = field(default_factory=ErrorContext)
    final_response: Optional[str] = None
    accumulated_results: List[Any] = field(default_factory=list)
