import math
from dataclasses import dataclass, field
from typing import Any, List, Optional
import traceback

from owlsight.utils.logger import logger


@dataclass
class ToolResult:
    """Represents the result of a tool execution, including success status and the result itself."""

    success: bool
    result: Any


@dataclass
class StepResult:
    """Represents the result of a PlanStep"""

    success: bool
    execution_result: Any = None


@dataclass
class PlanStep:
    """Represents a step in the execution plan, including the description, agent name, and reason for the step."""

    description: str
    agent_name: str
    reason: str
    result: Optional[StepResult] = None


@dataclass
class StepErrorInfo:
    """Represents an error that occurred during the execution of a step, including the step index, description, and traceback."""

    step_index: int
    step_description: str
    attempt_number: int
    traceback_str: str


@dataclass
class ErrorContext:
    """Represents the context of errors that occurred during the execution of a plan, including a list of StepErrorInfo records."""

    step_errors: List[StepErrorInfo] = field(default_factory=list)
    replan_attempts: int = 0

    def add_error(self, step_index: int, step_description: str, attempt_number: int, traceback_str: str):
        self.step_errors.append(StepErrorInfo(step_index, step_description, attempt_number, traceback_str))

    def __str__(self):
        if not self.step_errors:
            return "No errors"
        return "\n".join(
            f"Step {e.step_index} ({e.step_description}), Attempt {e.attempt_number}: {e.traceback_str}"
            for e in self.step_errors
        )


@dataclass
class ExecutionPlan:
    """
    Represents the execution plan for a set of PlanSteps, including the list of steps to be executed.
    This plan is made by the Planner agent.
    """

    steps: List[PlanStep]

    def get_step(self, index: int) -> Optional[PlanStep]:
        """
        Get a specific step from the execution plan by its index.
        This method retrieves the step at the specified index from the list of steps.
        If the index is out of range, it returns None.
        """
        if not self.steps:
            raise ValueError("Execution plan is empty.")
        return self.steps[index] if 0 <= index < len(self.steps) else None

    def __getitem__(self, index: int) -> PlanStep:
        return self.steps[index]

    def __len__(self):
        return len(self.steps)

    def __str__(self):
        return "\n".join(f"Step {i + 1}: {s.description} ({s.agent_name})" for i, s in enumerate(self.steps))


@dataclass
class AgentPrompt:
    """
    A flexible prompt template that can be formatted with various parameters.
    It is used to generate prompts for different agents in the system.
    The template can be a string with placeholders for parameters, which will be replaced with actual values when formatting.
    """

    template: str
    params: dict[str, Any] = field(default_factory=dict)

    def format(self, **kwargs) -> str:
        combined_params = {**self.params, **kwargs}
        formatted_prompt = self.template.format(**combined_params)
        logger.debug(self._get_amt_tokens(formatted_prompt, combined_params))
        return formatted_prompt

    def __str__(self) -> str:
        return self.template

    def _get_amt_tokens(self, formatted_prompt: str, params: dict[str, Any]) -> str:
        """
        Estimate token count by dividing character length by
        the average characters-per-token ratio and return log messages as a string.
        """
        AVG_CHARS_PER_TOKEN = 4
        log_lines = []

        prompt_chars = len(formatted_prompt)
        prompt_tokens = math.ceil(prompt_chars / AVG_CHARS_PER_TOKEN)
        log_lines.append(f"Total tokens in 'formatted_prompt': {prompt_chars} chars -> {prompt_tokens} tokens")

        for param, value in params.items():
            val_str = str(value)
            val_chars = len(val_str)
            val_tokens = math.ceil(val_chars / AVG_CHARS_PER_TOKEN)
            log_lines.append(f"Parameter '{param}': {val_chars} chars -> {val_tokens} tokens")

        return "\n".join(log_lines)


@dataclass
class AgentContext:
    """
    Represents the shared state (or central memory) passed among agents, including:
    - The user's original request
    - The index of the current step
    - The execution plan
    - An ErrorContext that can contain multiple StepErrorInfo records
    - A final_response (if any). This is the final output of the entire chain of agents.
    - Accumulated (summarized) results from previous steps
    - Planner feedback from guardrail validations
    """

    user_request: str
    current_step: int = 0
    execution_plan: Optional[ExecutionPlan] = None
    error_context: ErrorContext = field(default_factory=ErrorContext)
    final_response: Optional[str] = None
    accumulated_results: List[Any] = field(default_factory=list)
    planner_feedback_from_guardrails: Optional[str] = None

    def get_current_step(self) -> PlanStep:
        """
        Get the current step from the execution plan.
        This method retrieves the step at the current index from the execution plan.
        """
        if self.execution_plan is None:
            raise ValueError("Execution plan is not set.")
        try:
            current_step = self.execution_plan.get_step(self.current_step)
            if current_step is None:
                raise IndexError(f"Current step index {self.current_step} is out of range.")
            return current_step
        except Exception as e:
            raise ValueError(f"Failed to get current step {self.current_step} due to:\n{traceback.format_exc()}") from e

    def get_previous_results(self) -> str:
        """
        Format accumulated results from previous steps for inclusion in prompts.
        This method provides a summary of the results from all previous steps.
        """
        if not self.accumulated_results:
            return "No previous results"
        out = []
        for i, r in enumerate(self.accumulated_results):
            tag = "tool" if isinstance(r, ToolResult) else "note"
            out.append(f"Step {i + 1} ({tag}): {r.result if isinstance(r, ToolResult) else r}")
        return "\n".join(out)
