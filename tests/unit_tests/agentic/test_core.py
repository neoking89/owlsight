"""
Unit tests for the core agentic components like AgentOrchestrator.
"""

import pytest
from unittest.mock import MagicMock, patch

from owlsight.agentic.core import AgentOrchestrator
from owlsight.agentic.models import (
    AgentContext,
    ExecutionPlan,
    PlanStep,
)

# --- Mocks and placeholders for missing classes --- #
class PlannerAgent:
    pass

class StepResult:
    def __init__(self, success: bool, execution_result: str):
        self.success = success
        self.execution_result = execution_result

# Helper to create steps
def create_step(agent_name: str, description: str = "") -> PlanStep:
    return PlanStep(description=description or f"Step for {agent_name}", agent_name=agent_name, reason="Test reason")

# --- AgentOrchestrator Tests --- #

def test_orchestrator_replan_on_guardrail_violation():
    """
    Tests that the orchestrator replans if the initial plan violates a guardrail.
    This version mocks _plan directly and avoids fixtures.
    """
    # --- Setup Dependencies Directly ---
    mock_code_executor = MagicMock()
    mock_manager = MagicMock()  # Minimal mock for AgentManager if needed
    orchestrator = AgentOrchestrator(mock_code_executor, mock_manager, max_replans=1)

    # --- Test Setup ---
    user_question = "Test question"
    valid_plan_after_replan = ExecutionPlan([
        create_step("ToolSelectionAgent"),
        create_step("FinalAgent"),
    ])
    guardrail_error_msg = "FinalAgent not last"
    feedback_msg = f"Plan validation failed: {guardrail_error_msg}. Please revise the plan."
    expected_final_response = "Replanned execution successful"

    # --- Mock _plan Behavior ---
    plan_call_count = 0

    def plan_side_effect(context: AgentContext):
        nonlocal plan_call_count
        plan_call_count += 1
        if plan_call_count == 1:
            # First call: Simulate guardrail violation
            print("Simulating _plan call 1: Guardrail failure")
            context.planner_feedback_from_guardrails = feedback_msg
            context.execution_plan = None  # Ensure no plan is set on failure
            return False  # Indicate planning failed
        elif plan_call_count == 2:
            # Second call (replan): Simulate successful planning
            print("Simulating _plan call 2: Success")
            context.execution_plan = valid_plan_after_replan
            context.planner_feedback_from_guardrails = None
            return True  # Indicate planning succeeded
        else:
            pytest.fail("_plan called more than twice!")

    # --- Mock _execute Behavior ---
    execute_call_count = 0

    def execute_side_effect(context: AgentContext):
        nonlocal execute_call_count
        execute_call_count += 1
        print(f"Simulating _execute call {execute_call_count} with plan: {context.execution_plan}")
        # Check if called with the valid plan from the replan
        if context.execution_plan == valid_plan_after_replan:
            context.final_response = expected_final_response
            return True  # Simulate successful execution
        else:
            pytest.fail(f"_execute called with unexpected plan: {context.execution_plan}")

    # --- Patch and Act ---
    with patch.object(orchestrator, '_plan', side_effect=plan_side_effect) as mock_plan, \
         patch.object(orchestrator, '_execute', side_effect=execute_side_effect) as mock_execute:

        print("Calling orchestrator.process_user_question...")
        final_result = orchestrator.process_user_question(user_question)
        print(f"orchestrator.process_user_question returned: {final_result}")

    # --- Assert ---
    print("Asserting call counts...")
    # 1. _plan was called twice (initial attempt + replan)
    assert mock_plan.call_count == 2, f"Expected _plan to be called 2 times, but was called {mock_plan.call_count} times"

    # 2. _execute was called once (only after successful replan)
    assert mock_execute.call_count == 1, f"Expected _execute to be called 1 time, but was called {mock_execute.call_count} times"

    # 3. _execute was called with the *valid* plan
    # (The check is also inside the execute_side_effect)
    execute_call_args = mock_execute.call_args[0]
    assert execute_call_args[0].execution_plan == valid_plan_after_replan, "_execute was not called with the valid plan"

    # 4. Final result is based on successful execution of the replanned steps
    assert final_result == expected_final_response, f"Expected final result '{expected_final_response}', but got '{final_result}'"
