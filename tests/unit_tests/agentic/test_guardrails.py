"""
Unit tests for the agentic guardrails system.
"""

import pytest

from owlsight.agentic.models import ExecutionPlan, PlanStep
from owlsight.agentic.exceptions import GuardrailViolationError
from owlsight.agentic.guardrails import (
    GuardrailManager,
    ToolExecutionFollowsToolCreationGuardrail,
)

# Helper function to create PlanSteps
def create_step(agent_name: str, description: str = "") -> PlanStep:
    return PlanStep(description=description or f"Step for {agent_name}", agent_name=agent_name, reason="Test reason")


# --- Tests for ToolExecutionFollowsToolCreationGuardrail ---

def test_tool_creation_followed_by_selection_valid():
    """Tests a valid plan where ToolCreationAgent is followed by ToolSelectionAgent."""
    guardrail = ToolExecutionFollowsToolCreationGuardrail()
    plan = ExecutionPlan([
        create_step("ToolCreationAgent"),
        create_step("ToolSelectionAgent"),
        create_step("FinalAgent"),
    ])
    guardrail.validate(plan)  # Should not raise

def test_tool_selection_without_creation_valid():
    """Tests a valid plan where ToolSelectionAgent is used without ToolCreationAgent."""
    guardrail = ToolExecutionFollowsToolCreationGuardrail()
    plan = ExecutionPlan([
        create_step("ToolSelectionAgent"),
        create_step("FinalAgent"),
    ])
    guardrail.validate(plan)  # Should not raise

def test_multiple_tool_creation_sequences_valid():
    """Tests a valid plan with multiple ToolCreation -> ToolSelection sequences."""
    guardrail = ToolExecutionFollowsToolCreationGuardrail()
    plan = ExecutionPlan([
        create_step("ToolCreationAgent"),
        create_step("ToolSelectionAgent"),
        create_step("SomeOtherAgent"),
        create_step("ToolCreationAgent"),
        create_step("ToolSelectionAgent"),
        create_step("FinalAgent"),
    ])
    guardrail.validate(plan)  # Should not raise

def test_tool_creation_last_step_invalid():
    """Tests an invalid plan where ToolCreationAgent is the last step."""
    guardrail = ToolExecutionFollowsToolCreationGuardrail()
    plan = ExecutionPlan([
        create_step("ToolCreationAgent"),
    ])
    with pytest.raises(GuardrailViolationError, match="uses ToolCreationAgent but is the last step"):
        guardrail.validate(plan)

def test_tool_creation_followed_by_wrong_agent_invalid():
    """Tests an invalid plan where ToolCreationAgent is followed by the wrong agent."""
    guardrail = ToolExecutionFollowsToolCreationGuardrail()
    plan = ExecutionPlan([
        create_step("ToolCreationAgent"),
        create_step("FinalAgent"),  # Invalid step after creation
    ])
    with pytest.raises(GuardrailViolationError, match="not ToolSelectionAgent"):
        guardrail.validate(plan)


@pytest.fixture
def manager() -> GuardrailManager:
    """Fixture to provide a GuardrailManager with both guardrails registered."""
    mgr = GuardrailManager()
    mgr.register_guardrail(ToolExecutionFollowsToolCreationGuardrail())
    return mgr

def test_manager_valid_plan(manager):
    """Tests the manager with a plan valid for both guardrails."""
    plan = ExecutionPlan([
        create_step("ToolCreationAgent"),
        create_step("ToolSelectionAgent"),
        create_step("FinalAgent"),
    ])
    manager.validate_plan(plan)  # Should not raise

def test_manager_invalid_plan_tool_creation(manager):
    """Tests the manager with a plan invalid for the ToolCreation guardrail."""
    plan = ExecutionPlan([
        create_step("ToolCreationAgent"),
        create_step("FinalAgent"),
    ])
    with pytest.raises(GuardrailViolationError, match="not ToolSelectionAgent"):
        manager.validate_plan(plan)

def test_manager_register_guardrail():
    """Tests registering a guardrail."""
    mgr = GuardrailManager()
    assert len(mgr.guardrails) == 0
    mgr.register_guardrail(ToolExecutionFollowsToolCreationGuardrail())
    assert len(mgr.guardrails) == 1
    assert isinstance(mgr.guardrails[0], ToolExecutionFollowsToolCreationGuardrail)
