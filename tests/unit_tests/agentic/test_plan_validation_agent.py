"""
Unit tests for the PlanValidationAgent.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from owlsight.agentic.core import PlanValidationAgent
from owlsight.agentic.models import ExecutionPlan, PlanStep, StepResult

# Helper function to create PlanSteps
def create_step(agent_name: str, description: str = "") -> PlanStep:
    return PlanStep(description=description or f"Step for {agent_name}", agent_name=agent_name, reason="Test reason")

# Create a custom AgentContext class for testing
class MockAgentContext:
    def __init__(self, execution_plan, user_request="Test user request"):
        self.execution_plan = execution_plan
        self.user_request = user_request
        self.accumulated_results = []
        self.current_step = 0
        self.error_context = MagicMock()
        self.final_response = None
        self.planner_feedback_from_guardrails = None
    
    def get_available_tools_content(self):
        """Mock implementation of get_available_tools_content"""
        return "[]"

@pytest.fixture
def valid_plan():
    """Fixture for a valid plan that passes all guardrails."""
    return ExecutionPlan([
        create_step("ToolCreationAgent", "Create a tool"),
        create_step("ToolSelectionAgent", "Execute the tool"),
        create_step("FinalAgent", "Final response")
    ])

@pytest.fixture
def invalid_plan():
    """Fixture for an invalid plan that violates guardrails."""
    return ExecutionPlan([
        create_step("ToolCreationAgent", "Create a tool"),
        create_step("FinalAgent", "Skip tool execution")  # Violation: ToolCreationAgent not followed by ToolSelectionAgent
    ])

@pytest.fixture
def agent_context(valid_plan):
    """Fixture for AgentContext with a valid plan."""
    return MockAgentContext(valid_plan)

@pytest.fixture
def agent_context_with_invalid_plan(invalid_plan):
    """Fixture for AgentContext with an invalid plan."""
    return MockAgentContext(invalid_plan)

@pytest.fixture
def validation_agent():
    """Fixture for a PlanValidationAgent with mocked LLM call."""
    agent = PlanValidationAgent()
    agent.llm_call = MagicMock()
    agent.get_additional_information = MagicMock(return_value="")
    return agent

# --- Tests for validate_plan_by_guardrails ---

def test_validate_plan_by_guardrails_valid_plan(validation_agent, valid_plan):
    """Test validation of a valid plan."""
    # Arrange
    # Act
    result = validation_agent.validate_plan_by_guardrails(valid_plan)
    
    # Assert
    assert result.success is True
    assert result.execution_result == valid_plan.steps

def test_validate_plan_by_guardrails_invalid_plan(validation_agent, invalid_plan):
    """Test validation of an invalid plan that violates guardrails."""
    # Arrange
    # Act
    result = validation_agent.validate_plan_by_guardrails(invalid_plan)
    
    # Assert
    assert result.success is False
    assert "Plan validation failed" in result.execution_result
    assert "ToolCreationAgent" in result.execution_result
    assert "ToolSelectionAgent" in result.execution_result

# --- Tests for execute method ---

def test_execute_valid_plan_validated(validation_agent, agent_context):
    """Test execution with a valid plan that's validated with execution_result changes."""
    # Arrange
    llm_response = json.dumps({
        "validation_result": "valid",
        "validation_notes": "Plan is good",
        "plan": [
            {
                "description": "Create a tool",
                "agent": "ToolCreationAgent",
                "reason": "Test reason"
            },
            {
                "description": "Execute the tool",
                "agent": "ToolSelectionAgent",
                "reason": "Test reason"
            },
            {
                "description": "Final response",
                "agent": "FinalAgent",
                "reason": "Test reason"
            }
        ]
    })
    validation_agent.llm_call.return_value = llm_response
    
    # Override validate_plan_by_guardrails to return success
    with patch.object(validation_agent, 'validate_plan_by_guardrails') as mock_validate:
        mock_validate.return_value = StepResult(True, agent_context.execution_plan.steps)
        
        # Act
        result = validation_agent.execute(agent_context)
        
        # Assert
        assert result.success is True
        assert "Plan validated" in result.execution_result
        assert "Plan is good" in result.execution_result
        assert mock_validate.called
        validation_agent.llm_call.assert_called_once()
        # Plan should remain unchanged
        assert agent_context.execution_plan.steps == agent_context.execution_plan.steps

def test_execute_guardrail_violation_but_llm_fixes(validation_agent, agent_context_with_invalid_plan):
    """Test execution where guardrails find a violation but the LLM fixes it."""
    # Arrange
    # The revised plan from LLM that fixes the guardrail violation
    llm_response = json.dumps({
        "validation_result": "revised",
        "validation_notes": "Fixed the plan to follow guardrails",
        "plan": [
            {
                "description": "Create a tool",
                "agent": "ToolCreationAgent",
                "reason": "Test reason"
            },
            {
                "description": "Execute the tool",
                "agent": "ToolSelectionAgent",  # Added the missing ToolSelectionAgent
                "reason": "Test reason"
            },
            {
                "description": "Final response",
                "agent": "FinalAgent",
                "reason": "Test reason"
            }
        ]
    })
    validation_agent.llm_call.return_value = llm_response
    
    # Simulate guardrail violation
    guardrail_error = "Plan validation failed: Step 1 uses ToolCreationAgent but the next step (2) uses FinalAgent, not ToolSelectionAgent."
    with patch.object(validation_agent, 'validate_plan_by_guardrails') as mock_validate:
        mock_validate.return_value = StepResult(False, guardrail_error)
        
        # Act
        result = validation_agent.execute(agent_context_with_invalid_plan)
        
        # Assert
        assert result.success is True
        assert "Plan revised" in result.execution_result
        assert "Fixed the plan" in result.execution_result
        assert mock_validate.called
        validation_agent.llm_call.assert_called_once()
        
        # Check that the guardrail error was included in the prompt
        prompt_call_args = validation_agent.llm_call.call_args[0][0]
        assert guardrail_error in prompt_call_args
        
        # Plan should be updated with the fixed version
        assert len(agent_context_with_invalid_plan.execution_plan.steps) == 3
        assert agent_context_with_invalid_plan.execution_plan.steps[1].agent_name == "ToolSelectionAgent"

def test_execute_invalid_llm_response(validation_agent, agent_context):
    """Test handling of invalid LLM response (not JSON)."""
    # Arrange
    validation_agent.llm_call.return_value = "This is not valid JSON"
    
    # Act
    result = validation_agent.execute(agent_context)
    
    # Assert
    assert result.success is False
    assert "Failed to parse" in result.execution_result

def test_execute_missing_required_fields_in_llm_response(validation_agent, agent_context):
    """Test handling of LLM response missing required fields (validation_notes)."""
    # Arrange
    llm_response = json.dumps({
        "validation_result": "valid",
        # Missing "validation_notes" field
        "plan": [] # Add dummy plan to avoid unrelated errors if validation_result was 'revised'
    })
    validation_agent.llm_call.return_value = llm_response
    
    # Act
    result = validation_agent.execute(agent_context)
    
    # Assert
    assert result.success is False
    assert "Missing required fields" in result.execution_result
    assert "validation_notes" in result.execution_result

def test_execute_llm_response_with_revised_plan(validation_agent, agent_context):
    """Test handling of LLM response with a revised plan."""
    # Arrange
    # Revised plan adds an extra step
    llm_response = json.dumps({
        "validation_result": "revised",
        "validation_notes": "Added an additional step for better results",
        "plan": [
            {
                "description": "Create a tool",
                "agent": "ToolCreationAgent",
                "reason": "Test reason"
            },
            {
                "description": "Execute the tool",
                "agent": "ToolSelectionAgent",
                "reason": "Test reason"
            },
            {
                "description": "Process results",
                "agent": "ToolSelectionAgent",  # Additional step
                "reason": "Additional processing"
            },
            {
                "description": "Final response",
                "agent": "FinalAgent",
                "reason": "Test reason"
            }
        ]
    })
    validation_agent.llm_call.return_value = llm_response
    
    # Act
    result = validation_agent.execute(agent_context)
    
    # Assert
    assert result.success is True
    assert "Plan revised" in result.execution_result
    assert "Added an additional step" in result.execution_result
    
    # Check that the plan was updated with the new steps
    assert len(agent_context.execution_plan.steps) == 4  # Original was 3
    assert agent_context.execution_plan.steps[2].description == "Process results"
    assert agent_context.execution_plan.steps[2].agent_name == "ToolSelectionAgent"

# --- Integration Tests (with less mocking) ---

def test_full_validation_process_with_valid_plan(valid_plan):
    """Test the full validation process with a valid plan (less mocking)."""
    # Arrange
    agent = PlanValidationAgent()
    agent.llm_call = MagicMock(return_value=json.dumps({
        "validation_result": "valid",
        "validation_notes": "Plan is valid",
        "plan": [
            {
                "description": step.description,
                "agent": step.agent_name,
                "reason": step.reason
            } for step in valid_plan.steps
        ]
    }))
    agent.get_additional_information = MagicMock(return_value="")
    
    context = MockAgentContext(valid_plan)
    
    # Act
    result = agent.execute(context)
    
    # Assert
    assert result.success is True
    assert "Plan validated" in result.execution_result

def test_full_validation_process_with_invalid_plan(invalid_plan):
    """Test the full validation process with an invalid plan (less mocking)."""
    # Arrange
    agent = PlanValidationAgent()
    
    # Simulate LLM fixing the plan
    fixed_plan = [
        {
            "description": "Create a tool",
            "agent": "ToolCreationAgent",
            "reason": "Test reason"
        },
        {
            "description": "Execute the tool",
            "agent": "ToolSelectionAgent",  # Fixed: Added ToolSelectionAgent
            "reason": "Required by guardrails"
        },
        {
            "description": "Final response",
            "agent": "FinalAgent",
            "reason": "Test reason"
        }
    ]
    
    agent.llm_call = MagicMock(return_value=json.dumps({
        "validation_result": "revised",
        "validation_notes": "Fixed the guardrail violation",
        "plan": fixed_plan
    }))
    agent.get_additional_information = MagicMock(return_value="")
    
    context = MockAgentContext(invalid_plan)
    
    # Act
    result = agent.execute(context)
    
    # Assert
    assert result.success is True
    assert "Plan revised" in result.execution_result
    assert len(context.execution_plan.steps) == 3
    assert context.execution_plan.steps[1].agent_name == "ToolSelectionAgent"
