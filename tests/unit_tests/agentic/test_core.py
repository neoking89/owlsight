"""
Unit tests for the core agentic components like AgentOrchestrator.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from owlsight.agentic.core import AgentOrchestrator, BaseAgent, AgentPrompt
from owlsight.agentic.models import AgentContext, ExecutionPlan, PlanStep
from owlsight.configurations.config_manager import ConfigManager
from owlsight.processors.text_generation_manager import TextGenerationManager

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

        print("Calling orchestrator.process_user_request...")
        final_result = orchestrator.process_user_request(user_question)
        print(f"orchestrator.process_user_request returned: {final_result}")

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


# Create a concrete implementation of BaseAgent for testing
class ConcreteAgent(BaseAgent):
    def execute(self, context): # type: ignore
        raise NotImplementedError

    def __init__(self, manager):
        super().__init__("ConcreteAgent", AgentPrompt("Test prompt"))
        # Directly assign manager for testing, bypassing ClassVar setup complexity if needed
        self.manager = manager


@pytest.fixture
def mock_config_manager():
    """Fixture for a mock ConfigManager with a simple dict store."""
    manager = MagicMock(spec=ConfigManager)
    manager._config_store = {} # Use a simple dict to simulate storage

    def mock_get(key, default=None):
        return manager._config_store.get(key, default)

    def mock_set(key, value):
        manager._config_store[key] = value

    manager.get.side_effect = mock_get
    manager.set.side_effect = mock_set
    return manager

@pytest.fixture
def mock_manager(mock_config_manager):
    """Fixture for a mock TextGenerationManager holding the mock ConfigManager."""
    manager = MagicMock(spec=TextGenerationManager)
    manager.config_manager = mock_config_manager
    return manager

@pytest.fixture
def test_agent(mock_manager):
    """Fixture for a concrete agent instance with mocked manager."""
    # Set the ClassVar for the duration of the test if needed, or inject manager
    # BaseAgent.manager = mock_manager # Option 1: Modify ClassVar (might affect other tests)
    agent = ConcreteAgent(manager=mock_manager) # Option 2: Inject via constructor/attribute
    return agent

# --- Tests for get_additional_information ---

def test_get_additional_information_empty(test_agent, mock_config_manager):
    """Test getting info when none is set."""
    mock_config_manager._config_store = {}
    assert test_agent.get_additional_information() == ""

def test_get_additional_information_existing(test_agent, mock_config_manager):
    """Test getting existing info."""
    existing_info_str = "key1: value1\nkey2: value2"
    mock_config_manager._config_store["agentic.additional_information"] = existing_info_str
    assert test_agent.get_additional_information() == existing_info_str

def test_get_additional_information_no_config_manager(test_agent):
    """Test behavior when config_manager is missing."""
    # Simulate missing config_manager by accessing attribute that doesn't exist
    # or by setting it to None if the structure allows
    with patch.object(test_agent, 'manager', MagicMock(config_manager=None)):
        assert test_agent.get_additional_information() == ""

# --- Tests for set_additional_information ---

def test_set_additional_information_initial(test_agent, mock_config_manager):
    """Test setting info for the first time with a string."""
    mock_config_manager._config_store = {}
    new_info_str = "User ID: 123, Task: testing"
    test_agent.set_additional_information(new_info_str)
    assert mock_config_manager._config_store.get("agentic.additional_information") == new_info_str
    mock_config_manager.set.assert_called_once_with("agentic.additional_information", new_info_str)

def test_set_additional_information_update(test_agent, mock_config_manager):
    """Test updating existing info by appending a string."""
    initial_info_str = "First line of info."
    mock_config_manager._config_store["agentic.additional_information"] = initial_info_str

    update_str = "Second line appended."
    test_agent.set_additional_information(update_str)

    expected_full_str = f"{initial_info_str}\n{update_str}"

    assert mock_config_manager._config_store.get("agentic.additional_information") == expected_full_str
    mock_config_manager.set.assert_called_once_with("agentic.additional_information", expected_full_str)


@patch('owlsight.agentic.core.logger')
def test_set_additional_information_append_to_empty(mock_logger, test_agent, mock_config_manager):
    """Test appending when the existing stored value is empty."""
    mock_config_manager._config_store["agentic.additional_information"] = ""

    new_info_str = "First piece of info"
    test_agent.set_additional_information(new_info_str)

    assert mock_config_manager._config_store.get("agentic.additional_information") == new_info_str
    mock_config_manager.set.assert_called_once_with("agentic.additional_information", new_info_str)
    mock_logger.warning.assert_not_called()


@patch('owlsight.agentic.core.logger')
def test_set_additional_information_invalid_input_none(mock_logger, test_agent, mock_config_manager):
    """Test calling set_additional_information with None input."""
    initial_info = "Initial data"
    mock_config_manager._config_store["agentic.additional_information"] = initial_info

    test_agent.set_additional_information(None) # type: ignore

    # Check that the config wasn't updated and set wasn't called
    assert mock_config_manager._config_store["agentic.additional_information"] == initial_info
    mock_config_manager.set.assert_not_called()
    # Check that a warning was logged
    mock_logger.warning.assert_called_once()
    assert "invalid input (must be non-empty string)" in mock_logger.warning.call_args[0][0]


@patch('owlsight.agentic.core.logger')
def test_set_additional_information_invalid_input_empty_string(mock_logger, test_agent, mock_config_manager):
    """Test calling set_additional_information with an empty string input."""
    initial_info = "Initial data"
    mock_config_manager._config_store["agentic.additional_information"] = initial_info

    test_agent.set_additional_information("")

    # Check that the config wasn't updated and set wasn't called
    assert mock_config_manager._config_store["agentic.additional_information"] == initial_info
    mock_config_manager.set.assert_not_called()
    # Check that a warning was logged
    mock_logger.warning.assert_called_once()
    assert "invalid input (must be non-empty string)" in mock_logger.warning.call_args[0][0]

@patch('owlsight.agentic.core.logger')
def test_set_additional_information_invalid_input_integer(mock_logger, test_agent, mock_config_manager):
    """Test calling set_additional_information with a non-string input (int)."""
    initial_info = "Initial data"
    mock_config_manager._config_store["agentic.additional_information"] = initial_info

    test_agent.set_additional_information(123) # type: ignore

    # Check that the config wasn't updated and set wasn't called
    assert mock_config_manager._config_store["agentic.additional_information"] == initial_info
    mock_config_manager.set.assert_not_called()
    # Check that a warning was logged
    mock_logger.warning.assert_called_once()
    assert "invalid input (must be non-empty string)" in mock_logger.warning.call_args[0][0]
    assert "<class 'int'>" in mock_logger.warning.call_args[0][0]


@patch('owlsight.agentic.core.logger')
def test_set_additional_information_no_config_manager(mock_logger, test_agent):
    """Test setting info when config_manager is missing."""
    # Ensure the manager attached to the agent for this test has no config_manager
    test_agent.manager = MagicMock(config_manager=None)

    test_agent.set_additional_information("some info")

    mock_logger.warning.assert_called_once_with("ConfigManager not found on manager when setting additional information. Cannot save.")
    # No calls to a non-existent set method
