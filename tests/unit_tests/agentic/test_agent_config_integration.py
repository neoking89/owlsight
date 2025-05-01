"""
Integration tests for the agent configuration management in the agentic framework.
These tests focus on how the config per agent functionality integrates with
the agent execution flow.
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from owlsight.agentic.core import (
    BaseAgent, 
    AgentOrchestrator,
    PlanAgent,
    ObservationAgent,
)
from owlsight.agentic.constants import AGENT_INFORMATION
from owlsight.agentic.models import AgentContext, ExecutionPlan, PlanStep, StepResult


@pytest.fixture(autouse=True)
def reset_baseagent_class_variables():
    """Reset all BaseAgent class variables before each test."""
    BaseAgent.temp_config_filename = None
    BaseAgent.manager = None
    BaseAgent.config_per_agent = None
    yield
    # Reset after test completes too
    BaseAgent.temp_config_filename = None
    BaseAgent.manager = None
    BaseAgent.config_per_agent = None


@pytest.fixture
def mock_code_executor():
    """Create a mock code executor."""
    mock = MagicMock()
    mock.globals_dict = {}
    return mock


@pytest.fixture
def mock_manager():
    """Create a mock TextGenerationManager."""
    mock = MagicMock()
    mock.config_manager = MagicMock()
    mock._last_loaded_config = None
    mock.generate.return_value = "{}"  # Default response for LLM calls
    return mock


@pytest.fixture
def orchestrator(mock_code_executor, mock_manager):
    """Create an AgentOrchestrator with mocked dependencies."""
    return AgentOrchestrator(
        mock_code_executor, 
        mock_manager,
        max_retries_per_step=1,
        max_replans=1
    )


def test_get_config_per_agent_with_config(mock_manager):
    """Test _get_config_per_agent when config exists."""
    # Setup
    agent = PlanAgent()
    agent.manager = mock_manager
    
    # Mock config value
    expected_config = {
        "PlanAgent": "plan_config.json",
        "ObservationAgent": "observation_config.json"
    }
    mock_manager.config_manager.get.return_value = expected_config
    
    # Execute
    result = agent._get_config_per_agent()
    
    # Verify
    assert result == expected_config
    mock_manager.config_manager.get.assert_called_once_with("agentic.config_per_agent", {})


def test_get_config_per_agent_no_config_manager():
    """Test _get_config_per_agent when no config_manager is available."""
    # Setup
    agent = PlanAgent()
    agent.manager = MagicMock()
    agent.manager.config_manager = None  # No config_manager
    
    # Execute
    result = agent._get_config_per_agent()
    
    # Verify
    assert result == {}


def test_agent_config_path_exists_true(mock_manager):
    """Test _agent_config_path_exists when path exists."""
    # Setup
    agent = ObservationAgent()
    agent.manager = mock_manager
    
    # Mock config value
    mock_manager.config_manager.get.return_value = {
        "ObservationAgent": "existing_config.json"
    }
    
    # Mock Path.exists
    with patch.object(Path, 'exists', return_value=True):
        # Execute
        result = agent._agent_config_path_exists()
        
        # Verify
        assert result is True


def test_agent_config_path_exists_false_no_entry(mock_manager):
    """Test _agent_config_path_exists when agent has no config entry."""
    # Setup
    agent = ObservationAgent()
    agent.manager = mock_manager
    
    # Mock empty config
    mock_manager.config_manager.get.return_value = {}
    
    # Execute
    result = agent._agent_config_path_exists()
    
    # Verify
    assert result is False


def test_agent_config_path_exists_false_nonexistent_path(mock_manager):
    """Test _agent_config_path_exists when config path doesn't exist."""
    # Setup
    agent = ObservationAgent()
    agent.manager = mock_manager
    
    # Mock config value
    mock_manager.config_manager.get.return_value = {
        "ObservationAgent": "nonexistent_config.json"
    }
    
    # Mock Path.exists
    with patch.object(Path, 'exists', return_value=False):
        # Execute
        result = agent._agent_config_path_exists()
        
        # Verify
        assert result is False


@patch('owlsight.agentic.core.get_default_config_on_startup_path')
def test_shared_config_file_between_agents(mock_get_default_path, orchestrator, mock_manager):
    """
    Test that multiple agents can share the same temporary config file.
    """
    # Setup
    mock_get_default_path.return_value = None # Ensure default path doesn't interfere
    # Create plan with multiple steps using different agents
    context = AgentContext(user_request="Test request")
    steps = [
        PlanStep("Step 1", "PlanAgent", "Reason 1"),
        PlanStep("Step 2", "ObservationAgent", "Reason 2"),
    ]
    context.execution_plan = ExecutionPlan(steps)

    # 1. Mock _execute_impl for both agents
    mock_plan_execute_impl = patch.object(
        orchestrator.agents["PlanAgent"], '_execute_impl',
        return_value=StepResult(True, "Success")
    )
    mock_obs_execute_impl = patch.object(
        orchestrator.agents["ObservationAgent"], '_execute_impl',
        return_value=StepResult(True, "Success")
    )

    # 3. Spy on ObservationAgent's load_config_agent
    spy_obs_load = patch.object(
        orchestrator.agents["ObservationAgent"], 'load_config_agent',
        wraps=orchestrator.agents["ObservationAgent"].load_config_agent
    )
    # 4. Spy on PlanAgent's load_config_agent
    spy_plan_load = patch.object(
        orchestrator.agents["PlanAgent"], 'load_config_agent',
        wraps=orchestrator.agents["PlanAgent"].load_config_agent
    )

    # 2. Mock create_temp_config_filename directly
    mock_create_temp = patch(
        'owlsight.agentic.core.create_temp_config_filename',
        return_value="shared_temp_config.json"
    )
    # 5. Mock save_config
    mock_save_config = patch.object(mock_manager, 'save_config')
    # 6. Mock manager.get specifically for agentic.config_per_agent
    mock_get_config = patch.object(
        mock_manager.config_manager, 'get',
        return_value={} # Ensure it returns dict, not mock
    )

    # Use context managers for all patches
    with mock_plan_execute_impl, mock_obs_execute_impl, \
         spy_obs_load as mock_obs_load_spy, spy_plan_load as mock_plan_load_spy, \
         mock_create_temp as mock_create_temp_func, mock_save_config as mock_save_config_func, \
         mock_get_config as mock_get_config_func:

        try:
            assert BaseAgent.temp_config_filename is None
            assert BaseAgent.config_per_agent is None
            # Execute first step (PlanAgent)
            orchestrator._execute_step(context, steps[0], 0)

            # Verify temp filename was created and stored
            assert BaseAgent.temp_config_filename == "shared_temp_config.json"
            mock_create_temp_func.assert_called_once()
            mock_save_config_func.assert_called_once_with("shared_temp_config.json")
            mock_plan_load_spy.assert_called_once() # PlanAgent load called

            # Reset mocks for second step if needed (though save/create shouldn't be called again)
            mock_create_temp_func.reset_mock()
            mock_save_config_func.reset_mock()

            # Execute second step (ObservationAgent)
            orchestrator._execute_step(context, steps[1], 1)

            # Verify temp filename is unchanged
            assert BaseAgent.temp_config_filename == "shared_temp_config.json"
            mock_obs_load_spy.assert_called_once() # ObservationAgent load called
            mock_create_temp_func.assert_not_called() # Temp file not created again
            mock_save_config_func.assert_not_called() # Save not called again

            # Verify manager.get was called (once per agent load_config call)
            # It's called twice per load_config: once directly, once via _agent_config_path_exists
            assert mock_get_config_func.call_count == 4 # Once in PlanAgent, once in ObsAgent (each calls it twice)
            mock_get_config_func.assert_called_with("agentic.config_per_agent", {})

            # Verify final state of config_per_agent if necessary
            expected_config = {
                agent_name: "shared_temp_config.json"
                for agent_name in AGENT_INFORMATION.keys()
            }
            # Add any non-AGENT_INFORMATION agents if they exist in the orchestrator
            for agent_name in orchestrator.agents:
                if agent_name not in expected_config:
                    # Assuming they should also share the temp config if no specific one is defined
                    expected_config[agent_name] = "shared_temp_config.json"

            # Accessing the class variable directly for assertion
            assert BaseAgent.config_per_agent == expected_config

        finally:
            # Reset class variables to avoid state leakage
            BaseAgent.temp_config_filename = None
            BaseAgent.config_per_agent = None