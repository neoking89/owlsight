"""
Integration tests for the agent configuration management in the agentic framework.
These tests focus on how the config per agent functionality integrates with
the agent execution flow.
"""

import pytest
from unittest.mock import MagicMock, patch, call
from pathlib import Path

from owlsight.agentic.core import (
    BaseAgent, 
    AgentOrchestrator,
    PlanAgent,
    ObservationAgent
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


@patch('owlsight.agentic.helper_functions.create_temp_config_filename')
def test_orchestrator_execute_step_loads_agent_config(
    mock_create_temp, 
    orchestrator, 
    mock_manager
):
    """Test that _execute_step calls load_config_agent for agents in AGENT_INFORMATION."""
    # Setup
    mock_create_temp.return_value = "temp_config.json"
    BaseAgent.manager = mock_manager
    agent_name = "ObservationAgent"
    
    # Create context with a plan
    context = AgentContext(user_request="Test request")
    plan_step = PlanStep(
        description="Test step", 
        agent_name=agent_name,  # This is in AGENT_INFORMATION
        reason="Test reason"
    )
    context.execution_plan = ExecutionPlan([plan_step])
    
    # Mock the agent.execute to return success
    orchestrator.agents[agent_name].execute = MagicMock(
        return_value=StepResult(True, "Success")
    )
    
    # Create a spy on load_config_agent
    with patch.object(
        orchestrator.agents[agent_name], 
        'load_config_agent',
        wraps=orchestrator.agents[agent_name].load_config_agent
    ) as mock_load_config:
        # Execute
        orchestrator._execute_step(context, plan_step, 0)
        
        # Verify load_config_agent was called
        mock_load_config.assert_called_once()


def test_shared_config_file_between_agents(orchestrator, mock_manager):
    """
    Test that multiple agents can share the same temporary config file.
    """
    # Setup 
    # Create plan with multiple steps using different agents
    context = AgentContext(user_request="Test request")
    steps = [
        PlanStep("Step 1", "PlanAgent", "Reason 1"),
        PlanStep("Step 2", "ObservationAgent", "Reason 2"),
    ]
    context.execution_plan = ExecutionPlan(steps)
    
    # Mock agents' execute methods to return success
    for agent_name in ["PlanAgent", "ObservationAgent"]:
        orchestrator.agents[agent_name].execute = MagicMock(
            return_value=StepResult(True, "Success")
        )
    
    # Create spies on load_config_agent for ObservationAgent
    with patch.object(
        orchestrator.agents["ObservationAgent"], 
        'load_config_agent',
        wraps=orchestrator.agents["ObservationAgent"].load_config_agent
    ) as mock_obs_load:
        with patch.object(
            orchestrator.agents["PlanAgent"], 
            'load_config_agent',
            wraps=orchestrator.agents["PlanAgent"].load_config_agent
        ) as mock_plan:
            # Mock _set_classvar_config_per_agent to verify it stores the same file
            with patch.object(
                BaseAgent, 
                '_set_classvar_config_per_agent',
                wraps=BaseAgent._set_classvar_config_per_agent
            ) as mock_set_config:
                assert BaseAgent.temp_config_filename is None
                orchestrator._execute_step(context, steps[0], 0)
                # Verify temp filename is created
                assert BaseAgent.temp_config_filename.endswith(".json")
                config_name = BaseAgent.temp_config_filename
                orchestrator._execute_step(context, steps[1], 1)
            
                # Verify load_config_agent was called for both agents
                mock_obs_load.assert_called_once()
                mock_plan.assert_called_once()

                assert mock_set_config.call_count == 2
                assert config_name == BaseAgent.temp_config_filename
            