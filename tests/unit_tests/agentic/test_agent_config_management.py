"""
Tests for the agent configuration management functionality in core.py.
"""

import os
import pytest
from unittest.mock import MagicMock, patch, call

from owlsight.agentic.core import BaseAgent, AgentPrompt
from owlsight.agentic.constants import AGENT_INFORMATION
from owlsight.agentic.models import AgentContext
from owlsight.configurations.config_manager import ConfigManager
from owlsight.processors.text_generation_manager import TextGenerationManager


class TestAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing."""
    
    def execute(self, context: AgentContext):
        return "Test executed"


@pytest.fixture
def mock_config_manager():
    """Mock ConfigManager with a configurable store."""
    mock_cm = MagicMock(spec=ConfigManager)
    mock_cm.get.return_value = {}  # Default empty dict for config_per_agent
    return mock_cm


@pytest.fixture
def mock_manager(mock_config_manager):
    """Mock TextGenerationManager with the mock ConfigManager."""
    mock_mgr = MagicMock(spec=TextGenerationManager)
    mock_mgr.config_manager = mock_config_manager
    mock_mgr._last_loaded_config = None  # Add this attribute for tests
    return mock_mgr


@pytest.fixture
def test_agent(mock_manager):
    """Create a test agent instance with mocked dependencies."""
    agent = TestAgent("TestAgent", AgentPrompt("Test prompt"))
    agent.manager = mock_manager
    
    # Reset class variables to ensure clean test state
    BaseAgent.temp_config_filename = None
    BaseAgent.config_per_agent = None
    BaseAgent.manager = mock_manager
    
    return agent


def test_class_variables_initial_state():
    """Test that class variables are properly initialized."""
    # Reset to ensure test isolation
    BaseAgent.temp_config_filename = None
    BaseAgent.config_per_agent = None
    
    assert BaseAgent.temp_config_filename is None
    assert BaseAgent.config_per_agent is None


def test_reset_config_per_agent_classvars_without_file(test_agent):
    """Test reset_config_per_agent_classvars when no temp file exists."""
    # Set some values
    BaseAgent.config_per_agent = {"TestAgent": "test_config.json"}
    BaseAgent.temp_config_filename = None  # No file to remove
    
    # Call reset
    BaseAgent.reset_config_per_agent_classvars()
    
    # Verify reset
    assert BaseAgent.config_per_agent is None
    assert BaseAgent.temp_config_filename is None


@patch('os.remove')
def test_reset_config_per_agent_classvars_with_file(mock_remove, test_agent):
    """Test reset_config_per_agent_classvars when temp file exists."""
    # Set some values
    BaseAgent.config_per_agent = {"TestAgent": "test_config.json"}
    BaseAgent.temp_config_filename = "temp_config_1234.json"
    
    # Call reset
    BaseAgent.reset_config_per_agent_classvars()
    
    # Verify reset
    assert BaseAgent.config_per_agent is None
    assert BaseAgent.temp_config_filename is None
    mock_remove.assert_called_once_with("temp_config_1234.json")


@patch('owlsight.agentic.core.create_temp_config_filename')
def test_set_classvar_config_per_agent_first_call(mock_create_temp, test_agent):
    """Test _set_classvar_config_per_agent when called for the first time."""
    # Setup
    mock_create_temp.return_value = "tmp_test_config_123.json"
    BaseAgent.manager = test_agent.manager
    input_config = {"ObservationAgent": "existing_config.json"}
    
    # Execute
    result = BaseAgent._set_classvar_config_per_agent(input_config)
    
    # Verify temp filename created
    assert BaseAgent.temp_config_filename == "tmp_test_config_123.json"
    mock_create_temp.assert_called_once()
    
    # Verify config_per_agent populated with all agents from AGENT_INFORMATION
    for agent_name in AGENT_INFORMATION.keys():
        if agent_name not in input_config:
            assert result[agent_name] == "tmp_test_config_123.json"
    
    # Verify existing config entry preserved
    assert result["ObservationAgent"] == "existing_config.json"
    
    # Verify config saved
    test_agent.manager.save_config.assert_called_once_with("tmp_test_config_123.json")


def test_set_classvar_config_per_agent_subsequent_call(test_agent):
    """Test _set_classvar_config_per_agent when called after initialization."""
    # Setup - simulate previous initialization
    BaseAgent.temp_config_filename = "existing_temp.json"
    BaseAgent.config_per_agent = {"AgentA": "config1.json", "AgentB": "temp.json"}
    BaseAgent.manager = test_agent.manager
    
    # New config to merge
    new_config = {"AgentA": "new_config.json", "AgentC": "config3.json"}
    
    # Execute
    result = BaseAgent._set_classvar_config_per_agent(new_config)
    
    # Verify no changes to class variables - should use existing config
    assert BaseAgent.temp_config_filename == "existing_temp.json"
    assert result is BaseAgent.config_per_agent
    assert not test_agent.manager.save_config.called  # Shouldn't save again


def test_load_config_agent_with_existing_config(test_agent):
    """Test load_config_agent when a config exists for the agent."""
    # Setup
    test_agent.manager.config_manager.get.return_value = {"TestAgent": "agent_config.json"}
    with patch.object(test_agent, '_agent_config_path_exists', return_value=True):
        with patch.object(BaseAgent, '_set_classvar_config_per_agent', return_value={"TestAgent": "agent_config.json"}):
            # Set different last loaded config
            test_agent.manager._last_loaded_config = "different_config.json"
            
            # Execute
            test_agent.load_config_agent()
            
            # Verify config loading attempted
            test_agent.manager.load_config.assert_called_once_with("agent_config.json")


def test_load_config_agent_with_same_config(test_agent):
    """Test load_config_agent when the config is already loaded."""
    # Setup
    test_agent.manager.config_manager.get.return_value = {"TestAgent": "agent_config.json"}
    with patch.object(test_agent, '_agent_config_path_exists', return_value=True):
        with patch.object(BaseAgent, '_set_classvar_config_per_agent', return_value={"TestAgent": "agent_config.json"}):
            # Set same last loaded config
            test_agent.manager._last_loaded_config = "agent_config.json"
            
            # Execute
            test_agent.load_config_agent()
            
            # Verify no config loading attempted
            assert not test_agent.manager.load_config.called


def test_load_config_agent_with_nonexistent_config(test_agent):
    """Test load_config_agent when the config file doesn't exist."""
    # Setup
    test_agent.manager.config_manager.get.return_value = {"TestAgent": "nonexistent.json"}
    with patch.object(test_agent, '_agent_config_path_exists', return_value=False):
        with patch.object(BaseAgent, '_set_classvar_config_per_agent', return_value={"TestAgent": "nonexistent.json"}):
            # Execute
            test_agent.load_config_agent()
            
            # Verify no config loading attempted
            assert not test_agent.manager.load_config.called


@patch('os.remove')
def test_agent_orchestrator_cleanup(mock_remove):
    """Test that the agent orchestrator cleans up config vars."""
    # This is testing indirectly since we can see the finally block calls reset_config_per_agent_classvars
    
    # Setup
    from owlsight.agentic.core import AgentOrchestrator
    
    # Set some values to be cleaned
    BaseAgent.config_per_agent = {"TestAgent": "test_config.json"}
    BaseAgent.temp_config_filename = "temp_to_remove.json"
    
    # Create mock dependencies for AgentOrchestrator
    mock_code_executor = MagicMock()
    mock_manager = MagicMock()
    
    # Create orchestrator
    orchestrator = AgentOrchestrator(mock_code_executor, mock_manager)
    
    # Patch _plan and _execute to avoid actual processing
    with patch.object(orchestrator, '_plan', return_value=False):
        # Test that process_user_request calls reset_config_per_agent_classvars
        orchestrator.process_user_request("Test request")
        
        # Verify cleanup performed
        assert BaseAgent.config_per_agent is None
        assert BaseAgent.temp_config_filename is None
        if BaseAgent.temp_config_filename:  # This would only be true if reset didn't run
            mock_remove.assert_called_once_with("temp_to_remove.json")
