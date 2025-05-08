"""
Unit tests specifically for the additional_information handling in the agentic framework.
These tests focus on the step-specific information vs base context management.
"""

import pytest
from unittest.mock import MagicMock, patch

from owlsight.agentic.core import BaseAgent, AgentPrompt, AgentOrchestrator
from owlsight.agentic.models import AgentContext, PlanStep, StepResult

# Import fixtures from test_core for consistency
from tests.unit_tests.agentic.test_core import (
    mock_config_manager,
    mock_manager,
    test_agent,
    ConcreteAgent,
)


class TestStepSpecificAdditionalInformation:
    """Tests for the new step-specific additional information mechanism."""

    def test_initial_step_specific_info_empty(self, test_agent):
        """Test that a new agent has empty step-specific additional info."""
        assert test_agent.step_specific_additional_info == ""

    def test_set_additional_information_modifies_only_step_specific(self, test_agent, mock_config_manager):
        """Test that set_additional_information only modifies step-specific info, not config_manager."""
        # Setup initial state
        base_info = "This is base context that should remain untouched"
        mock_config_manager._config_store["agentic.additional_information"] = base_info
        
        # Add step-specific info
        new_info = "This is step-specific information"
        test_agent.set_additional_information(new_info)
        
        # Verify the base context in config_manager was not modified
        assert mock_config_manager._config_store["agentic.additional_information"] == base_info
        # Verify step-specific info was updated
        assert test_agent.step_specific_additional_info == new_info
        # Ensure config_manager.set was never called
        mock_config_manager.set.assert_not_called()

    def test_get_additional_information_combines_contexts(self, test_agent, mock_config_manager):
        """Test that get_additional_information combines base context and step-specific info."""
        # Setup initial state
        base_info = "This is base context from config_manager"
        mock_config_manager._config_store["agentic.additional_information"] = base_info
        
        # Set step-specific info
        step_info = "This is step-specific information"
        test_agent.step_specific_additional_info = step_info
        
        # Get combined info
        combined_info = test_agent.get_additional_information()
        
        # Verify combined result contains both pieces of information
        assert base_info in combined_info
        assert step_info in combined_info
        assert combined_info == f"{base_info}\n{step_info}"

    def test_get_additional_information_only_base_context(self, test_agent, mock_config_manager):
        """Test get_additional_information when only base context exists."""
        # Setup base context only
        base_info = "This is base context from config_manager"
        mock_config_manager._config_store["agentic.additional_information"] = base_info
        test_agent.step_specific_additional_info = ""
        
        # Get info
        result = test_agent.get_additional_information()
        
        # Verify result is just the base context
        assert result == base_info

    def test_get_additional_information_only_step_specific(self, test_agent, mock_config_manager):
        """Test get_additional_information when only step-specific info exists."""
        # Setup step-specific info only
        mock_config_manager._config_store["agentic.additional_information"] = ""
        step_info = "This is step-specific information"
        test_agent.step_specific_additional_info = step_info
        
        # Get info
        result = test_agent.get_additional_information()
        
        # Verify result is just the step-specific info
        assert result == step_info

    def test_clear_step_specific_additional_information(self, test_agent, mock_config_manager):
        """Test clearing step-specific additional information."""
        # Setup
        base_info = "This is base context that should remain"
        mock_config_manager._config_store["agentic.additional_information"] = base_info
        step_info = "This is step-specific info that should be cleared"
        test_agent.step_specific_additional_info = step_info
        
        # Clear step-specific info
        test_agent.clear_step_specific_additional_information()
        
        # Verify
        assert test_agent.step_specific_additional_info == ""
        assert mock_config_manager._config_store["agentic.additional_information"] == base_info
        assert test_agent.get_additional_information() == base_info

    @patch('owlsight.agentic.core.logger')
    def test_set_additional_information_updates_step_specific_only(self, mock_logger, test_agent, mock_config_manager):
        """Test that set_additional_information appends to existing step-specific info."""
        # Setup
        base_info = "Base context"
        mock_config_manager._config_store["agentic.additional_information"] = base_info
        
        # First update
        first_info = "First step-specific info"
        test_agent.set_additional_information(first_info)
        
        # Second update
        second_info = "Second step-specific info"
        test_agent.set_additional_information(second_info)
        
        # Verify
        expected_step_info = f"{first_info}\n{second_info}"
        assert test_agent.step_specific_additional_info == expected_step_info
        assert mock_config_manager._config_store["agentic.additional_information"] == base_info
        assert test_agent.get_additional_information() == f"{base_info}\n{expected_step_info}"
        
        # Verify logging
        assert mock_logger.debug.call_count >= 2  # Called for each set_additional_information
        mock_config_manager.set.assert_not_called()


class TestOrchestatorAdditionalInformationIntegration:
    """Tests for how AgentOrchestrator interacts with the new additional_information model."""
    
    def test_orchestrator_clears_step_specific_info_before_execution(self):
        """Test that AgentOrchestrator clears step-specific info before executing an agent."""
        # Setup
        mock_code_executor = MagicMock()
        mock_manager = MagicMock()
        mock_agent = MagicMock()
        mock_agent.step_specific_additional_info = "Existing step info that should be cleared"
        
        orchestrator = AgentOrchestrator(mock_code_executor, mock_manager)
        orchestrator.agents = {"TestAgent": mock_agent}
        
        # Create a test step and context
        step = PlanStep(description="Test step", agent_name="TestAgent", reason="Testing")
        context = AgentContext(user_request="Test request")
        
        # Initialize execution_plan in the context (needed by _execute_step)
        from owlsight.agentic.models import ExecutionPlan
        context.execution_plan = ExecutionPlan(steps=[step])
        
        # Mock agent.execute to return success
        mock_agent.execute.return_value = StepResult(True, "Success")
        
        # Call _execute_step
        orchestrator._execute_step(context, step, 0)
        
        # Verify clear_step_specific_additional_information was called before execute
        mock_agent.clear_step_specific_additional_information.assert_called_once()
        # Verify the execution order: clear first, then execute
        assert mock_agent.method_calls[0][0] == 'clear_step_specific_additional_information'
        assert mock_agent.method_calls[1][0] == 'execute'
