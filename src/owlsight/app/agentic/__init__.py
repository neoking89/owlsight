"""
Agentic framework module for Owlsight.

This module provides a framework for creating and orchestrating AI agents
that can work together to solve complex tasks. The framework supports
tool creation, tool selection, planning, and response synthesis.
"""

# Re-export all public components for backward compatibility
from owlsight.app.agentic.helpers import (
    get_agent_information,
    get_available_tools,
    parse_tool_response,
    execute_tool,
)

from owlsight.app.agentic.models import (
    ToolResult,
    EventType,
    StepResult,
    PlanStep,
    StepErrorInfo,
    ErrorContext,
    ExecutionPlan,
    AgentPrompt,
    AgentContext,
)

from owlsight.app.agentic.agents.base import BaseAgent
from owlsight.app.agentic.agents.planner import PlannerAgent
from owlsight.app.agentic.agents.tool_creation import ToolCreationAgent
from owlsight.app.agentic.agents.tool_selection import ToolSelectionAgent
from owlsight.app.agentic.agents.observation import ObservationAgent
from owlsight.app.agentic.agents.final import FinalAgent

from owlsight.app.agentic.orchestrator import AgentOrchestrator

# Preserve the AGENT_INFORMATION constant
from owlsight.app.agentic.prompts import AGENT_INFORMATION
