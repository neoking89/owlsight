from typing import Dict

# exclude these agents from injection into prompts so that models never knows about them
EXCLUDED_AGENTS = ["ObservationAgent", "PlanAgent", "PlanValidationAgent", "FinalAgent"]

# Value in this dictionary is used to inject information about each agent into agents prompts
# This dictionary is populated dynamically at runtime from agent docstrings via BaseAgent.__init_subclass__ in core.py
AGENT_INFORMATION: Dict[str, str] = {}