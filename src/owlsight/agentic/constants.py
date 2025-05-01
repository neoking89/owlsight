from typing import Dict

# exclude these agents from injection into prompts so that models never knows about them
EXCLUDED_AGENTS = ["ObservationAgent", "PlanAgent", "PlanValidationAgent"]

# Value in this dictionary is used to inject information about each agent into agents prompts
AGENT_INFORMATION: Dict[str, str] = {
    "PlanAgent": "Creates a complete execution plan based on the user's request",
    "PlanValidationAgent": "Validates the execution plan created by PlanAgent and ensures it aligns with the original request.",
    "ToolSelectionAgent": "Chooses & executes ONE existing tool. The tool output will be summarized before the next step.",
    "ToolCreationAgent": "Writes a python function, serving as a *new* tool. MUST be followed immediately by ToolSelectionAgent to run the new tool.",
    "ObservationAgent": "Observes the output of the previous step and provides a summary.",
    "FinalAgent": "Drafts the ultimate reply based on all accumulated context. Always the last step.",
}