from typing import Dict

AGENT_INFORMATION: Dict[str, str] = {
    "ToolSelectionAgent": "Chooses & executes ONE existing tool. The tool output will be summarized before the next step.",
    "ToolCreationAgent": "Writes a python function, serving as a *new* tool. MUST be followed immediately by ToolSelectionAgent to run the new tool.",
    "FinalAgent": "Drafts the ultimate reply based on all accumulated context. Always the last step.",
    # Add other agents here if needed, following the same pattern.
}