from typing import Dict

AGENT_INFORMATION: Dict[str, str] = {
    "ToolSelectionAgent": "Chooses & executes ONE existing tool. The tool output will be summarized before the next step.",
    "ToolCreationAgent": "Writes a python function, serving as a *new* tool. MUST be followed immediately by ToolSelectionAgent to run the new tool.",
    "ObservationAgent": "Observes the output of the previous step and provides a summary.",
    "FinalAgent": "Drafts the ultimate reply based on all accumulated context. Always the last step.",
}