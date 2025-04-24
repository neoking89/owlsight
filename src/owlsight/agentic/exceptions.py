"""
Exceptions used within the agentic framework.
"""

class PlanningError(Exception):
    """Exception raised for errors during the planning phase."""
    pass


class GuardrailViolationError(PlanningError):
    """Exception raised when a plan violates a guardrail constraint."""
    
    def __init__(self, message: str, guardrail_name: str):
        self.guardrail_name = guardrail_name
        self.message = message
        super().__init__(f"[{guardrail_name}] {message}")
