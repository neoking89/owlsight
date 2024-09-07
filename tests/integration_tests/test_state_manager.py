import pytest
from src.utils.custom_classes import StateManager


class CodeExecutor:
    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager

    def execute_python_code(self, code: str) -> None:
        """
        Executes the provided code and updates the StateManager with any new variables.
        """
        # Execute the code in the prepared environment
        exec(code, {}, self.state_manager.get_state())


@pytest.fixture
def code_executor():
    return CodeExecutor(StateManager())


def test_execute_python_code(code_executor: CodeExecutor):
    # Test that variables are correctly set in the state after execution
    code_executor.execute_python_code("x = 10")
    assert code_executor.state_manager.get("x") == 10

    # Test executing another block of code that uses the state
    code_executor.execute_python_code("y = x + 5")
    assert code_executor.state_manager.get("y") == 15

    # Test that the state is persistent across executions
    assert code_executor.state_manager.get("x") == 10  # x should still be available


def test_clear_state(code_executor: CodeExecutor):
    # Set some state
    code_executor.execute_python_code("x = 20")
    assert code_executor.state_manager.get("x") == 20

    # Clear the state and check if it's removed
    code_executor.state_manager.clear_state()
    assert code_executor.state_manager.get("x") is None  # State should be cleared
