import pytest
import sys
import tempfile
import os

sys.path.append(".")
sys.path.append("tests")
from src.utils.code_execution import CodeExecutor, execute_code_with_feedback
from src.utils.custom_classes import StateManager
from src.utils.venv_manager import create_venv

try:
    from conftest import MockTextGenerationProcessor
except ImportError:
    from tests.conftest import MockTextGenerationProcessor


@pytest.fixture
def code_executor(request):
    mock_responses = getattr(request, "param", ["Default Response"])
    return CodeExecutor(
        MockTextGenerationProcessor(
            "model_id",
            save_history=True,
            mock_responses=mock_responses,
        ),
        state_manager=StateManager(),
        pip_path="pip",
        venv_path="venv",
        max_new_tokens=512,
        max_retries=3,
    )


def test_code_executor_execute_python_code_succesfully(code_executor: CodeExecutor):
    result = code_executor.execute_and_retry(
        "python", "print('Hello')", "original question"
    )
    assert result

def test_code_executor_python_state_is_saved(code_executor: CodeExecutor):
    code_executor.execute_python_code(
        "a=5"
    )
    assert code_executor.state_manager.get("a") == 5

def test_code_executor_install_missing_module_in_venv():
    # arrange
    module_name = "tinydb"
    question = f"Use python to generate code which uses the '{module_name}' module"
    model_response = f"""
    Sure! Here is a Python code snippet that imports the '{module_name}' module:
    ```python
    import {module_name} as md\na = 5
    ```
    """.strip()
    processor = MockTextGenerationProcessor(
        "model_id",
        save_history=True,
        mock_responses=[model_response],
    )
    venv_dir = "venv"
    state_manager = StateManager()
    max_retries = 3
    max_new_tokens = 256

    with tempfile.TemporaryDirectory() as temp_dir, create_venv(
        os.path.join(temp_dir, venv_dir)
    ) as pip_path:
        venv_path = os.path.join(temp_dir, venv_dir)
        code_executor = CodeExecutor(
            processor, venv_path, pip_path, state_manager, max_retries, max_new_tokens
        )
        results = execute_code_with_feedback(
            model_response,
            question,
            code_executor,
        )

        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], dict)
        assert results[0]["success"]

        # lib is installed in venv
        assert module_name in os.listdir(code_executor.lib_path)

        # state is saved correctly
        assert state_manager.get("a") == 5


if __name__ == "__main__":
    pytest.main([__file__])
    # test_code_executor_install_missing_module_in_venv()
