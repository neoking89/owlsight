import os
import re
from typing import Dict, List
import shlex
import tempfile
import subprocess

from src.processors.text_generation import TextGenerationProcessor
from src.utils.custom_classes import StateManager
from src.utils.subprocess_utils import execute_shell_command, run_subprocess
from src.utils.venv_manager import install_module
from src.utils.helper_functions import extract_markdown

from src.utils.logger_manager import LoggerManager

logger = LoggerManager.get_logger(__name__)


def extract_missing_module(stderr: str) -> str:
    match = re.search(r"No module named '(\w+)'", stderr)
    return match.group(1) if match else None


class CodeExecutor:
    def __init__(
        self,
        processor: TextGenerationProcessor,
        venv_path: str,
        pip_path: str,
        state_manager: StateManager,
        max_retries: int,
        max_new_tokens: int,
    ):
        self.processor = processor
        self.state_manager = state_manager
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries

        self._init_python_properties(venv_path, pip_path)
        self._reset_retries()

    def execute_and_retry(
        self, lang: str, code_block: str, original_question: str
    ) -> bool:
        """
        Execute code block in the specified language and retry if an error occurs.
        """
        while self.retries_left > 0:
            logger.info(
                f"Executing {lang.capitalize()} code (Attempt {self._get_nth_attempt()}/{self.max_retries})..."
            )
            try:
                self.execute_code_block(lang, code_block)
                logger.info(
                    f"Code executed successfully on attempt {self._get_nth_attempt()}."
                )
                return True
            except Exception as e:
                self.retries_left -= 1
                if self.retries_left > 0:
                    logger.warning(f"Error on attempt {self._get_nth_attempt()}: {e}")
                    logger.info(
                        f"Retrying... ({self._get_nth_attempt()}/{self.max_retries})"
                    )
                    code_block = self._generate_fixed_code(
                        original_question, code_block, str(e)
                    )
                else:
                    logger.error(
                        f"Failed to execute {lang} code after {self.max_retries} attempts."
                    )

        self._reset_retries()

        return False

    def execute_code_block(self, lang: str, code_block: str) -> None:
        if lang == "python":
            self.execute_python_code(code_block)
        elif lang in ["cmd", "bash"]:
            execute_shell_command(code_block, self.venv_path)
        else:
            logger.warning(f"Unsupported language: {lang}")

    # def execute_python_code(self, code_block: str, python_executable: str) -> None:
    #     with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
    #         temp_file.write(code_block)
    #         temp_file_path = temp_file.name

    #     try:
    #         stdout, stderr = run_subprocess([python_executable, temp_file_path])
    #         self._handle_python_output(stdout, stderr, code_block, python_executable)
    #     finally:
    #         os.unlink(temp_file_path)

    # def _handle_python_output(self, stdout: str, stderr: str, code_block: str, python_executable: str) -> None:
    #     if stdout:
    #         logger.info(stdout)
    #     if stderr:
    #         logger.error(f"Error: {stderr}")
    #         missing_module = extract_missing_module(stderr)
    #         if missing_module:
    #             logger.info(f"Module '{missing_module}' not found. Attempting to install...")
    #             if install_module(missing_module, self.pip_path):
    #                 self.execute_python_code(code_block, python_executable)
    #         else:
    #             raise Exception(stderr)
    #     else:
    #         logger.info("Code executed successfully.")

    def execute_python_code(self, code_block: str) -> None:
        """Execute Python code block."""
        try:
            # python_command = f'{self.python_executable} -c {code_block}'
            # python_command = f'{shlex.quote(self.python_executable)} -c {shlex.quote(code_block)}'
            # result = execute_shell_command(python_command, self.venv_path)
            exec(code_block, {}, self.state_manager.get_state())
            logger.info("Code executed successfully.")
        except ModuleNotFoundError as e:
            logger.error(f"Module not found: {e}")
            module_name = str(e).split("'")[1]
            logger.info(f"Attempting to install module: {module_name}")
            if install_module(module_name, self.pip_path):
                logger.info(f"Retrying execution after installing {module_name}")
                self.execute_python_code(
                    code_block
                )  # Retry execution
            else:
                logger.error(
                    f"Failed to install {module_name}. Cannot execute the code."
                )
        except Exception as e:
            logger.error(f"Error executing code: {e}")

    def _reset_retries(self) -> None:
        self.retries_left = self.max_retries

    def _get_nth_attempt(self) -> int:
        return self.max_retries - self.retries_left + 1

    def _generate_fixed_code(
        self, original_question: str, code_block: str, error: str
    ) -> str:
        new_question = f"""
        # ORIGINAL QUESTION:
        {original_question}

        # ANSWER WHICH GENERATED THE ERROR:
        {code_block}

        # ERROR:
        {error}

        # TASK: 
        1. Look at the error message and identify the issue.
        2. Do NOT make the same mistake again.
        3. Please provide updated Python code that addresses this error.
        """
        return self.processor.generate(new_question, max_new_tokens=self.max_new_tokens)

    def _init_python_properties(self, venv_path: str, pip_path: str):
        self.venv_path = venv_path
        self.lib_path = os.path.join(self.venv_path, "Lib/site-packages")
        self.python_executable = os.path.join(
            self.venv_path, "Scripts" if os.name == "nt" else "bin", "python"
        )
        self.pip_path = pip_path


def execute_code_with_feedback(
    response: str,
    original_question: str,
    code_executor: CodeExecutor,
) -> List[Dict]:
    """
    Extract code blocks from a response and execute them with feedback and retry logic.
    """
    results = []
    for lang, code_block in extract_markdown(response):
        is_succes = code_executor.execute_and_retry(lang, code_block, original_question)
        result = {"success": is_succes, "language": lang, "code": code_block}
        results.append(result)

    return results
