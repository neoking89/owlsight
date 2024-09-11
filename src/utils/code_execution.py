import os
import re
from typing import Dict, List

from src.processors.text_generation import TextGenerationProcessor
from src.utils.custom_exceptions import ModuleNotFoundInVenvError
from src.utils.subprocess_utils import execute_shell_command
from src.utils.helper_functions import extract_markdown
from src.utils.venv_manager import install_module, get_lib_path, get_python_executable

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
        temp_dir: str,
        max_retries: int,
        max_new_tokens: int,
    ):
        self.processor = processor
        self.temp_dir = temp_dir
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries
        self.global_dict = {}

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

    def execute_python_code(self, code_block: str) -> None:
        """Execute Python code block."""
        try:
            exec(code_block, self.global_dict)
        except ModuleNotFoundError as e:
            logger.error(f"Module not found: {e}")
            missing_module = extract_missing_module(str(e))
            logger.info(f"Attempting to install module: {missing_module}")
            if install_module(missing_module, self.pip_path, self.temp_dir):
                if not missing_module in os.listdir(self.temp_dir):
                    raise ModuleNotFoundInVenvError(
                        missing_module,
                        self.venv_path,
                        os.listdir(self.temp_dir),
                    )
                logger.info(f"Retrying execution after installing {missing_module}")
                self.execute_python_code(code_block)  # Retry execution
            else:
                logger.error(
                    f"Failed to install {missing_module}. Cannot execute the code."
                )
        except NameError as e:  # non-fatal errors here
            logger.error(f"Error executing code: {e}")
        except Exception as e:  # fatal errors here
            logger.error(f"Error executing code: {e}")
            raise e

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
        self.lib_path = get_lib_path(venv_path)
        self.python_executable = get_python_executable(venv_path)
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
