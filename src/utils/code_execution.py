import os
import tempfile
import re

from src.processors.onnx import TextGenerationProcessorOnnx
from src.utils.custom_classes import StateManager
from src.utils.subprocess_utils import _run_subprocess, execute_shell_command
from src.utils.venv_manager import install_module
from src.utils.helper_functions import extract_markdown

from src.utils.logger_manager import LoggerManager
logger = LoggerManager.get_logger(__name__)

class CodeExecutor:
    def __init__(self, processor: TextGenerationProcessorOnnx, venv_path: str, pip_path: str, state_manager: StateManager, max_retries: int, max_new_tokens: int):
        self.processor = processor
        self.venv_path = venv_path
        self.pip_path = pip_path
        self.state_manager = state_manager
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries
        self._reset_retries()

    def execute_and_retry(self, lang: str, code_block: str, original_question: str) -> bool:
        """
        Execute code block in the specified language and retry if an error occurs.
        """
        while self.retries_left > 0:
            logger.info(f"Executing {lang.capitalize()} code (Attempt {self._get_nth_attempt()}/{self.max_retries})...")
            try:
                self._execute_code_block(lang, code_block)
                logger.info(f"Code executed successfully on attempt {self._get_nth_attempt()}.")
                return True
            except Exception as e:
                self.retries_left -= 1
                if self.retries_left > 0:
                    logger.warning(f"Error on attempt {self._get_nth_attempt()}: {e}")
                    logger.info(f"Retrying... ({self._get_nth_attempt()}/{self.max_retries})")
                    code_block = self._generate_fixed_code(original_question, code_block, str(e))
                else:
                    logger.error(f"Failed to execute {lang} code after {self.max_retries} attempts.")

        self._reset_retries()
        
        return False

    def _execute_code_block(self, lang: str, code_block: str) -> None:
        python_executable = os.path.join(self.venv_path, "Scripts" if os.name == "nt" else "bin", "python")
        if lang == "python":
            self._execute_python_code(code_block, python_executable)
        elif lang in ["cmd", "bash"]:
            execute_shell_command(code_block, self.venv_path)
        else:
            logger.warning(f"Unsupported language: {lang}")

    def _execute_python_code(self, code_block: str, python_executable: str) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
            temp_file.write(code_block)
            temp_file_path = temp_file.name

        try:
            stdout, stderr = _run_subprocess([python_executable, temp_file_path])
            self._handle_python_output(stdout, stderr, code_block, python_executable)
        finally:
            os.unlink(temp_file_path)

    def _handle_python_output(self, stdout: str, stderr: str, code_block: str, python_executable: str) -> None:
        if stdout:
            print(stdout)
        if stderr:
            logger.error(f"Error: {stderr}")
            missing_module = self._extract_missing_module(stderr)
            if missing_module:
                logger.info(f"Module '{missing_module}' not found. Attempting to install...")
                if install_module(missing_module, self.pip_path):
                    self._execute_python_code(code_block, python_executable)
            else:
                raise Exception(stderr)
        else:
            logger.info("Code executed successfully.")

    def _reset_retries(self) -> None:
        self.retries_left = self.max_retries

    def _get_nth_attempt(self) -> int:
        return self.max_retries - self.retries_left + 1

    @staticmethod
    def _extract_missing_module(stderr: str) -> str:
        match = re.search(r"No module named '(\w+)'", stderr)
        return match.group(1) if match else None

    def _generate_fixed_code(self, original_question: str, code_block: str, error: str) -> str:
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

def execute_code_with_feedback(
    response: str,
    processor: TextGenerationProcessorOnnx,
    original_question: str,
    max_retries: int,
    max_new_tokens: int,
    venv_path: str,
    pip_path: str,
    state_manager: StateManager,
) -> None:
    """
    Extract code blocks from a response and execute them with feedback and retry logic.
    """
    executor = CodeExecutor(processor, venv_path, pip_path, state_manager, max_retries, max_new_tokens)
    for lang, code_block in extract_markdown(response):
        executor.execute_and_retry(lang, code_block, original_question)

# import os
# import tempfile
# import re

# from src.processors.onnx import TextGenerationProcessorOnnx
# from src.utils.custom_classes import StateManager
# from src.utils.subprocess_utils import _run_subprocess, execute_shell_command
# from src.utils.venv_manager import install_module
# from src.utils.helper_functions import extract_markdown

# from src.utils.logger_manager import LoggerManager
# logger = LoggerManager.get_logger(__name__)


# def execute_python_code(
#     code_block: str, python_executable: str, pip_path: str, state_manager: StateManager
# ) -> None:
#     """
#     Execute Python code in the current virtual environment and handle missing modules.

#     Parameters
#     ----------
#     code_block : str
#         The Python code to be executed.
#     python_executable : str
#         Path to the Python executable within the virtual environment.
#     pip_path : str
#         Path to the pip executable within the virtual environment.
#     state_manager : StateManager
#         The state manager used for managing application state.

#     Returns
#     -------
#     None
#     """
#     with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
#         temp_file.write(code_block)
#         temp_file_path = temp_file.name

#     try:
#         stdout, stderr = _run_subprocess([python_executable, temp_file_path])
#         _handle_python_output(
#             stdout, stderr, code_block, python_executable, pip_path, state_manager
#         )
#     finally:
#         os.unlink(temp_file_path)


# def execute_code_block(
#     language: str,
#     code_block: str,
#     venv_path: str,
#     pip_path: str,
#     state_manager: StateManager,
# ) -> None:
#     """
#     Execute code in the appropriate language (Python, shell) and handle retries.

#     Parameters
#     ----------
#     language : str
#         The language of the code block to be executed (e.g., "python", "bash").
#     code_block : str
#         The code block to be executed.
#     venv_path : str
#         Path to the virtual environment.
#     pip_path : str
#         Path to the pip executable within the virtual environment.
#     state_manager : StateManager
#         The state manager used for managing application state.

#     Returns
#     -------
#     None
#     """
#     python_executable = os.path.join(
#         venv_path, "Scripts" if os.name == "nt" else "bin", "python"
#     )
#     if language == "python":
#         execute_python_code(code_block, python_executable, pip_path, state_manager)
#     elif language in ["cmd", "bash"]:
#         execute_shell_command(code_block, venv_path)
#     else:
#         logger.warning(f"Unsupported language: {language}")


# def _handle_python_output(
#     stdout: str,
#     stderr: str,
#     code_block: str,
#     python_executable: str,
#     pip_path: str,
#     state_manager: StateManager,
# ) -> None:
#     """
#     Handle the output from executing Python code, and retry on missing module errors.

#     Parameters
#     ----------
#     stdout : str
#         Standard output from the executed code.
#     stderr : str
#         Standard error from the executed code.
#     code_block : str
#         The Python code block that was executed.
#     python_executable : str
#         Path to the Python executable within the virtual environment.
#     pip_path : str
#         Path to the pip executable within the virtual environment.
#     state_manager : StateManager
#         The state manager used for managing application state.

#     Returns
#     -------
#     None
#     """
#     if stdout:
#         print(stdout)
#     if stderr:
#         logger.error(f"Error: {stderr}")
#         missing_module = _extract_missing_module(stderr)
#         if missing_module:
#             logger.info(
#                 f"Module '{missing_module}' not found. Attempting to install..."
#             )
#             if install_module(missing_module, pip_path):
#                 execute_python_code(
#                     code_block, python_executable, pip_path, state_manager
#                 )
#     else:
#         logger.info("Code executed successfully.")


# def _extract_missing_module(stderr: str) -> str:
#     """
#     Extract the missing module name from an error message.

#     Parameters
#     ----------
#     stderr : str
#         The error message returned by Python execution.

#     Returns
#     -------
#     str
#         The name of the missing module, if found. Otherwise, None.
#     """
#     match = re.search(r"No module named '(\w+)'", stderr)
#     return match.group(1) if match else None


# def generate_fixed_code(
#     processor: TextGenerationProcessorOnnx,
#     original_question: str,
#     code_block: str,
#     error: str,
#     max_new_tokens: int,
# ) -> str:
#     """
#     Generate a corrected version of code based on the error output.

#     Parameters
#     ----------
#     processor : TextGenerationProcessorOnnx
#         The ONNX-based text generation processor used to generate new code.
#     original_question : str
#         The original question provided by the user.
#     code_block : str
#         The code block that caused an error.
#     error : str
#         The error message received during code execution.
#     max_new_tokens : int
#         The maximum number of new tokens to generate.

#     Returns
#     -------
#     str
#         The newly generated, corrected code.
#     """
#     new_question = f"""
#     # ORIGINAL QUESTION:
#     {original_question}

#     # ANSWER WHICH GENERATED THE ERROR:
#     {code_block}

#     # ERROR:
#     {error}

#     # TASK: 
#     1. Look at the error message and identify the issue.
#     2. Do NOT make the same mistake again.
#     3. Please provide updated Python code that addresses this error.
#     """
#     return processor.generate(new_question, max_new_tokens=max_new_tokens)


# def _execute_code_with_retry(
#     lang: str,
#     code_block: str,
#     processor: TextGenerationProcessorOnnx,
#     original_question: str,
#     venv_path: str,
#     pip_path: str,
#     state_manager: StateManager,
#     max_retries: int,
#     max_new_tokens: int,
#     current_attempt: int = 1,
# ) -> bool:
#     total_attempts = current_attempt + max_retries - 1

#     while current_attempt <= total_attempts:
#         logger.info(
#             f"Executing {lang.upper()} code (Attempt {current_attempt}/{total_attempts})..."
#         )
#         try:
#             raise ValueError("Test error message")
#             execute_code_block(lang, code_block, venv_path, pip_path, state_manager)
#             logger.info(f"Code executed successfully on attempt {current_attempt}.")
#             return True  # Success
#         except Exception as e:
#             if current_attempt < total_attempts:
#                 logger.warning(f"Error on attempt {current_attempt}: {e}")
#                 logger.info(f"Retrying... ({current_attempt}/{total_attempts})")
#                 # Generate new code to retry based on the error
#                 new_response = generate_fixed_code(
#                     processor, original_question, code_block, str(e), max_new_tokens
#                 )
#                 new_code_blocks = extract_markdown(new_response)
#                 for new_lang, new_code_block in new_code_blocks:
#                     # Retry with new code blocks generated by the processor
#                     if _execute_code_with_retry(
#                         new_lang,
#                         new_code_block,
#                         processor,
#                         original_question,
#                         venv_path,
#                         pip_path,
#                         state_manager,
#                         total_attempts - current_attempt,  # Remaining retries
#                         max_new_tokens,
#                         current_attempt + 1,
#                     ):
#                         return True  # If any new code block succeeds, return True
#             else:
#                 logger.error(
#                     f"Failed to execute {lang} code after {current_attempt} attempts."
#                 )
#             current_attempt += 1

#     return False


# def extract_and_execute_code_with_feedback(
#     response: str,
#     processor: TextGenerationProcessorOnnx,
#     original_question: str,
#     max_retries: int,
#     max_new_tokens: int,
#     venv_path: str,
#     pip_path: str,
#     state_manager: StateManager,
# ) -> None:
#     """
#     Extract code blocks from a response and execute them with feedback and retry logic.

#     Parameters
#     ----------
#     response : str
#         The generated response containing the code to execute.
#     processor : TextGenerationProcessorOnnx
#         The ONNX-based text generation processor used to generate new code.
#     original_question : str
#         The original question provided by the user.
#     max_retries : int
#         Maximum number of retries for code execution.
#     max_new_tokens : int
#         The maximum number of new tokens to generate when fixing code.
#     venv_path : str
#         Path to the virtual environment.
#     pip_path : str
#         Path to the pip executable within the virtual environment.
#     state_manager : StateManager
#         The state manager used for managing application state.

#     Returns
#     -------
#     None
#     """
#     for lang, code_block in extract_markdown(response):
#         _execute_code_with_retry(
#             lang,
#             code_block,
#             processor,
#             original_question,
#             venv_path,
#             pip_path,
#             state_manager,
#             max_retries,
#             max_new_tokens,
#         )
