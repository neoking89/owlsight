import shutil
import os
import tempfile
import signal
import sys
from src.utils.deep_learning import check_gpu_and_cuda
from src.processors.text_generation import (
    TextGenerationProcessorOnnx,
)
from src.utils.logger_manager import LoggerManager
from src.utils.venv_manager import get_venv_path, get_pip_path, get_lib_path
from src.utils.code_execution import CodeExecutor, execute_code_with_feedback

logger = LoggerManager.get_logger(__name__)


def handle_exit(signum, frame):
    logger.info("Received exit signal. Exiting gracefully...")
    sys.exit(0)


def force_delete(temp_dir: str) -> None:
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
        except PermissionError as e:
            os.chmod(temp_dir, 0o777)
            os.rmdir(temp_dir)
        except OSError as e:
            logger.error("Error: %s - %s." % (e.filename, e.strerror))

    if os.path.exists(temp_dir):
        logger.error(f"Failed to delete temporary directory: {temp_dir}")


# Set up signal handling
signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)


def main() -> None:
    """
    Main function to run the interactive loop for code generation and execution.
    """
    venv_path = get_venv_path()
    lib_path = get_lib_path(venv_path)
    pip_path = get_pip_path(venv_path)

    model_path = r"models\small\cuda\cuda-int4-rtn-block-32"
    processor = TextGenerationProcessorOnnx(
        model_id=model_path, verbose=True, save_history=False
    )

    max_retries = 3
    max_new_tokens = 2048

    # create temp dir in venv to install packages
    with tempfile.TemporaryDirectory(dir=lib_path) as temp_dir:
        logger.info(f"Temporary directory created at: {temp_dir}")

        code_executor = CodeExecutor(
            processor, venv_path, pip_path, temp_dir, max_retries, max_new_tokens
        )

        try:
            while True:
                question = input(
                    "What can I do for you (Type 'q' or 'quit' to exit)?\n"
                )
                if question.lower() in ["q", "quit"]:
                    logger.info("Quitting...")
                    break

                if question.strip().lower() == "#python":
                    code_input = input(
                        "Enter Python code to execute using current state:\n"
                    )
                    code_executor.execute_code_block("python", code_input)
                elif question.strip().lower() == "#clear":
                    code_executor.global_dict.clear()
                    processor.history.clear()
                    logger.info("State and history cleared.")
                else:
                    response = processor.generate(
                        question, max_new_tokens=max_new_tokens, stopwords=["```\n"]
                    )
                    execute_code_with_feedback(response, question, code_executor)
        finally:
            logger.info(f"Removing temporary directory: {temp_dir}")

        force_delete(temp_dir)


if __name__ == "__main__":
    check_gpu_and_cuda()
    main()
