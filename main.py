import shutil
import os
import tempfile
import signal
import sys
import stat
from src.utils.deep_learning import check_gpu_and_cuda
from src.processors.text_generation import (
    TextGenerationProcessorOnnx,
    TextGenerationProcessorTransformers,
)
from src.utils.logger_manager import LoggerManager
from src.utils.venv_manager import get_venv_path, get_pip_path, get_lib_path
from src.utils.code_execution import CodeExecutor, execute_code_with_feedback

logger = LoggerManager.get_logger(__name__)


def handle_exit(signum, frame):
    logger.info("Received exit signal. Exiting gracefully...")
    sys.exit(0)


# Function to handle removing files that are marked as read-only or locked
def handle_remove_readonly(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)  # Change the file to writable
    func(path)  # Retry the function that raised the error (either rmtree or os.remove)


def force_delete(temp_dir: str) -> None:
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir, onerror=handle_remove_readonly)
        except Exception as e:
            logger.error(f"Error deleting directory {temp_dir}: {e}")

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

    # Remove any lingering temporary directories
    for d in os.listdir(lib_path):
        if d.startswith("tmp"):
            logger.info(f"Removing temporary directory: {d}")
            force_delete(os.path.join(lib_path, d))

    model_path = r"models\small\cuda\cuda-int4-rtn-block-32"
    model_hf_id = "microsoft/Phi-3-mini-4k-instruct"

    # processor = TextGenerationProcessorTransformers(
    #     model_id=model_hf_id,
    #     quantization_bits=4,
    #     save_history=True,
    #     )

    processor = TextGenerationProcessorOnnx(
        model_id=model_path,
        huggingface_id=model_hf_id,
        verbose=True,
        save_history=False,
    )

    max_retries = 3
    max_new_tokens = 1024

    # Create temp dir in venv to install packages
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

                # Acces Python global state in interactive console
                if question.strip().lower() == "#python":
                    code_executor.init_interactive_py_console()

                # Clear all past states and history
                elif question.strip().lower() == "#clear":
                    code_executor.globals_dict.clear()
                    processor.history.clear()
                    logger.info("State and history cleared.")
                else:
                    response = processor.generate(
                        question,
                        max_new_tokens=max_new_tokens,
                        stopwords=["```\n"],
                        generation_kwargs={"repetition_penalty": 1.2},
                    )
                    execute_code_with_feedback(response, question, code_executor, prompt_execution=True)
        finally:
            logger.info(f"Removing temporary directory: {temp_dir}")
            force_delete(temp_dir)


if __name__ == "__main__":
    check_gpu_and_cuda()
    main()
