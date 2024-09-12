import shutil
import os
import tempfile
import sys
import subprocess
from typing import List, Optional

from src.utils.deep_learning import check_gpu_and_cuda
from src.processors.text_generation import (
    TextGenerationProcessorOnnx,
    TextGenerationProcessorTransformers,
    TextGenerationProcessor,
)
from src.utils.logger_manager import LoggerManager
from src.utils.venv_manager import get_venv_path, get_pip_path, get_lib_path
from src.utils.code_execution import CodeExecutor, execute_code_with_feedback

logger = LoggerManager.get_logger(__name__)


def force_delete(temp_dir: str) -> None:
    """Forcefully deletes a directory if it exists."""
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.error(f"Error deleting directory {temp_dir}: {e}")


def remove_temp_directories(lib_path: str) -> None:
    """Removes lingering temporary directories in the virtual environment's library path."""
    for d in os.listdir(lib_path):
        if d.startswith("tmp"):
            logger.info(f"Removing temporary directory: {d}")
            force_delete(os.path.join(lib_path, d))


def handle_interactive_shell(question: str) -> None:
    """Handles interactive shell sessions based on user input."""
    if question.lower() == "!cmd":
        subprocess.call("cmd.exe", shell=True)
    elif question.lower() == "!bash":
        subprocess.call("/bin/bash", shell=True)


def handle_interactive_code_execution(code_executor: CodeExecutor) -> None:
    """Handles the interactive Python console execution."""
    try:
        code_executor.init_interactive_py_console()
    except Exception as e:
        logger.error(f"Unexpected error in interactive console: {e}")
    # Reopen stdin if it's closed
    if sys.stdin.closed:
        logger.warning("stdin is closed. Reopening for further input.")
        sys.stdin = open(0)


def run_code_generation_loop(
    code_executor: CodeExecutor,
    processor: TextGenerationProcessorOnnx,
    max_new_tokens: int,
    stopwords: Optional[List[str]],
    generation_kwargs: Optional[dict],
    prompt_code_execution: bool,
) -> None:
    """Runs the main loop for code generation and user interaction."""
    try:
        while True:
            question = input("What can I do for you (Type 'q' or 'quit' to exit)?\n")
            if question.lower() in ["q", "quit"]:
                logger.info("Quitting...")
                break

            if question.strip().lower() == "!python":
                handle_interactive_code_execution(code_executor)
            elif question.strip().lower() in ["!cmd", "!bash"]:
                logger.info("Starting an interactive shell. Type 'exit' to return.")
                handle_interactive_shell(question)
            elif question.strip().lower() == "!clear":
                code_executor.globals_dict.clear()
                processor.history.clear()
                logger.info("State and history cleared.")
            else:
                response = processor.generate(
                    question,
                    max_new_tokens=max_new_tokens,
                    stopwords=stopwords,
                    generation_kwargs=generation_kwargs,
                )
                execute_code_with_feedback(
                    response,
                    question,
                    code_executor,
                    prompt_code_execution=prompt_code_execution,
                )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


def main(
    processor: TextGenerationProcessor,
    max_retries: int = 3,
    max_new_tokens: int = 1024,
    stopwords: Optional[List[str]] = None,
    generation_kwargs: Optional[dict] = None,
    prompt_code_execution: bool = True,
) -> None:
    """
    Main function to run the interactive loop for code generation and execution

    Parameters
    ----------
    processor : TextGenerationProcessor
        The text generation processor to use for generating code.
    max_retries : int, optional
        The maximum number of retries for code execution, by default 3
    max_new_tokens : int, optional
        The maximum number of new tokens to generate, by default 1024
    stopwords : Optional[List[str]], optional
        List of stopwords to stop generation at, by default None
    generation_kwargs : Optional[dict], optional
        Additional keyword arguments for model generation, by default None
        For example: {"top_k": 50, "top_p": 0.95}
    prompt_code_execution : bool, optional
        Whether to prompt the user before executing the generated code, by default True
    """
    venv_path = get_venv_path()
    lib_path = get_lib_path(venv_path)
    pip_path = get_pip_path(venv_path)

    # Remove lingering temporary directories
    remove_temp_directories(lib_path)

    # Create temporary directory in venv to install packages
    with tempfile.TemporaryDirectory(dir=lib_path) as temp_dir:
        logger.info(f"Temporary directory created at: {temp_dir}")

        code_executor = CodeExecutor(
            processor, venv_path, pip_path, temp_dir, max_retries, max_new_tokens
        )

        run_code_generation_loop(
            code_executor,
            processor,
            max_new_tokens,
            stopwords,
            generation_kwargs,
            prompt_code_execution,
        )

    logger.info(f"Removing temporary directory: {temp_dir}")
    force_delete(temp_dir)


if __name__ == "__main__":
    check_gpu_and_cuda()

    model_path = r"models\small\cuda\cuda-int4-rtn-block-32"
    model_hf_id = "microsoft/Phi-3-mini-4k-instruct"

    # Initialize processor (uncomment if Transformers processor is needed)
    # processor = TextGenerationProcessorTransformers(
    #     model_id=model_hf_id,
    #     quantization_bits=4,
    #     save_history=True,
    # )

    processor = TextGenerationProcessorOnnx(
        model_id=model_path,
        huggingface_id=model_hf_id,
        verbose=True,
        save_history=False,
    )

    main(processor, max_retries=3, max_new_tokens=1024, prompt_code_execution=True)
