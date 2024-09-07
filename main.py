import os
import tempfile

from src.utils.deep_learning import check_gpu_and_cuda
from src.processors.text_generation import (
    TextGenerationProcessorOnnx,
    TextGenerationProcessorTransformers,
)
from src.utils.custom_classes import StateManager
from src.utils.logger_manager import LoggerManager
from src.utils.venv_manager import create_venv
from src.utils.code_execution import CodeExecutor, execute_code_with_feedback


logger = LoggerManager.get_logger(__name__)

check_gpu_and_cuda()


def main() -> None:
    """
    Main function to run the interactive loop for code generation and execution.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    model_path = r"models\small\cuda\cuda-int4-rtn-block-32"
    # model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    # filename = "tinyllama-1.1b-chat-v1.0.Q6_K.gguf"
    processor = TextGenerationProcessorOnnx(
        model_id=model_path, verbose=True, save_history=False
    )
    # processor = TextGenerationProcessorTransformers(model_id=model_id, quantization_bits=None, save_history=True, gguf_file=filename)

    max_retries = 3
    max_new_tokens = 2048

    venv_dir = "venv"
    state_manager = StateManager()

    with tempfile.TemporaryDirectory() as temp_dir, create_venv(
        os.path.join(temp_dir, venv_dir)
    ) as pip_path:
        venv_path = os.path.join(temp_dir, venv_dir)
        code_executor = CodeExecutor(
            processor, venv_path, pip_path, state_manager, max_retries, max_new_tokens
        )

        while True:
            question = input("What can I do for you (Type 'q' or 'quit' to exit)?\n")
            if question.lower() in ["q", "quit"]:
                logger.info("Quitting...")
                break

            if question.strip().lower() == "#python":
                code_input = input(
                    "Enter Python code to execute using current state:\n"
                )
                code_executor.execute_code_block("python", code_input)
            elif question.strip().lower() == "#clear":
                state_manager.clear_state()
                processor.history.clear()
                logger.info("State and history cleared.")
            else:
                response = processor.generate(
                    question, max_new_tokens=max_new_tokens, stopwords=["```\n"]
                )
                execute_code_with_feedback(response, question, code_executor)

if __name__ == "__main__":
    main()
