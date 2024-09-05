import os
import re
import subprocess
import platform
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple
import venv
import logging
from contextlib import contextmanager

import onnxruntime_genai as og

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class StateManager:
    def __init__(self):
        self.state = {}

    def get(self, key, default=None):
        return self.state.get(key, default)

    def set(self, key, value):
        self.state[key] = value

    def clear(self):
        self.state.clear()


state_manager = StateManager()


def extract_markdown(md_string: str) -> List[Tuple[str, str]]:
    """
    Extract language and code blocks from a markdown string.
    """
    pattern = r"```(\w+)([\s\S]*?)```"
    return [
        (match[0].strip(), match[1].strip()) for match in re.findall(pattern, md_string)
    ]


class TextGenerationProcessorOnnx:
    def __init__(
        self,
        model_path: str,
        verbose: bool = False,
        num_threads: int = 1,
        save_history: bool = False,
    ):
        self.model_path = model_path
        self.verbose = verbose
        self.num_threads = num_threads
        self.save_history = save_history
        self.history = []

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        self._set_environment_variables()
        self._initialize_model()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def cleanup(self):
        import gc

        gc.collect()

    def _set_environment_variables(self) -> None:
        os.environ.update(
            {
                "OMP_NUM_THREADS": str(self.num_threads),
                "OMP_WAIT_POLICY": "ACTIVE",
                "OMP_SCHEDULE": "STATIC",
                "ONNXRUNTIME_INTRA_OP_NUM_THREADS": str(self.num_threads),
                "ONNXRUNTIME_INTER_OP_NUM_THREADS": "1",
            }
        )

    def _initialize_model(self) -> None:
        logger.info("Loading model...")
        self.model = og.Model(self.model_path)
        self.tokenizer = og.Tokenizer(self.model)
        self.tokenizer_stream = self.tokenizer.create_stream()
        logger.info(f"Model loaded using {self.num_threads} threads")
        logger.info("Tokenizer created")

    def generate(
        self,
        input_text: str,
        chat_template: str = "<|user|>\n{input}\n<|assistant|>\n{output}",
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        stopwords: Optional[List[str]] = None,
        buffer_wordsize: int = 10,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        search_options = {
            "do_sample": temperature > 0.0,
            "max_length": max_new_tokens,
            "temperature": temperature,
            **(generation_kwargs or {}),
        }

        prompt = ""
        for i in range(0, len(self.history), 2):
            user_message = self.history[i]
            assistant_message = self.history[i + 1] if i + 1 < len(self.history) else ""
            prompt += chat_template.format(input=user_message, output=assistant_message)

        # Add the new user input
        prompt += chat_template.format(input=input_text, output="")

        input_tokens = self.tokenizer.encode(prompt)

        params = og.GeneratorParams(self.model)
        params.set_search_options(**search_options)
        params.input_ids = input_tokens
        generator = og.Generator(self.model, params)

        logger.info("Running generation loop ...")

        generated_text, buffer = "", ""
        token_counter = 0
        start = time.time()

        try:
            while not generator.is_done():
                generator.compute_logits()
                generator.generate_next_token()
                new_text = self.tokenizer_stream.decode(generator.get_next_tokens()[0])
                buffer += new_text
                token_counter += 1
                print(new_text, end="", flush=True)

                if len(buffer.split()) > buffer_wordsize:
                    generated_text += buffer
                    buffer = ""

                    if stopwords and any(
                        stop_word in generated_text for stop_word in stopwords
                    ):
                        break

        except KeyboardInterrupt:
            logger.warning("Control+C pressed, aborting generation")

        generated_text += buffer
        del generator

        total_time = time.time() - start
        logger.info(f"Generation took {total_time:.2f} seconds")
        logger.info(f"Tokens per second: {token_counter / total_time:.2f}")

        if self.save_history:
            self.history.append(input_text)
            self.history.append(generated_text.strip())

        return generated_text.strip()


@contextmanager
def create_venv(venv_path: str) -> str:
    venv.create(venv_path, with_pip=True)
    pip_path = os.path.join(venv_path, "Scripts" if os.name == "nt" else "bin", "pip")
    yield pip_path


def install_module(module_name: str, pip_path: str) -> bool:
    try:
        subprocess.check_call([pip_path, "install", module_name])
        logger.info(f"Successfully installed {module_name}")
        return True
    except subprocess.CalledProcessError:
        logger.error(f"Failed to install {module_name}")
        return False


def execute_python_code(code_block: str, pip_path: str) -> None:
    try:
        exec(code_block, state_manager.state)
        logger.info("Code executed successfully.")
    except ModuleNotFoundError as e:
        logger.error(f"Module not found: {e}")
        module_name = str(e).split("'")[1]
        logger.info(f"Attempting to install module: {module_name}")
        if install_module(module_name, pip_path):
            logger.info(f"Retrying execution after installing {module_name}")
            execute_python_code(code_block, pip_path)  # Retry execution
        else:
            logger.error(f"Failed to install {module_name}. Cannot execute the code.")
    except Exception as e:
        logger.error(f"Error executing code: {e}")
        raise


def execute_shell_command(code_block: str, venv_path: str) -> None:
    try:
        # Detect the current platform (Windows or Unix-like)
        current_os = platform.system().lower()

        if current_os == "windows":  # Windows-specific logic
            activate_script = os.path.join(venv_path, "Scripts", "activate.bat")
            full_command = f'call "{activate_script}" && {code_block}'
            shell = True  # Using the default cmd shell on Windows

        else:  # Unix-like systems (Linux, macOS)
            activate_script = os.path.join(venv_path, "bin", "activate")
            full_command = f'bash -c "source {activate_script} && {code_block}"'
            shell = True  # Using bash

        # Execute the command
        result = subprocess.run(
            full_command,
            shell=shell,
            capture_output=True,
            text=True,
            check=True,
        )

        # Output the result
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            logger.warning(f"Command produced stderr output: {result.stderr}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}: {e.stderr}")
        raise


def execute_code_block(
    language: str, code_block: str, pip_path: str, venv_path: str
) -> None:
    if language == "python":
        execute_python_code(code_block, pip_path)
    elif language in ["cmd", "bash"]:
        execute_shell_command(code_block, venv_path)
    else:
        logger.warning(f"Unsupported language: {language}")


def generate_fixed_code(
    processor: TextGenerationProcessorOnnx,
    original_question: str,
    code_block: str,
    error: str,
    max_new_tokens: int,
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
    return processor.generate(new_question, max_new_tokens=max_new_tokens)


def execute_code_block_with_feedback(
    language: str,
    code_block: str,
    processor: TextGenerationProcessorOnnx,
    original_question: str,
    retry_count: int,
    max_retries: int,
    max_new_tokens: int,
    pip_path: str,
    venv_path: str,
) -> None:
    if retry_count >= max_retries:
        logger.warning(
            f"Maximum retries ({max_retries}) reached. Unable to generate working code."
        )
        return

    logger.info(f"Executing {language.upper()} code (Attempt {retry_count + 1})...")
    try:
        execute_code_block(language, code_block, pip_path, venv_path)
    except Exception as e:
        if retry_count < max_retries:
            retry_count += 1
            logger.info(f"Retrying... ({retry_count}/{max_retries})")
            new_response = generate_fixed_code(
                processor, original_question, code_block, str(e), max_new_tokens
            )
            new_code_blocks = extract_markdown(new_response)
            for new_lang, new_code_block in new_code_blocks:
                execute_code_block_with_feedback(
                    new_lang,
                    new_code_block,
                    processor,
                    original_question,
                    retry_count,
                    max_retries,
                    max_new_tokens,
                    pip_path,
                    venv_path,
                )
        else:
            logger.error(
                f"Failed to execute {language} code after {max_retries} attempts."
            )


def extract_and_execute_code_with_feedback(
    response: str,
    processor: TextGenerationProcessorOnnx,
    original_question: str,
    max_retries: int,
    max_new_tokens: int,
    pip_path: str,
    venv_path: str,
) -> None:
    for lang, code_block in extract_markdown(response):
        execute_code_block_with_feedback(
            lang,
            code_block,
            processor,
            original_question,
            0,
            max_retries,
            max_new_tokens,
            pip_path,
            venv_path,
        )


def main():
    model_path = os.environ.get(
        "MODEL_PATH", r"models\small\cuda\cuda-int4-rtn-block-32"
    )
    max_retries = 3
    max_new_tokens = 1024

    with TextGenerationProcessorOnnx(
        model_path=model_path, verbose=True, save_history=True
    ) as processor, tempfile.TemporaryDirectory() as temp_dir, create_venv(
        os.path.join(temp_dir, "venv")
    ) as pip_path:
        venv_path = os.path.join(temp_dir, "venv")

        while True:
            question = input("What can I do for you (Type 'q' or 'quit' to exit)?\n")
            if question.lower() in ["q", "quit"]:
                logger.info("Quitting...")
                break

            if question.strip().lower() == "#python":
                code_input = input(
                    "Enter Python code to execute using current state:\n"
                )
                execute_python_code(code_input, pip_path)
            elif question.strip().lower() == "#clear":
                state_manager.clear()
                processor.history.clear()
                logger.info("State and history cleared.")
            else:
                response = processor.generate(
                    question, max_new_tokens=max_new_tokens, stopwords=["```\n"]
                )
                extract_and_execute_code_with_feedback(
                    response,
                    processor,
                    question,
                    max_retries,
                    max_new_tokens,
                    pip_path,
                    venv_path,
                )


if __name__ == "__main__":
    main()
