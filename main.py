import re
import subprocess
import tempfile
import time
import venv
from typing import Any, Dict, List, Optional, Tuple
import os

import onnxruntime_genai as og

INTERACTIVE_MODE = True
MAX_NEW_TOKENS = 2048
MAX_RETRIES = 3


def extract_markdown(md_string: str) -> List[Tuple[str, str]]:
    """
    Extract code blocks from a markdown string.

    Args:
        md_string (str): The markdown string containing code blocks.

    Returns:
        List[Tuple[str, str]]: A list of tuples containing the language and code content of each block.
    """
    pattern = r"```(\w+)([\s\S]*?)```"
    return [
        (match[0].strip(), match[1].strip()) for match in re.findall(pattern, md_string)
    ]


class TextGenerationProcessorOnnx:
    """
    A class for text generation using ONNX runtime.

    Args:
        model_path (str): Path to the ONNX model file.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        num_threads (int, optional): Number of threads to use. Defaults to None (uses all available CPU cores).
    """

    def __init__(
        self, model_path: str, verbose: bool = False, num_threads: Optional[int] = None
    ):
        self.model_path = model_path
        self.verbose = verbose
        self.num_threads = num_threads or os.cpu_count()
        if self.verbose:
            print("Loading model...")

        self._set_environment_variables()
        self._initialize_model()

        if self.verbose:
            print(f"Model loaded using {self.num_threads} threads")
            print("Tokenizer created")

    def _set_environment_variables(self) -> None:
        """Set environment variables for optimal ONNX runtime performance."""
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
        """Initialize the ONNX model and tokenizer."""
        self.model = og.Model(self.model_path)
        self.tokenizer = og.Tokenizer(self.model)
        self.tokenizer_stream = self.tokenizer.create_stream()

    def generate(
        self,
        input_text: str,
        chat_template: str = "<|user|>\n{input} <|end|>\n<|assistant|>",
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        stopwords: Optional[List[str]] = None,
        buffer_wordsize: int = 10,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate text based on the input.

        Args:
            input_text (str): The input text to generate from.
            chat_template (str, optional): The template for formatting the chat input.
            max_new_tokens (int, optional): Maximum number of new tokens to generate.
            temperature (float, optional): The temperature for text generation.
            stopwords (List[str], optional): List of words to stop generation.
            buffer_wordsize (int, optional): Number of words to buffer before output.
            generation_kwargs (Dict[str, Any], optional): Additional generation parameters.

        Returns:
            str: The generated text.
        """
        search_options = {
            "do_sample": temperature > 0.0,
            "max_length": max_new_tokens,
            "temperature": temperature,
            **(generation_kwargs or {}),
        }

        prompt = chat_template.format(input=input_text)
        input_tokens = self.tokenizer.encode(prompt)

        params = og.GeneratorParams(self.model)
        params.set_search_options(**search_options)
        params.input_ids = input_tokens
        generator = og.Generator(self.model, params)

        if self.verbose:
            print("Generator created")
            print("Running generation loop ...")

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
            print("  --control+c pressed, aborting generation--")

        generated_text += buffer
        del generator

        total_time = time.time() - start
        print(f"\nTook {total_time:.2f} seconds")
        print(f"Tokens per second: {token_counter / total_time:.2f}")

        return generated_text.strip()


def create_venv(venv_path: str) -> str:
    """
    Create a virtual environment at the specified path.

    Args:
        venv_path (str): The path where the virtual environment should be created.

    Returns:
        str: The path to the pip executable in the created virtual environment.
    """
    venv.create(venv_path, with_pip=True)
    pip_path = (
        os.path.join(venv_path, "bin", "pip")
        if os.name != "nt"
        else os.path.join(venv_path, "Scripts", "pip.exe")
    )
    return pip_path


def install_module(module_name: str, pip_path: str) -> bool:
    """
    Install a Python module using pip.

    Args:
        module_name (str): The name of the module to install.
        pip_path (str): The path to the pip executable.

    Returns:
        bool: True if installation was successful, False otherwise.
    """
    try:
        subprocess.check_call([pip_path, "install", module_name])
        print(f"Successfully installed {module_name}")
        return True
    except subprocess.CalledProcessError:
        print(f"Failed to install {module_name}")
        return False


def execute_code_block_with_feedback(
    language: str,
    code_block: str,
    venv_path: str,
    pip_path: str,
    processor: TextGenerationProcessorOnnx,
    original_question: str,
    retry_count: int = 0,
) -> None:
    """
    Execute a code block and provide feedback, with the ability to retry on errors.

    Args:
        language (str): The programming language of the code block.
        code_block (str): The code to execute.
        venv_path (str): The path to the virtual environment.
        pip_path (str): The path to the pip executable.
        processor (TextGenerationProcessorOnnx): The text generation processor.
        original_question (str): The original question that prompted this code generation.
        retry_count (int, optional): The number of times this execution has been retried.
    """
    if retry_count >= MAX_RETRIES:
        print(
            f"Maximum retries ({MAX_RETRIES}) reached. Unable to generate working code."
        )
        return

    print(f"\nExecuting {language.upper()} code (Attempt {retry_count + 1})...")
    try:
        if language == "python":
            python_executable = (
                os.path.join(venv_path, "bin", "python")
                if os.name != "nt"
                else os.path.join(venv_path, "Scripts", "python.exe")
            )

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as temp_file:
                temp_file.write(code_block)
                temp_file_path = temp_file.name

            process = subprocess.Popen(
                [python_executable, temp_file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = process.communicate()

            os.unlink(temp_file_path)

            if stdout:
                print(stdout)
            if stderr:
                print(f"Error: {stderr}")

                # Check for missing modules and install them
                match = re.search(r"No module named '(\w+)'", stderr)
                if match:
                    missing_module = match.group(1)
                    print(
                        f"Module '{missing_module}' not found. Attempting to install..."
                    )
                    if install_module(missing_module, pip_path):
                        return execute_code_block_with_feedback(
                            language,
                            code_block,
                            venv_path,
                            pip_path,
                            processor,
                            original_question,
                            retry_count,
                        )

                # Generate new code based on the error
                new_question = f"""
                # ORIGINAL QUESTION:
                {original_question}

                # ANSWER WHICH GENERATED THE ERROR:
                {code_block}

                # ERROR:
                {stderr}

                # TASK: 
                1. Look at the error message and identify the issue.
                2. Do NOT make the same mistake again.
                3. Please provide updated Python code that addresses this error.
                """
                new_response = processor.generate(
                    new_question, max_new_tokens=MAX_NEW_TOKENS, stopwords=["```\n"]
                )

                # Extract and execute the new code
                new_code_blocks = extract_markdown(new_response)
                for new_lang, new_code_block in new_code_blocks:
                    if new_lang == "python":
                        return execute_code_block_with_feedback(
                            new_lang,
                            new_code_block,
                            venv_path,
                            pip_path,
                            processor,
                            original_question,
                            retry_count + 1,
                        )

                print("No valid Python code found in the new response.")
        elif language in ["cmd", "bash"]:
            result = subprocess.run(
                code_block, shell=True, capture_output=True, text=True
            )
            print(result.stdout)
            if result.stderr:
                print("Error:", result.stderr)
        else:
            print(f"Unsupported language: {language}")
    except Exception as e:
        print(f"An error occurred while executing {language} code: {e}")


def extract_and_execute_code_with_feedback(
    response: str,
    venv_path: str,
    pip_path: str,
    processor: TextGenerationProcessorOnnx,
    original_question: str,
) -> None:
    """
    Extract code blocks from a response and execute them with feedback.

    Args:
        response (str): The response containing code blocks.
        venv_path (str): The path to the virtual environment.
        pip_path (str): The path to the pip executable.
        processor (TextGenerationProcessorOnnx): The text generation processor.
        original_question (str): The original question that prompted this code generation.
    """
    for lang, code_block in extract_markdown(response):
        execute_code_block_with_feedback(
            lang, code_block, venv_path, pip_path, processor, original_question
        )


if __name__ == "__main__":
    model_path = r"models/cuda/cuda-fp16"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    processor = TextGenerationProcessorOnnx(num_threads=1, model_path=model_path)

    with tempfile.TemporaryDirectory() as temp_dir:
        venv_path = os.path.join(temp_dir, "venv")
        pip_path = create_venv(venv_path)

        if INTERACTIVE_MODE:
            while True:
                question = input(
                    "What can I do for you (Type 'q' or 'quit' to exit)?\n"
                )
                if question.lower() in ["q", "quit"]:
                    print("Quitting...")
                    break
                response = processor.generate(
                    question, max_new_tokens=MAX_NEW_TOKENS, stopwords=["```\n"]
                )
                extract_and_execute_code_with_feedback(
                    response, venv_path, pip_path, processor, question
                )
        else:
            question = "Use python to 1: find the last opened pdf-file on the entire computer. 2: read the contents of this pdf file."
            response = processor.generate(
                question, max_new_tokens=MAX_NEW_TOKENS, stopwords=["```\n"]
            )
            extract_and_execute_code_with_feedback(
                response, venv_path, pip_path, processor, question
            )
