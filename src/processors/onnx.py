import os
import time
from typing import Any, Dict, List, Optional
import onnxruntime_genai as og

from src.utils.logger_manager import LoggerManager
logger = LoggerManager.get_logger(__name__)


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