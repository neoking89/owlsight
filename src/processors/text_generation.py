from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple
import os
import time
import onnxruntime_genai as og

import torch
from transformers import (
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
    TextIteratorStreamer,
    AutoTokenizer,
)
from src.utils.threads import KillableThread
from src.utils.custom_classes import StopWordCriteria
from src.utils.logger_manager import LoggerManager

logger = LoggerManager.get_logger(__name__)


def is_flash_attention_available() -> bool:
    try:
        from flash_attn import flash_attn_fn

        return True
    except ImportError:
        return False


class TextGenerationProcessor(ABC):
    def __init__(
        self, model_id: str, save_history: bool, system_prompt: str
    ):
        """
        Abstract class for text generation processors.

        Parameters
        ----------
        model_id: str
            The model ID to use for generation.
            Uusally the name of the model or the path to the model.
        save_history : bool
            Whether or not to save the history of inputs and outputs.
        """
        self.model_id = model_id
        self.save_history = save_history
        self.history = []
        self.system_prompt = system_prompt

    @abstractmethod
    def generate(
        self,
        input_text: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        stopwords: Optional[List[str]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate text based on the input text and other optional parameters.
        This method must be implemented by all subclasses.

        Parameters
        ----------
        input_text : str
            The input text to use for generation.
        max_new_tokens : int
            Maximum number of new tokens to generate.
        temperature : float
            Sampling temperature. A higher temperature leads to more random outputs.
        stopwords : List[str], optional
            List of stop words to use during generation.
        generation_kwargs : dict, optional
            Additional arguments for generation.

        Returns
        -------
        str
            Generated text.
        """


class TextGenerationProcessorTransformers(TextGenerationProcessor):
    def __init__(
        self,
        model_id: str,
        device: str = None,
        quantization_bits: Optional[int] = None,
        bnb_kwargs: Optional[dict] = None,
        gguf_file: Optional[str] = None,
        tokenizer_kwargs: Optional[dict] = None,
        model_kwargs: Optional[dict] = None,
        save_history: bool = False,
        system_prompt: str = None
    ):
        """
        Text generation processor using Hugging Face Transformers library.

        Parameters
        ----------
        model_id : str
            The model ID to use for generation.
            Usually the name of the model or the path to the model.
        device : str, optional
            The device to use for generation (default is "cuda" if available).
        quantization_bits : int, optional
            Number of quantization bits to use for the model.
            Available options: 4, 8, None (default is None).
        bnb_kwargs : dict, optional
            Additional keyword arguments for BitsAndBytesConfig.
        gguf_file : str, optional
            Path to the GGUF file. Experimental feature.

            See: https://huggingface.co/docs/transformers/en/gguf

            Example Use:
            >>> from transformers import AutoTokenizer, AutoModelForCausalLM
            >>> model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
            >>> gguf_file = "tinyllama-1.1b-chat-v1.0.Q6_K.gguf"

            >>> TextGenerationProcessorTransformers(model_id, gguf_file=gguf_file)
        tokenizer_kwargs : dict, optional
            Additional keyword arguments for the tokenizer.
        model_kwargs : dict, optional
            Additional keyword arguments for the model.
        save_history : bool
            Set to True if you want model to generate responses based on previous inputs.
        """
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._attention_implementation = (
            "flash" if is_flash_attention_available() else "eager"
        )
        self.save_history = save_history
        self.history = []
        self.system_prompt = system_prompt

        tokenizer, model = self._load_tokenizer_model(
            quantization_bits,
            gguf_file=gguf_file,
            tokenizer_kwargs=tokenizer_kwargs or {},
            bnb_kwargs=bnb_kwargs or {},
            model_kwargs=model_kwargs or {},
        )
        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto" if self.device != "cpu" else {"": "cpu"},
        )
        self.streamer = TextIteratorStreamer(
            self.pipe.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

    def _load_tokenizer_model(
        self,
        quantization_bits: Optional[int],
        gguf_file: Optional[str],
        tokenizer_kwargs: Dict,
        bnb_kwargs: Dict,
        model_kwargs: Dict,
    ):
        quantization_config = None
        if quantization_bits == 4 and self.device != "cpu":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                **bnb_kwargs,
            )
        elif quantization_bits == 8 and self.device != "cpu":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True, **bnb_kwargs)

        model_kwargs.update(
            {
                "device_map": "auto" if self.device != "cpu" else {"": "cpu"},
                "trust_remote_code": True,
                "torch_dtype": "auto" if self.device != "cpu" else torch.float32,
                "quantization_config": quantization_config,
                "_attn_implementation": self._attention_implementation,
            }
        )

        if gguf_file:
            tokenizer_kwargs["gguf_file"] = gguf_file
            model_kwargs["gguf_file"] = gguf_file

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, **tokenizer_kwargs)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)

        return tokenizer, model

    @torch.inference_mode()
    def generate(
        self,
        input_text: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        stopwords: Optional[List[str]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        if self.system_prompt is not None:
            input_text = f"{self.system_prompt}\n\n{input_text}"

        _generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "streamer": self.streamer,
            "eos_token_id": self.pipe.tokenizer.eos_token_id,
            "temperature": temperature if temperature > 0.0 else None,
            "do_sample": temperature > 0.0,
        }

        if stopwords is not None:
            _generation_kwargs["stopping_criteria"] = StopWordCriteria(
                prompts=[input_text],
                stop_words=stopwords,
                tokenizer=self.pipe.tokenizer,
            )

        if generation_kwargs is not None:
            _generation_kwargs.update(generation_kwargs)

        generation_thread = KillableThread(
            target=self.pipe, args=(input_text,), kwargs=_generation_kwargs
        )
        generation_thread.start()

        generated_text = ""
        for new_text in self.streamer:
            generated_text += new_text
            print(new_text, end="", flush=True)

        print()  # Print newline after generation is done

        generation_thread.kill()
        generation_thread.join()

        if self.save_history:
            self.history.append(input_text)
            self.history.append(generated_text.strip())

        return generated_text


class TextGenerationProcessorOnnx(TextGenerationProcessor):
    def __init__(
        self,
        model_id: str,
        verbose: bool = False,
        num_threads: int = 1,
        save_history: bool = False,
        system_prompt: str = None
    ):
        self.model_id = model_id
        self.verbose = verbose
        self.num_threads = num_threads
        self.save_history = save_history
        self.history = []
        self.system_prompt = system_prompt

        if not os.path.exists(model_id):
            raise FileNotFoundError(f"Model not found at {model_id}")

        self._set_environment_variables()
        self._initialize_model()

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
        self.model = og.Model(self.model_id)
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
        if self.system_prompt is not None:
            input_text = f"{self.system_prompt}\n\n{input_text}"
        
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
