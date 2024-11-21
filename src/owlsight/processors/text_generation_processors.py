from typing import Optional, List, Dict, Any, Union, Tuple, Generator
import os
import time
import traceback
import threading
from ast import literal_eval
from functools import lru_cache

import torch
from transformers import (
    BitsAndBytesConfig,
    TextIteratorStreamer,
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    pipeline,
    Pipeline,
)
from owlsight.processors.base import TextGenerationProcessor
from owlsight.processors.constants import (
    DEFAULT_TASK,
    GENERATION_THREAD_TIMEOUT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
)
from owlsight.utils.threads import ThreadNotKilledError
from owlsight.utils.custom_exceptions import QuantizationNotSupportedError, InvalidGGUFFileError
from owlsight.utils.custom_classes import StopWordCriteria
from owlsight.utils.deep_learning import get_best_device, bfloat16_is_supported
from owlsight.hugging_face.constants import TASK_TO_AUTO_MODEL, SUPPORTED_TASKS
from owlsight.utils.helper_functions import check_invalid_input_parameters
from owlsight.utils.logger import logger

ONNX_MSG = "ONNX Runtime is disabled. Use 'pip install owlsight[onnx]' or install [onnxruntime-genai, onnxruntime-genai-cuda] seperately"

try:
    import onnxruntime_genai as og
except ImportError:
    logger.warning("Support for ONNX models is disabled.")
    og = None

try:
    from llama_cpp import Llama
except ImportError:
    logger.warning(
        "Support for GGUF models is disabled, because llama-cpp is not found. Install it using 'pip install llama-cpp-python'."
    )
    Llama = None


class TextGenerationProcessorTransformers(TextGenerationProcessor):
    def __init__(
        self,
        model_id: str,
        transformers__device: Optional[str] = None,
        transformers__quantization_bits: Optional[int] = None,
        transformers__stream: bool = True,
        transformers__use_fp16: bool = False,
        bnb_kwargs: Optional[dict] = None,
        tokenizer_kwargs: Optional[dict] = None,
        model_kwargs: Optional[dict] = None,
        task: Optional[str] = None,
        save_history: bool = False,
        system_prompt: str = "",
        **kwargs,
    ):
        """
        Text generation processor using transformers library.

        Parameters
        ----------
        model_id : str
            The model ID to use for generation.
            Usually the name of the model or the path to the model.
        transformers__device : str
            The device to use for generation. Default is None, where the best available device is checked out of the possible devices.
        transformers__quantization_bits : Optional[int]
            The number of quantization bits to use for the model. Default is None.
        transformers__stream : bool
            Whether to use streaming generation. Default is True.
        transformers__use_fp16 : bool
            Whether to use FP16 for the model. This will not work for cpu, as FP16 is not supported on CPU.
            Checks if bfloat16 is supported and will use this if available, else uses torch.float16.
        bnb_kwargs : Optional[dict]
            Additional keyword arguments for BitsAndBytesConfig. Default is None.
        tokenizer_kwargs : Optional[dict]
            Additional keyword arguments for the tokenizer. Default is None.
        model_kwargs : Optional[dict]
            Additional keyword arguments for the model. Default is None.
        task : Optional[str]
            The task to use for the pipeline. Default is None, where the task is set to "text-generation".
        save_history : bool
            Set to True if you want model to generate responses based on previous inputs.
        system_prompt : str
            The system prompt to prepend to the input text.
        """
        if task and task not in SUPPORTED_TASKS:
            raise ValueError(f"Task '{task}' is not supported. Supported tasks are: {list(SUPPORTED_TASKS.keys())}")

        super().__init__(model_id, save_history, system_prompt)

        # Initialize configuration
        self.transformers__device = transformers__device or get_best_device()
        self.transformers__stream = transformers__stream
        self.transformers__quantization_bits = transformers__quantization_bits
        self.transformers__use_fp16 = transformers__use_fp16
        self.bnb_kwargs = bnb_kwargs or {}
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.model_kwargs = model_kwargs or {}
        self.task = task or DEFAULT_TASK

        # Set device and dtype configuration
        self._torch_dtype = self._determine_torch_dtype()
        self._auto_model_cls: AutoModel = TASK_TO_AUTO_MODEL.get(self.task, AutoModelForCausalLM)

        # Initialize model components
        self._setup_tokenizer_and_model_kwargs()
        self.pipe = self._setup_pipeline()
        self.streamer = self._setup_streamer() if self.transformers__stream else None

    def _determine_torch_dtype(self) -> Any:
        """Determine appropriate torch dtype based on configuration."""
        if self.transformers__use_fp16:
            if self.transformers__device == "cpu":
                raise TypeError("FP16 is not supported on CPU.")
            return self._get_correct_fp16_dtype()
        return torch.float32 if self.transformers__device == "cpu" else "auto"

    @lru_cache(maxsize=1)
    def _get_correct_fp16_dtype(self) -> torch.dtype:
        """Get correct FP16 dtype based on hardware support."""
        return torch.bfloat16 if bfloat16_is_supported() else torch.float16

    def _setup_tokenizer_and_model_kwargs(self) -> Tuple[AutoTokenizer, AutoModel]:
        """Load and configure tokenizer and model."""
        if self.transformers__quantization_bits and self.transformers__device in ["cpu", "mps"]:
            raise QuantizationNotSupportedError("Quantization not supported on CPU or MPS.")

        quantization_config = self._get_quantization_config()
        self.model_kwargs = self._prepare_model_kwargs(quantization_config)

        self.tokenizer = None
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=True, **self.tokenizer_kwargs
            )
        except Exception:
            logger.error(f"Failed to load tokenizer for model {self.model_id}: {traceback.format_exc()}")

    def _flash_attention_is_available(self) -> bool:
        """Check if flash attention is available."""
        try:
            from flash_attn import flash_attn_fn

            return True
        except ImportError:
            return False

    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get quantization configuration if applicable."""
        if self.transformers__quantization_bits == 4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self._get_correct_fp16_dtype(),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                **self.bnb_kwargs,
            )
        elif self.transformers__quantization_bits == 8:
            return BitsAndBytesConfig(load_in_8bit=True, **self.bnb_kwargs)
        return None

    def _prepare_model_kwargs(self, quantization_config: Optional[BitsAndBytesConfig]) -> Dict[str, Any]:
        """Prepare model initialization kwargs."""
        kwargs = {
            "torch_dtype": self._torch_dtype,
            "quantization_config": quantization_config,
            "_attn_implementation": "flash" if self._flash_attention_is_available() else "eager",
        }
        kwargs.update(self.model_kwargs)
        return kwargs

    def _setup_pipeline(self) -> Pipeline:
        """Set up the generation pipeline."""
        return pipeline(
            task=self.task,
            model=self.model_id,
            tokenizer=self.tokenizer,
            device=self.transformers__device,
            trust_remote_code=True,
            model_kwargs=self.model_kwargs,
        )

    def _setup_streamer(self) -> TextIteratorStreamer:
        """Set up text streaming if enabled."""
        return TextIteratorStreamer(
            self.pipe.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_data: str,
        max_new_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        stopwords: Optional[List[str]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate text response."""
        if self.transformers__stream:
            response = ""
            for text_chunk in self.generate_stream(
                input_data, max_new_tokens, temperature, stopwords, generation_kwargs
            ):
                print(text_chunk, end="", flush=True)
                response += text_chunk
            print()  # Print newline after generation is done
            return response
        return self._generate_non_stream(input_data, max_new_tokens, temperature, stopwords, generation_kwargs)

    @torch.inference_mode()
    def generate_stream(
        self,
        input_data: str,
        max_new_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        stopwords: Optional[List[str]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Generator[str, None, None]:
        """Generate streaming text response."""
        if not self.transformers__stream:
            raise ValueError("Streaming is disabled. Enable with transformers__stream=True.")

        yield from self._generate_stream(input_data, max_new_tokens, temperature, stopwords, generation_kwargs)

    def prepare_generation(
        self,
        input_data: str,
        max_new_tokens: int,
        temperature: float,
        stopwords: Optional[List[str]],
        generation_kwargs: Optional[Dict[str, Any]],
        streaming: bool = False,
        apply_chat_template: bool = True,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Prepare generation parameters.

        Returns:
        --------
        Tuple[str, Dict[str, Any]]
            The input text and generation_kwargs.
        """
        if apply_chat_template:
            input_data = self.apply_chat_template(input_data, self.pipe.tokenizer)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "eos_token_id": self.pipe.tokenizer.eos_token_id,
            "temperature": temperature if temperature > 0.0 else None,
            "do_sample": temperature > 0.0,
        }

        if stopwords:
            gen_kwargs["stopping_criteria"] = StopWordCriteria(
                prompts=[input_data],
                stop_words=stopwords,
                tokenizer=self.pipe.tokenizer,
            )

        if generation_kwargs:
            gen_kwargs.update(generation_kwargs)

        if streaming:
            gen_kwargs["streamer"] = self.streamer
            # gen_kwargs["num_beams"] = 1  # Required for streaming

        return input_data, gen_kwargs

    def _generate_non_stream(
        self,
        input_data: str,
        max_new_tokens: int,
        temperature: float,
        stopwords: Optional[List[str]],
        generation_kwargs: Optional[Dict[str, Any]],
    ) -> str:
        """Generate text without streaming."""
        templated_text, gen_kwargs = self.prepare_generation(
            input_data, max_new_tokens, temperature, stopwords, generation_kwargs
        )
        output = self.pipe(templated_text, **gen_kwargs)
        generated_text = output[0]["generated_text"][len(templated_text) :].strip()
        self.update_history(input_data, generated_text)
        return generated_text

    def _generate_stream(
        self,
        input_data: str,
        max_new_tokens: int,
        temperature: float,
        stopwords: Optional[List[str]],
        generation_kwargs: Optional[Dict[str, Any]],
    ) -> Generator[str, None, None]:
        """Generate streaming text."""
        templated_text, gen_kwargs = self.prepare_generation(
            input_data, max_new_tokens, temperature, stopwords, generation_kwargs, streaming=True
        )

        stop_event = threading.Event()
        generation_thread = threading.Thread(
            target=self._run_generation_thread,
            args=(templated_text, gen_kwargs, stop_event),
        )
        generation_thread.start()

        try:
            yield from self._stream_generator(stop_event, input_data)
        except Exception as e:
            logger.error(f"Streaming error: {traceback.format_exc()}")
            raise e
        finally:
            stop_event.set()
            generation_thread.join(timeout=GENERATION_THREAD_TIMEOUT)

            if generation_thread.is_alive():
                raise ThreadNotKilledError("Generation thread wasn't killed in time.")

    def _stream_generator(self, stop_event: threading.Event, input_data: str) -> Generator[str, None, None]:
        """Handle text stream generation."""
        generated_text = ""
        try:
            while not stop_event.is_set():
                try:
                    new_text = next(self.streamer)
                    generated_text += new_text
                    yield new_text
                except StopIteration:
                    break
                # Check for error after each iteration
                if hasattr(self.streamer, "error") and self.streamer.error is not None:
                    raise self.streamer.error
        finally:
            if generated_text:
                self.update_history(input_data, generated_text.strip())

    def _run_generation_thread(
        self, templated_text: str, gen_kwargs: Dict[str, Any], stop_event: threading.Event
    ) -> None:
        """Run generation in a separate thread."""
        try:
            self.pipe(templated_text, **gen_kwargs)
        except Exception as e:
            self.streamer.error = e  # Store error in streamer
            self.streamer.end()
        finally:
            stop_event.set()


class TextGenerationProcessorOnnx(TextGenerationProcessor):
    def __init__(
        self,
        model_id: str,
        onnx__tokenizer: Union[str, PreTrainedTokenizer],
        onnx__verbose: bool = False,
        onnx__num_threads: int = 1,
        save_history: bool = False,
        system_prompt: str = None,
        **kwargs,
    ):
        """
        Text generation processor using ONNX Runtime.

        Parameters
        ----------
        model_id : str
            The model ID to use for generation.
            Usually the name of the model or the path to the model.
        onnx__tokenizer : Union[str, PreTrainedTokenizer]
            The tokenizer to use for generation.
            If str, it should be the model ID of the tokenizer.
            else, it should be a PreTrainedTokenizer object.
            This tokenizer allows universal use of chat templates.
        onnx__verbose : bool
            Whether to print verbose logs.
        onnx__num_threads : int
            Number of threads to use for generation.
        save_history : bool
            Set to True if you want model to generate responses based on previous inputs.
        system_prompt : str
            The system prompt to prepend to the input text.
        """
        self._validate_model_tokenizer(model_id, onnx__tokenizer)

        super().__init__(model_id, save_history, system_prompt)
        self.onnx__verbose = onnx__verbose
        self.onnx__num_threads = onnx__num_threads

        self._set_tokenizer(onnx__tokenizer)
        self._set_environment_variables()
        self._initialize_model()

    def generate(
        self,
        input_data: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        stopwords: Optional[List[str]] = None,
        buffer_wordsize: int = 10,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate text using the ONNX model.

        Parameters
        ----------
        input_data : str
            The input text to generate a response for.
        max_new_tokens : int
            The maximum number of tokens to generate.
        temperature : float
            The temperature for sampling.
        stopwords : List[str], optional
            List of stop words to stop generation at.
        buffer_wordsize : int
            The buffer word size for generation.
            Larger buffer sizes will check later for stop words.
        generation_kwargs : Dict[str, Any], optional
            Additional keyword arguments for generation.
            Example: {"top_k": 50, "top_p": 0.95}
        """
        generator = self._prepare_generate(input_data, max_new_tokens, temperature, generation_kwargs)

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

                    if stopwords and any(stop_word in generated_text for stop_word in stopwords):
                        break

        except KeyboardInterrupt:
            logger.warning("Control+C pressed, aborting generation")

        generated_text += buffer
        del generator

        total_time = time.time() - start
        logger.info(f"Generation took {total_time:.2f} seconds")
        logger.info(f"Tokens per second: {token_counter / total_time:.2f}")

        self.update_history(input_data, generated_text.strip())

        return generated_text.strip()

    def generate_stream(
        self,
        input_data: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Generate text using the ONNX model.

        Parameters
        ----------
        input_data : str
            The input text to generate a response for.
        max_new_tokens : int
            The maximum number of tokens to generate.
        temperature : float
            The temperature for sampling.
        generation_kwargs : Dict[str, Any], optional
            Additional keyword arguments for generation.
            Example: {"top_k": 50, "top_p": 0.95}
        """
        generator = self._prepare_generate(input_data, max_new_tokens, temperature, generation_kwargs)

        logger.info("Running generation loop ...")
        generated_text = ""

        try:
            while not generator.is_done():
                generator.compute_logits()
                generator.generate_next_token()
                new_text = self.tokenizer_stream.decode(generator.get_next_tokens()[0])
                generated_text += new_text
                yield new_text

        except KeyboardInterrupt:
            logger.warning("Control+C pressed, aborting generation")

        del generator

        self.update_history(input_data, generated_text.strip())

    def _prepare_generate(self, input_data, max_new_tokens, temperature, generation_kwargs):
        templated_text = self.apply_chat_template(input_data, self.transformers_tokenizer)

        search_options = {
            "max_length": max_new_tokens,
            "temperature": temperature,
            **(generation_kwargs or {}),
        }

        input_tokens = self.tokenizer.encode(templated_text)

        params = og.GeneratorParams(self.model)
        params.set_search_options(**search_options)
        params.input_ids = input_tokens
        generator = og.Generator(self.model, params)

        return generator

    def _set_tokenizer(self, onnx__tokenizer):
        if isinstance(onnx__tokenizer, str):
            self.transformers_tokenizer = AutoTokenizer.from_pretrained(onnx__tokenizer)
        else:
            self.transformers_tokenizer = onnx__tokenizer

    def _set_environment_variables(self) -> None:
        os.environ.update(
            {
                "OMP_NUM_THREADS": str(self.onnx__num_threads),
                "OMP_WAIT_POLICY": "ACTIVE",
                "OMP_SCHEDULE": "STATIC",
                "ONNXRUNTIME_INTRA_OP_NUM_THREADS": str(self.onnx__num_threads),
                "ONNXRUNTIME_INTER_OP_NUM_THREADS": "1",
            }
        )

    def _initialize_model(self) -> None:
        logger.info("Loading model...")
        self.model = og.Model(self.model_id)
        self.tokenizer = og.Tokenizer(self.model)
        self.tokenizer_stream = self.tokenizer.create_stream()
        logger.info(f"Model loaded using {self.onnx__num_threads} threads")
        logger.info("Tokenizer created")

    def _validate_model_tokenizer(self, model_id, onnx__tokenizer):
        if og is None:
            raise ImportError(ONNX_MSG)

        if not os.path.exists(model_id):
            raise FileNotFoundError(f"{model_id} does not exist! Ensure the model path is an existing local directory.")

        if not os.path.isdir(model_id):
            raise NotADirectoryError(f"{model_id} is not a directory! Ensure the model path is a directory.")

        model_path_contents = os.listdir(model_id)
        if not "genai_config.json" in model_path_contents:
            raise FileNotFoundError(
                f"{model_id} does not contain a genai_config.json! This file is required for ONNX models."
            )

        if not onnx__tokenizer:
            raise ValueError(
                "No tokenizer found! "
                "A tokenizer from the transformers library is required "
                "for ONNX models, to standardize chat templates."
                "Look into HuggingFace (https://huggingface.co) and find the fitting model to use."
            )


class TextGenerationProcessorGGUF(TextGenerationProcessor):
    def __init__(
        self,
        model_id: str,
        gguf__filename: str = "",
        gguf__verbose: bool = False,
        gguf__n_ctx: int = 512,
        gguf__n_gpu_layers: int = 0,
        gguf__n_batch: int = 512,
        gguf__n_cpu_threads: int = 1,
        save_history: bool = False,
        system_prompt: str = "",
        model_kwargs: Dict[str, Any] = None,
        **kwargs,
    ):
        """
        Text generation processor using GGUF models. Uses llama-cpp.Llama class under the hood.

        Parameters
        ----------
        model_id : str
            The model ID to use for generation.
            Usually the name of the model (on HuggingFace) or the path to the model.
        gguf__filename : str
            The filename of the model to load. This is required when loading a model from huggingface.
        gguf__verbose : bool
            Whether to print verbose logs from llama_cpp.LLama class.
        gguf__n_ctx : int
            The context size for the model.
        gguf__n_gpu_layers : int
            The number of layers to offload to the GPU.
        gguf__n_batch : int
            The batch size for generation. Increase for faster generation, at the cost of memory.
        gguf__n_cpu_threads : int
            The number of CPU threads to use for generation. Increase for much faster generation if multiple cores are available.
        save_history : bool
            Set to True if you want model to generate responses based on previous inputs (eg. chat history).
        system_prompt : str
            The system prompt to prepend to the input text.
        model_kwargs : Dict[str, Any]
            Additional keyword arguments for the model. These get passed directly to llama-cpp.Llama.__init__.
        """
        super().__init__(model_id, save_history, system_prompt)

        if Llama is None:
            raise ImportError(
                """llama-cpp not found. Install it using 'pip install llama-cpp-python'.
                              Please see https://github.com/abetlen/llama-cpp-python for more information."""
            )

        _model_kwargs = {
            "verbose": gguf__verbose,
            "n_ctx": gguf__n_ctx,
            "n_gpu_layers": gguf__n_gpu_layers,
            "n_batch": gguf__n_batch,
            "n_threads": gguf__n_cpu_threads,
            **(model_kwargs or {}),
        }

        check_invalid_input_parameters(Llama.__init__, _model_kwargs)

        if os.path.exists(model_id):
            self.llm = Llama(
                model_path=model_id,
                **_model_kwargs,
            )
        else:
            try:
                self.llm = Llama.from_pretrained(
                    repo_id=model_id,
                    filename=gguf__filename,
                    **_model_kwargs,
                )
            except ValueError as exc:
                error_msg = traceback.format_exc()
                if "Available Files:" in error_msg:
                    files_str = error_msg.split("Available Files:")[1].strip()
                    try:
                        files_list = literal_eval(files_str)
                        gguf_files = sorted(f for f in files_list if f.endswith(".gguf"))

                        logger.error("Specify a valid GGUF file in the 'gguf__filename' parameter")
                        logger.error("Available .gguf files:")
                        for file in gguf_files:
                            logger.error(file)
                    except (ValueError, SyntaxError):
                        logger.error("Could not parse available files list")
                raise InvalidGGUFFileError() from exc

    def generate(
        self,
        input_data: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        stopwords: Optional[List[str]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        templated_text, _generation_kwargs = self._prepare_generate(
            input_data, max_new_tokens, temperature, stopwords, generation_kwargs
        )

        generated_text = ""

        try:
            output = self.llm.create_chat_completion(templated_text, **_generation_kwargs)
            for item in output:
                new_text = item["choices"][0]["delta"].get("content", "")
                generated_text += new_text
                print(new_text, end="", flush=True)
        except KeyboardInterrupt:
            logger.warning("Control+C pressed, aborting generation")
        except Exception:
            logger.error(f"Error occured during generation: \n{traceback.format_exc()}")
        finally:
            print()  # Print newline after generation is done

        self.update_history(input_data, generated_text.strip())

        return generated_text.strip()

    def generate_stream(
        self,
        input_data: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        templated_text, _generation_kwargs = self._prepare_generate(
            input_data, max_new_tokens, temperature, None, generation_kwargs
        )

        generated_text = ""
        try:
            output = self.llm.create_chat_completion(templated_text, **_generation_kwargs)
            for item in output:
                new_text = item["choices"][0]["delta"].get("content", "")
                generated_text += new_text
                yield new_text
        except KeyboardInterrupt:
            logger.warning("Control+C pressed, aborting generation")
        except Exception:
            logger.error(f"Error occured during generation: \n{traceback.format_exc()}")

        self.update_history(input_data, generated_text.strip())

    def _prepare_generate(self, input_data, max_new_tokens, temperature, stopwords, generation_kwargs):
        templated_text = self.apply_chat_template(input_data)

        _generation_kwargs = {
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "stream": True,
        }

        if stopwords:
            _generation_kwargs["stop"] = stopwords

        if generation_kwargs:
            _generation_kwargs.update(generation_kwargs)

        check_invalid_input_parameters(self.llm.create_chat_completion, _generation_kwargs)

        return templated_text, _generation_kwargs

    # override the original apply_chat_template method
    def apply_chat_template(self, input_data: str) -> List[Dict[str, str]]:
        messages = []
        if self.save_history:
            messages = self.history.copy()
        messages.append({"role": "user", "content": input_data})
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        return messages
