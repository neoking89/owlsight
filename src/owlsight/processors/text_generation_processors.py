from typing import Optional, List, Dict, Any, Union, Tuple, Generator
import os
import time
import traceback
import threading
from ast import literal_eval

import torch
from huggingface_hub import snapshot_download, list_repo_files
from transformers import (
    BitsAndBytesConfig,
    TextIteratorStreamer,
    AutoTokenizer,
    AutoModel,
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
from owlsight.processors.custom_classes import GGUF_Utils
from owlsight.utils.threads import ThreadNotKilledError
from owlsight.utils.custom_exceptions import QuantizationNotSupportedError, InvalidGGUFFileError
from owlsight.utils.custom_classes import StopWordCriteria
from owlsight.utils.deep_learning import get_best_device, bfloat16_is_supported
from owlsight.huggingface.constants import SUPPORTED_TASKS
from owlsight.utils.helper_functions import validate_input_params
from owlsight.utils.logger import logger

try:
    import onnxruntime_genai as og
except ImportError:
    logger.warning(
        "Support for ONNX models is disabled, because onnxruntime-genai is not found. Install it using 'pip install onnxruntime-genai'."
    )
    og = None

try:
    from llama_cpp import Llama
except ImportError:
    logger.warning(
        "Support for GGUF models is disabled, because llama-cpp is not found. Install it using 'pip install llama-cpp-python'."
    )
    Llama = None


class TextGenerationProcessorTransformers(TextGenerationProcessor):
    """Text generation processor using transformers library."""

    def __init__(
        self,
        model_id: str,
        transformers__device: Optional[str] = None,
        transformers__quantization_bits: Optional[int] = None,
        transformers__stream: bool = True,
        bnb_kwargs: Optional[dict] = None,
        tokenizer_kwargs: Optional[dict] = None,
        task: Optional[str] = None,
        apply_chat_history: bool = False,
        system_prompt: str = "",
        apply_tools: Optional[List[dict]] = None,
        model_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Initialize the transformers text generation processor.

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
        bnb_kwargs : Optional[dict]
            Additional keyword arguments for BitsAndBytesConfig. Default is None.
        tokenizer_kwargs : Optional[dict]
            Additional keyword arguments for the tokenizer. Default is None.
        task : Optional[str]
            The task to use for the pipeline. Default is None, where the task is set to "text-generation".
        apply_chat_history : bool
            Set to True if you want model to generate responses based on previous inputs.
        system_prompt : str
            The system prompt to prepend to the input text.
        apply_tools : Optional[List[dict]]
            A list of tools to call from the processor.
            Default is None.
            Also see: https://medium.com/@malumbea/function-tool-calling-using-gemma-transform-instruction-tuned-it-model-bc8b05585377
        model_kwargs : Optional[dict]
            Additional keyword arguments for the model.
            These get passed to `transformers.pipeline` function as `model_kwargs` argument.
            Default is None.
        """
        if task and task not in SUPPORTED_TASKS:
            raise ValueError(f"Task '{task}' is not supported. Supported tasks are: {list(SUPPORTED_TASKS.keys())}")

        super().__init__(model_id, apply_chat_history, system_prompt, model_kwargs, apply_tools)

        # Initialize configuration
        self.transformers__device = transformers__device or get_best_device()
        self.transformers__stream = transformers__stream
        self.transformers__quantization_bits = transformers__quantization_bits
        self.bnb_kwargs = bnb_kwargs or {}
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.task = task or DEFAULT_TASK

        # Set device and dtype configuration
        self._torch_dtype = self._determine_torch_dtype()

        # Initialize model components
        self._setup_tokenizer_and_model_kwargs()
        self.pipe = self._setup_pipeline()
        self.streamer = self._setup_streamer() if self.transformers__stream else None

    def _determine_torch_dtype(self) -> Any:
        """Determine appropriate torch dtype based on configuration."""
        if self.transformers__quantization_bits == 16:
            if self.transformers__device == "cpu":
                raise TypeError("FP16 is not supported on CPU.")
            return self._get_correct_fp16_dtype()
        return torch.float32 if self.transformers__device == "cpu" else "auto"

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
            "_attn_implementation": "flash" if self._flash_attention_is_available() else "eager",
        }
        if quantization_config is not None:
            kwargs["quantization_config"] = quantization_config
        kwargs.update(self.model_kwargs)
        return kwargs

    def _setup_pipeline(self) -> Pipeline:
        """Set up the generation pipeline using EAFP pattern.

        Attempts to create pipeline with device specification first.
        If that fails due to Accelerate, creates pipeline without device parameter.
        """
        pipeline_kwargs = {
            "task": self.task,
            "model": self.model_id,
            "tokenizer": self.tokenizer,
            "trust_remote_code": True,
            "model_kwargs": self.model_kwargs,
            "device": self.transformers__device,  # Try with device first
        }

        try:
            return pipeline(**pipeline_kwargs)
        except ValueError as e:
            if "model has been loaded with `accelerate`" in str(e):
                # Remove device parameter and retry if using accelerate
                del pipeline_kwargs["device"]
                return pipeline(**pipeline_kwargs)
            raise  # Re-raise if it's a different ValueError

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

    def pipe_call(self, input_data: Union[str, List[str]], **gen_kwargs) -> Any:
        """
        Call the pipeline with input data and kwargs, supporting batch processing.

        Parameters:
        -----------
            input_data (Union[str, List[str]]): Input text or a list of texts to process.
            batch_size (Optional[int]): The size of the batches for pipeline processing.
            **gen_kwargs: Additional keyword arguments passed to the pipeline.

        Returns:
            Any: Processed output from the pipeline.
        """
        # Handle a single string input directly
        if not isinstance(input_data, (list, tuple)):
            logger.debug("Processing single input with pipeline...")
            return self.pipe(input_data, **gen_kwargs)

        # Handle empty list
        if not input_data:
            logger.debug("Received empty input list.")
            return []

        data_len = 1 if not isinstance(input_data, (list, tuple)) else len(input_data)

        batch_size = None
        if isinstance(gen_kwargs, dict):
            batch_size = next(iter(gen_kwargs.values()), {}).pop("batch_size", None)

        # If batch_size is specified and valid, let pipeline handle batching by using a generator
        if batch_size is not None and batch_size > 0:
            logger.info(f"Processing {data_len} inputs in pipeline-managed batches of size {batch_size}...")

            def data_generator():
                yield from input_data

            results = []
            # Passing a generator to the pipeline with batch_size will make it process in batches internally
            for out in self.pipe(data_generator(), batch_size=batch_size, **gen_kwargs):
                results.append(out)
            return results
        else:
            # No batch_size provided, process all at once
            logger.info(f"Processing {data_len} inputs without explicit batching...")
            return self.pipe(input_data, **gen_kwargs)

    def get_max_context_length(self) -> Optional[int]:
        if self.tokenizer:
            return self.tokenizer.model_max_length

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
        output = self.pipe_call(templated_text, **gen_kwargs)
        # get the generated text from the output dictionary
        text_from_output = next(iter(output[0].values()))
        generated_text = text_from_output[len(templated_text) :].strip()
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
            self.pipe_call(templated_text, **gen_kwargs)
        except Exception as e:
            self.streamer.error = e  # Store error in streamer
            self.streamer.end()
        finally:
            stop_event.set()


class TextGenerationProcessorOnnx(TextGenerationProcessor):
    """Text generation processor using ONNX Runtime optimized models.

    This processor enables text generation using ONNX-optimized models,
    which can run on both CPU and GPU. Supports both local models and models from
    Hugging Face Hub.

    Parameters
    ----------
    model_id : str
        Path to local ONNX model or Hugging Face model ID
    onnx__verbose : bool, default=False
        Enable verbose ONNX Runtime logging
    onnx__n_cpu_threads : int, default=8
        Number of CPU threads for computation
    onnx__model_dir : str, optional
        Specific model directory when multiple valid ones exist
    token : str, optional
        Hugging Face token for private models
    apply_chat_history : bool, default=False
        Whether to maintain conversation history
    system_prompt : str, optional
        System prompt prepended to all inputs
    model_kwargs : dict, optional
        Additional keyword arguments to pass to the model.
        Default is None.

    Notes
    -----
    - ONNX models typically offer better CPU performance than PyTorch
    - Thread count affects CPU performance significantly
    - Models must be ONNX-optimized versions of transformers models

    Examples
    --------
    >>> # Load local ONNX model
    >>> processor = TextGenerationProcessorOnnx("path/to/model")
    >>>
    >>> # Load from Hugging Face
    >>> processor = TextGenerationProcessorOnnx(
    ...     "onnx-community/Llama-2-7B-Instruct-ONNX",
    ...     onnx__n_cpu_threads=12
    ... )
    """

    def __init__(
        self,
        model_id: str,
        onnx__verbose: bool = False,
        onnx__n_cpu_threads: int = 8,
        onnx__model_dir: Optional[str] = None,
        token: Optional[str] = None,
        apply_chat_history: bool = False,
        system_prompt: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
        apply_tools: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> None:
        if og is None:
            raise ImportError("ONNX Runtime is disabled. Install with: pip install owlsight[onnx]")
        logger.warning("model_kwargs is currently ignored for ONNX models")

        self.model_id = self._validate_model_id(model_id, onnx__model_dir, token)
        self.transformers_tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=token)

        super().__init__(self.model_id, apply_chat_history, system_prompt, model_kwargs, apply_tools)
        self.onnx__verbose = onnx__verbose
        self.onnx__n_cpu_threads = onnx__n_cpu_threads

        self._set_environment_variables()
        self._initialize_model()

    def _validate_model_id(
        self, model_id: str, onnx__model_dir: Optional[str] = None, token: Optional[str] = None
    ) -> str:
        """
        Initialize the model and tokenizer from either a local path or Hugging Face repo ID.

        Parameters
        ----------
        model_id : str
            The model ID to use for generation.
            Can be either:
            - A local path to an ONNX model directory
            - A HuggingFace Hub model ID
        onnx__model_dir : Optional[str]
            The directory containing the ONNX model.
            Apply this if there are multiple valid directories in the model repository.
        token : Optional[str]
            HuggingFace token for private models.

        Returns
        -------
        str
            The validated model ID path

        Raises
        ------
        ValueError
            If the model_id is neither a valid local path nor a valid Hugging Face repository ID
        """
        # Handle local paths vs Hugging Face repo IDs
        if os.path.exists(model_id):
            # Local path - use directly
            self.model_id = model_id
        else:
            # Try as Hugging Face repo ID
            try:
                self.pre_validate_model_id(model_id, onnx__model_dir)
                allow_patterns = [f"{onnx__model_dir}/*"] if onnx__model_dir else None
                self.model_id = snapshot_download(
                    model_id, token=token, repo_type="model", allow_patterns=allow_patterns
                )
            except Exception as e:
                raise ValueError(
                    f"Invalid model_id: {model_id} is neither a valid local path nor a valid Hugging Face repository ID"
                ) from e

        # Validate and find correct model directory
        self.model_id = self._post_validate_model_id(self.model_id)

        return self.model_id

    def generate(
        self,
        input_data: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        stopwords: Optional[List[str]] = None,
        buffer_wordsize: int = 10,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate text response for the given input.

        Parameters
        ----------
        input_data : str
            Input text to generate from
        max_new_tokens : int, default=512
            Maximum number of tokens to generate
        temperature : float, default=0.0
            Sampling temperature (0.0 = deterministic, higher = more random)
        stopwords : List[str], optional
            Words that will stop generation when encountered
        buffer_wordsize : int, default=10
            Size of word buffer for stopword checking
        generation_kwargs : Dict[str, Any], optional
            Additional generation parameters for ONNX Runtime

        Returns
        -------
        str
            Generated text response

        Examples
        --------
        >>> response = processor.generate(
        ...     "Explain quantum computing:",
        ...     max_new_tokens=200,
        ...     temperature=0.7
        ... )
        """
        generator = self._prepare_generate(input_data, max_new_tokens, temperature, generation_kwargs)

        logger.info("Starting generation...")
        generated_text, buffer = "", ""
        token_counter = 0
        start = time.time()

        try:
            while not generator.is_done():
                new_text = self._get_text_from_generator(generator)
                buffer += new_text
                token_counter += 1

                print(new_text, end="", flush=True)

                if len(buffer.split()) > buffer_wordsize:
                    generated_text += buffer
                    buffer = ""

                    if stopwords and any(stop_word in generated_text for stop_word in stopwords):
                        break

        except KeyboardInterrupt:
            logger.warning("Generation interrupted by user")

        generated_text += buffer
        del generator

        total_time = time.time() - start
        if self.onnx__verbose:
            logger.info(f"Generation took {total_time:.2f}s ({token_counter / total_time:.2f} tokens/s)")

        self.update_history(input_data, generated_text.strip())
        return generated_text.strip()

    def generate_stream(
        self,
        input_data: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Stream generated text tokens one by one.

        Parameters
        ----------
        input_data : str
            Input text to generate from
        max_new_tokens : int, default=512
            Maximum number of tokens to generate
        temperature : float, default=0.0
            Sampling temperature (0.0 = deterministic, higher = more random)
        generation_kwargs : Dict[str, Any], optional
            Additional generation parameters for ONNX Runtime

        Yields
        ------
        str
            Generated text tokens

        Examples
        --------
        >>> for token in processor.generate_stream("Tell me a story"):
        ...     print(token, end="", flush=True)
        """
        generator = self._prepare_generate(input_data, max_new_tokens, temperature, generation_kwargs)
        generated_text = ""

        try:
            while not generator.is_done():
                new_text = self._get_text_from_generator(generator)
                generated_text += new_text
                yield new_text

        except KeyboardInterrupt:
            logger.warning("Generation interrupted by user")

        del generator
        self.update_history(input_data, generated_text.strip())

    def get_max_context_length(self) -> Optional[int]:
        """Get maximum context length for the model."""
        if self.tokenizer:
            return self.transformers_tokenizer.model_max_length
        return None

    def _set_environment_variables(self) -> None:
        """Set ONNX runtime environment variables."""
        os.environ.update(
            {
                "OMP_NUM_THREADS": str(self.onnx__n_cpu_threads),
                "OMP_WAIT_POLICY": "ACTIVE",
                "OMP_SCHEDULE": "STATIC",
                "ONNXRUNTIME_INTRA_OP_NUM_THREADS": str(self.onnx__n_cpu_threads),
                "ONNXRUNTIME_INTER_OP_NUM_THREADS": str(self.onnx__n_cpu_threads),
            }
        )

    def _initialize_model(self) -> None:
        """Initialize the ONNX model and tokenizer."""
        logger.info("Loading ONNX model...")
        try:
            self.model = og.Model(self.model_id)
            self.tokenizer = og.Tokenizer(self.model)
            self.tokenizer_stream = self.tokenizer.create_stream()
            if self.onnx__verbose:
                logger.info(f"Model loaded with {self.onnx__n_cpu_threads} threads")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {str(e)}")

    def _prepare_generate(self, input_data, max_new_tokens, temperature, generation_kwargs):
        """Prepare the generator for text generation."""
        templated_text = self.apply_chat_template(input_data, self.transformers_tokenizer)

        search_options = {
            "max_length": max_new_tokens,
            "temperature": temperature,
            **(generation_kwargs or {}),
        }

        input_tokens = self.tokenizer.encode(templated_text)
        params = og.GeneratorParams(self.model)
        params.set_search_options(**search_options)

        generator = og.Generator(self.model, params)
        generator.append_tokens(input_tokens)
        return generator

    def _post_validate_model_id(self, model_id: str) -> str:
        """Validate the model_id and model_directory after using `snapshot_download`."""
        if not os.path.exists(model_id):
            raise FileNotFoundError(f"Model directory not found: {model_id}")

        configs = ("tokenizer.json", "genai_config.json")
        config_paths = [os.path.join(model_id, config) for config in configs]

        if not all(os.path.exists(p) for p in config_paths):
            for root, _, files in os.walk(model_id):
                if all(config in files for config in configs):
                    return root
        else:
            return model_id

        raise FileNotFoundError(f"Config files not found in model directory: {model_id}")

    def _get_text_from_generator(self, generator):
        generator.generate_next_token()
        new_token = generator.get_next_tokens()[0]
        new_text = self.tokenizer_stream.decode(new_token)
        return new_text

    @staticmethod
    def list_valid_repo_files(repo_id: str) -> List[str]:
        file_list = list_repo_files(repo_id)

        valid_directories = []

        for file in file_list:
            if file.endswith("genai_config.json"):
                directory = file.rsplit("/", 1)[0]
                if f"{directory}/tokenizer.json" in file_list:
                    valid_directories.append(directory)

        return valid_directories

    @staticmethod
    def pre_validate_model_id(model_id: str, onnx__model_dir: str):
        """Validate the model_id and model_directory before using `snapshot_download`."""
        repo_files = TextGenerationProcessorOnnx.list_valid_repo_files(model_id)
        if len(repo_files) > 1:
            if not onnx__model_dir:
                raise ValueError(
                    f"Multiple valid directories found in model repository {model_id}: {repo_files}. Please specify a valid onnx__model_dir."
                )
            if onnx__model_dir not in repo_files:
                raise ValueError(
                    f"Model directory {onnx__model_dir} not found in model repository {model_id}. Valid directories are: {repo_files}"
                )


class TextGenerationProcessorGGUF(TextGenerationProcessor):
    """Text generation processor for GGUF models using llama-cpp.

    This processor enables efficient text generation using GGUF-quantized models,
    which are optimized for CPU and GPU inference. Supports both local models and
    models from Hugging Face Hub.

    Parameters
    ----------
    model_id : str
        Path to local GGUF model or Hugging Face model ID
    gguf__filename : str, optional
        Specific GGUF file to load when using Hugging Face model ID
    gguf__verbose : bool, default=False
        Enable verbose logging from llama-cpp
    gguf__n_ctx : int, optional
        Context window size. Larger values allow longer conversations but use more memory
    gguf__n_gpu_layers : int, default=0
        Number of layers to offload to GPU. Set >0 for GPU acceleration
    gguf__n_batch : int, optional
        Batch size for generation. Increase for faster generation, at the cost of memory.
    gguf__n_cpu_threads : int, optional
        The number of CPU threads to use for generation. Increase for much faster generation if multiple cores are available.
    apply_chat_history : bool, default=False
        Whether to maintain conversation history
    system_prompt : str, default=""
        System prompt prepended to all inputs
    model_kwargs : Optional[Dict[str, Any]]
        Additional arguments passed for the model.
        These get passed to `transformers.pipeline` function as `model_kwargs` argument.
        Default is None.

    Notes
    -----
    - GPU acceleration requires llama-cpp-python build specifically with CUDA support
    - Context size (n_ctx) affects memory usage significantly
    - For optimal performance, adjust n_batch and n_cpu_threads based on hardware

    Examples
    --------
    >>> # Load local GGUF model
    >>> processor = TextGenerationProcessorGGUF("path/to/model.gguf", gguf__n_gpu_layers=20)
    >>>
    >>> # Load from Hugging Face with GPU
    >>> processor = TextGenerationProcessorGGUF(
    ...     "TheBloke/Llama-2-7B-GGUF",
    ...     gguf__filename="llama-2-7b.Q4_K_M.gguf",
    ...     gguf__n_gpu_layers=32
    ... )
    """

    def __init__(
        self,
        model_id: str,
        gguf__filename: str = "",
        gguf__verbose: bool = False,
        gguf__n_ctx: Optional[int] = None,
        gguf__n_gpu_layers: int = 0,
        gguf__n_batch: Optional[int] = None,
        gguf__n_cpu_threads: Optional[int] = None,
        apply_chat_history: bool = False,
        system_prompt: str = "",
        model_kwargs: Dict[str, Any] = None,
        apply_tools: Optional[List[dict]] = None,
        **kwargs,
    ):
        """
        Initialize the GGUF text generation processor.
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
        apply_chat_history : bool
            Set to True if you want model to generate responses based on previous inputs (eg. chat history).
        system_prompt : str
            The system prompt to prepend to the input text.
        model_kwargs : Optional[Dict[str, Any]]
            Additional keyword arguments for the model. These get passed directly to llama-cpp.Llama.__init__.
        """
        super().__init__(model_id, apply_chat_history, system_prompt, model_kwargs, apply_tools)

        if Llama is None:
            raise ImportError(
                """llama-cpp not found. Install it using 'pip install llama-cpp-python'.
                              Please see https://github.com/abetlen/llama-cpp-python for more information."""
            )

        n_batch = gguf__n_batch or GGUF_Utils.get_optimal_n_batch()
        n_cpu_threads = gguf__n_cpu_threads or GGUF_Utils.get_optimal_n_threads()
        n_ctx = gguf__n_ctx or 512

        _model_kwargs = {
            "verbose": gguf__verbose,
            "n_ctx": n_ctx,
            "n_gpu_layers": gguf__n_gpu_layers,
            "n_batch": n_batch,
            "n_threads": n_cpu_threads,
            "n_threads_batch": GGUF_Utils.get_optimal_n_threads_batch(),
            **(self.model_kwargs),
        }

        validate_input_params(Llama.__init__, _model_kwargs)

        # load model
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

                        logger.info("Specify a valid GGUF file in the 'gguf__filename' parameter")
                        logger.info("Available .gguf files:")
                        for file in gguf_files:
                            logger.info(file)
                    except (ValueError, SyntaxError):
                        logger.error("Could not parse available files list")
                raise InvalidGGUFFileError(message=error_msg) from exc

    def generate(
        self,
        input_data: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        stopwords: Optional[List[str]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate text response for the given input.

        Parameters
        ----------
        input_data : str
            Input text to generate from
        max_new_tokens : int, default=512
            Maximum number of tokens to generate
        temperature : float, default=0.1
            Sampling temperature (0.0 = deterministic, higher = more random)
        stopwords : List[str], optional
            Words that will stop generation when encountered
        generation_kwargs : Dict[str, Any], optional
            Additional generation parameters passed to llama-cpp

        Returns
        -------
        str
            Generated text response

        Examples
        --------
        >>> response = processor.generate(
        ...     "What is Python?",
        ...     max_new_tokens=100,
        ...     temperature=0.7,
        ...     stopwords=["END"]
        ... )
        """
        if self.apply_tools:
            generation_kwargs["tools"] = self.apply_tools
            generation_kwargs["tool_choice"] = "auto"
        else:
            if generation_kwargs is not None:
                generation_kwargs.pop("tools", None)
                generation_kwargs.pop("tool_choice", None)

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
    ) -> Generator[str, None, None]:
        """Stream generated text tokens one by one.

        Parameters
        ----------
        input_data : str
            Input text to generate from
        max_new_tokens : int, default=512
            Maximum number of tokens to generate
        temperature : float, default=0.1
            Sampling temperature (0.0 = deterministic, higher = more random)
        generation_kwargs : Dict[str, Any], optional
            Additional generation parameters passed to llama-cpp

        Yields
        ------
        str
            Generated text tokens

        Examples
        --------
        >>> for token in processor.generate_stream("Tell me a story"):
        ...     print(token, end="", flush=True)
        """
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

    def get_max_context_length(self) -> Optional[int]:
        context_length_key = next(filter(lambda metadata: "context_length" in metadata, self.llm.metadata), None)
        if context_length_key:
            val = self.llm.metadata.get(context_length_key, None)
            if val is not None:
                return int(val)
        return None

    # override the original apply_chat_template method
    def apply_chat_template(self, input_data: str) -> List[Dict[str, str]]:
        messages = []
        if self.apply_chat_history:
            messages = self.chat_history.copy()
        messages.append({"role": "user", "content": input_data})
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        return messages

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

        validate_input_params(self.llm.create_chat_completion, _generation_kwargs)

        return templated_text, _generation_kwargs
