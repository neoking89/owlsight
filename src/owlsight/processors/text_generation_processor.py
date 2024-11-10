from abc import ABC
from typing import Optional, List, Dict, Any, Type, Union
import os
import time
import traceback
from ast import literal_eval

import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
    AutoTokenizer,
    PreTrainedTokenizer,
    pipeline,
)
from owlsight.utils.threads import KillableThread
from owlsight.utils.custom_exceptions import QuantizationNotSupportedError
from owlsight.utils.custom_classes import StopWordCriteria
from owlsight.utils.logger import logger
from owlsight.utils.deep_learning import get_best_device
from owlsight.utils.helper_functions import check_invalid_input_parameters



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


def select_processor_type(model_id: str) -> Type["TextGenerationProcessor"]:
    """
    Utilityfunction which selects the appropriate TextGenerationProcessor class based on the model ID or directory.

    If the model_id is a directory, the function will inspect the contents of the directory
    to decide the processor type. Otherwise, it will use the model_id string to make the decision.
    """
    # Check if the model_id is a directory
    if os.path.isdir(model_id):
        # Check if any file in the directory ends with .onnx
        if any(f.endswith("onnx") for f in os.listdir(model_id)):
            return TextGenerationProcessorOnnx
        elif model_id.lower().endswith("gguf") or any(f.endswith("gguf") for f in os.listdir(model_id)):
            return TextGenerationProcessorGGUF
        else:
            return TextGenerationProcessorTransformers
    else:
        # If model_id is not a directory, use the model_id string
        if model_id.lower().endswith("gguf"):
            return TextGenerationProcessorGGUF
        elif "onnx" in model_id.lower():
            return TextGenerationProcessorOnnx
        else:
            return TextGenerationProcessorTransformers


def flash_attention_is_available() -> bool:
    try:
        from flash_attn import flash_attn_fn

        return True
    except ImportError:
        return False


class TextGenerationProcessor(ABC):
    def __init__(
        self,
        model_id: str,
        save_history: bool,
        system_prompt: str,
    ):
        """
        Abstract class for text generation processors.

        Parameters
        ----------
        model_id: str
            The model ID to use for generation.
            Usually the name of the model or the path to the model.
        save_history : bool
            Whether or not to save the history of inputs and outputs.
        """
        if not model_id:
            raise ValueError("Model ID cannot be empty.")

        self.model_id = model_id
        self.save_history = save_history
        self.history = []
        self.system_prompt = system_prompt

    def apply_chat_template(
        self,
        input_text: str,
        tokenizer: PreTrainedTokenizer,
    ) -> str:
        """
        Apply chat template to the input text.
        This is used to format the input text before generating a response and should be universal across all models.
        """
        if tokenizer.chat_template is not None:
            messages = self.get_history()
            messages.append({"role": "user", "content": input_text})
            templated_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            logger.warning("Chat template not found in tokenizer. Using input text as is.")
            templated_text = input_text

        return templated_text

    def update_history(self, input_text: str, generated_text: str):
        """Update the history with the input and generated text."""
        if self.save_history:
            self.history.append({"role": "user", "content": input_text})
            self.history.append({"role": "assistant", "content": generated_text.strip()})

    def get_history(self) -> List[Dict[str, str]]:
        """Get complete chathistory of inputs and outputs and system prompt."""
        messages = self.history.copy()
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        return messages

    def generate(self) -> str:
        raise NotImplementedError("generate method must be implemented in the subclass.")

    def generate_stream(self) -> str:
        raise NotImplementedError("generate_stream method must be implemented in the subclass.")


class TextGenerationProcessorTransformers(TextGenerationProcessor):
    def __init__(
        self,
        model_id: str,
        transformers__device: str = None,
        transformers__quantization_bits: Optional[int] = None,
        bnb_kwargs: Optional[dict] = None,
        tokenizer_kwargs: Optional[dict] = None,
        model_kwargs: Optional[dict] = None,
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
        bnb_kwargs : Optional[dict]
            Additional keyword arguments for BitsAndBytesConfig. Default is None.
        tokenizer_kwargs : Optional[dict]
            Additional keyword arguments for the tokenizer. Default is None.
        model_kwargs : Optional[dict]
            Additional keyword arguments for the model. Default is None.
        save_history : bool
            Set to True if you want model to generate responses based on previous inputs.
        system_prompt : str
            The system prompt to prepend to the input text.
        """
        super().__init__(model_id, save_history, system_prompt)
        self.transformers__device = transformers__device or get_best_device()
        self._attention_implementation = "flash" if flash_attention_is_available() else "eager"
        self.history = []

        tokenizer, model = self._load_tokenizer_model(
            transformers__quantization_bits,
            tokenizer_kwargs=tokenizer_kwargs or {},
            bnb_kwargs=bnb_kwargs or {},
            model_kwargs=model_kwargs or {},
        )
        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto" if self.transformers__device != "cpu" else {"": "cpu"},
        )
        self.streamer = TextIteratorStreamer(self.pipe.tokenizer, skip_prompt=True, skip_special_tokens=True)

    def _load_tokenizer_model(
        self,
        transformers__quantization_bits: Optional[int],
        tokenizer_kwargs: Dict,
        bnb_kwargs: Dict,
        model_kwargs: Dict,
    ):
        if transformers__quantization_bits and self.transformers__device in [
            "cpu",
            "mps",
        ]:
            raise QuantizationNotSupportedError("Quantization is not supported on CPU or MPS.")

        quantization_config = None
        if transformers__quantization_bits == 4:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                **bnb_kwargs,
            )
        elif transformers__quantization_bits == 8:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True, **bnb_kwargs)

        model_kwargs.update(
            {
                "device_map": ("auto" if self.transformers__device != "cpu" else {"": "cpu"}),
                "trust_remote_code": True,
                "torch_dtype": ("auto" if self.transformers__device != "cpu" else torch.float32),
                "quantization_config": quantization_config,
                "_attn_implementation": self._attention_implementation,
            }
        )

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
        templated_text, _generation_kwargs = self._prepare_generate(
            input_text, max_new_tokens, temperature, stopwords, generation_kwargs
        )

        generation_thread = KillableThread(target=self.pipe, args=(templated_text,), kwargs=_generation_kwargs)
        generation_thread.start()
        generated_text = ""

        try:
            for new_text in self.streamer:
                generated_text += new_text
                print(new_text, end="", flush=True)
        except KeyboardInterrupt:
            logger.warning("Control+C pressed, aborting generation")
        finally:
            print()  # Print newline after generation is done
            generation_thread.kill()
            generation_thread.join()

        self.update_history(input_text, generated_text.strip())

        return generated_text

    @torch.inference_mode()
    def generate_stream(
        self,
        input_text: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        templated_text, _generation_kwargs = self._prepare_generate(
            input_text, max_new_tokens, temperature, None, generation_kwargs
        )

        generated_text = ""
        generation_thread = KillableThread(target=self.pipe, args=(templated_text,), kwargs=_generation_kwargs)
        generation_thread.start()

        try:
            for new_text in self.streamer:
                generated_text += new_text
                yield new_text
        except KeyboardInterrupt:
            logger.warning("Control+C pressed, aborting generation")
        finally:
            generation_thread.kill()
            generation_thread.join()

        self.update_history(input_text, generated_text.strip())

    def _prepare_generate(self, input_text, max_new_tokens, temperature, stopwords, generation_kwargs):
        templated_text = self.apply_chat_template(input_text, self.pipe.tokenizer)

        _generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "streamer": self.streamer,
            "eos_token_id": self.pipe.tokenizer.eos_token_id,
            "temperature": temperature if temperature > 0.0 else None,
            "do_sample": temperature > 0.0,
        }

        if stopwords is not None:
            _generation_kwargs["stopping_criteria"] = StopWordCriteria(
                prompts=[templated_text],
                stop_words=stopwords,
                tokenizer=self.pipe.tokenizer,
            )

        if generation_kwargs is not None:
            _generation_kwargs.update(generation_kwargs)
        return templated_text, _generation_kwargs


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
        if og is None:
            raise ImportError(ONNX_MSG)

        if not os.path.exists(model_id):
            raise FileNotFoundError(f"Model not found at {model_id}")
        if not onnx__tokenizer:
            raise ValueError(
                "No tokenizer found! "
                "A tokenizer from the transformers library is required "
                "for ONNX models, to standardize chat templates."
                "Look into HuggingFace (https://huggingface.co) and find the fitting model to use."
            )

        super().__init__(model_id, save_history, system_prompt)
        self.onnx__verbose = onnx__verbose
        self.onnx__num_threads = onnx__num_threads
        self.history = []

        self._set_tokenizer(onnx__tokenizer)
        self._set_environment_variables()
        self._initialize_model()

    def generate(
        self,
        input_text: str,
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
        input_text : str
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
        generator = self._prepare_generate(input_text, max_new_tokens, temperature, generation_kwargs)

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

        self.update_history(input_text, generated_text.strip())

        return generated_text.strip()

    def generate_stream(
        self,
        input_text: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate text using the ONNX model.

        Parameters
        ----------
        input_text : str
            The input text to generate a response for.
        max_new_tokens : int
            The maximum number of tokens to generate.
        temperature : float
            The temperature for sampling.
        generation_kwargs : Dict[str, Any], optional
            Additional keyword arguments for generation.
            Example: {"top_k": 50, "top_p": 0.95}
        """
        generator = self._prepare_generate(input_text, max_new_tokens, temperature, generation_kwargs)

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

        self.update_history(input_text, generated_text.strip())

    def _prepare_generate(self, input_text, max_new_tokens, temperature, generation_kwargs):
        templated_text = self.apply_chat_template(input_text, self.transfomers_tokenizer)

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
            self.transfomers_tokenizer = AutoTokenizer.from_pretrained(onnx__tokenizer)
        else:
            self.transfomers_tokenizer = onnx__tokenizer

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
            raise ImportError("llama-cpp not found. Install it using 'pip install llama-cpp-python'.")

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
            # if not gguf__filename:
            #     raise ValueError("gguf__filename is required when loading a model from HuggingFace.")
            try:
                self.llm = Llama.from_pretrained(
                    repo_id=model_id,
                    filename=gguf__filename,
                    **_model_kwargs,
                )
            except ValueError as e:
                error_msg = str(e)
                if "Available Files:" in error_msg:
                    files_str = error_msg.split("Available Files:")[1].strip()
                    try:
                        files_list = literal_eval(files_str)
                        gguf_files = sorted([f for f in files_list if f.endswith('.gguf')])
                        
                        print("Specify gguf__filename in config:model from the following list:")
                        print("Available .gguf files:")
                        for file in gguf_files:
                            print(f"- {file}")
                    except (ValueError, SyntaxError):
                        logger.error("Could not parse available files list")
                raise  # Re-raise the original error in all cases

    def generate(
        self,
        input_text: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        stopwords: Optional[List[str]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        templated_text, _generation_kwargs = self._prepare_generate(
            input_text, max_new_tokens, temperature, stopwords, generation_kwargs
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

        self.update_history(input_text, generated_text.strip())

        return generated_text.strip()

    def generate_stream(
        self,
        input_text: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        templated_text, _generation_kwargs = self._prepare_generate(
            input_text, max_new_tokens, temperature, None, generation_kwargs
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

        self.update_history(input_text, generated_text.strip())

    def _prepare_generate(self, input_text, max_new_tokens, temperature, stopwords, generation_kwargs):
        templated_text = self.apply_chat_template(input_text)

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
    def apply_chat_template(self, input_text) -> List[Dict[str, str]]:
        messages = []
        if self.save_history:
            messages = self.history.copy()
        messages.append({"role": "user", "content": input_text})
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        return messages
