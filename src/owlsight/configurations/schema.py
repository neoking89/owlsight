"""Complete schema for the application."""

import os
from typing import Any, Dict, List, Union
import json

from owlsight.rag.constants import SENTENCETRANSFORMER_DEFAULT_MODEL
from owlsight.hugging_face.constants import HUGGINGFACE_TASKS
from owlsight.processors.custom_classes import GGUF_Utils
from owlsight.ui.custom_classes import MenuItem, OptionType


class Schema:
    """Configuration schema definition and validation."""

    CONFIG = {
        "main": {
            "back": MenuItem(type=OptionType.ACTION, description="Main settings for the application"),
            "max_retries_on_error": MenuItem(
                type=OptionType.TOGGLE,
                description="Maximum number of retries for Python code error recovery. This parameter is only used when `prompt_retry_on_error` is set to True.",
                default=3,
                choices=list(range(0, 10)),
            ),
            "prompt_retry_on_error": MenuItem(
                type=OptionType.TOGGLE,
                description="Whether to prompt before retrying on error. Set this to True to avoid direct Python code execution on error!",
                default=True,
                choices=[False, True],
            ),
            "prompt_code_execution": MenuItem(
                type=OptionType.TOGGLE,
                description="Whether to prompt before executing code. Set this to True to avoid direct Python code execution!",
                default=True,
                choices=[False, True],
            ),
            "track_model_usage": MenuItem(
                type=OptionType.TOGGLE,
                description="Show metrics, which tracks GPU/CPU usage, amount of generated words and responsetime of model",
                default=False,
                choices=[False, True],
            ),
            "extra_index_url": MenuItem(
                type=OptionType.EDITABLE,
                description="Additional URL for Python package installation. Useful for example when installing python packages (through pip) from private repositories",
                default="",
                choices=None,
            ),
            "python_compile_mode": MenuItem(
                type=OptionType.TOGGLE,
                description="Compile mode in the Python Interpreter (main menu): 'exec' is suited for defining code blocks, 'single' for direct execution",
                default="single",
                choices=["exec", "single"],
            ),
            "dynamic_system_prompt": MenuItem(
                type=OptionType.TOGGLE,
                description="Experimental feature: update the system prompt based on user input.",
                default=False,
                choices=[False, True],
            ),
            "sequence_on_loading": MenuItem(
                type=OptionType.EDITABLE,
                description="A list of key sequences to execute when loading the configuration. Uses owl_press functionality.",
                default=[],
                choices=None,
            ),
        },
        "model": {
            "back": MenuItem(type=OptionType.ACTION, description="Settings for model loading and configuration"),
            "model_id": MenuItem(
                type=OptionType.EDITABLE,
                description="Model identifier or path. The most important parameter in the configuration, as this will load the model to be used",
                default="",
                choices=None,
            ),
            "apply_chat_history": MenuItem(
                type=OptionType.TOGGLE,
                description="Toggle the inclusion of saved chat history in the prompt. Enable for chat models, disable for instruct models.",
                default=True,
                choices=[False, True],
            ),
            "system_prompt": MenuItem(
                type=OptionType.EDITABLE, description="System prompt defining model behavior", default="", choices=None
            ),
            "model_kwargs": MenuItem(
                type=OptionType.EDITABLE,
                description="Additional parameters passed during model initialization. For llama-cpp, these get passed to llama_cpp.Llama. For transformers, these get passed to transformers.pipeline",
                default={},
                choices=None,
            ),
            "transformers__device": MenuItem(
                type=OptionType.TOGGLE,
                description="Device for transformers model",
                default=None,
                choices=[None, "cpu", "cuda", "mps"],
            ),
            "transformers__quantization_bits": MenuItem(
                type=OptionType.TOGGLE,
                description="Quantization bits for transformers model",
                default=None,
                choices=[None, 4, 8, 16],
            ),
            "transformers__stream": MenuItem(
                type=OptionType.TOGGLE,
                description="Whether to stream input to transformers model",
                default=True,
                choices=[False, True],
            ),
            "gguf__filename": MenuItem(
                type=OptionType.EDITABLE, description="GGUF model filename", default="", choices=None
            ),
            "gguf__verbose": MenuItem(
                type=OptionType.TOGGLE,
                description="Verbose output for GGUF model",
                default=False,
                choices=[False, True],
            ),
            "gguf__n_ctx": MenuItem(
                type=OptionType.TOGGLE,
                description="Context length for GGUF model",
                default=2048,
                choices=[32 * (2**n) for n in range(15)],
            ),
            "gguf__n_gpu_layers": MenuItem(
                type=OptionType.TOGGLE,
                description="Number of layers from the model which are offloaded to the GPU",
                default=0,
                choices=[-1, 0, 1] + [(2**n) for n in range(1, 11)],
            ),
            "gguf__n_batch": MenuItem(
                type=OptionType.TOGGLE,
                description="Batch size to be used by GGUF model",
                default=GGUF_Utils.get_optimal_n_batch(),
                choices=[4 * (2**n) for n in range(13)],
            ),
            "gguf__n_cpu_threads": MenuItem(
                type=OptionType.TOGGLE,
                description="Number of CPU threads to be used by GGUF model.",
                default=GGUF_Utils.get_optimal_n_threads(),
                choices=list(range(1, os.cpu_count() + 1)),
            ),
            "onnx__model_dir": MenuItem(
                type=OptionType.EDITABLE, description="Directory containing local ONNX model", default="", choices=None
            ),
            "onnx__verbose": MenuItem(
                type=OptionType.TOGGLE,
                description="Verbose output for ONNX model",
                default=False,
                choices=[False, True],
            ),
            "onnx__n_cpu_threads": MenuItem(
                type=OptionType.TOGGLE,
                description="Number of CPU threads to be used by ONNX model",
                default=GGUF_Utils.get_optimal_n_threads(),
                choices=list(range(1, os.cpu_count() + 1)),
            ),
        },
        "generate": {
            "back": MenuItem(type=OptionType.ACTION, description="Settings for model generation"),
            "stopwords": MenuItem(
                type=OptionType.EDITABLE,
                description="Stopwords that stop text generation. This can be useful for getting more control over when modelgeneration should stop. Pass these like `['stop', 'word']`",
                default=[],
                choices=None,
            ),
            "max_new_tokens": MenuItem(
                type=OptionType.TOGGLE,
                description="Maximum amount of tokens to generate",
                default=2048,
                choices=[32 * (2**n) for n in range(15)],
            ),
            "temperature": MenuItem(
                type=OptionType.TOGGLE,
                description="Temperature for model generation",
                default=0.7,
                choices=[round(x * 0.05, 2) for x in range(21)],
            ),
            "generation_kwargs": MenuItem(
                type=OptionType.EDITABLE,
                description="Additional generation parameters, like top_k, top_p, etc. Pass these like `{'top_k': 4, 'top_p': 0.9}`",
                default={},
                choices=None,
            ),
        },
        "rag": {
            "back": MenuItem(
                type=OptionType.ACTION,
                description="Use RAG for installed python libraries with a combination of TFIDF and an optional embedding model. Default weights are TFIDF-weight=1, embedding-weight=0",
            ),
            "active": MenuItem(
                type=OptionType.TOGGLE,
                description="Whether RAG for python libraries is active. If True, the search-results will be implicitly added as context to the modelprompt and when pressing ENTER, search-results will be shown",
                default=False,
                choices=[False, True],
            ),
            "target_library": MenuItem(
                type=OptionType.EDITABLE,
                description="Target python library for to use for RAG. If the library is not installed in the active environment, a warning will be showed with available options",
                default="",
                choices=None,
            ),
            "top_k": MenuItem(
                type=OptionType.TOGGLE,
                description="Number of most matching RAG results to return, based on `search` query",
                default=10,
                choices=list(range(1, 51)),
            ),
            "sentence_transformer_weight": MenuItem(
                type=OptionType.TOGGLE,
                description="Weight for the embedding model. TFIDF-weight is 1 - `sentence_transformer_weight`",
                default=0.0,
                choices=[round(x * 0.05, 2) for x in range(21)],
            ),
            "sentence_transformer_name_or_path": MenuItem(
                type=OptionType.EDITABLE,
                description="Name or path to a sentence-transformer model, which is used for embedding",
                default=SENTENCETRANSFORMER_DEFAULT_MODEL,
                choices=None,
            ),
            "search": MenuItem(
                type=OptionType.EDITABLE,
                description="RAG search query. Press ENTER to show the `top_k` results. Only used when `active` is True",
                default="",
                choices=None,
            ),
        },
        "huggingface": {
            "back": MenuItem(
                type=OptionType.ACTION, description="Connect with the Hugging Face Hub for model search and loading"
            ),
            "search": MenuItem(
                type=OptionType.EDITABLE,
                description="Search for a model on the Hugging Face Hub by pressing ENTER. Keywords can be used optionally to finetune searchresults, e.g. 'llama 3b gguf'",
                default="",
                choices=None,
            ),
            "top_k": MenuItem(
                type=OptionType.TOGGLE,
                description="Top number of Hugging Face results to return. The results will be sorted by highest score first",
                default=10,
                choices=list(range(1, 51)),
            ),
            "select_model": MenuItem(
                type=OptionType.TOGGLE,
                description="Select and load a model from the Hugging Face Hub by toggling through the options found by `search`",
                default="",
                choices=None,
            ),
            "task": MenuItem(
                type=OptionType.TOGGLE,
                description="Filter Hugging Face models by task. When using `search`, the results will be filtered directly by chosen task",
                default=None,
                choices=HUGGINGFACE_TASKS,
            ),
        },
    }

    MENU = {
        "assistant": MenuItem(
            type=OptionType.EDITABLE,
            description="Chat with the loaded model. Use {{expression}} to pass python code directly. Or e.g. [[image: path/to/image.jpg]] to pass an image to the model",
            default="How can I assist you?",
        ),
        "shell": MenuItem(type=OptionType.EDITABLE, description="Execute shell commands", default=""),
        "python": MenuItem(type=OptionType.ACTION, description="Enter Python interpreter"),
        "config": MenuItem(
            type=OptionType.TOGGLE,
            description="Configuration settings",
            choices=list(CONFIG.keys()),
        ),
        "save": MenuItem(type=OptionType.EDITABLE, description="Save current configuration as JSON-file", default=""),
        "load": MenuItem(type=OptionType.EDITABLE, description="Load a configuration from a JSON-file", default=""),
        "clear history": MenuItem(type=OptionType.ACTION, description="Clear cache and chat history"),
        "quit": MenuItem(type=OptionType.ACTION, description="Exit application"),
    }

    @classmethod
    def get_config_defaults(cls, as_json: bool = False) -> Union[str, Dict[str, Dict[str, Any]]]:
        """
        Extract default values from schema.

        Parameters:
        ----------
        as_json : bool
            Whether to return the default values in json format as a string.

        Returns:
        -------
        Union[str, Dict[str, Dict[str, Any]]]
            Default values as a dictionary or json string if `as_json` is True.
        """
        d = {
            section: {key: value.default for key, value in options.items() if value.type != OptionType.ACTION}
            for section, options in cls.CONFIG.items()
        }
        if as_json:
            return json.dumps(d, indent=4)

        return d

    @classmethod
    def get_config_choices(cls) -> Dict[str, Dict[str, Any]]:
        """Extract choices from schema, adding 'back' option to each section."""
        return {
            section: {
                "back": None,
                **{
                    key: value.choices if value.choices is not None else value.default for key, value in options.items()
                },
            }
            for section, options in cls.CONFIG.items()
        }

    @classmethod
    def get_config_descriptions(cls) -> Dict[str, Dict[str, str]]:
        """Extract descriptions from config schema."""
        return {
            section: {key: value.description for key, value in options.items()}
            for section, options in cls.CONFIG.items()
        }

    @classmethod
    def get_config_types(cls) -> Dict[str, Dict[str, OptionType]]:
        """Extract types from config schema."""
        return {section: {key: value.type for key, value in options.items()} for section, options in cls.CONFIG.items()}

    @classmethod
    def get_main_menu(cls) -> Dict[str, Union[str, None, List]]:
        """Extract main menu options from schema."""
        menu_dict = {
            key: (menu.choices if menu.type == OptionType.TOGGLE else menu.default) for key, menu in cls.MENU.items()
        }
        new_key = menu_dict.pop("assistant")
        menu_dict = {new_key: "", **menu_dict}
        return menu_dict

    @classmethod
    def generate_diagram(cls) -> str:
        """Generate a human-readable diagram of the application structure."""
        lines = ["Main Menu:"]

        for key, item in cls.MENU.items():
            lines.append(f"- {key}: {item.description}")
            if key == "config" and item.choices:
                for section in item.choices:
                    lines.append(f"  - {section} settings:")
                    lines.append("    - back: Return to previous menu")
                    if section in cls.CONFIG:
                        for setting, details in cls.CONFIG[section].items():
                            if details.type == OptionType.ACTION:
                                continue
                            choices = details.choices
                            desc = details.description
                            choice_str = f"Options: {', '.join(str(c) for c in choices)}" if choices else ""
                            optiontype_str = f"Type: {details.type}" if details.type else ""
                            line = f"    - {setting}: {desc}"
                            if choice_str:
                                line += f", {choice_str}"
                            if optiontype_str:
                                line += f", {optiontype_str}"
                            lines.append(line)

        return "\n".join(lines)
