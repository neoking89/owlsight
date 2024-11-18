import os
from pathlib import Path
from typing import Union, Optional, Any, Dict, TypeVar

from owlsight.hugging_face.constants import HUGGINGFACE_TASKS

# ANSI color codes for terminal output
COLOR_CODES = {
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "reset": "\033[0m",  # Resets to default color
}

PROMPT_COLOR = "blue"
CHOICE_COLOR = "green"

T = TypeVar("T")


class ConfigSchema:
    """Configuration schema definition and validation."""

    SCHEMA = {
        "main": {
            "max_retries_on_error": {
                "default": 3,
                "choices": list(range(0, 10)),
                "description": "Maximum number of retries for error recovery",
            },
            "prompt_retry_on_error": {
                "default": False,
                "choices": [False, True],
                "description": "Whether to prompt before retrying on error",
            },
            "prompt_code_execution": {
                "default": True,
                "choices": [False, True],
                "description": "Whether to prompt before executing code",
            },
            "extra_index_url": {
                "default": "",
                "choices": None,
                "description": "Additional URL for package installation",
            },
        },
        "model": {
            "model_id": {"default": "", "choices": None, "description": "Model identifier or path"},
            "save_history": {
                "default": False,
                "choices": [False, True],
                "description": "Whether to save conversation history",
            },
            "system_prompt": {
                "default": """
# ROLE:
You are an advanced problem-solving AI with expert-level knowledge in various programming languages, particularly Python.

# TASK:
- Prioritize Python solutions when appropriate.
- Present code in markdown format.
- Clearly state when non-Python solutions are necessary.
- Break down complex problems into manageable steps and think through the solution step-by-step.
- Adhere to best coding practices, including error handling and consideration of edge cases.
- Acknowledge any limitations in your solutions.
- Always aim to provide the best solution to the user's problem, whether it involves Python or not.
                """.strip(),
                "choices": None,
                "description": "System prompt defining model behavior",
            },
            # Transformers specific
            "transformers__device": {
                "default": None,
                "choices": [None, "cpu", "cuda", "mps"],
                "description": "Device for transformers model",
            },
            "transformers__quantization_bits": {
                "default": None,
                "choices": [None, 8, 4],
                "description": "Quantization bits for transformers model",
            },
            "transformers__stream": {
                "default": True,
                "choices": [False, True],
                "description": "Whether to stream input to transformers model",
            },
            "transformers__use_fp16": {
                "default": False,
                "choices": [False, True],
                "description": "Whether to use FP16 for transformers model",
            },
            # GGUF specific
            "gguf__filename": {"default": "", "choices": None, "description": "GGUF model filename"},
            "gguf__verbose": {
                "default": False,
                "choices": [False, True],
                "description": "Verbose output for GGUF model",
            },
            "gguf__n_ctx": {
                "default": 512,
                "choices": [32 * (2**n) for n in range(15)],
                "description": "Context length for GGUF model",
            },
            "gguf__n_gpu_layers": {
                "default": 0,
                "choices": [-1, 0, 1] + [(2**n) for n in range(1, 9)],
                "description": "Number of GPU layers for GGUF model",
            },
            "gguf__n_batch": {
                "default": 512,
                "choices": [32 * (2**n) for n in range(11)],
                "description": "Batch size for GGUF model",
            },
            "gguf__n_cpu_threads": {
                "default": 1,
                "choices": list(range(1, os.cpu_count() + 1)),
                "description": "Number of CPU threads for GGUF model",
            },
            # ONNX specific
            "onnx__tokenizer": {"default": "", "choices": None, "description": "Tokenizer for ONNX model"},
            "onnx__verbose": {
                "default": False,
                "choices": [False, True],
                "description": "Verbose output for ONNX model",
            },
            "onnx__num_threads": {
                "default": 1,
                "choices": list(range(1, os.cpu_count() + 1)),
                "description": "Number of threads for ONNX model",
            },
        },
        "generate": {
            "stopwords": {"default": [], "choices": None, "description": "Words that stop text generation"},
            "max_new_tokens": {
                "default": 512,
                "choices": [32 * (2**n) for n in range(15)],
                "description": "Maximum tokens to generate",
            },
            "temperature": {
                "default": 0.0,
                "choices": [round(x * 0.05, 2) for x in range(21)],
                "description": "Temperature for text generation",
            },
            "generation_kwargs": {"default": {}, "choices": None, "description": "Additional generation parameters"},
        },
        "rag": {
            "active": {"default": False, "choices": [False, True], "description": "Whether RAG is active"},
            "target_library": {"default": "", "choices": None, "description": "Target python library for RAG"},
            "top_k": {"default": 3, "choices": list(range(1, 51)), "description": "Number of RAG results to return"},
            "search_query": {"default": "", "choices": None, "description": "RAG search query"},
        },
        "huggingface": {
            "search": {"default": "", "choices": None, "description": "search for a model on huggingface"},
            "top_k": {
                "default": 5,
                "choices": list(range(1, 51)),
                "description": "Number of huggingface results to return",
            },
            "select_model": {"default": "", "choices": None, "description": "select a model from huggingface"},
            "task": {
                "default": None,
                "choices": HUGGINGFACE_TASKS,
                "description": "task for huggingface",
            },
        },
    }

    @classmethod
    def get_defaults(cls) -> Dict[str, Dict[str, Any]]:
        """Extract default values from schema."""
        return {
            section: {key: value["default"] for key, value in options.items()}
            for section, options in cls.SCHEMA.items()
        }

    @classmethod
    def get_choices(cls) -> Dict[str, Dict[str, Any]]:
        """Extract choices from schema, adding 'back' option to each section."""
        return {
            section: {
                "back": None,
                **{
                    key: value["choices"] if value["choices"] is not None else value["default"]
                    for key, value in options.items()
                },
            }
            for section, options in cls.SCHEMA.items()
        }

    @classmethod
    def get_descriptions(cls) -> Dict[str, Dict[str, str]]:
        """Extract descriptions from schema."""
        return {
            section: {key: value["description"] for key, value in options.items()}
            for section, options in cls.SCHEMA.items()
        }

    # @classmethod
    # def validate_value(cls, section: str, key: str, value: Any) -> bool:
    #     """Validate a configuration value against the schema."""
    #     if section not in cls.SCHEMA or key not in cls.SCHEMA[section]:
    #         return False

    #     choices = cls.SCHEMA[section][key]["choices"]
    #     if choices is None:
    #         return isinstance(value, type(cls.SCHEMA[section][key]["default"]))
    #     return value in choices

    # @classmethod
    # def get_next_value(cls, section: str, key: str, current_value: T) -> T:
    #     """Get next value from choices, cycling back to first if at end."""
    #     choices = cls.SCHEMA[section][key]["choices"]
    #     if choices is None or not choices:
    #         return current_value

    #     try:
    #         current_idx = choices.index(current_value)
    #         next_idx = (current_idx + 1) % len(choices)
    #         return choices[next_idx]
    #     except ValueError:
    #         return choices[0] if choices else current_value

    # @classmethod
    # def get_prev_value(cls, section: str, key: str, current_value: T) -> T:
    #     """Get previous value from choices, cycling to last if at start."""
    #     choices = cls.SCHEMA[section][key]["choices"]
    #     if choices is None or not choices:
    #         return current_value

    #     try:
    #         current_idx = choices.index(current_value)
    #         prev_idx = (current_idx - 1) % len(choices)
    #         return choices[prev_idx]
    #     except ValueError:
    #         return choices[-1] if choices else current_value


# Create global constants from schema
DEFAULTS = ConfigSchema.get_defaults()
CHOICES = ConfigSchema.get_choices()
DESCRIPTIONS = ConfigSchema.get_descriptions()

# Menu configuration
MENU_KEYS = {
    "assistant": "how can I assist you?",
}

MAIN_MENU = {
    MENU_KEYS["assistant"]: "",
    "shell": "",
    "python": None,
    "config": list(DEFAULTS.keys()),
    "save": "",
    "load": "",
    "clear history": None,
    "quit": None,
}


# Directory and file handling functions
def get_cache_dir() -> Path:
    """Returns the base directory for storing cached data."""
    data_dir = Path.home() / ".owlsight"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def create_directory(path: Union[str, Path], base: Optional[Path] = None) -> Path:
    """Creates a directory if it does not exist and returns the path."""
    full_path = Path(base or get_cache_dir()) / path
    full_path.mkdir(parents=True, exist_ok=True)
    return full_path


def create_file(path: Union[str, Path], base: Optional[Path] = None) -> Path:
    """Creates an empty file if it does not exist and returns the file path."""
    full_path = Path(base or get_cache_dir()) / path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.touch(exist_ok=True)
    return full_path


def get_prompt_cache() -> Path:
    """Returns the path to the prompt history cache file."""
    return create_file(".prompt_history")


def get_py_cache() -> Path:
    """Returns the path to the python history cache file."""
    return create_file(".python_history")


def get_pickle_cache() -> Path:
    """Returns the path to the pickle cache directory."""
    return create_directory(".pickle")
