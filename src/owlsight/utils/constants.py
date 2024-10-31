import os
from pathlib import Path
from typing import Union, Optional

PROMPT_COLOR = "blue"
CHOICE_COLOR = "green"

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

MENU_KEYS = {
    "assistant": "how can I assist you?",
}

DEFAULTS = {
    "main": {
        "max_retries_on_error": 3,
        "prompt_retry_on_error": False,
        "prompt_code_execution": True,
        "extra_index_url": "",
    },
    "model": {
        "model_id": "",
        "save_history": False,
        "system_prompt": """
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
        # specific parameters for the different processors:
        # transformers
        "transformers__device": None,
        "transformers__quantization_bits": None,
        # gguf
        "gguf__filename": "",
        "gguf__verbose": False,
        "gguf__n_ctx": 512,
        "gguf__n_gpu_layers": 0,
        "gguf__n_batch": 512,
        "gguf__n_cpu_threads": 1,
        # onnx
        "onnx__tokenizer": "",
        "onnx__verbose": False,
        "onnx__num_threads": 1,
    },
    "generate": {
        "stopwords": [],
        "max_new_tokens": 512,
        "temperature": 0.0,
        "generation_kwargs": {},
    },
    "rag": {
        "active": False,
        "target_library": "",
        "top_k": 3,
        "search_query": "",
    },
}

CHOICES = {
    "main": {
        "back": None,
        "max_retries_on_error": list(range(0, 10)),
        "prompt_retry_on_error": [False, True],
        "prompt_code_execution": [False, True],
        "extra_index_url": DEFAULTS["main"]["extra_index_url"],
    },
    "model": {
        "back": None,
        "model_id": DEFAULTS["model"]["model_id"],
        "save_history": [False, True],
        "system_prompt": DEFAULTS["model"]["system_prompt"],
        "transformers__device": [None, "cpu", "cuda", "mps"],
        "transformers__quantization_bits": [None, 8, 4],
        "gguf__filename": DEFAULTS["model"]["gguf__filename"],
        "gguf__verbose": [False, True],
        "gguf__n_ctx": [32 * (2**n) for n in range(15)],
        "gguf__n_gpu_layers": [-1, 0, 1] + [(2**n) for n in range(1, 9)],
        "gguf__n_batch": [32 * (2**n) for n in range(11)],
        "gguf__n_cpu_threads": list(range(1, os.cpu_count() + 1)),
        "onnx__tokenizer": DEFAULTS["model"]["onnx__tokenizer"],
        "onnx__verbose": [False, True],
        "onnx__num_threads": DEFAULTS["model"]["onnx__num_threads"],
    },
    "generate": {
        "back": None,
        "stopwords": DEFAULTS["generate"]["stopwords"],
        "max_new_tokens": [32 * (2**n) for n in range(15)],
        "temperature": [round(x * 0.05, 2) for x in range(21)],
        "generation_kwargs": DEFAULTS["generate"]["generation_kwargs"],
    },
    "rag": {
        "back": None,
        "active": [False, True],
        "target_library": DEFAULTS["rag"]["target_library"],
        "top_k": list(range(1, 51)),
        "search_query": DEFAULTS["rag"]["search_query"],
    },
}


def get_cache_dir() -> Path:
    """Returns the base directory for storing cached data."""
    data_dir = Path.home() / ".owlsight"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def create_directory(path: Union[str, Path], base: Optional[Path] = None) -> Path:
    """
    Creates a directory if it does not exist and returns the path.

    Parameters:
    ----------
    path : Union[str, Path]
        The directory path to create.
    base : Optional[Path]
        The base directory for relative paths. Defaults to get_cache_dir().

    Returns:
    -------
    Path
        The created directory path.
    """
    full_path = Path(base or get_cache_dir()) / path
    full_path.mkdir(parents=True, exist_ok=True)
    return full_path


def create_file(path: Union[str, Path], base: Optional[Path] = None) -> Path:
    """
    Creates an empty file if it does not exist and returns the file path.

    Parameters:
    ----------
    path : Union[str, Path]
        The file path to create.
    base : Optional[Path]
        The base directory for relative paths. Defaults to get_cache_dir().

    Returns:
    -------
    Path
        The created file path.
    """
    full_path = Path(base or get_cache_dir()) / path
    full_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    full_path.touch(exist_ok=True)  # Create file if it doesn't exist
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
