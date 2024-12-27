from pathlib import Path
from typing import Union, Optional


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
