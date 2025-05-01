from pathlib import Path
from typing import Union, Optional


KB_AUTOCOMPLETE = ("escape", "v")


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


def get_default_config_on_startup_path(return_cache_path: bool = False) -> str:
    """
    Returns the path to the JSON configuration file which is used for the default config on startup.
    Use this as value for the main.default_config_on_startup key.

    Parameters
    ----------
    return_cache_path : bool, optional
        Whether to return the path to the cache file where the path to the config file is stored instead of the path to the config file, by default False.

    Returns
    -------
    str
        The path to the configuration file OR the path to the cache file depending on the return_cache_path parameter.
    """
    cache_path = create_file(".default_config")
    if return_cache_path:
        return cache_path

    with open(cache_path, "r") as f:
        default_config_path = f.read().strip()
    return default_config_path


def get_prompt_cache() -> Path:
    """Returns the path to the prompt history cache file."""
    return create_file(".prompt_history")


def get_py_cache() -> Path:
    """Returns the path to the python history cache file."""
    return create_file(".python_history")


def get_pickle_cache() -> Path:
    """Returns the path to the pickle cache directory."""
    return create_directory(".pickle")
