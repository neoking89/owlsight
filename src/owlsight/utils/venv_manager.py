import os
import sys
from typing import Any, List, Union
import venv
from contextlib import contextmanager
import subprocess
import sysconfig
import tempfile
from pathlib import Path

from owlsight.utils.helper_functions import os_is_windows
from owlsight.utils.logger_manager import LoggerManager


logger = LoggerManager.get_logger(__name__)


@contextmanager
def create_venv(pyenv_path: str) -> str:
    """
    Context manager to create and manage a Python virtual environment.

    Parameters
    ----------
    pyenv_path : str
        The path where the virtual environment will be created.

    Yields
    ------
    str
        Path to the pip executable within the created virtual environment.
    """
    venv.create(pyenv_path, with_pip=True)
    pip_path = os.path.join(pyenv_path, "Scripts" if os_is_windows() else "bin", "pip")
    yield pip_path


def in_venv() -> bool:
    """
    Check if the current Python process is running inside a virtual environment.

    Returns
    -------
    bool
    True if the current process is running inside a virtual environment, False otherwise.
    """
    return hasattr(sys, "real_prefix") or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)


def get_lib_path(pyenv_path: str) -> str:
    """
    Get the path to the lib directory within the virtual environment.

    Parameters
    ----------
    pyenv_path : str
        The path to the (virtual) python environment.

    Returns
    -------
    str
        The path to the lib directory.
    """
    # Get the name of the site-packages directory
    site_packages = sysconfig.get_path("purelib", vars={"base": pyenv_path})
    return site_packages


def get_python_executable(pyenv_path: str) -> str:
    """
    Get the path to the Python executable within the virtual environment.

    Parameters
    ----------
    pyenv_path : str
        The path to the virtual environment.

    Returns
    -------
    str
        The path to the Python executable.
    """
    return os.path.join(pyenv_path, "Scripts" if os_is_windows() else "bin", "python")


def get_pyenv_path() -> str:
    """
    Get the path to the current (virtual) python environment.

    Returns
    -------
    bool
        The path to the current (virtual) python environment.
    """
    # if not in_venv():
    #     raise RuntimeError("Not running inside a virtual environment.")
    return sys.prefix


def get_pip_path(pyenv_path: str) -> str:
    """
    Get the path to the pip executable within the (virtual) python environment.

    Parameters
    ----------
    pyenv_path : str
        The path to the (virtual) python environment.

    Returns
    -------
    str
        The path to the pip executable.
    """
    return os.path.join(pyenv_path, "Scripts" if os_is_windows() else "bin", "pip")

def get_temp_dir(suffix: str) -> str:
    """
    Get an appropriate temporary directory path that the user has write permissions for.
    
    Parameters
    ----------
    suffix : str
        The suffix to be appended to the temporary directory path.
        e.g., ".owlsight_temp"

    Returns
    -------
    str
        The path to a writable temporary directory
    """
    # Try user's home directory first
    user_temp = Path.home() / suffix
    try:
        os.makedirs(user_temp, exist_ok=True)
        return user_temp
    except Exception:
        # Fall back to system temp directory
        return tempfile.gettempdir()

def install_python_modules(module_names: Union[str, List[str]], pip_path: str, target_dir: str, *args: Any) -> bool:
    """
    Install one or more Python modules using pip into a specified directory and add it to sys.path.

    Parameters
    ----------
    module_names : Union[str, List[str]]
        The name of the module(s) to install. Can be a single module as a string or a list of modules.
    pip_path : str
        The path to the pip executable.
    target_dir : str
        The directory where the module(s) should be installed.
    *args : Any
        Additional arguments to pass to the pip install command (e.g., --extra-index-url).

    Returns
    -------
    bool
        True if all installations are successful, False otherwise.

    Examples
    --------
    >>> install_python_modules("some-package", pip_path, temp_dir, "--extra-index-url", "https://private-repo.com/simple")
    >>> install_python_modules(["some-package", "another-package"], pip_path, temp_dir)
    """

    # Convert module_name to a list if it's a string (comma-separated or space-separated)
    if isinstance(module_names, str):
        module_names = [name.strip() for name in module_names.split(" ")]

    success = True

    for module in module_names:
        pip_command = [pip_path, "install", "--target", target_dir, module] + list(args)
        try:
            # Install the module
            subprocess.check_call(pip_command)
            logger.info(f"Successfully installed {module} into {target_dir}")

            # Add target_dir to sys.path so that installed modules can be imported
            if target_dir not in sys.path:
                sys.path.insert(0, target_dir)

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {module}. Error: {e}")
            success = False

    return success
