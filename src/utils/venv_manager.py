import os
import sys
from typing import Any
import venv
from contextlib import contextmanager
import subprocess

from src.utils.logger_manager import LoggerManager

logger = LoggerManager.get_logger(__name__)


@contextmanager
def create_venv(venv_path: str) -> str:
    """
    Context manager to create and manage a Python virtual environment.

    Parameters
    ----------
    venv_path : str
        The path where the virtual environment will be created.

    Yields
    ------
    str
        Path to the pip executable within the created virtual environment.
    """
    venv.create(venv_path, with_pip=True)
    pip_path = os.path.join(venv_path, "Scripts" if os.name == "nt" else "bin", "pip")
    yield pip_path


def in_venv() -> bool:
    """
    Check if the current Python process is running inside a virtual environment.

    Returns
    -------
    bool
    True if the current process is running inside a virtual environment, False otherwise.
    """
    return hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )


def get_lib_path(venv_path: str) -> str:
    """
    Get the path to the lib directory within the virtual environment.

    Parameters
    ----------
    venv_path : str
        The path to the virtual environment.

    Returns
    -------
    str
        The path to the lib directory.
    """
    return os.path.join(venv_path, "Lib", "site-packages")


def get_python_executable(venv_path: str) -> str:
    """
    Get the path to the Python executable within the virtual environment.

    Parameters
    ----------
    venv_path : str
        The path to the virtual environment.

    Returns
    -------
    str
        The path to the Python executable.
    """
    return os.path.join(venv_path, "Scripts" if os.name == "nt" else "bin", "python")


def get_venv_path() -> str:
    """
    Get the path to the current virtual environment.

    Returns
    -------
    bool
        The path to the current virtual environment.
    """
    if not in_venv():
        raise RuntimeError("Not running inside a virtual environment.")
    return sys.prefix


def get_pip_path(venv_path: str) -> str:
    """
    Get the path to the pip executable within the virtual environment.

    Parameters
    ----------
    venv_path : str
        The path to the virtual environment.

    Returns
    -------
    str
        The path to the pip executable.
    """
    return os.path.join(venv_path, "Scripts" if os.name == "nt" else "bin", "pip")


def install_module(
    module_name: str, pip_path: str, target_dir: str, *args: Any
) -> bool:
    """
    Install a Python module using pip into a temporary directory and add it to sys.path.

    Parameters
    ----------
    module_name : str
        The name of the module to install.
    pip_path : str
        The path to the pip executable.
    temp_dir : str
        The temporary directory where the module should be installed.
    *args : Any
        Additional arguments to pass to the pip install command (e.g., --extra-index-url).

    Returns
    -------
    bool
        True if the installation is successful, False otherwise.

    Examples
    --------
    >>> install_module("some-package", pip_path, temp_dir, "--extra-index-url", "https://private-repo.com/simple")
    """
    pip_command = [pip_path, "install", "--target", target_dir, module_name] + list(
        args
    )
    try:
        # Install the module to the specified temp_dir
        subprocess.check_call(pip_command)
        logger.info(f"Successfully installed {module_name} into {target_dir}")

        # Add target_dir to sys.path so that installed modules can be imported
        if target_dir not in sys.path:
            sys.path.insert(0, target_dir)

        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install {module_name}. Error: {e}")
        return False
