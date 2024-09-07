import os
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


def install_module(module_name: str, pip_path: str, *args: Any) -> bool:
    """
    Install a Python module using pip with optional additional arguments.

    Parameters
    ----------
    module_name : str
        The name of the module to install.
    pip_path : str
        The path to the pip executable.
    *args : Any
        Additional arguments to pass to the pip install command (e.g., --extra-index-url).

    Returns
    -------
    bool
        True if the installation is successful, False otherwise.

    Examples
    --------
    >>> install_module("some-package", pip_path, "--extra-index-url", "https://private-repo.com/simple")
    """
    pip_command = [pip_path, "install", module_name] + list(args)
    try:
        subprocess.check_call(pip_command)
        logger.info(f"Successfully installed {module_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install {module_name}. Error: {e}")
        return False
