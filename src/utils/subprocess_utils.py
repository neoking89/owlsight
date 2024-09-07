import subprocess
from typing import Tuple
import os
import platform

from src.utils.logger_manager import LoggerManager

logger = LoggerManager.get_logger(__name__)


def run_subprocess(command: list) -> Tuple[str, str]:
    """
    Run subprocess command and capture stdout and stderr.

    Parameters
    ----------
    command : list
        List of command arguments to be executed.

    Returns
    -------
    tuple of (str, str)
        The stdout and stderr outputs from the subprocess.
    """
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    stdout, stderr = process.communicate()
    return stdout, stderr


def _build_shell_command(activate_script: str, command: str) -> str:
    """
    Build the full shell command for different platforms.

    Parameters
    ----------
    activate_script : str
        Path to the activation script.
    command : str
        The shell command to execute after activating the virtual environment.

    Returns
    -------
    str
        The full shell command including virtual environment activation.
    """
    if platform.system().lower() == "windows":
        # For Windows, use `call` to activate and `&&` to run the Python command
        return f'call "{activate_script}" && {command}'
    else:
        # For Unix-like systems, use `source` to activate and `&&` to run the Python command
        return f'bash -c "source {activate_script} && {command}"'


def _log_shell_output(result: subprocess.CompletedProcess) -> None:
    """
    Log the output of a shell command.

    Parameters
    ----------
    result : subprocess.CompletedProcess
        The result of the executed shell command.

    Returns
    -------
    None
    """
    if result.stdout:
        logger.info(result.stdout)
    elif result.stderr:
        logger.warning(f"Command produced stderr output: {result.stderr}")
    else:
        raise ValueError("No output from shell command.")


def _get_activate_script(venv_path: str) -> str:
    """
    Get the path to the virtual environment's activation script.

    Parameters
    ----------
    venv_path : str
        Path to the virtual environment.

    Returns
    -------
    str
        The path to the activation script for the virtual environment.
    """
    return os.path.join(
        venv_path,
        "Scripts" if platform.system().lower() == "windows" else "bin",
        "activate",
    )


def execute_shell_command(command: str, venv_path: str) -> None:
    """
    Execute a shell command inside the virtual environment.

    Parameters
    ----------
    command : str
        The shell command to execute.
    venv_path : str
        Path to the virtual environment.

    Returns
    -------
    None
    """
    activate_script = _get_activate_script(venv_path)
    full_command = _build_shell_command(activate_script, command)
    try:
        result = subprocess.run(
            full_command, shell=True, capture_output=True, text=True, check=True
        )
        _log_shell_output(result)
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}: {e.stderr}")
