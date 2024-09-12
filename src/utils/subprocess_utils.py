import subprocess
from typing import Tuple
import os
import platform
import re
from ast import literal_eval

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
    if result.stderr:
        logger.warning(f"Command produced stderr output: {result.stderr}")
    if hasattr(result, "output") and result.output:
        logger.warning(f"Command produced output: {result.output}")


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


def execute_shell_command(command: str, venv_path: str) -> subprocess.CompletedProcess:
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
    subprocess.CompletedProcess
        The result of the subprocess run or the exception if failed.
    """
    # first activate venv and then run the command
    activate_venv = _get_activate_script(venv_path)
    full_command = _build_shell_command(activate_venv, command)

    result = None
    try:
        result = subprocess.run(
            full_command, shell=True, capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}: {e.stderr}")
        logger.error(f"Output: {e.output}")
        result = e
    finally:
        _log_shell_output(
            result
        )  # Log the output of the command regardless of success or failure

    return result


def parse_globals_from_stdout(stdout: str) -> dict:
    """
    Parse the globals dictionary from the stdout of a Python command.
    """
    # Remove newline characters and strip outer curly braces
    stdout = stdout.strip().strip("{}")

    # Regular expression to match key-value pairs
    pattern = r"'(\w+)':\s*([^,]+)(?:,|$)"

    result = {}
    for match in re.finditer(pattern, stdout):
        key, value = match.groups()

        # Try to evaluate the value, if it fails, keep it as a string
        try:
            parsed_value = literal_eval(value)
        except Exception as e:
            parsed_value = value.strip()
            logger.error(
                f"Failed to parse value '{value}' for key '{key}' because:\n{e}"
            )

        result[key] = parsed_value

    return result
