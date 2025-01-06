import subprocess
from typing import Tuple, Union, List
import os
import re
import traceback
from ast import literal_eval

from owlsight.utils.helper_functions import os_is_windows
from owlsight.utils.logger import logger


def run_subprocess(command: List[str]) -> Tuple[str, str]:
    """
    Run subprocess command and capture stdout and stderr.

    Parameters
    ----------
    command : List[str]
        List of command arguments to be executed.

    Returns
    -------
    tuple of (str, str)
        The stdout and stderr outputs from the subprocess.
    """
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    return stdout, stderr


def _get_activate_script(pyenv_path: str) -> str:
    """
    Get the path to the virtual environment's activation script.

    Parameters
    ----------
    pyenv_path : str
        Path to the virtual environment.

    Returns
    -------
    str
        The path to the activation script for the virtual environment.
    """
    # Normalize path separators for the current OS
    pyenv_path = os.path.normpath(pyenv_path)
    
    # Get the correct activation script path
    if os_is_windows():
        return os.path.join(pyenv_path, "Scripts", "activate.bat")
    else:
        # For WSL/Linux, use the bin/activate script
        return os.path.join(pyenv_path, "bin", "activate")


def execute_shell_command(command: str, pyenv_path: str) -> subprocess.CompletedProcess:
    """
    Execute a shell command inside the (virtual) python environment.

    Parameters
    ----------
    command : str
        The shell command to execute.
    pyenv_path : str
        Path to the (virtual) python environment.

    Returns
    -------
    subprocess.CompletedProcess
        The result of the subprocess run or the exception if failed.
    """
    activate_venv = _get_activate_script(pyenv_path)
    logger.info(f"Virtual env path: {pyenv_path}")
    logger.info(f"Activation script path: {activate_venv}")
    logger.info(f"Checking if activation script exists: {os.path.exists(activate_venv)}")

    if os_is_windows():
        command_list = ["cmd", "/c", f"call {activate_venv} && {command}"]
    else:
        command_list = ["bash", "-c", f"source {activate_venv} && {command}"]
    
    logger.info(f"Executing command: {' '.join(command_list)}")

    result = None
    try:
        result = subprocess.run(command_list, capture_output=True, text=True, check=True)
        logger.info("Command executed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(f"Command stderr: {e.stderr}")
        logger.error(f"Command stdout: {e.stdout}")
        result = e
    except Exception as e:
        logger.error(f"Unexpected error during command execution: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    finally:
        _log_shell_output(result)

    return result


def _log_shell_output(result: Union[subprocess.CompletedProcess, None]) -> None:
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
    if result is not None:
        if hasattr(result, "stdout") and result.stdout:
            logger.info(result.stdout)
        if hasattr(result, "stderr") and result.stderr:
            logger.warning(f"Command produced stderr output: {result.stderr}")
        if hasattr(result, "output") and result.output:
            logger.warning(f"Command produced output: {result.output}")


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
        except Exception:
            parsed_value = value.strip()
            logger.error(f"Failed to parse value '{value}' for key '{key}' because:\n{traceback.format_exc()}")

        result[key] = parsed_value

    return result
