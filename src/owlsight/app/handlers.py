import subprocess
import sys
import traceback

from owlsight.utils.code_execution import CodeExecutor
from owlsight.utils.logger import logger


def handle_interactive_shell(question: str) -> None:
    """Handles interactive shell sessions based on user input."""
    if question.lower() == "!cmd":
        subprocess.call("cmd.exe", shell=True)
    elif question.lower() == "!bash":
        subprocess.call("/bin/bash", shell=True)


def handle_interactive_code_execution(code_executor: CodeExecutor) -> None:
    """Handles the interactive Python console execution."""
    try:
        code_executor.init_interactive_py_console()
    except Exception:
        logger.error(f"Unexpected error in interactive console: {traceback.format_exc()}")
    # Reopen stdin if it's closed
    if sys.stdin.closed:
        logger.warning("stdin is closed. Reopening for further input.")
        sys.stdin = open(0)
