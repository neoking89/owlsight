"""
Logger module for the project. Use as follows:

```python

from owlsight.utils.logger_manager import LoggerManager
logger = LoggerManager.get_logger(__name__)
```

"""

import os
import logging
from typing import Optional
import psutil
from datetime import datetime


class EnhancedFormatter(logging.Formatter):
    """Enhanced formatter that adds CPU and memory usage to the log message."""

    def format(self, record):
        process = psutil.Process(os.getpid())
        record.cpu_usage = f"{psutil.cpu_percent(interval=None)}%"
        record.memory_usage = f"{psutil.virtual_memory().percent}%"
        record.process_mem = f"{process.memory_info().rss / (1024 ** 2):.2f} MB"
        return super().format(record)


class LoggerManager:
    """Manager class for the logger."""

    _log_path = None
    _log_filename = None
    _add_memory_logging = False

    @classmethod
    def configure_logger(
        cls,
        log_path: Optional[str] = None,
        log_filename: Optional[str] = None,
        add_memory_logging: bool = False,
    ):
        """
        Configure the logger with a log path and filename.

        :param log_path: Path to the log file. If None, output will be printed to the console.
        :param log_filename: Filename of the log file. If None, the current date and time will be used.
        :param add_memory_logging: Add memory logging to the logger.

        :return: None
        """
        cls._validate_args(log_path, log_filename, add_memory_logging)

        cls._add_memory_logging = add_memory_logging

        if log_path is not None:
            if not os.path.exists(log_path):
                raise FileNotFoundError(f"Log path {log_path} does not exist.")
            if not os.path.isdir(log_path):
                raise NotADirectoryError(f"Log path {log_path} is not a directory.")
            cls._log_path = log_path

        if log_filename is not None:
            cls._log_filename = log_filename
        elif cls._log_path and cls._log_filename is None:
            cls._log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")

    @classmethod
    def get_logger(cls, name: str, level: int = logging.DEBUG) -> logging.Logger:
        """
        Get a logger instance.

        :param str name: Name of the logger.
        :param int level: Logging level of the logger.
        :return: LoggerManager instance.
        :rtype: logging.Logger
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)
        formatter = EnhancedFormatter(
            (
                "%(asctime)s - %(name)s - %(levelname)s - CPU: %(cpu_usage)s - Memory: %(memory_usage)s - Process Memory: %(process_mem)s - %(message)s"
                if cls._add_memory_logging
                else "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        if cls._log_path and cls._log_filename:
            file_handler = logging.FileHandler(os.path.join(cls._log_path, cls._log_filename))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        return logger

    @classmethod
    def _validate_args(cls, log_path, log_filename, add_memory_logging):
        if log_path is not None and not isinstance(log_path, str):
            raise TypeError(f"Log path must be a string, not {type(log_path)}")
        if log_filename is not None and not isinstance(log_filename, str):
            raise TypeError(f"Log filename must be a string, not {type(log_filename)}")
        if not isinstance(add_memory_logging, bool):
            raise TypeError(f"Add memory logging must be a boolean, not {type(add_memory_logging)}")
