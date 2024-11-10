"""Define a custom logger class with colored output for logging to the console and/or a file and instantiate it"""


import logging
import os
from typing import Optional

import psutil

LEVELNAME_TO_LEVELVALUE = {
    "NOT SET": 0,
    "INFO": 20,
    "NOTICE": 25,
    "WARNING": 30,
}

VERBOSE_TO_LEVELNAME = {
    0: "NOT SET",
    1: "WARNING",
    2: "NOTICE",
    3: "INFO",
}

COLOR_CODES = {
    "DEBUG": "\033[1;36m",  # Bright Cyan
    "INFO": "\033[1;37m",  # Bright White
    "NOTICE": "\033[1;34m",  # Bright Blue
    "WARNING": "\033[1;33m",  # Bright Yellow
    "ERROR": "\033[1;31m",  # Bright Red
    "CRITICAL": "\033[1;35m",  # Bright Purple
}

RESET_CODE = "\033[0m"  # Resets to default terminal color


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds color to the log message"""

    def format(self, record):
        levelname = record.levelname
        if levelname in COLOR_CODES:
            color_code = COLOR_CODES[levelname]
            message = super().format(record)
            colored_message = f"{color_code}{message}{RESET_CODE}"
            return colored_message
        return super().format(record)


class EnhancedFormatter(logging.Formatter):
    """Enhanced formatter that adds CPU and memory usage to the log message"""

    def format(self, record):
        process = psutil.Process(os.getpid())
        record.cpu_usage = f"{psutil.cpu_percent(interval=None)}%"
        record.memory_usage = f"{psutil.virtual_memory().percent}%"
        record.process_mem = f"{process.memory_info().rss / (1024 ** 2):.2f} MB"
        return super().format(record)


class EnhancedColoredFormatter(ColoredFormatter, EnhancedFormatter):
    """Enhanced formatter that adds CPU and memory usage to the log message and colors the log message"""

# class Singleton(type):
#     _instances = {}
#     def __call__(cls, *args, **kwargs):
#         if cls not in cls._instances:
#             cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
#         return cls._instances[cls]

class ColoredLogger(logging.Logger):
    """Custom logger class with colored output for console and for logging to the console and/or a file"""

    def __init__(
        self,
        name: Optional[str] = None,
        verbose: int = 0,
        filename: Optional[str] = None,
        mode: str = "a+",
        log_cpu_and_memory_usage: bool = False,
    ):
        level_value = self._get_level_value_from_verbose(verbose)
        super().__init__(name, level_value)

        self.filename = filename
        self.mode = mode
        self.log_cpu_and_memory_usage = log_cpu_and_memory_usage

        self.add_custom_level(level_name="NOTICE", level_value=25)
        self.set_verbose(verbose)

    def info(self, msg: str, *args, **kwargs) -> None:
        if self.isEnabledFor(20):
            self._log(20, msg, args, **kwargs)

    def notice(self, msg: str, *args, **kwargs) -> None:
        if self.isEnabledFor(25):
            self._log(25, msg, args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        if self.isEnabledFor(30):
            self._log(30, msg, args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        if self.isEnabledFor(40):
            self._log(40, msg, args, **kwargs)

    def warn_always(self, msg: str, *args, **kwargs) -> None:
        """Always log the message, regardless of the verbosity level"""
        _ = self.get_verbose()
        self.set_verbose(50)
        self._log(30, msg, args, **kwargs)
        self.set_verbose(_)

    def set_cpu_and_memory_usage(self, log_cpu_and_memory_usage: bool) -> None:
        """Set whether to additionaly log CPU and memory usage in the log message"""
        self.log_cpu_and_memory_usage = log_cpu_and_memory_usage
        self.formatter = self._get_formatter(self.name)
        self._infer_handler_from_verbose_and_add_it(self.verbose, self.level)

    def set_verbose(self, verbose: int) -> None:
        """Set the global verbosity level of the logger. If verbose is 0, the logger is disabled."""
        if verbose < 0:
            raise ValueError("verbose must be >= 0")

        self.verbose = verbose
        level_value = self._get_level_value_from_verbose(verbose)
        self.setLevel(level_value)
        self.formatter = self._get_formatter(self.name)
        self._infer_handler_from_verbose_and_add_it(verbose, level_value)

    def get_verbose(self) -> int:
        """Get the global verbosity level of the logger. If verbose is 0, the logger is disabled."""
        return self.verbose

    def set_log_file(self, filename: Optional[str], mode: str = "a+") -> None:
        """Set the log file to write to. If filename is None, the logger will write to the console."""
        if filename is not None:
            self.filename = filename
            self.mode = mode
            self._add_log_file_handler(self.level, self.filename, self.mode, self.formatter)

    def add_custom_level(self, level_name: str, level_value: int) -> None:
        """Add a custom log level to the logger"""
        if level_name not in logging._levelToName:
            setattr(logging, level_name, level_value)
            logging.addLevelName(level_value, level_name)

    def _get_formatter(self, name: str) -> logging.Formatter:
        date_fmt, fmt = self._get_formatting(name)
        if self.log_cpu_and_memory_usage:
            console_formatter = EnhancedColoredFormatter(fmt, datefmt=date_fmt)
            file_formatter = EnhancedFormatter(fmt, datefmt=date_fmt)
        else:
            console_formatter = ColoredFormatter(fmt, datefmt=date_fmt)
            file_formatter = logging.Formatter(fmt, datefmt=date_fmt)
        return console_formatter if self.filename is None else file_formatter

    def _get_formatting(self, name: Optional[str]) -> tuple:
        date_fmt = "%Y-%m-%d %H:%M:%S"
        if self.log_cpu_and_memory_usage:
            if name is None:
                fmt = "%(asctime)s - %(levelname)s - CPU: %(cpu_usage)s - Memory: %(memory_usage)s - Process Memory: %(process_mem)s - %(message)s"
            else:
                fmt = "%(asctime)s - %(name)s - %(levelname)s - CPU: %(cpu_usage)s - Memory: %(memory_usage)s - Process Memory: %(process_mem)s - %(message)s"
        else:
            if name is None:
                fmt = "%(asctime)s - %(levelname)s - %(message)s"
            else:
                fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        return date_fmt, fmt

    def _get_level_value_from_verbose(self, verbose):
        level_name = VERBOSE_TO_LEVELNAME.get(abs(verbose), "INFO")
        level_value = LEVELNAME_TO_LEVELVALUE[level_name]
        return level_value

    def _add_console_handler(self, level: int, colored_formatter: ColoredFormatter) -> None:
        self.handlers.clear()
        ch = logging.StreamHandler()
        self._add_handler(level, colored_formatter, ch)

    def _add_log_file_handler(self, level: int, filename: str, mode: str, formatter: logging.Formatter) -> None:
        self.handlers.clear()
        fh = logging.FileHandler(filename=filename, mode=mode)
        self._add_handler(level, formatter, fh)

    def _add_handler(self, level: int, formatter: logging.Formatter, ch: logging.Handler):
        ch.setLevel(level)
        ch.setFormatter(formatter)
        self.addHandler(ch)

    def _infer_handler_from_verbose_and_add_it(self, verbose, level_value):
        if verbose != 0:
            self.disabled = False
            if self.filename is None:
                self._add_console_handler(level_value, self.formatter)
            else:
                self._add_log_file_handler(level_value, self.filename, self.mode, self.formatter)
        else:
            self.disabled = True
            self.handlers.clear()


logger = ColoredLogger(verbose=3)