"""Define a custom logger class with colored output for logging to the console and/or a file and instantiate it"""

import logging
import os
from typing import Optional, Union

import psutil

COLOR_CODES = {
    "DEBUG": "\033[1;36m",  # Bright Cyan
    "INFO": "\033[1;37m",  # Bright White
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
        record.process_mem = f"{process.memory_info().rss / (1024**2):.2f} MB"
        return super().format(record)


class EnhancedColoredFormatter(ColoredFormatter, EnhancedFormatter):
    """Enhanced formatter that adds CPU and memory usage to the log message and colors the log message"""


class ColoredLogger(logging.Logger):
    """Custom logger class with colored output for console and file logging"""

    def __init__(
        self,
        name: Optional[str] = None,
        level: int = logging.INFO,
        log_cpu_and_memory_usage: bool = False,
    ):
        super().__init__(name, level)
        self.log_cpu_and_memory_usage = log_cpu_and_memory_usage
        self.formatter = self._get_formatter(name)
        self._configure_handlers()

    def debug(self, msg: str, *args, **kwargs) -> None:
        if self.isEnabledFor(10):  # DEBUG is 10
            self._log(10, msg, args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        if self.isEnabledFor(20):
            self._log(20, msg, args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        if self.isEnabledFor(30):
            self._log(30, msg, args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        if self.isEnabledFor(40):
            self._log(40, msg, args, **kwargs)

    def warn_always(self, msg: str, *args, **kwargs) -> None:
        """Always log the warning message"""
        original_level = self.level
        self.setLevel(logging.WARNING)
        self.warning(msg, *args, **kwargs)
        self.setLevel(original_level)

    def set_cpu_and_memory_usage(self, log_cpu_and_memory_usage: bool) -> None:
        """Set whether to additionaly log CPU and memory usage in the log message"""
        self.log_cpu_and_memory_usage = log_cpu_and_memory_usage
        self.formatter = self._get_formatter(self.name)
        self._configure_handlers()

    def _configure_handlers(self):
        """Configure console logging handler"""
        self.handlers.clear()
        handler = logging.StreamHandler()
        handler.setFormatter(self._get_formatter(self.name))
        self.addHandler(handler)

    def log_to_file(self, filename: str, mode: str = "a+") -> None:
        """Add file logging to the current logger.

        Args:
            filename: Path to the log file
            mode: File open mode, defaults to "a+" (append and read)
        """
        handler = logging.FileHandler(filename, mode=mode, encoding='utf-8')
        handler.setFormatter(self._get_formatter(self.name, for_file=True))
        self.addHandler(handler)

    def configure_file_logging(
        self,
        filename: str,
        level: Union[int, str] = logging.INFO,
        mode: str = "a+",
    ) -> None:
        """Configure file logging with custom debug level.

        Args:
            filename: Path to the log file
            level: Log level for the file handler. Can be an integer (e.g. logging.DEBUG)
                  or a string (e.g. "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
            mode: File open mode, defaults to "a+" (append and read)
        """
        # Convert string level to int if necessary
        if isinstance(level, str):
            level = getattr(logging, level.upper())

        handler = logging.FileHandler(filename, mode=mode, encoding='utf-8')
        handler.setFormatter(self._get_formatter(self.name, for_file=True))
        handler.setLevel(level)
        self.addHandler(handler)

    def setLevel(self, level: int) -> None:
        """Set logging level for all handlers"""
        super().setLevel(level)
        for handler in self.handlers:
            handler.setLevel(level)

    def _get_formatter(self, name: Optional[str], for_file: bool = False) -> logging.Formatter:
        """Get the appropriate formatter based on settings and handler type"""
        fmt, date_fmt = self._get_formatting(name)

        if self.log_cpu_and_memory_usage:
            if for_file:
                return EnhancedFormatter(fmt, datefmt=date_fmt)
            return EnhancedColoredFormatter(fmt, datefmt=date_fmt)
        else:
            if for_file:
                return logging.Formatter(fmt, datefmt=date_fmt)
            return ColoredFormatter(fmt, datefmt=date_fmt)

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
        return fmt, date_fmt

    def add_custom_level(self, level_name: str, level_value: int) -> None:
        """Add a custom log level to the logger"""
        if level_name not in logging._levelToName:
            setattr(logging, level_name, level_value)
            logging.addLevelName(level_value, level_name)


logger = ColoredLogger()
