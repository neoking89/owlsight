import pytest
import logging
from owlsight.utils.logger import ColoredLogger


@pytest.fixture
def console_logger(caplog):
    """Create a logger that works with pytest's capture system"""
    logger = ColoredLogger(name="test_logger")
    # Clear any existing handlers
    logger.handlers.clear()
    # Add pytest's handler
    logger.addHandler(caplog.handler)
    logger.propagate = True  # Ensure messages propagate
    return logger


@pytest.fixture
def file_logger(tmp_path):
    log_file = tmp_path / "test.log"
    logger = ColoredLogger(name="file_logger")
    logger.log_to_file(str(log_file))
    return logger


def test_initialization():
    """Test logger initialization with default parameters"""
    logger = ColoredLogger(name="test_logger")
    assert logger.name == "test_logger"
    assert logger.level == logging.INFO
    assert len(logger.handlers) == 1


def test_log_levels(console_logger, caplog):
    """Test all log level outputs"""
    console_logger.setLevel(logging.DEBUG)
    caplog.set_level(logging.DEBUG)  # Ensure caplog captures all levels

    test_messages = [
        (logging.DEBUG, "Debug message"),
        (logging.INFO, "Info message"),
        (logging.WARNING, "Warning message"),
        (logging.ERROR, "Error message"),
        (logging.CRITICAL, "Critical message"),
    ]

    for level, msg in test_messages:
        console_logger.log(level, msg)

    assert len(caplog.records) == 5
    for record, (level, msg) in zip(caplog.records, test_messages):
        assert record.levelno == level
        assert record.message == msg


def test_warn_always(console_logger, caplog):
    """Test warn_always bypasses level setting"""
    console_logger.setLevel(logging.ERROR)
    caplog.set_level(logging.WARNING)  # Ensure caplog captures warnings

    console_logger.warn_always("Important warning")
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert "Important warning" in caplog.records[0].message


def test_color_codes():
    """Verify color codes in console output"""
    logger = ColoredLogger(name="color_test")
    color_check = {
        logging.DEBUG: "\033[1;36m",  # Cyan
        logging.INFO: "\033[1;37m",  # White
        logging.WARNING: "\033[1;33m",  # Yellow
        logging.ERROR: "\033[1;31m",  # Red
        logging.CRITICAL: "\033[1;35m",  # Purple
    }

    for level, color in color_check.items():
        record = logger.makeRecord(name=logger.name, level=level, fn="", lno=0, msg="test", args=(), exc_info=None)
        formatted = logger.handlers[0].format(record)
        assert color in formatted
        assert "\033[0m" in formatted  # Reset code


def test_file_logging(file_logger, tmp_path):
    """Test file logging without color codes"""
    test_msg = "File log message"
    file_logger.info(test_msg)

    log_file = tmp_path / "test.log"
    assert log_file.exists()

    with open(log_file) as f:
        content = f.read()
        assert test_msg in content
        assert "\033[" not in content  # No color codes


def test_log_to_file(tmp_path):
    """Test adding file logging after initialization"""
    logger = ColoredLogger(name="test_logger")
    assert len(logger.handlers) == 1  # Only console handler

    log_file = tmp_path / "new.log"
    logger.log_to_file(str(log_file))
    assert len(logger.handlers) == 2  # Console and file handler

    test_msg = "Test both handlers"
    logger.info(test_msg)

    with open(log_file) as f:
        content = f.read()
        assert test_msg in content


def test_level_changes(console_logger, caplog):
    """Test dynamic level changes"""

    # Try a message at each level to see what gets through
    console_logger.debug("Debug test")
    console_logger.info("Info test")
    console_logger.warning("Warning test")
    console_logger.error("Error test")
    caplog.clear()

    # Now proceed with the actual test
    console_logger.setLevel(logging.WARNING)
    caplog.handler.setLevel(logging.WARNING)

    console_logger.info("Should not appear")
    console_logger.warning("Warning should appear")
    caplog.clear()

    console_logger.setLevel(logging.INFO)
    caplog.handler.setLevel(logging.INFO)

    console_logger.info("Should appear")
    print(f"Final records: {caplog.records}")

    assert len(caplog.records) == 1


def test_configure_file_logging(tmp_path):
    """Test configure_file_logging with different debug levels"""
    logger = ColoredLogger(name="test_logger")
    logger.setLevel(logging.DEBUG)  # Set logger to DEBUG level
    log_file = tmp_path / "test.log"

    # Configure file logging with DEBUG level
    logger.configure_file_logging(str(log_file), level="DEBUG")

    # Test different log levels
    debug_msg = "Debug message"
    info_msg = "Info message"
    warn_msg = "Warning message"

    logger.debug(debug_msg)  # This should be logged to file
    logger.info(info_msg)  # This should be logged to file
    logger.warning(warn_msg)  # This should be logged to file

    with open(log_file) as f:
        content = f.read()
        assert debug_msg in content
        assert info_msg in content
        assert warn_msg in content
        assert "\033[" not in content  # No color codes


def test_configure_file_logging_with_int_level(tmp_path):
    """Test configure_file_logging with integer log level"""
    logger = ColoredLogger(name="test_logger")
    log_file = tmp_path / "test.log"

    # Configure file logging with INFO level using integer
    logger.configure_file_logging(str(log_file), level=logging.INFO)

    debug_msg = "Debug message"
    info_msg = "Info message"

    logger.debug(debug_msg)  # This should NOT be logged to file
    logger.info(info_msg)  # This should be logged to file

    with open(log_file) as f:
        content = f.read()
        assert debug_msg not in content
        assert info_msg in content
