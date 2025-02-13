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
    return ColoredLogger(name="file_logger", filename=str(log_file))

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
        logging.DEBUG: "\033[1;36m",    # Cyan
        logging.INFO: "\033[1;37m",     # White
        logging.WARNING: "\033[1;33m",  # Yellow
        logging.ERROR: "\033[1;31m",    # Red
        logging.CRITICAL: "\033[1;35m"  # Purple
    }
    
    for level, color in color_check.items():
        record = logger.makeRecord(
            name=logger.name,
            level=level,
            fn="",
            lno=0,
            msg="test",
            args=(),
            exc_info=None
        )
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
