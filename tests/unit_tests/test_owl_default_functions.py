import pytest
import tempfile
import os
from pathlib import Path
import sys

from unittest.mock import patch, Mock

sys.path.append("src")
from owlsight.app.default_functions import OwlDefaultFunctions
from owlsight.utils.custom_classes import SingletonDict


@pytest.fixture
def owl_instance():
    globals_dict = SingletonDict()
    return OwlDefaultFunctions(globals_dict)


@pytest.fixture
def temp_dir():
    temp_path = tempfile.mkdtemp()
    yield temp_path
    # Cleanup after tests
    for file in Path(temp_path).glob("*"):
        file.unlink()
    os.rmdir(temp_path)


def test_owl_read_write(owl_instance: OwlDefaultFunctions, temp_dir: Path):
    """Test the owl_read and owl_write functions"""
    test_file = os.path.join(temp_dir, "test.txt")
    test_content = "Hello, World!"

    # Test writing
    owl_instance.owl_write(test_file, test_content)
    assert os.path.exists(test_file)

    # Test reading
    read_content = owl_instance.owl_read(test_file)
    assert read_content == test_content

    # Test reading non-existent file
    non_existent = os.path.join(temp_dir, "nonexistent.txt")
    result = owl_instance.owl_read(non_existent)
    assert result.startswith("File not found:")


def test_owl_show(owl_instance: OwlDefaultFunctions):
    """Test the owl_show function with a simple variable"""
    owl_instance.globals_dict["test_var"] = 42
    # Since owl_show prints to stdout, we're just testing it doesn't raise exceptions
    owl_instance.owl_show(docs=False)
    owl_instance.owl_show(docs=True)


def test_method_naming_convention(owl_instance: OwlDefaultFunctions):
    """Test that all public methods follow the owl_ naming convention"""
    methods = [
        method for method in dir(owl_instance) if not method.startswith("_") and callable(getattr(owl_instance, method))
    ]
    for method in methods:
        assert method.startswith("owl_"), f"Method {method} does not follow owl_ naming convention"


def test_owl_press_executed_successfully(owl_instance: OwlDefaultFunctions):
    """Test that owl_press executes successfully with mocked subprocess."""
    # Create mock so that _start_child_process_owl_press does not actually press the keys
    mock_start_process = Mock(return_value=None)
    
    # Patch the method
    with patch.object(owl_instance, "_start_child_process_owl_press", mock_start_process):
        # Create a test sequence
        sequence = ["test", "ENTER"]

        # Execute owl_press
        executed_successfully = owl_instance.owl_press(
            sequence=sequence,
            exit_python_from_interpreter=False,
        )

        # Assert method was called once
        mock_start_process.assert_called_once()

        # Assert return value
        assert executed_successfully is True


if __name__ == "__main__":
    pytest.main([__file__])
