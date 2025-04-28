import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, Mock
from pynput.keyboard import Controller

from owlsight.app.default_functions import OwlDefaultFunctions
from owlsight.app._child_process_owl_press import KEY_MAP, execute_key_sequence
import platform
import subprocess
import sys

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
    """Test that owl_press executes successfully with mocked subprocess and returns True."""
    # Create mock so that _start_child_process_owl_press does not actually press the keys
    mock_start_process = Mock(return_value=None)

    # Patch the method
    with patch.object(owl_instance, "_start_child_process_owl_press", mock_start_process):
        # Create a test sequence
        sequence = ["test", "ENTER"]

        # Execute owl_press
        executed_successfully = owl_instance.owl_press(
            sequence=sequence,
            exit_python_before_sequence=False,
        )

        # Assert method was called once
        mock_start_process.assert_called_once()

        # Assert return value
        assert executed_successfully is True

def test_owl_press_keys_executed_successfully(owl_instance: OwlDefaultFunctions):
    """Test that owl_press executes successfully with mocked key presses."""
    with patch.object(Controller, "press") as mock_press, patch.object(Controller, "release") as mock_release:
        for key_string, key in KEY_MAP.items():
            # skip keycombinations (tuple) for now, as this test assumes single key presses (str).
            if isinstance(key, tuple):
                continue
            # Call the method that triggers the key press
            execute_key_sequence([key_string], time_before_sequence=0, time_between_keys=0)
            # Assert that the press and release methods were called with the correct key
            mock_press.assert_called_with(key)
            mock_release.assert_called_with(key)

def test_owl_tools_executed_successfully(owl_instance: OwlDefaultFunctions):
    """Test that owl_tools returns a list of defined functions as strings."""
    def test():
        return 42
    owl_instance.globals_dict["test"] = test
    tools = owl_instance.owl_tools()
    assert isinstance(tools, list)
    assert len(tools) > 0
    for tool in tools:
        assert isinstance(tool, dict)

def test_owl_terminal(owl_instance: OwlDefaultFunctions):
    """Test the owl_terminal function for various scenarios."""
    # 1. Test simple command without shell=True (should work cross-platform)
    #    Use sys.executable to get the current python interpreter path
    python_executable = sys.executable
    result = owl_instance.owl_terminal([python_executable, "--version"], shell=False)
    assert result["returncode"] == 0
    # Handle platform-specific newline characters in echo output
    assert "Python" in result["stdout"].strip()
    assert result["stderr"] == ""

    # 2. Test platform-specific shell built-in with shell=True
    is_windows = platform.system() == "Windows"
    shell_command = "dir" if is_windows else "ls"
    result_shell = owl_instance.owl_terminal(shell_command, shell=True)
    assert result_shell["returncode"] == 0
    assert result_shell["stdout"] != ""  # Should list some files
    assert result_shell["stderr"] == ""

    # 3. Test non-existent command (should fail with returncode -1)
    non_existent_cmd = "this_command_should_not_exist_anywhere"
    result_fail = owl_instance.owl_terminal(non_existent_cmd, shell=False)
    assert result_fail["returncode"] == -1 # Specific code for FileNotFoundError
    # Stderr should contain the FileNotFoundError message
    assert "The system cannot find the file specified" in result_fail["stderr"] or \
           "No such file or directory" in result_fail["stderr"] # Linux/Mac message

    # 4. Test non-existent command with raise_on_error=True (should still return -1, not raise)
    # FileNotFoundError happens before the process runs, so CalledProcessError is not raised.
    result_fail_raise = owl_instance.owl_terminal(non_existent_cmd, shell=False, raise_on_error=True)
    assert result_fail_raise["returncode"] == -1
    assert "The system cannot find the file specified" in result_fail_raise["stderr"] or \
           "No such file or directory" in result_fail_raise["stderr"]

    # 5. Test shell=True is required for built-ins (should fail with returncode -1 without it)
    result_no_shell = owl_instance.owl_terminal(shell_command, shell=False)
    assert result_no_shell["returncode"] == -1 # Specific code for FileNotFoundError
    assert "The system cannot find the file specified" in result_no_shell["stderr"] or \
           "No such file or directory" in result_no_shell["stderr"]

    # 6. Test raise_on_error=True with a command that exists but returns non-zero
    # Use python to run a script that exits with code 1
    fail_command = [sys.executable, "-c", "import sys; sys.exit(1)"]
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        owl_instance.owl_terminal(fail_command, shell=False, raise_on_error=True)
    # Check that the raised exception has the correct return code
    assert excinfo.value.returncode == 1


if __name__ == "__main__":
    pytest.main([__file__])
