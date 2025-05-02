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

@pytest.fixture
def owl(tmp_path: Path):
    """Return a fresh OwlDefaultFunctions instance for each test."""
    # We pass an empty globals dict because owl_edit_file does not rely on it.
    return OwlDefaultFunctions({})


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    """Create a sample text file for editing and return its Path."""
    content = "hello world\nfoo123\n"
    file_path = tmp_path / "sample.txt"
    file_path.write_text(content, encoding="utf-8")
    return file_path


def test_literal_replace(owl: OwlDefaultFunctions, sample_file: Path):
    """Literal (non-regex) replacement works."""
    owl.owl_edit_file(
        sample_file,
        edits=[{"pattern": "world", "replacement": "universe"}],
        regex=False,
    )
    assert sample_file.read_text() == "hello universe\nfoo123\n"


def test_regex_replace(owl: OwlDefaultFunctions, sample_file: Path):
    """Regex replacement works with default regex=True."""
    owl.owl_edit_file(
        sample_file,
        edits=[{"pattern": r"foo\d+", "replacement": "bar"}],
    )
    assert sample_file.read_text() == "hello world\nbar\n"


def test_backup_created(owl: OwlDefaultFunctions, sample_file: Path):
    """Backup file is created and contains the original content."""
    backup_path = sample_file.with_suffix(sample_file.suffix + ".bak")

    owl.owl_edit_file(
        sample_file,
        edits=[{"pattern": "hello", "replacement": "hi"}],
    )

    assert backup_path.exists()
    assert backup_path.read_text() == "hello world\nfoo123\n"


def test_no_backup(owl: OwlDefaultFunctions, sample_file: Path):
    """No backup is created when create_backup=False."""
    backup_path = sample_file.with_suffix(sample_file.suffix + ".bak")

    owl.owl_edit_file(
        sample_file,
        edits=[{"pattern": "hello", "replacement": "hi"}],
        create_backup=False,
    )

    assert not backup_path.exists()


def test_file_not_found(owl: OwlDefaultFunctions, tmp_path: Path):
    """Editing a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        owl.owl_edit_file(tmp_path / "missing.txt", edits=[{"pattern": "x", "replacement": "y"}])


def test_empty_edits(owl: OwlDefaultFunctions, sample_file: Path):
    """An empty edits list raises ValueError."""
    with pytest.raises(ValueError):
        owl.owl_edit_file(sample_file, edits=[])


def test_invalid_edit_schema(owl: OwlDefaultFunctions, sample_file: Path):
    """Each edit dict must contain both 'pattern' and 'replacement'."""
    with pytest.raises(ValueError):
        owl.owl_edit_file(sample_file, edits=[{"pattern": "x"}])

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
    """Comprehensive behaviour checks for `owl_terminal`."""
    is_windows = platform.system() == "Windows"
    shell_command = "dir" if is_windows else "ls"

    # 1. Simple command, no shell
    result = owl_instance.owl_terminal([sys.executable, "--version"], shell=False)
    assert result["returncode"] == 0
    assert "Python" in result["stdout"].strip()
    assert result["stderr"] == ""

    # 2. Shell built-in with shell=True
    result_shell = owl_instance.owl_terminal(shell_command, shell=True)
    assert result_shell["returncode"] == 0
    assert result_shell["stdout"] != ""
    assert result_shell["stderr"] == ""

    # 3 & 4. Non-existent command should raise
    missing = "this_command_should_not_exist_anywhere"
    with pytest.raises(FileNotFoundError):
        owl_instance.owl_terminal(missing, shell=False)

    with pytest.raises(FileNotFoundError):
        owl_instance.owl_terminal(missing, shell=False, raise_on_error=True)

    # 5. Built-in *without* shell=True → OS-specific expectation
    if is_windows:
        with pytest.raises(FileNotFoundError):
            owl_instance.owl_terminal(shell_command, shell=False)
    else:  # POSIX: `ls` is a real executable, so expect success
        result_no_shell = owl_instance.owl_terminal(shell_command, shell=False)
        assert result_no_shell["returncode"] == 0
        assert result_no_shell["stdout"] != ""
        assert result_no_shell["stderr"] == ""

    # 6. Command that exists but returns non-zero with raise_on_error=True
    fail_cmd = [sys.executable, "-c", "import sys; sys.exit(1)"]
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        owl_instance.owl_terminal(fail_cmd, shell=False, raise_on_error=True)
    assert excinfo.value.returncode == 1


if __name__ == "__main__":
    pytest.main([__file__])
