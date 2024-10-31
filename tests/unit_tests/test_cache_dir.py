import os
import pytest
import shutil
from unittest.mock import patch
import tempfile
import sys

sys.path.append("src")
from owlsight.utils.constants import create_or_get_path, get_cache_dir
from owlsight.utils.helper_functions import os_is_windows


@pytest.fixture
def test_env():
    """Create and manage test environment using tempfile for cross-platform compatibility."""
    # Create temporary directory using tempfile (handles cross-platform paths)
    temp_dir = tempfile.mkdtemp()
    test_home = os.path.join(temp_dir, "home")
    test_cache_dir = os.path.join(test_home, ".owlsight")

    # Create temporary home directory
    os.makedirs(test_home, exist_ok=True)

    # Create a fixture return value with all needed test data
    env_data = {
        "test_home": test_home,
        "test_cache_dir": test_cache_dir,
    }

    # Use context manager for patching
    with patch("os.path.expanduser", return_value=test_home):
        yield env_data

    # Cleanup after tests
    try:
        shutil.rmtree(temp_dir)
    except PermissionError:
        # Handle Windows file permission issues
        pass


def test_create_or_get_path_creates_subdirectory(test_env):
    """Test that create_or_get_path creates subdirectories."""
    test_path = "test_subdir"
    result = create_or_get_path(test_path)

    # Use normpath to handle path separators consistently across platforms
    actual_full_path = os.path.normpath(os.path.join(get_cache_dir(), test_path))

    assert result == test_path
    assert os.path.exists(actual_full_path), f"Path does not exist: {actual_full_path}"
    assert os.path.isdir(actual_full_path)


def test_get_cache_dir_permissions(test_env):
    """Test that created directories have correct permissions."""
    cache_dir = get_cache_dir()
    mode = os.stat(cache_dir).st_mode & 0o777

    if os_is_windows():
        # Windows permissions are different, typically 666 or 777
        assert mode & 0o400  # Check if readable
        assert mode & 0o200  # Check if writable
    else:
        # Unix permissions
        assert mode in (0o755, 0o775)


def test_path_separators():
    """Test that path separators are handled correctly for the current platform."""
    test_path = os.path.join("nested", "path")
    result = create_or_get_path(test_path)
    
    # Adjust expected path to match OS behavior
    expected_path = os.path.normpath(test_path)
    assert os.path.normpath(result) == expected_path, f"Unexpected path formatting: {result}"


@pytest.mark.skipif(os_is_windows(), reason="Symlink tests not supported on Windows without admin privileges")
def test_symlink_handling(test_env):
    """Test handling of symlinked directories (Skip on Windows)."""
    # Create a real directory
    real_dir = os.path.join(test_env["test_home"], "real_dir")
    os.makedirs(real_dir, exist_ok=True)

    # Create a symlink
    symlink_path = os.path.join(test_env["test_home"], "symlink_dir")
    os.symlink(real_dir, symlink_path)

    result = create_or_get_path(symlink_path)
    assert os.path.exists(result)


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
