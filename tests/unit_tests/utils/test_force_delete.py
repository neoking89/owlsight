import os
import tempfile
import uuid
import stat
import time
from pathlib import Path
import pytest

from owlsight.utils.helper_functions import force_delete, os_is_windows


@pytest.fixture
def unique_temp_dir():
    """
    Create a completely isolated temporary directory with a unique name.
    This ensures tests don't interfere with any existing directories.
    """
    # Using uuid4 to ensure uniqueness across test runs
    unique_name = f"owlsight_test_{uuid.uuid4().hex}"
    # Create a temp dir in the system's temp directory, not in the project directory
    temp_path = Path(tempfile.gettempdir()) / unique_name
    try:
        temp_path.mkdir(exist_ok=False)  # Fail if dir already exists
    except Exception as e:
        print(f"Error creating temp dir: {e}")
        pytest.fail("Failed to create temp dir")
    
    yield temp_path
    
    # Clean up if something went wrong and the directory still exists
    if temp_path.exists():
        try:
            import shutil
            # On Windows, we might need to reset permissions first
            for root, dirs, files in os.walk(temp_path, topdown=False):
                for file_path in files:
                    full_path = Path(root) / file_path
                    try:
                        os.chmod(full_path, stat.S_IWRITE)
                    except Exception as e:
                        print(f"Error resetting permissions: {e}")
            shutil.rmtree(temp_path, ignore_errors=True)
        except Exception as e:
            print(f"Error cleaning up temp dir: {e}")
            try:
                # If shutil.rmtree fails, try to delete each file individually
                for root, dirs, files in os.walk(temp_path, topdown=False):
                    for file_path in files:
                        full_path = Path(root) / file_path
                        try:
                            os.remove(full_path)
                        except Exception as e:
                            print(f"Error deleting file: {e}")
                for root, dirs, files in os.walk(temp_path, topdown=False):
                    for dir_path in dirs:
                        full_path = Path(root) / dir_path
                        try:
                            os.rmdir(full_path)
                        except Exception as e:
                            print(f"Error deleting dir: {e}")
                os.rmdir(temp_path)
            except Exception as e:
                print(f"Error cleaning up temp dir (final attempt): {e}")


def test_force_delete_existing_directory(unique_temp_dir):
    """Test that force_delete properly removes an existing directory."""
    try:
        # Create a file in the temp directory to verify deletion works
        test_file = unique_temp_dir / "test.txt"
        test_file.write_text("test content")
        
        # Create a nested directory with content
        nested_dir = unique_temp_dir / "nested"
        nested_dir.mkdir()
        (nested_dir / "nested_file.txt").write_text("nested content")
        
        assert unique_temp_dir.exists()
        assert test_file.exists()
        assert nested_dir.exists()
        
        # Call the function under test
        force_delete(unique_temp_dir)
        
        # Verify the directory is gone
        assert not unique_temp_dir.exists()
    except Exception as e:
        print(f"Error in test_force_delete_existing_directory: {e}")
        pytest.fail("Test failed")


def test_force_delete_nonexistent_directory():
    """Test that force_delete handles nonexistent directories gracefully."""
    try:
        # Create a path that definitely doesn't exist
        nonexistent_path = Path(tempfile.gettempdir()) / f"nonexistent_{uuid.uuid4().hex}"
        
        # This should not raise an exception
        force_delete(nonexistent_path)
        
        # Directory should still not exist
        assert not nonexistent_path.exists()
    except Exception as e:
        print(f"Error in test_force_delete_nonexistent_directory: {e}")
        pytest.fail("Test failed")


def test_force_delete_string_path(unique_temp_dir):
    """Test that force_delete works with string paths."""
    try:
        # Create a file to verify deletion
        test_file = unique_temp_dir / "test.txt"
        test_file.write_text("test content")
        
        # Convert Path to string
        dir_str = str(unique_temp_dir)
        
        # Call the function with a string path
        force_delete(dir_str)
        
        # Verify the directory is gone
        assert not Path(dir_str).exists()
    except Exception as e:
        print(f"Error in test_force_delete_string_path: {e}")
        pytest.fail("Test failed")

@pytest.mark.skipif(os_is_windows(), reason="Skipping readonly test on Windows")
def test_force_delete_with_readonly_files(unique_temp_dir):
    """Test that force_delete can remove directories with readonly files."""
    try:
        # Create a readonly file
        readonly_file = unique_temp_dir / "readonly.txt"
        readonly_file.write_text("readonly content")
        
        # Make the file readonly 
        os.chmod(readonly_file, stat.S_IREAD)
        
        # Call the function under test
        force_delete(unique_temp_dir)
        
        # Verify the directory is gone
        assert not unique_temp_dir.exists()
    except Exception as e:
        print(f"Error in test_force_delete_with_readonly_files: {e}")
        pytest.fail("Test failed")
