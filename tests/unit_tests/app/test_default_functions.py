import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os

from owlsight.app.default_functions import OwlDefaultFunctions, is_url
from owlsight.utils.logger import logger # Assuming logger is used and needs to be available

# Fixture for OwlDefaultFunctions instance
@pytest.fixture
def owl_funcs():
    # Mock globals_dict if necessary for other owl_ Punctions, not strictly needed for owl_read standalone tests
    return OwlDefaultFunctions(globals_dict={})

# Fixture to mock DocumentReader
@pytest.fixture
def mock_document_reader_instance():
    mock_reader = MagicMock()
    # Common setup for read_file and read_directory mocks can go here if desired
    # e.g., mock_reader.read_file.return_value = "mocked content"
    return mock_reader

@pytest.fixture
def mock_get_document_reader(owl_funcs, mock_document_reader_instance):
    with patch.object(owl_funcs, '_get_document_reader', return_value=mock_document_reader_instance) as mock_method:
        yield mock_method, mock_document_reader_instance


def test_is_url():
    # Arrange
    test_cases = [
        # Valid URLs
        ("https://claude.ai/new", True),
        ("http://example.com", True),
        ("ftp://ftp.example.com/file.txt", True),
        ("https://sub.domain.example.com/path?query=123#fragment", True),
        ("http://localhost:8000", True),
        ("http://127.0.0.1", True),
        ("https://123.123.123.123", True),
        ("https://example.com:8080", True),
        ("https://example.com/path/to/page", True),

        # Invalid URLs
        ("www.google.nl", False),  # Missing protocol
        ("htp://missing-t.com", False),  # Invalid protocol
        ("http:/missing-slash.com", False),  # Malformed protocol
        ("www.google", False),  # Missing top-level domain
        ("https://", False),  # Incomplete URL
        ("ftp://", False),  # Incomplete URL with only protocol
        ("http://?", False),  # Missing domain
        ("//example.com", False),  # Missing protocol
        ("example", False),  # Not a URL
        ("https://.com", False),  # Missing domain name
        ("https://example..com", False),  # Invalid domain with double dot
    ]

    # Act & Assert
    for url, expected in test_cases:
        result = is_url(url)
        assert result == expected, f"Test failed for URL: {url}. Expected: {expected}, Got: {result}"

# --- Tests for owl_read --- 

@pytest.mark.usefixtures("mock_get_document_reader")
def test_owl_read_single_file_success(owl_funcs, mock_get_document_reader, tmp_path):
    """Test owl_read with a single file successfully read by DocumentReader."""
    mock_method, mock_reader = mock_get_document_reader
    file_path = tmp_path / "test.txt"
    file_path.write_text("actual content") # Fallback content

    mock_reader.read_file.return_value = "tika content"
    
    result = owl_funcs.owl_read(str(file_path))
    assert result == "tika content"
    mock_reader.read_file.assert_called_once_with(str(file_path))

@pytest.mark.usefixtures("mock_get_document_reader")
def test_owl_read_single_file_empty_from_tika(owl_funcs, mock_get_document_reader, tmp_path):
    """Test owl_read when DocumentReader.read_file returns an empty string."""
    mock_method, mock_reader = mock_get_document_reader
    file_path = tmp_path / "empty_tika.txt"
    file_path.write_text("actual content")

    mock_reader.read_file.return_value = ""
    
    result = owl_funcs.owl_read(str(file_path))
    assert result == "" # Expect empty string from Tika
    mock_reader.read_file.assert_called_once_with(str(file_path))

@pytest.mark.usefixtures("mock_get_document_reader")
def test_owl_read_single_file_tika_fails_fallback_succeeds(owl_funcs, mock_get_document_reader, tmp_path):
    """Test owl_read when DocumentReader.read_file returns None (Tika error), fallback to basic read succeeds."""
    mock_method, mock_reader = mock_get_document_reader
    file_path = tmp_path / "tika_fail_fallback.txt"
    expected_content = "this is fallback content"
    file_path.write_text(expected_content)

    mock_reader.read_file.return_value = None # Simulate Tika processing error
    
    result = owl_funcs.owl_read(str(file_path))
    assert result == expected_content # Expect content from basic file read
    mock_reader.read_file.assert_called_once_with(str(file_path))

@pytest.mark.usefixtures("mock_get_document_reader")
def test_owl_read_single_file_tika_fails_fallback_not_found(owl_funcs, mock_get_document_reader, tmp_path):
    """Test owl_read when Tika fails (returns None) and fallback file doesn't exist."""
    mock_method, mock_reader = mock_get_document_reader
    non_existent_file = tmp_path / "non_existent.txt"

    mock_reader.read_file.return_value = None
    
    result = owl_funcs.owl_read(str(non_existent_file))
    assert f"File not found: {non_existent_file}" in result
    mock_reader.read_file.assert_called_once_with(str(non_existent_file))


@pytest.mark.usefixtures("mock_get_document_reader")
def test_owl_read_directory_success(owl_funcs, mock_get_document_reader, tmp_path):
    """Test owl_read with a directory, DocumentReader.read_directory succeeds."""
    mock_method, mock_reader = mock_get_document_reader
    dir_path = tmp_path / "test_dir"
    dir_path.mkdir()
    file1_path = dir_path / "file1.txt"
    file2_path = dir_path / "file2.pdf"

    # Actual files not strictly needed if read_directory is fully mocked,
    # but good for completeness if any part relies on file existence.
    file1_path.write_text("content1_actual")
    file2_path.write_text("content2_actual") 

    mock_read_directory_return = {
        str(file1_path): "tika content1",
        str(file2_path): "tika content2",
        str(dir_path / "file3_error.txt"): None, # Simulate error for one file
        str(dir_path / "file4_empty.txt"): ""    # Simulate empty for another
    }
    mock_reader.read_directory.return_value = mock_read_directory_return.items()
    
    result = owl_funcs.owl_read(str(dir_path), recursive=True)
    assert result == mock_read_directory_return
    mock_reader.read_directory.assert_called_once_with(str(dir_path), recursive=True)

@pytest.mark.usefixtures("mock_get_document_reader")
def test_owl_read_iterable_files(owl_funcs, mock_get_document_reader, tmp_path):
    """Test owl_read with an iterable of file paths."""
    mock_method, mock_reader = mock_get_document_reader
    file1 = tmp_path / "iter_file1.txt"
    file2_tika_error = tmp_path / "iter_file2_error.txt" # Tika will return None for this
    file3_empty = tmp_path / "iter_file3_empty.txt"   # Tika will return "" for this

    file1.write_text("content file1 basic")
    file2_tika_error.write_text("content file2 basic") # This won't be read if Tika returns None
    file3_empty.write_text("content file3 basic")

    # Configure side_effect for multiple calls to read_file
    def read_file_side_effect(file_path_str):
        if file_path_str == str(file1):
            return "tika content for file1"
        elif file_path_str == str(file2_tika_error):
            return None # Simulate Tika error
        elif file_path_str == str(file3_empty):
            return "" # Simulate Tika empty
        return "default fallback text if not matched"
    
    mock_reader.read_file.side_effect = read_file_side_effect
    
    file_list = [str(file1), str(file2_tika_error), str(file3_empty)]
    result = owl_funcs.owl_read(file_list)
    
    expected_result = {
        str(file1): "tika content for file1",
        str(file2_tika_error): None, # Expect None as per updated logic
        str(file3_empty): "",
    }
    assert result == expected_result
    assert mock_reader.read_file.call_count == len(file_list)


def test_owl_read_url_raises_value_error(owl_funcs):
    """Test owl_read with a URL raises ValueError."""
    with pytest.raises(ValueError, match=r"owl_read requires local files\. Use owl_scrape\(\) for URLs like 'http://example\.com'"):
        owl_funcs.owl_read("http://example.com")

@pytest.mark.usefixtures("mock_get_document_reader")
def test_owl_read_buffer_input(owl_funcs, mock_get_document_reader):
    """Test owl_read with bytes input (buffer)."""
    mock_method, mock_reader = mock_get_document_reader
    buffer_content = b"This is binary content"
    expected_text = "text from buffer"

    mock_reader.read_file.return_value = expected_text

    result = owl_funcs.owl_read(buffer_content)
    assert result == expected_text
    mock_reader.read_file.assert_called_once_with(buffer_content)

if __name__ == "__main__":
    pytest.main([__file__])