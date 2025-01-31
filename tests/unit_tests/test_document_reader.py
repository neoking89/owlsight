"""Tests for the DocumentReader class."""

import pytest
from unittest.mock import patch, Mock
import os
import socket

from owlsight.rag.document_reader import DocumentReader, _check_internet_connection

# Test data
SAMPLE_TEXT = "This is sample text content"
SAMPLE_PDF_CONTENT = {"content": SAMPLE_TEXT, "status": 200}
FAILED_PARSE = {"content": None, "status": 500}


@pytest.fixture
def reader():
    """Create a DocumentReader instance for testing."""
    return DocumentReader()


@pytest.fixture
def test_dir(tmp_path):
    """Create a temporary directory with test files."""
    # Create test files
    test_files = {
        "doc1.pdf": SAMPLE_TEXT,
        "doc2.txt": "Another sample text",
        "subdir/doc3.docx": "Document in subdirectory",
    }

    for filepath, content in test_files.items():
        full_path = tmp_path / filepath
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)

    return tmp_path


def test_init_default():
    """Test DocumentReader initialization with default parameters."""
    reader = DocumentReader()
    assert reader.supported_extensions is None
    assert reader.ocr_enabled is True
    assert reader.timeout == 5


def test_init_custom():
    """Test DocumentReader initialization with custom parameters."""
    extensions = [".pdf", ".doc"]
    reader = DocumentReader(supported_extensions=extensions, ocr_enabled=False, timeout=600)
    assert reader.supported_extensions == extensions
    assert reader.ocr_enabled is False
    assert reader.timeout == 600


def test_is_supported_file(reader):
    """Test file extension checking."""
    # Without extension restrictions
    assert reader.is_supported_file("test.pdf") is True
    assert reader.is_supported_file("test.xyz") is True

    # With extension restrictions
    reader = DocumentReader(supported_extensions=[".pdf", ".doc"])
    assert reader.is_supported_file("test.pdf") is True
    assert reader.is_supported_file("test.PDF") is True
    assert reader.is_supported_file("test.xyz") is False


@patch("owlsight.rag.document_reader.parser")
def test_read_file_success(mock_parser, reader):
    """Test successful file reading."""
    mock_parser.from_file.return_value = SAMPLE_PDF_CONTENT

    content = reader.read_file("test.pdf")
    assert content == SAMPLE_TEXT
    mock_parser.from_file.assert_called_once()


@patch("owlsight.rag.document_reader.parser")
def test_read_file_failure(mock_parser, reader):
    """Test failed file reading."""
    mock_parser.from_file.return_value = FAILED_PARSE

    content = reader.read_file("test.pdf")
    assert content is None


@patch("owlsight.rag.document_reader.parser")
def test_read_file_exception(mock_parser, reader):
    """Test exception handling during file reading."""
    mock_parser.from_file.side_effect = Exception("Test error")

    content = reader.read_file("test.pdf")
    assert content is None


@patch("owlsight.rag.document_reader.parser")
def test_read_directory(mock_parser, reader, test_dir):
    """Test directory reading functionality."""
    mock_parser.from_file.return_value = SAMPLE_PDF_CONTENT

    # Test recursive reading
    files = list(reader.read_directory(test_dir))
    assert len(files) == 3  # All files including subdirectory

    # Test non-recursive reading
    files = list(reader.read_directory(test_dir, recursive=False))
    assert len(files) == 2  # Only files in root directory


def test_read_directory_nonexistent(reader):
    """Test reading from a nonexistent directory."""
    with pytest.raises(FileNotFoundError):
        list(reader.read_directory("/nonexistent/path"))


@patch('owlsight.rag.document_reader._check_internet_connection')
def test_init_offline_no_jar(mock_check_internet):
    """Test DocumentReader initialization in offline mode without TIKA_SERVER_JAR."""
    # Simulate offline mode
    mock_check_internet.return_value = False
    
    # Ensure TIKA_SERVER_JAR is not set
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(RuntimeError) as exc_info:
            DocumentReader()
        
        assert "No internet connection detected" in str(exc_info.value)
        assert "TIKA_SERVER_JAR environment variable is not set" in str(exc_info.value)


@patch('owlsight.rag.document_reader._check_internet_connection')
def test_init_offline_with_jar(mock_check_internet):
    """Test DocumentReader initialization in offline mode with TIKA_SERVER_JAR set."""
    # Simulate offline mode
    mock_check_internet.return_value = False
    
    # Set TIKA_SERVER_JAR environment variable
    with patch.dict(os.environ, {'TIKA_SERVER_JAR': 'file:////tika-server-standard.jar'}):
        reader = DocumentReader()
        assert reader.is_offline is True
        assert reader.tika_jar_path == 'file:////tika-server-standard.jar'


@patch('owlsight.rag.document_reader._check_internet_connection')
def test_init_online(mock_check_internet):
    """Test DocumentReader initialization in online mode."""
    # Simulate online mode
    mock_check_internet.return_value = True
    
    # Should work regardless of TIKA_SERVER_JAR being set or not
    reader = DocumentReader()
    assert reader.is_offline is False
    
    with patch.dict(os.environ, {'TIKA_SERVER_JAR': 'file:////tika-server-standard.jar'}):
        reader = DocumentReader()
        assert reader.is_offline is False


def test_check_internet_connection():
    """Test the internet connection check function."""
    with patch('socket.socket') as mock_socket:
        # Test successful connection
        mock_socket.return_value.connect.return_value = None
        assert _check_internet_connection() is True
        
        # Test connection timeout
        mock_socket.return_value.connect.side_effect = socket.timeout()
        assert _check_internet_connection() is False
        
        # Test connection error
        mock_socket.return_value.connect.side_effect = socket.gaierror()
        assert _check_internet_connection() is False
