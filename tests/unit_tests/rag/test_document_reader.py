"""Tests for the DocumentReader class."""

import os
import pytest
import shutil
from unittest.mock import patch
import glob
from pathlib import Path

from owlsight.rag.document_reader import DocumentReader, _has_internet_connection

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
    reader = DocumentReader(supported_extensions=extensions)
    assert reader.supported_extensions == extensions


def test_is_supported_file(reader):
    """Test file extension checking."""
    assert reader.is_supported_file("test.pdf") is True
    assert reader.is_supported_file("test.txt") is True
    assert reader.is_supported_file("test.xyz") is True  # No extension filter by default


@patch("tika.parser.from_file")
def test_read_file_success(mock_parser, reader):
    """Test successful file reading."""
    mock_parser.return_value = SAMPLE_PDF_CONTENT
    content = reader.read_file("test.pdf")
    assert content == SAMPLE_TEXT


@patch("tika.parser.from_file")
def test_read_file_failure(mock_parser, reader):
    """Test failed file reading."""
    mock_parser.return_value = FAILED_PARSE
    content = reader.read_file("test.pdf")
    assert content == ""


@patch("tika.parser.from_file")
def test_read_file_exception(mock_parser, reader):
    """Test exception handling during file reading."""
    mock_parser.side_effect = Exception("Test error")
    content = reader.read_file("test.pdf")
    assert content == ""


@patch("tika.parser.from_file")
def test_read_directory(mock_parser, reader, test_dir):
    """Test directory reading functionality."""
    mock_parser.return_value = SAMPLE_PDF_CONTENT
    results = list(reader.read_directory(test_dir))
    assert len(results) == 3
    assert all(isinstance(content, str) for _, content in results)


def test_read_directory_nonexistent(reader):
    """Test reading from a nonexistent directory."""
    with pytest.raises(FileNotFoundError):
        list(reader.read_directory("nonexistent_dir"))


@pytest.fixture
def cleanup_unzipped():
    yield
    # Clean up any unzipped (jar) files
    for d in glob.glob("src/owlsight/blobs/*"):
        if os.path.isdir(d):
            shutil.rmtree(d)


@patch("owlsight.rag.document_reader._has_internet_connection")
def test_init_offline_zipped_jar_exists(mock_check_internet, cleanup_unzipped):
    """Test DocumentReader initialization in offline mode without TIKA_SERVER_JAR."""
    mock_check_internet.return_value = False
    reader = DocumentReader()
    assert os.path.exists(reader.tika_server_jar_path)
    assert reader.tika_server_jar_path.endswith(".jar")


@patch("owlsight.rag.document_reader._has_internet_connection")
@patch("owlsight.rag.document_reader.logger")
def test_init_online(mock_logger, mock_check_internet):
    """Test DocumentReader initialization in online mode."""
    mock_check_internet.return_value = True
    DocumentReader()
    mock_logger.info.assert_called_once_with("Using remote Tika server")


def test_has_internet_connection():
    """Test the internet connection check function."""
    # Test with valid host
    assert _has_internet_connection(host="8.8.8.8", timeout=1) in (True, False)

    # Test with invalid host
    assert _has_internet_connection(host="invalid.host", timeout=1) is False


@patch("owlsight.rag.document_reader._has_internet_connection")
def test_offline_invalid_jar_path(mock_check_internet):
    """Test offline mode with invalid user-provided JAR path"""
    mock_check_internet.return_value = False
    with pytest.raises(FileNotFoundError):
        DocumentReader(tika_server_jar_path="invalid/path/tika-server.jar")


@pytest.mark.asyncio
async def test_init_no_blobs_with_internet():
    """Test DocumentReader initialization when blobs directory doesn't exist but internet is available."""
    # Ensure blobs directory doesn't exist
    blobs_dir = Path(__file__).parent.parent.parent / "src" / "owlsight" / "blobs"
    if blobs_dir.exists():
        temp_dir = Path(__file__).parent / "temp_blobs_backup"
        shutil.move(str(blobs_dir), str(temp_dir))
        try:
            # Mock internet connection check to return True
            with patch('owlsight.rag.document_reader._has_internet_connection', return_value=True):
                # Initialize DocumentReader
                reader = DocumentReader()
                
                # Verify reader was initialized correctly
                assert reader.ocr_enabled is True
                assert reader.timeout == 5
                
                # Test basic functionality
                with patch('tika.parser.from_file', return_value=SAMPLE_PDF_CONTENT):
                    content = reader.read_file("test.pdf")
                    assert content == SAMPLE_TEXT
        finally:
            # Restore blobs directory if it existed
            if temp_dir.exists():
                shutil.move(str(temp_dir), str(blobs_dir))


@patch("tika.parser.from_buffer")
def test_read_file_bytes_success(mock_parser, reader):
    """Test successful file reading from bytes buffer."""
    mock_parser.return_value = SAMPLE_PDF_CONTENT
    content = reader.read_file(b"sample bytes content")
    assert content == SAMPLE_TEXT
    mock_parser.assert_called_once_with(b"sample bytes content", requestOptions={"timeout": 5})


@patch("tika.parser.from_buffer")
def test_read_file_bytes_failure(mock_parser, reader):
    """Test failed file reading from bytes buffer."""
    mock_parser.return_value = FAILED_PARSE
    content = reader.read_file(b"invalid bytes content")
    assert content == ""
    mock_parser.assert_called_once_with(b"invalid bytes content", requestOptions={"timeout": 5})


@patch("tika.parser.from_buffer")
def test_read_file_bytes_exception(mock_parser, reader):
    """Test exception handling during bytes buffer reading."""
    mock_parser.side_effect = Exception("Test error")
    content = reader.read_file(b"problematic bytes content")
    assert content == ""
    mock_parser.assert_called_once_with(b"problematic bytes content", requestOptions={"timeout": 5})
