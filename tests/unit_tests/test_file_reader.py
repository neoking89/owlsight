"""Tests for the TikaFileReader class."""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from owlsight.rag.file_reader import TikaFileReader

# Test data
SAMPLE_TEXT = "This is sample text content"
SAMPLE_PDF_CONTENT = {"content": SAMPLE_TEXT, "status": 200}
FAILED_PARSE = {"content": None, "status": 500}

@pytest.fixture
def reader():
    """Create a TikaFileReader instance for testing."""
    return TikaFileReader()

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
    """Test TikaFileReader initialization with default parameters."""
    reader = TikaFileReader()
    assert reader.supported_extensions is None
    assert reader.ocr_enabled is True
    assert reader.timeout == 300

def test_init_custom():
    """Test TikaFileReader initialization with custom parameters."""
    extensions = ['.pdf', '.doc']
    reader = TikaFileReader(supported_extensions=extensions, 
                          ocr_enabled=False, 
                          timeout=600)
    assert reader.supported_extensions == extensions
    assert reader.ocr_enabled is False
    assert reader.timeout == 600

def test_is_supported_file(reader):
    """Test file extension checking."""
    # Without extension restrictions
    assert reader.is_supported_file("test.pdf") is True
    assert reader.is_supported_file("test.xyz") is True
    
    # With extension restrictions
    reader = TikaFileReader(supported_extensions=['.pdf', '.doc'])
    assert reader.is_supported_file("test.pdf") is True
    assert reader.is_supported_file("test.PDF") is True
    assert reader.is_supported_file("test.xyz") is False

@patch('owlsight.rag.file_reader.parser')
def test_read_file_success(mock_parser, reader):
    """Test successful file reading."""
    mock_parser.from_file.return_value = SAMPLE_PDF_CONTENT
    
    content = reader.read_file("test.pdf")
    assert content == SAMPLE_TEXT
    mock_parser.from_file.assert_called_once()

@patch('owlsight.rag.file_reader.parser')
def test_read_file_failure(mock_parser, reader):
    """Test failed file reading."""
    mock_parser.from_file.return_value = FAILED_PARSE
    
    content = reader.read_file("test.pdf")
    assert content is None

@patch('owlsight.rag.file_reader.parser')
def test_read_file_exception(mock_parser, reader):
    """Test exception handling during file reading."""
    mock_parser.from_file.side_effect = Exception("Test error")
    
    content = reader.read_file("test.pdf")
    assert content is None

@patch('owlsight.rag.file_reader.parser')
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

def test_repr(reader):
    """Test string representation of TikaFileReader."""
    expected = ("TikaFileReader(supported_extensions=None, "
               "ocr_enabled=True, timeout=300)")
    assert repr(reader) == expected
