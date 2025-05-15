"""Tests for the DocumentReader class."""

import os
import pytest
from unittest.mock import patch
from pathlib import Path

from owlsight.rag.document_reader import DocumentReader

# Test data
SAMPLE_TEXT = "This is sample text content"
SAMPLE_PDF_TEXT = "This is sample PDF text content"
SAMPLE_OTHER_TEXT = "This is text in another file"

# Mock Tika parser responses
SUCCESSFUL_TEXT_PARSE = {"content": SAMPLE_TEXT, "status": 200, "metadata": {"resourceName": "test.txt"}}
SUCCESSFUL_PDF_PARSE = {"content": SAMPLE_PDF_TEXT, "status": 200, "metadata": {"resourceName": "test.pdf"}}
SUCCESSFUL_OTHER_PARSE = {"content": SAMPLE_OTHER_TEXT, "status": 200, "metadata": {"resourceName": "other.txt"}}
FAILED_PARSE = {"content": None, "status": 500, "metadata": {}}
EMPTY_CONTENT_PARSE = {"content": "", "status": 200, "metadata": {}}


@pytest.fixture
def reader(request): 
    """Create a DocumentReader instance for testing, ensuring shutdown."""
    _reader = DocumentReader(supported_extensions=['.txt', '.pdf'], ignore_patterns=['*.ignored'])
    
    def finalizer():
        _reader.shutdown()
    request.addfinalizer(finalizer)
    return _reader

@pytest.fixture
def test_dir(tmp_path):
    """Create a temporary directory with a diverse set of test files."""
    (tmp_path / "file1.txt").write_text(SAMPLE_TEXT)
    (tmp_path / "document.pdf").write_text(SAMPLE_PDF_TEXT) 
    (tmp_path / "image.jpg").write_text("dummy image data") 
    (tmp_path / "temp.ignored").write_text("this should be ignored")

    sub_dir = tmp_path / "subdir"
    sub_dir.mkdir()
    (sub_dir / "file2.txt").write_text(SAMPLE_OTHER_TEXT)
    (sub_dir / "archive.zip").write_text("dummy zip data") 

    return tmp_path


@pytest.fixture
def mock_parser_config():
    """Provides a configuration for the tika parser mock."""
    def side_effect_func(file_path, **kwargs):
        file_path_str = str(file_path)
        if file_path_str.endswith("file1.txt"):
            return SUCCESSFUL_TEXT_PARSE
        elif file_path_str.endswith("document.pdf"):
            return SUCCESSFUL_PDF_PARSE
        elif file_path_str.endswith("file2.txt"):
            return {**SUCCESSFUL_OTHER_PARSE, "metadata": {"resourceName": os.path.basename(file_path_str)}}
        elif file_path_str.endswith("error.txt"):
            raise Exception("Simulated Tika processing error")
        elif file_path_str.endswith("empty.txt"):
            return EMPTY_CONTENT_PARSE
        print(f"Warning: tika.parser.from_file called with unmocked path: {file_path_str}")
        return FAILED_PARSE 
    return side_effect_func

@patch("owlsight.rag.document_reader._has_internet_connection", return_value=True) 
def test_init_default(mock_internet, reader): 
    assert reader.supported_extensions == ['.txt', '.pdf']
    assert reader.ignore_patterns == ['*.ignored']


def test_is_supported_file(reader): 
    assert reader.is_supported_file("test.txt") is True
    assert reader.is_supported_file("document.pdf") is True
    assert reader.is_supported_file("image.jpg") is False 
    assert reader.is_supported_file("temp.ignored") is False 
    reader_custom_ignore = DocumentReader(ignore_patterns=["**/ignored_dir/*", "*.specific_ignore"])
    assert reader_custom_ignore.should_ignore_file("path/to/ignored_dir/file.txt") is True
    assert reader_custom_ignore.should_ignore_file("path/to/file.specific_ignore") is True
    reader_custom_ignore.shutdown() 


@patch("owlsight.rag.document_reader.parser")
def test_read_file_success(mock_parser, reader, test_dir):
    """Test successful file reading for different types (txt, pdf)."""
    headers = None
    mock_parser.from_file.return_value = SUCCESSFUL_TEXT_PARSE
    txt_path = str(test_dir / "file1.txt")
    content = reader.read_file(txt_path)
    assert content == SAMPLE_TEXT
    mock_parser.from_file.assert_called_with(txt_path, service='text', requestOptions={'timeout': reader.timeout}, headers=headers)

    mock_parser.from_file.return_value = SUCCESSFUL_PDF_PARSE
    pdf_path = str(test_dir / "document.pdf")
    content = reader.read_file(pdf_path)
    assert content == SAMPLE_PDF_TEXT
    mock_parser.from_file.assert_called_with(pdf_path, service='text', requestOptions={'timeout': reader.timeout}, headers=headers)

    mock_parser.from_file.return_value = EMPTY_CONTENT_PARSE
    empty_file_path = test_dir / "empty.txt"
    empty_file_path.write_text("")
    content = reader.read_file(str(empty_file_path))
    assert content == ""
    mock_parser.from_file.assert_called_with(str(empty_file_path), service='text', requestOptions={'timeout': reader.timeout}, headers=headers)

@patch("owlsight.rag.document_reader.parser")
def test_read_file_unsupported(mock_parser, reader, test_dir):
    """Test reading an unsupported file type returns empty string."""
    unsupported_path = str(test_dir / "image.jpg")
    content = reader.read_file(unsupported_path)
    assert content == ""
    mock_parser.from_file.assert_not_called() 

@patch("owlsight.rag.document_reader.parser")
def test_read_file_ignored(mock_parser, reader, test_dir):
    """Test reading an ignored file type returns empty string."""
    ignored_path = str(test_dir / "temp.ignored")
    content = reader.read_file(ignored_path)
    assert content == ""
    mock_parser.from_file.assert_not_called()

@patch("owlsight.rag.document_reader.parser")
def test_read_directory(mock_parser_actual_patch, reader, test_dir, mock_parser_config):
    """Test directory reading functionality with concurrency, mixed files, and error handling."""
    mock_parser_actual_patch.from_file.side_effect = mock_parser_config

    (test_dir / "error.txt").write_text("this file will cause an error")

    expected_files = {
        os.path.join("subdir", "file2.txt"): SAMPLE_OTHER_TEXT,
        "file1.txt": SAMPLE_TEXT,
        "document.pdf": SAMPLE_PDF_TEXT,
        "error.txt": None  
    }

    results_relative = {}
    for filepath, content in reader.read_directory(str(test_dir)):
        results_relative[filepath.replace(os.sep, '/')] = content 
    
    expected_files_normalized_keys = {k.replace(os.sep, '/'): v for k,v in expected_files.items()}

    assert len(results_relative) == len(expected_files_normalized_keys)
    for k_exp, v_exp in expected_files_normalized_keys.items():
        assert k_exp in results_relative
        assert results_relative[k_exp] == v_exp, f"Content mismatch for {k_exp}"

    assert "image.jpg" not in results_relative
    assert "temp.ignored" not in results_relative
    assert os.path.join("subdir", "archive.zip").replace(os.sep, '/') not in results_relative

    results_absolute = {}
    reader_abs = DocumentReader(supported_extensions=['.txt', '.pdf'], ignore_patterns=['*.ignored'])
    mock_parser_actual_patch.from_file.side_effect = mock_parser_config 
    try:
        for filepath, content in reader_abs.read_directory(str(test_dir), relative_paths=False):
            results_absolute[filepath] = content
    finally:
        reader_abs.shutdown()

    abs_expected_files = {
        str(test_dir / "subdir" / "file2.txt"): SAMPLE_OTHER_TEXT,
        str(test_dir / "file1.txt"): SAMPLE_TEXT,
        str(test_dir / "document.pdf"): SAMPLE_PDF_TEXT,
        str(test_dir / "error.txt"): None
    }
    assert len(results_absolute) == len(abs_expected_files)
    for k_exp, v_exp in abs_expected_files.items():
        assert k_exp in results_absolute
        s1 = results_absolute[k_exp]
        s2 = v_exp
        if isinstance(s1, str) and isinstance(s2, str):
            s1 = s1.strip()
            s2 = s2.strip()
        assert s1 == s2, f"Content mismatch for {k_exp}"

    assert mock_parser_actual_patch.from_file.call_count >= len(expected_files) 


@patch("owlsight.rag.document_reader._has_internet_connection", return_value=True)
@patch("owlsight.rag.document_reader.logger")
def test_init_online(mock_logger, mock_internet): 
    """Test DocumentReader initialization in online mode."""
    with patch.object(DocumentReader, 'shutdown') as mock_shutdown: 
        reader_online = DocumentReader()
        mock_logger.info.assert_any_call("Using remote Tika server")
        assert reader_online.tika_server_jar_path is None
        reader_online.shutdown()


def test_document_reader_max_workers():
    """Test DocumentReader initialization with max_workers.""" 
    with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor:
        with patch("owlsight.rag.document_reader._has_internet_connection", return_value=True):
            reader_mw = DocumentReader(max_workers=10)
            mock_executor.assert_called_once_with(max_workers=10)
            reader_mw.shutdown() 
