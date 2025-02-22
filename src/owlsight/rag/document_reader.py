"""
Module for reading text content from files using Apache Tika.
This module provides a class that can extract text from various file formats including:
- PDF documents
- Microsoft Office documents (Word, Excel, PowerPoint)
- OpenOffice documents
- Images (via OCR)
- HTML/XML
- Plain text
- And many more formats supported by Apache Tika
"""

import os
import fnmatch
import socket
from pathlib import Path
from typing import Optional, List, Generator, Tuple, Union
import zipfile
import logging
import glob
import hashlib
from functools import partial

import tika
from tika import parser

from owlsight.utils.logger import logger

TIKA_SERVER_JAR = None


def _has_internet_connection(host="8.8.8.8", port=53, timeout=3):
    """
    Check if there is an internet connection by trying to connect to Google's DNS.
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except (socket.timeout, socket.gaierror, OSError):
        return False


# Disable Tika logging
tika_logger = logging.getLogger("tika.tika")
tika_logger.setLevel(logging.ERROR)

# Configure Tika to run in client-only mode
tika.TikaClientOnly = True


class DocumentReader:
    """
    A class for reading text content from files using Apache Tika.

    Supports a wide variety of file formats and provides streaming capabilities
    for processing large directories.

    Examples
    --------
    >>> reader = DocumentReader()
    >>> for filename, content in reader.read_directory("path/to/docs"):
    ...     print(f"Processing {filename}...")
    ...     process_content(content)
    """

    def __init__(
        self,
        supported_extensions: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
        ocr_enabled: bool = True,
        timeout: int = 5,
        text_only: bool = True,
        tika_server_jar_path: Optional[str] = None,
    ):
        """
        Initialize the DocumentReader.

        Parameters
        ----------
        supported_extensions : List[str], optional
            List of file extensions to process. If None, will attempt to process all files.
            Example: ['.pdf', '.doc', '.docx']
        ignore_patterns : List[str], optional
            List of gitignore-style patterns to exclude.
            Example: ['*.pyc', '__pycache__/*', '.venv/**/*']
        ocr_enabled : bool, default=True
            Whether to enable OCR for image files
        timeout : int, default=5
            Timeout in seconds for Tika processing
        text_only : bool, default=True
            Whether to request only text content from Tika.
            If False, will request both text and metadata.
        tika_server_jar_path : str, optional
            Path to the Tika server JAR file. If not provided, will use the default Tika server.
            For offline usage, set this to 'file:///path/to/tika-server.jar'
        """
        global TIKA_SERVER_JAR
        self.supported_extensions = supported_extensions
        self.ignore_patterns = ignore_patterns or []
        self.ocr_enabled = ocr_enabled
        self.timeout = timeout
        self.text_only = text_only

        # Handle TIKA_SERVER_JAR configuration
        self.tika_server_jar_path = None

        if not _has_internet_connection():
            if tika_server_jar_path:
                if not tika_server_jar_path.endswith(".jar"):
                    raise ValueError(f"TIKA_SERVER_JAR must be a .jar file, but got {tika_server_jar_path}")
                if not os.path.exists(tika_server_jar_path):
                    raise FileNotFoundError(f"Tika server jar not found at {tika_server_jar_path}")
                self.tika_server_jar_path = tika_server_jar_path
            else:
                zip_files = glob.glob(
                    os.path.join(os.path.dirname(os.path.dirname(__file__)), "blobs", "tika-server-standard*.zip")
                )
                if not zip_files:
                    raise RuntimeError(
                        "No internet connection detected and no Tika server zip found in blobs/\n"
                        "Please either:\n"
                        "1. Download tika-server-standard-*.zip from https://tika.apache.org/download.html\n"
                        "2. Place it in the blobs/ directory\n"
                        "3. Set the TIKA_SERVER_JAR environment variable"
                    )

                # Find latest version
                zip_files.sort(reverse=True)
                try:
                    self.tika_server_jar_path = self._extract_tika_server(zip_files[0])
                    if not os.path.exists(self.tika_server_jar_path):
                        raise FileNotFoundError(f"Extracted Tika server not found at {self.tika_server_jar_path}")
                except Exception as e:
                    logger.error(f"Tika server extraction failed: {str(e)}")
                    raise RuntimeError(f"Failed to extract Tika server: {str(e)}")

            if not os.path.exists(self.tika_server_jar_path):
                logger.error(f"TIKA_SERVER_JAR path invalid: {self.tika_server_jar_path}")
                raise FileNotFoundError(f"TIKA_SERVER_JAR not found: {self.tika_server_jar_path}")

            TIKA_SERVER_JAR = self.tika_server_jar_path
            logger.info(f"Using local Tika server: {TIKA_SERVER_JAR}")
        else:
            logger.info("Using remote Tika server")

    def should_ignore_file(self, filepath: str) -> bool:
        """
        Check if a file should be ignored based on gitignore-style patterns.

        Parameters
        ----------
        filepath : str
            Path to the file to check

        Returns
        -------
        bool
            True if the file should be ignored, False otherwise
        """
        if not self.ignore_patterns:
            return False

        # Convert to relative path for pattern matching
        filepath = os.path.normpath(filepath)

        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(filepath, pattern):
                return True
            # Handle directory wildcards (e.g., '**/test/')
            if "**" in pattern:
                parts = filepath.split(os.sep)
                pattern_parts = pattern.split("/")
                if any(fnmatch.fnmatch("/".join(parts[i:]), "/".join(pattern_parts)) for i in range(len(parts))):
                    return True
        return False

    def is_supported_file(self, filepath: str) -> bool:
        """
        Check if a file is supported based on its extension and ignore patterns.

        Parameters
        ----------
        filepath : str
            Path to the file to check

        Returns
        -------
        bool
            True if the file should be processed, False otherwise
        """
        if self.should_ignore_file(filepath):
            return False

        if not self.supported_extensions:
            return True

        return any(filepath.lower().endswith(ext.lower()) for ext in self.supported_extensions)

    def read_file(self, file_source: Union[str, bytes]) -> str:
        """
        Read and extract text content from either a file path or file content buffer.

        Parameters
        ----------
        file_source : Union[str, bytes]
            Either a path to the file to read (str) or the raw file content buffer (bytes).
            For file paths, the file must exist and be readable.
            For bytes content, it should be the raw file content buffer.

        Returns
        -------
        str
            Extracted text content, or an empty string if reading fails
        --------
        >>> reader = DocumentReader()
        >>> # Reading from file path
        >>> content = reader.read_file("path/to/document.pdf")
        >>> # Reading from bytes buffer
        >>> with open("document.pdf", "rb") as f:
        ...     content = reader.read_file(f.read())
        """
        is_file = not isinstance(file_source, bytes)
        if not is_file:
            parse_func = parser.from_buffer
        else:
            parse_func = partial(
                parser.from_file,
                service="text" if self.text_only else "all",
            )

        try:
            # Parse the file using Tika with timeout, requesting only text content
            parsed = parse_func(
                file_source,
                requestOptions={"timeout": self.timeout},
            )

            if parsed.get("status") != 200:
                logger.warning(f"Failed to parse {'file buffer' if not is_file else file_source}. Status: {parsed.get('status')}")
                return ""

            content = parsed.get("content", "")

            # Clean up the extracted text
            if content:
                content = content.strip()
                # Remove any null characters
                content = content.replace("\x00", "")
                # Normalize newlines
                content = content.replace("\r\n", "\n")
                return content

            return ""

        except Exception as e:
            logger.error(f"Error processing {'file buffer' if not is_file else file_source}: {str(e)}")
            return ""

    def read_directory(self, directory: str, recursive: bool = True) -> Generator[Tuple[str, str], None, None]:
        """
        Read all supported files in a directory and yield their content.

        Parameters
        ----------
        directory : str
            Path to the directory to process
        recursive : bool, default=True
            Whether to recursively process subdirectories

        Yields
        ------
        tuple of (str, str)
            Pairs of (filename, content) for each successfully processed file

        Examples
        --------
        >>> reader = DocumentReader()
        >>> for filepath, content in reader.read_directory("docs"):
        ...     print(f"Found {len(content)} characters in {filepath}")
        """
        directory: Path = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Walk through the directory
        for root, _, files in os.walk(directory):
            # Skip processing subdirectories if not recursive
            if not recursive and root != str(directory):
                continue

            for filename in files:
                filepath = os.path.join(root, filename)

                # Skip unsupported or ignored files
                if not self.is_supported_file(filepath):
                    continue

                # Try to read the file
                content = self.read_file(filepath)
                if content:
                    yield filepath, content

    def _extract_tika_server(self, zip_path: str) -> str:
        """Extract Tika server JAR from zip file and validate contents."""
        extract_dir = Path(zip_path).parent / "extracted"
        jar_pattern = "**/tika-server*.jar"
        md5_pattern = "**/tika-server*.jar.md5"

        try:
            # Create extraction directory
            extract_dir.mkdir(parents=True, exist_ok=True)

            # Extract zip contents
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            # Find extracted JAR file
            jar_files = list(extract_dir.glob(jar_pattern))
            if not jar_files:
                raise FileNotFoundError(f"No tika-server JAR found in {zip_path}")

            # Find MD5 file if exists
            md5_files = list(extract_dir.glob(md5_pattern))
            if md5_files:
                self._verify_md5(jar_files[0], md5_files[0])

            return str(jar_files[0])

        except zipfile.BadZipFile:
            raise ValueError(f"Invalid zip file: {zip_path}")

        # finally:
        #     shutil.rmtree(extract_dir)

    def _verify_md5(self, jar_path: Path, md5_path: Path) -> None:
        """Verify JAR file against MD5 checksum."""
        expected_hash = md5_path.read_text().strip()
        actual_hash = hashlib.md5(jar_path.read_bytes()).hexdigest()

        if actual_hash != expected_hash:
            raise ValueError(f"MD5 mismatch for {jar_path.name}\nExpected: {expected_hash}\nActual:   {actual_hash}")


if __name__ == "__main__":
    # Create a DocumentReader instance
    reader = DocumentReader()

    # Read the current directory
    for filepath, content in reader.read_directory(".", recursive=False):
        print(f"Found {len(content)} characters in {filepath}")
