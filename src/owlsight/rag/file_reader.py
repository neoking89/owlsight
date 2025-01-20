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
from pathlib import Path
from typing import Generator, Tuple, Optional, List


from tika import parser
import tika

from owlsight.utils.logger import logger

# Disable Tika logging
import logging
tika_logger = logging.getLogger('tika.tika')
tika_logger.setLevel(logging.ERROR)

# Configure Tika to run in client-only mode
tika.TikaClientOnly = True

class TikaFileReader:
    """
    A class for reading text content from files using Apache Tika.
    Supports a wide variety of file formats and provides streaming capabilities
    for processing large directories.

    Examples:
        reader = TikaFileReader()
        for filename, content in reader.read_directory("path/to/docs"):
            print(f"Processing {filename}...")
            process_content(content)
    """

    def __init__(self, 
                 supported_extensions: Optional[List[str]] = None,
                 ocr_enabled: bool = True,
                 timeout: int = 300):
        """
        Initialize the TikaFileReader.

        Args:
            supported_extensions: Optional list of file extensions to process.
                               If None, will attempt to process all files.
                               Example: ['.pdf', '.doc', '.docx']
            ocr_enabled: Whether to enable OCR for image files
            timeout: Timeout in seconds for Tika processing
        """
        self.supported_extensions = supported_extensions
        self.ocr_enabled = ocr_enabled
        self.timeout = timeout

    def is_supported_file(self, filepath: str) -> bool:
        """
        Check if a file is supported based on its extension.

        Args:
            filepath: Path to the file to check

        Returns:
            bool: True if the file should be processed, False otherwise
        """
        if not self.supported_extensions:
            return True
        
        return any(filepath.lower().endswith(ext.lower()) 
                  for ext in self.supported_extensions)

    def read_file(self, filepath: str) -> Optional[str]:
        """
        Read and extract text content from a single file.

        Args:
            filepath: Path to the file to read

        Returns:
            Optional[str]: Extracted text content if successful, None otherwise
        """
        try:
            # Parse the file using Tika
            parsed = parser.from_file(filepath, 
                                    requestOptions={'timeout': self.timeout})
            
            if parsed.get("status") != 200:
                logger.warning(f"Failed to parse {filepath}. Status: {parsed.get('status')}")
                return None

            content = parsed.get("content", "")
            
            # Clean up the extracted text
            if content:
                content = content.strip()
                # Remove any null characters
                content = content.replace('\x00', '')
                # Normalize newlines
                content = content.replace('\r\n', '\n')
                return content
            
            return None

        except Exception as e:
            logger.error(f"Error processing {filepath}: {str(e)}")
            return None

    def read_directory(self, 
                      directory: str, 
                      recursive: bool = True) -> Generator[Tuple[str, str], None, None]:
        """
        Read all supported files in a directory and yield their content.

        Args:
            directory: Path to the directory to process
            recursive: Whether to recursively process subdirectories

        Yields:
            Tuple[str, str]: Pairs of (filename, content) for each successfully processed file

        Examples:
            reader = TikaFileReader()
            for filepath, content in reader.read_directory("docs"):
                print(f"Found {len(content)} characters in {filepath}")
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Walk through the directory
        for root, _, files in os.walk(directory):
            # Skip processing subdirectories if not recursive
            if not recursive and root != str(directory):
                continue

            for filename in files:
                filepath = os.path.join(root, filename)
                
                # Skip unsupported files
                if not self.is_supported_file(filepath):
                    continue

                # Try to read the file
                content = self.read_file(filepath)
                if content:
                    yield filepath, content

    def __repr__(self) -> str:
        """Return string representation of the reader configuration."""
        return (f"TikaFileReader(supported_extensions={self.supported_extensions}, "
                f"ocr_enabled={self.ocr_enabled}, timeout={self.timeout})")


if __name__ == "__main__":
    reader = TikaFileReader()
    for filepath, content in reader.read_directory("docs"):
        print(f"Found {len(content)} characters in {filepath}")