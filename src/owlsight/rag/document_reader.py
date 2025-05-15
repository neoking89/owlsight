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
from typing import Optional, List, Tuple, Generator, Union
import zipfile
import logging
import glob
import hashlib
import concurrent.futures
import threading

import tika  # ▶ critical change ◀  import tika first; parser is imported later

from owlsight.utils.logger import logger

TIKA_SERVER_JAR = None


def _has_internet_connection(host: str = "8.8.8.8", port: int = 53, timeout: int = 3) -> bool:
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


class DocumentReader:
    """
    A class for reading text content from files using Apache Tika.

    Supports a wide variety of file formats and provides streaming capabilities
    for processing large directories.

    Examples
    --------
    >>> with DocumentReader() as reader:
    ...     for filename, content in reader.read_directory("path/to/docs"):
    ...         print(f"Processing {filename}...")
    ...         process_content(content)
    """

    def __init__(
        self,
        supported_extensions: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
        timeout: int = 5,
        text_only: bool = True,
        tika_server_jar_path: Optional[str] = None,
        tika_headers: Optional[dict] = None,
        max_workers: Optional[int] = None,
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
        timeout : int, default=5
            Timeout in seconds for Tika processing
        text_only : bool, default=True
            Whether to request only text content from Tika.
            If False, will request both text and metadata.
        tika_server_jar_path : str, optional
            Path to the Tika server JAR file. If not provided, will use the default Tika server.
            For offline usage, set this to 'file:///path/to/tika-server.jar'
        tika_headers : dict, optional
            Additional headers to send with Tika requests.
            Example: {'X-Tika-PDFextractInlineImages': 'true', 'Accept-Encoding': 'gzip, deflate'}
        max_workers : int, optional
            Maximum number of threads to use for concurrent processing.
            If None, defaults to os.cpu_count() * 5.
        """
        global TIKA_SERVER_JAR
        global parser  # ▶ critical change ◀ the parser symbol will be injected below

        self.supported_extensions = supported_extensions
        self.ignore_patterns = ignore_patterns or []
        self.timeout = timeout
        self.tika_headers = tika_headers
        self.text_only = text_only

        # ────────────────────────────────────────────────────────────────
        # Critical fix: decide ONLINE vs OFFLINE and configure Tika flags
        # ────────────────────────────────────────────────────────────────
        if _has_internet_connection():
            # Online ⇒ rely on an external server (default localhost:9998 or env var)
            tika.TikaClientOnly = True
            self.tika_server_jar_path = None
            logger.info("Internet detected - using remote Tika server (client-only mode).")
        else:
            # Offline ⇒ locate/extract local JAR and let tika-python auto-launch it
            self.tika_server_jar_path = self._prepare_offline_jar(tika_server_jar_path)
            TIKA_SERVER_JAR = self.tika_server_jar_path
            os.environ["TIKA_SERVER_JAR"] = TIKA_SERVER_JAR
            tika.TikaClientOnly = False
            logger.info(f"Offline: will auto-launch local Tika server from {TIKA_SERVER_JAR}")

        # Import parser *after* client-only mode has been set so it honours the flag
        from tika import parser as _tika_parser  # type: ignore
        parser = _tika_parser  # inject into module namespace for existing calls

        # Initialize ThreadPoolExecutor with sensible default
        if max_workers is None:
            max_workers = (os.cpu_count() or 1) * 5
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._tika_server_start_lock = threading.Lock()
        self._tika_server_started_successfully = False

    # ------------------------------------------------------------------ #
    # Helper for offline JAR handling (critical addition)                #
    # ------------------------------------------------------------------ #
    def _prepare_offline_jar(self, explicit_path: Optional[str]) -> str:
        """
        Locate or extract a Tika server JAR for offline use.

        Returns
        -------
        str
            Absolute path to the JAR file ready to be launched by tika-python.
        """
        if explicit_path:
            if not explicit_path.endswith(".jar"):
                raise ValueError("tika_server_jar_path must point to a .jar file.")
            if not os.path.exists(explicit_path):
                raise FileNotFoundError(explicit_path)
            return explicit_path

        # Search blobs/ for a tika-server zip
        zip_files = glob.glob(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "blobs", "tika-server-standard*.zip")
        )
        if not zip_files:
            raise RuntimeError(
                "No internet connection and no tika-server zip found in blobs/. "
                "Download it or pass tika_server_jar_path."
            )

        zip_files.sort(reverse=True)
        jar_path = self._extract_tika_server(zip_files[0])
        if not os.path.exists(jar_path):
            raise FileNotFoundError(f"Extracted Tika server not found at {jar_path}")
        return jar_path

    # (the remainder of the class definition is unchanged)  # ← keeps all docstrings & behaviour intact
    # --------------------------------------------------------------------- #
    # Context-manager helpers                                               #
    # --------------------------------------------------------------------- #
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the internal thread pool executor."""
        if hasattr(self, "executor") and self.executor:
            self.executor.shutdown(wait=wait)
            logger.debug("DocumentReader's ThreadPoolExecutor has been shut down.")

    # --------------------------------------------------------------------- #
    # File-filter helpers                                                   #
    # --------------------------------------------------------------------- #
    def should_ignore_file(self, filepath: str) -> bool:
        """
        Check if a file should be ignored based on gitignore-style patterns.
        """
        if not self.ignore_patterns:
            return False

        filepath_norm = os.path.normpath(filepath)
        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(filepath_norm, pattern):
                return True
            if "**" in pattern:
                parts = filepath_norm.split(os.sep)
                pattern_parts = pattern.split("/")
                if any(fnmatch.fnmatch("/".join(parts[i:]), "/".join(pattern_parts)) for i in range(len(parts))):
                    return True
        return False

    def is_supported_file(self, filepath: str) -> bool:
        """
        Check if a file is supported based on its extension and ignore patterns.
        """
        if self.should_ignore_file(filepath):
            return False
        if not self.supported_extensions:
            return True
        return any(filepath.lower().endswith(ext.lower()) for ext in self.supported_extensions)

    # --------------------------------------------------------------------- #
    # Tika server helpers                                                   #
    # --------------------------------------------------------------------- #
    def _ensure_tika_server_started(self) -> None:
        """
        Ensure the Tika server has been started exactly once in a thread-safe way.
        """
        if self._tika_server_started_successfully:
            return
        with self._tika_server_start_lock:
            if self._tika_server_started_successfully:
                return
            try:
                logger.info("Attempting to start/verify Tika server…")
                parser.from_buffer("")  # benign ping; auto-starts if needed
                logger.info("Tika server is responsive.")
                self._tika_server_started_successfully = True
            except Exception as e:
                logger.error(f"Failed to start or verify Tika server: {e}")
                raise RuntimeError(f"Tika server could not be initialized or started: {e}") from e

    # --------------------------------------------------------------------- #
    # Core file-reading APIs                                                #
    # --------------------------------------------------------------------- #
    def read_file(self, file_source: Union[str, bytes]) -> Optional[str]:
        """
        Read text content from a single file path or bytes buffer.
        """
        request_options = {"timeout": self.timeout}
        parser_params = {
            "service": "text" if self.text_only else "all",
            "requestOptions": request_options,
            "headers": self.tika_headers,
        }

        try:
            if isinstance(file_source, str):
                filepath = file_source
                if not os.path.exists(filepath):
                    raise FileNotFoundError(filepath)
                if not self.is_supported_file(filepath):
                    logger.debug(f"Skipping unsupported or ignored file: {filepath}")
                    return ""
                self._ensure_tika_server_started()
                parsed = parser.from_file(filepath, **parser_params)

            elif isinstance(file_source, bytes):
                self._ensure_tika_server_started()
                parsed = parser.from_buffer(file_source, **parser_params)
            else:
                raise TypeError("file_source must be a file path (str) or bytes.")

            if parsed and parsed.get("status") == 200:
                return parsed.get("content", "") or ""
            status = parsed.get("status") if parsed else "N/A"
            logger.warning(f"Tika failed to extract text from {file_source!r}. Status: {status}")
            return ""
        except (FileNotFoundError, TypeError):
            raise
        except Exception as e:
            logger.error(f"Exception during Tika processing for {file_source!r}: {e}")
            return None

    def _read_file_task(self, filepath_abs: str, yield_path: str) -> Tuple[str, Optional[str]]:
        """Task wrapper for executor submission."""
        try:
            return yield_path, self.read_file(filepath_abs)
        except Exception as e:
            logger.error(f"Exception in _read_file_task for {filepath_abs}: {e}")
            return yield_path, None

    def read_directory(
        self,
        directory_path: str,
        relative_paths: bool = True,
        recursive: bool = True,
        ignore_hidden: bool = False,
    ) -> Generator[Tuple[str, Optional[str]], None, None]:
        """
        Read text content from all supported files in a directory concurrently.
        """
        if not os.path.isdir(directory_path):
            raise ValueError(f"Provided path is not a directory: {directory_path}")

        tasks = []
        submitted_paths_abs: set[str] = set()

        for root, dirs, files_in_root in os.walk(directory_path, topdown=True):
            current_files = files_in_root
            if ignore_hidden:
                current_files = [f for f in files_in_root if not f.startswith(".")]
                dirs[:] = [d for d in dirs if not d.startswith(".")]

            if root == directory_path or recursive:
                for filename in current_files:
                    filepath_abs = os.path.join(root, filename)
                    if filepath_abs in submitted_paths_abs:
                        continue
                    if not self.is_supported_file(filepath_abs):
                        logger.debug(f"Skipping unsupported or ignored file: {filepath_abs}")
                        continue
                    yield_path = os.path.relpath(filepath_abs, directory_path) if relative_paths else filepath_abs
                    tasks.append(self.executor.submit(self._read_file_task, filepath_abs, yield_path))
                    submitted_paths_abs.add(filepath_abs)

            if not recursive:
                dirs[:] = []

        if tasks:
            self._ensure_tika_server_started()

        for future in concurrent.futures.as_completed(tasks):
            try:
                yield future.result()
            except Exception as e:
                logger.error(f"Unexpected error processing a future: {e}")

    # --------------------------------------------------------------------- #
    # Helper utilities                                                      #
    # --------------------------------------------------------------------- #
    def _extract_tika_server(self, zip_path: str) -> str:
        """Extract Tika server JAR from a zip file and validate contents."""
        extract_dir = Path(zip_path).parent / "extracted"
        jar_pattern = "**/tika-server*.jar"
        md5_pattern = "**/tika-server*.jar.md5"

        try:
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            jar_files = list(extract_dir.glob(jar_pattern))
            if not jar_files:
                raise FileNotFoundError(f"No tika-server JAR found in {zip_path}")

            md5_files = list(extract_dir.glob(md5_pattern))
            if md5_files:
                self._verify_md5(jar_files[0], md5_files[0])

            return str(jar_files[0])

        except zipfile.BadZipFile:
            raise ValueError(f"Invalid zip file: {zip_path}")

    def _verify_md5(self, jar_path: Path, md5_path: Path) -> None:
        """Verify JAR file against MD5 checksum."""
        expected_hash = md5_path.read_text().strip()
        actual_hash = hashlib.md5(jar_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"MD5 mismatch for {jar_path.name}\nExpected: {expected_hash}\nActual:   {actual_hash}")
