import importlib.util
import asyncio
import os
import inspect
import traceback
import re
import subprocess
import sys
import json
import dill
import time
import random
from typing import Optional, List, Dict, Union, Iterable, Callable, TypeVar, Any
from datetime import datetime
from pathlib import Path
import shutil
import platform

from huggingface_hub import CachedRepoInfo

from owlsight.utils.helper_functions import safe_lru_cache
from owlsight.utils.logger import logger

document_searcher_type = TypeVar("DocumentSearcher")
document_reader_type = TypeVar("DocumentReader")

MAX_CACHE_SIZE = 128
EXCLUDE_TOOLS = [
    "owl_tools",
    "owl_show",
    "owl_save_namespace",
    "owl_load_namespace",
    "owl_create_document_searcher",
    "owl_context_length",
    "owl_press",
    "owl_history",
    "owl_models",
    "owl_import",
    "owl_search_and_scrape",
]


class OwlDefaultFunctions:
    """
    Define default functions that can be used in the Python interpreter.
    This provides the user with some utility functions to interact with the interpreter.
    Convention is that the functions start with 'owl_' to avoid conflicts with built-in functions.

    This class is open for extension, as possibly more useful functions can be added in the future.
    """

    def __init__(self, globals_dict: Dict):
        """
        Initialize the OwlDefaultFunctions class.

        Parameters
        ----------
        globals_dict : Union[dict, GlobalPythonVarsDict]
            Dictionary of global variables.
            If it is only relevant to use methods in this class, pass an empty dict.

        """
        # Add check to make sure every function starts with 'owl_'
        self._check_method_naming_convention()

        self.globals_dict = globals_dict
        self._document_reader = None
        self._document_searcher_cache = {}  # Cache for DocumentSearcher instances

    def owl_tools(self, as_json: bool = True) -> List[Union[Callable, Dict]]:
        """
        Retrieve available tool-callable functions in OpenAI-compatible format.

        Returns
        -------
        List[Union[Callable, Dict]]
            List of tools/functions available for execution. Example JSON format:
            {{"name": "tool_name", "description": "...", "parameters": {{...}}}}
        as_json : bool, default=True
            When True, returns tools in JSON schema format compatible with OpenAI's
            function calling API. When False, returns raw function objects.

        Notes
        -----
        - Maintains compatibility with OpenAI's tool calling specifications
        """
        if not hasattr(self.globals_dict, "get_tools"):
            raise ValueError("get_tools method not found in globals_dict. Use GlobalPythonVarsDict instead.")
        tools = self.globals_dict.get_tools(exclude_keys=EXCLUDE_TOOLS, as_json=as_json).copy()
        return tools

    def owl_read(
        self,
        file_source: Union[str, Path, bytes, Iterable[Union[str, Path]]],
        recursive: bool = False,
        ignore_patterns: Optional[List[str]] = None,
        timeout: int = 5,
        tika_headers: Optional[Dict[str, str]] = None,
    ) -> Union[str, Dict[str, Optional[str]]]:
        """
        Read content from local files or directories using advanced parsing (e.g., Tika).

        This function is designed for **local** file system paths. For web URLs,
        use `owl_scrape()`.

        Parameters
        ----------
        file_source : str | Path | bytes | Iterable[Union[str, Path]]
            A single file path, a file-like object (bytes), a directory path,
            or an iterable of file paths.
        recursive : bool, default False
            If `file_source` is a directory, scan sub-folders recursively.
        ignore_patterns : Optional[List[str]], default None
            A list of gitignore-style glob patterns to exclude files.
            These patterns apply to both file and directory names.
        timeout : int, default 5
            Timeout in seconds for the advanced parsing process (Tika) for
            each individual file.
        tika_headers : Optional[Dict[str, str]], default None
            Additional headers to send with Tika requests.

        Returns
        -------
        Union[str, Dict[str, Optional[str]]]
            - If `file_source` is a single file path or bytes:
                - `str`: The extracted text content of the file.
                - Returns an empty string (`""`) if the file is unsupported,
                  ignored by `ignore_patterns`, or successfully parsed by Tika
                  but contains no extractable text.
                - If advanced parsing (Tika) fails for a supported file,
                  a fallback to basic text reading is attempted. If this also
                  fails, an error message might be returned or an exception raised.
            - If `file_source` is a directory or an iterable of paths:
                - `Dict[str, Optional[str]]`: A dictionary mapping each processed
                  file path (str) to its extracted content (str or None).
                - Content for a file will be `None` if advanced parsing (Tika)
                  failed for that specific file and no fallback was successful.
                - Content will be an empty string (`""`) for unsupported/ignored
                  files or files with no extractable text after Tika processing.

        Raises
        ------
        ValueError
            If `file_source` or any path in an iterable `file_source` is detected
            as a web URL.
        RuntimeError
            If critical errors occur during directory reading or fallback file reading.

        Notes
        -----
        - The function first attempts to use advanced parsing (via Apache Tika)
          to extract rich text content.
        - If a file is explicitly unsupported by Tika (e.g., based on its extension
          and not in `DocumentReader.SUPPORTED_EXTENSIONS`) or matches an
          `ignore_patterns`, it's skipped, and an empty string (`""`) is returned/stored.
        - If Tika encounters an internal error while processing a *supported* file,
          `owl_read` logs a warning and attempts a basic fallback read (plain text).
          If `file_source` is a single file, the result of this fallback is returned.
          If `file_source` is a directory/iterable, `None` is stored for that file
          in the results dictionary if Tika failed.
        - The `timeout` applies to the Tika parsing stage.
        """
        if isinstance(file_source, (str, Path)) and is_url(file_source):
            raise ValueError(f"owl_read requires local files. Use owl_scrape() for URLs like '{file_source}'")

        if isinstance(file_source, Iterable) and not isinstance(file_source, (str, bytes)):
            for p in file_source:
                if isinstance(p, (str, Path)) and is_url(p):
                    raise ValueError("Detected web URL in paths. Use owl_scrape() instead.")

        try:
            reader = self._get_document_reader(
                timeout=timeout, ignore_patterns=ignore_patterns, tika_headers=tika_headers
            )

            if isinstance(file_source, bytes):
                print("Given file_source is a buffer, trying to read as a buffer...")
                try:
                    content = reader.read_file(file_source)
                    return content
                except Exception as e:
                    raise RuntimeError(f"Error reading buffer: {e}")

            # handle directory
            if isinstance(file_source, (str, Path)):
                file_source = Path(file_source)
                if file_source.is_dir():
                    results = {}
                    try:
                        for filepath, content in reader.read_directory(str(file_source), recursive=recursive):
                            # content can be None if _read_file_task had an exception for that file
                            # or "" if Tika processed it to empty, or skipped.
                            # The dict will store None or "" correctly.
                            results[filepath] = content
                        return results
                    except Exception as e:
                        print(f"DocumentReader failed to read directory {file_source}: {str(e)}")
                        raise RuntimeError(f"Error reading directory {file_source}: {str(e)}")
                else:
                    # Handle single file
                    try:
                        content = reader.read_file(str(file_source))
                        if content is not None: # Tika processed it (or skipped returning "")
                            return content
                        else: # content is None, meaning a Tika processing exception occurred
                            # Decide how to handle Tika processing error for a single file.
                            # Option 1: Return a specific error message, maintaining str return type for success/empty.
                            # Option 2: Let it fall through to basic read, but log Tika's specific failure.
                            # Option 3: Raise an error or return None (would change owl_read's type hint for single file).
                            # For now, let's try to return an error message if Tika fails catastrophically for a single file.
                            # If content is "" (empty string from Tika), it will be returned by 'if content is not None' (as "" is not None)
                            # and the fallback won't be triggered unless Tika explicitly errored (returned None).
                            logger.warning(f"Tika processing failed for {file_source}. Attempting basic read.") # Or return error string
                            # If we want to indicate Tika error and not fallback:
                            # return f"Error during advanced processing of {file_source}"
                    except Exception as e: # Catch exceptions from reader.read_file() itself, though most are handled inside
                        logger.warning(f"Exception calling DocumentReader.read_file for {file_source}: {e}. Attempting basic read.")
                        pass # Silently fall back to basic file reading

                    # Fallback to basic file reading if Tika returned None (error) or if above try block had other issues
                    try:
                        with open(file_source, encoding="utf-8") as file:
                            return file.read()
                    except FileNotFoundError:
                        return f"File not found: {file_source}"
                    except Exception as e:
                        raise RuntimeError(f"Error reading file {file_source}: {str(e)}")
            else:
                # Handle iterable of files
                results = {}
                for file_path_item in file_source: # Renamed to avoid conflict with outer scope 'file_path'
                    file_path = Path(file_path_item)
                    try:
                        content = reader.read_file(str(file_path))
                        if content is not None: # Tika processed (or skipped returning ""), or errored returning None
                            results[str(file_path)] = content
                            # If content is "" (empty string from Tika) it's stored.
                            # If content is None (Tika exception), it's stored as None.
                        else: # content is None, Tika processing exception
                            # Store a specific marker or None for this file
                            results[str(file_path)] = None # Explicitly store None if Tika errored
                            logger.warning(f"Tika processing failed for {file_path}. Marked as None.")
                            # No fallback to basic read here, to ensure results dict accurately reflects Tika's outcome per file.
                            # If fallback is desired per file, it needs more complex logic here.

                    except Exception as e: # Catch exceptions from reader.read_file() itself
                        logger.error(f"Error calling DocumentReader.read_file for {file_path}: {e}. Marking as error.")
                        results[str(file_path)] = f"Error processing file: {e}" # Or None, or specific error marker
                return results

        except FileNotFoundError as e:
            print(f"File not found in owl_read: {str(e)}")
            return f"File not found: {str(e).split(': ', 1)[1] if ': ' in str(e) else str(e)}"
        except Exception as e:
            print(f"Critical error in owl_read: {str(e)}")
            raise RuntimeError(f"Critical error: {str(e)}")

    @safe_lru_cache(maxsize=MAX_CACHE_SIZE)
    def owl_search(self, query: str, max_results: int = 10, max_retries: int = 3) -> Dict[str, str]:
        """
        DuckDuckGo text search with simple back-off.

        Parameters
        ----------
        query : str
            Search phrase.
        max_results : int, default 10
            Limit between 1-20.
        max_retries : int, default 3
            Retry attempts on failure.

        Returns
        -------
        dict
            ``{url: "title. snippet"}``.

        Examples
        --------
        >>> owl_search("numpy masked array", max_results=5)
        """
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            error_msg = (
                "The duckduckgo_search package is required for web search functionality "
                "but is not installed. Please install using 'pip install owlsight[search]'"
            )
            print(error_msg)
            raise ImportError(error_msg)

        errors = []
        for attempt in range(max_retries):
            try:
                print(f"Searching for query: {query} (attempt {attempt + 1}/{max_retries})")

                with DDGS() as ddgs:
                    # Use a generator to avoid loading all results at once
                    results = {}
                    for result in ddgs.text(query, max_results=max_results):
                        main_text = f"{result['title']}. {result['body']}"
                        results[result["href"]] = main_text
                        if len(results) >= max_results:
                            break

                if not results:
                    print(f"No results found for query: {query}")
                    return {}

                print(f"Found {len(results)} results")
                return results

            except Exception as e:
                error_msg = f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}"
                print(error_msg)
                errors.append(error_msg)

                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    wait_time = min(2**attempt + random.random(), 10)
                    print(f"Waiting {wait_time:.1f} seconds before retry {attempt + 2}/{max_retries}")
                    time.sleep(wait_time)
                else:
                    print(f"All {max_retries} attempts failed")
                    raise RuntimeError(f"Search failed after {max_retries} attempts: {'; '.join(errors)}")

    def owl_terminal(
        self,
        command: Union[str, List[str]],
        shell: bool,
        cwd: Union[str, Path] = ".",
        capture_output: bool = True,
        timeout: Optional[int] = None,
        raise_on_error: bool = True,
        encoding: str = "utf-8",
    ) -> Dict[str, Union[str, int]]:
        """
        Cross-platform shell command runner.

        Parameters
        ----------
        command : str | list[str]
            String when *shell=True*; else list.
        shell : bool
            Allow shell built-ins when *True*.
        cwd : str | Path, default "."
            Working directory.
        capture_output : bool, default True
            Capture stdout/stderr.
        timeout : int, optional
            Kill after *timeout* seconds.
        raise_on_error : bool, default True
            Raise on non-zero exit.
        encoding : str, default "utf-8"
            Decode byte output.

        Returns
        -------
        dict
            ``{"stdout": str, "stderr": str, "returncode": int}``.

        Examples
        --------
        >>> owl_terminal(["echo", "hi"], shell=False)
        >>> owl_terminal("dir", shell=True)
        """
        import shlex  # local import to avoid polluting global namespace
        if isinstance(command, str):
            if not shell:
                # shlex uses POSIX rules by default; on Windows switch to native.
                posix_mode = platform.system() != "Windows"
                command = shlex.split(command, posix=posix_mode)
        elif not isinstance(command, list):
            raise TypeError("command must be a str or list[str]")

        cmd: Union[str, List[str]] = command  # after normalisation, keep the type.

        # ── 2. Prepare subprocess.run kwargs ──────────────────────────────────────
        run_kwargs = {
            "cwd": os.fspath(cwd),
            "shell": shell,
            "timeout": timeout,
            "check": raise_on_error,
        }

        if capture_output:
            run_kwargs.update(
                {
                    "capture_output": True,
                    "text": True,
                    "encoding": encoding,
                }
            )

        # ── 3. Execute ────────────────────────────────────────────────────────────
        try:
            proc = subprocess.run(cmd, **run_kwargs)
        except subprocess.TimeoutExpired:
            # Let the caller decide how to handle a real timeout.
            raise
        # FileNotFoundError is raised automatically if the executable is missing.

        # ── 4. Package result ─────────────────────────────────────────────────────
        return {
            "stdout": proc.stdout if capture_output else None,
            "stderr": proc.stderr if capture_output else None,
            "returncode": proc.returncode,
        }
    

    def owl_edit(
        self,
        file_path: Union[str, Path],
        edits: List[Dict[str, str]],
        *,
        regex: bool = True,
        create_backup: bool = True,
        backup_suffix: str = ".bak",
        encoding: str = "utf-8",
    ) -> str:
        """
        Apply multiple substitutions to one local file.

        Parameters
        ----------
        file_path : str | Path
            Target file.
        edits : list of dict
            Each dict needs ``"pattern"`` and ``"replacement"``.
        regex : bool, default True
            Interpret *pattern* as regex.
        create_backup : bool, default True
            Save a copy with *backup_suffix*.
        backup_suffix : str, default ".bak"
            Extension for backups.
        encoding : str, default "utf-8"
            File encoding.

        Returns
        -------
        str
            Edited file path.

        Examples
        --------
        >>> owl_edit("notes.txt", [{"pattern": r"foo\\d+", "replacement": "bar"}])
        """
        file_path = Path(file_path)

        # --- pre-flight checks -------------------------------------------------
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not edits or not isinstance(edits, list):
            raise ValueError("Parameter 'edits' must be a non-empty list.")

        for op in edits:
            if "pattern" not in op or "replacement" not in op:
                raise ValueError(
                    "Each edit operation must contain 'pattern' and 'replacement' keys."
                )

        try:
            original_text = file_path.read_text(encoding=encoding)
        except Exception as exc:
            raise RuntimeError(f"Failed to read {file_path}: {exc}")

        # --- create backup if requested ---------------------------------------
        if create_backup:
            backup_path = file_path.with_suffix(file_path.suffix + backup_suffix)
            try:
                shutil.copyfile(file_path, backup_path)
            except Exception as exc:
                raise RuntimeError(f"Could not create backup file: {exc}")

        # --- apply edits -------------------------------------------------------
        new_text = original_text
        for op in edits:
            pattern = op["pattern"]
            replacement = op["replacement"]

            if regex:
                new_text = re.sub(pattern, replacement, new_text, flags=re.MULTILINE)
            else:
                new_text = new_text.replace(pattern, replacement)

        # --- write back to disk -----------------------------------------------
        try:
            file_path.write_text(new_text, encoding=encoding)
        except Exception as exc:
            raise RuntimeError(f"Failed to write edited file {file_path}: {exc}")

        print(f"Applied {len(edits)} edit(s) to {file_path}")
        if create_backup:
            print(f"Backup saved to {backup_path}")

        return str(file_path)


    def owl_import(self, file_path: str):
        """
        Import Python module into the current execution environment.

        Parameters
        ----------
        file_path : str
            Absolute path to Python (.py) file

        Notes
        -----
        - Makes all module symbols available in global namespace
        - Overwrites existing names with same identifiers
        - Handles relative imports within the module

        Raises
        ------
        Exception
            If there is an error importing the module
        Examples
        --------
        >>> owl_import("my_utils.py")
        """
        try:
            module_name = os.path.splitext(os.path.basename(file_path))[0]
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.globals_dict.update(vars(module))
            print(f"Module '{module_name}' imported successfully.")
        except Exception:
            print(f"Error importing module:\n{traceback.format_exc()}")
            raise

    def owl_show(self, docs: bool = True, return_str: bool = False) -> List[str]:
        """
        Display active namespace objects with documentation.

        Parameters
        ----------
        docs : bool, default=True
            Include docstrings in output
        return_str : bool, default=False
            Return formatted string instead of printing

        Returns
        -------
        List[str]
            Formatted inventory of objects when return_str=True

        Notes
        -----
        - Filters out builtins and internal objects (starting with '_')
        - Displays name, parameters and docstring (if available)

        Raises
        ------
        Exception
            If there is an error displaying the namespace objects
        """
        current_globals = self.globals_dict
        active_objects = self.globals_dict._filter_globals(current_globals)

        output = []
        brackets = "#" * 50
        output.append("Active imported objects:")
        output.append(brackets)
        for name, obj in active_objects.items():
            obj_type = type(obj).__name__
            output.append(f"{name} ({obj_type})")

            if docs:
                docstring = obj.__doc__
                if docstring:
                    output.append(f"Doc: {docstring.strip()}")
                else:
                    output.append("Doc: No documentation available")
            output.append(brackets)

        output = "\n".join(output)
        print(output)
        if return_str:
            return output

    def owl_write(self, file_path: str, content: str) -> None:
        """
        Write *content* to *file_path* (UTF-8, overwrite).

        Parameters
        ----------
        file_path : str
            Destination.
        content : str
            Data to write.

        Examples
        --------
        >>> owl_write("output.txt", "hello")
        """
        try:
            with open(file_path, "w") as file:
                file.write(content)
            print(f"Content successfully written to {file_path}")
        except Exception as e:
            print(f"Error writing to file: {e}")
            raise

    def owl_save_namespace(self, file_path: str):
        """
        Serialize current namespace state to disk.

        Parameters
        ----------
        file_path : str
            Output path with .dill extension

        Notes
        -----
        - Excludes internal variables (starting with '_' or 'owl_')
        - Serialization uses dill package
        - Not all object types can be serialized

        Raises
        ------
        Exception
            If there is an error during serialization
        """
        if not file_path.endswith(".dill"):
            file_path += ".dill"

        global_dict = {key: value for key, value in self.globals_dict.items() if not key.startswith(("_", "owl_"))}

        try:
            with open(file_path, "wb") as file:
                dill.dump(global_dict, file)
            print(f"Namespace successfully saved to {file_path}")
        except Exception as e:
            print(f"An error occurred while saving: {e}")
            raise

    def owl_load_namespace(self, file_path: str):
        """
        Load namespace using dill.

        Parameters
        ----------
        file_path : str
            The path to the file to load the namespace from.

        Raises
        ------
        FileNotFoundError
            If the specified file cannot be found
        Exception
            If there is an error during deserialization
        """

        if not file_path.endswith(".dill"):
            file_path += ".dill"
        try:
            with open(file_path, "rb") as file:
                loaded_data = dill.load(file)
            self.globals_dict.update(loaded_data)
            print(f"Namespace successfully loaded from {file_path}")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            raise
        except Exception as e:
            print(f"An error occurred while loading: {e}")
            raise

    @safe_lru_cache(maxsize=MAX_CACHE_SIZE)
    def owl_scrape(
        self,
        urls: List[str],
        max_concurrent: int = 5,
        timeout: int = 10,
    ) -> Dict[str, str]:
        """
        Download and parse the main text from web pages.

        Parameters
        ----------
        urls : list of str
            HTTP/HTTPS addresses.
        max_concurrent : int, default 5
            Maximum simultaneous requests.
        timeout : int, default 10
            Seconds before any single request aborts.

        Returns
        -------
        dict
            ``{url: content (in markdown)}``.

        Examples
        --------
        >>> scraped_content = owl_scrape(["https://pypi.org/project/requests/"])
        """
        from owlsight.app.url_processor import fetch_and_parse_urls, AIOHTTP_AVAILABLE, LXML_AVAILABLE

        if not AIOHTTP_AVAILABLE or not LXML_AVAILABLE:
            missing_packages = []
            if not AIOHTTP_AVAILABLE:
                missing_packages.append("aiohttp")
            if not LXML_AVAILABLE:
                missing_packages.append("lxml")

            error_msg = (
                f"The following packages are required for web scraping functionality "
                f"but are not installed: {', '.join(missing_packages)}. "
                f"Please install using 'pip install owlsight[search]'"
            )
            print(error_msg)
            raise ImportError(error_msg)

        content_dict = asyncio.run(fetch_and_parse_urls(urls, max_concurrent, timeout))
        content_dict = {url: content.strip() for url, content in content_dict.items() if content.strip()}
        return content_dict

    @safe_lru_cache(maxsize=MAX_CACHE_SIZE)
    def owl_search_and_scrape(
        self,
        query: str,
        max_results: int = 10,
        max_concurrent: int = 5,
        timeout: int = 10,
        max_retries: int = 3,
    ) -> Dict[str, str]:
        """
        Search the web then scrape the resulting URLs.

        Parameters
        ----------
        query : str
            DuckDuckGo query.
        max_results : int, default 10
            URL limit.
        max_concurrent : int, default 5
            Concurrent scrapes.
        timeout : int, default 10
            Seconds per scrape.
        max_retries : int, default 3
            Retries for search.

        Returns
        -------
        dict
            ``{url: content (in markdown)}``.

        Examples
        --------
        >>> scraped_content = owl_search_and_scrape("python walrus operator", max_results=3)
        """
        # Check for required packages
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            error_msg = (
                "The duckduckgo_search package is required for web search functionality "
                "but is not installed. Please install using 'pip install owlsight[search]'"
            )
            print(error_msg)
            raise ImportError(error_msg)

        # Try importing from url_processor to check if aiohttp and lxml are available
        from owlsight.app.url_processor import AIOHTTP_AVAILABLE, LXML_AVAILABLE

        if not AIOHTTP_AVAILABLE or not LXML_AVAILABLE:
            missing_packages = []
            if not AIOHTTP_AVAILABLE:
                missing_packages.append("aiohttp")
            if not LXML_AVAILABLE:
                missing_packages.append("lxml")

            error_msg = (
                f"The following packages are required for web scraping functionality "
                f"but are not installed: {', '.join(missing_packages)}. "
                f"Please install using 'pip install owlsight[search]'"
            )
            print(error_msg)
            raise ImportError(error_msg)

        # First perform the search to get URLs
        search_results = self.owl_search(query, max_results=max_results, max_retries=max_retries)

        # Get the URLs from the search results
        urls = list(search_results.keys())

        # Then scrape content from those URLs
        scraped_results = self.owl_scrape(urls, max_concurrent=max_concurrent, timeout=timeout)

        return scraped_results

    def owl_models(self, cache_dir: Optional[str] = None, show_task: bool = False) -> List[str]:
        """
        Audit Hugging Face model cache.

        Parameters
        ----------
        cache_dir : Optional[str], default=None
            Custom cache path override
        show_task : bool, default=False
            Include model task/purpose information

        Returns
        -------
        List[str]
            Formatted report containing:
            - Model IDs
            - Storage sizes
            - Last modified timestamps
            - File locations

        Raises
        ------
        FileNotFoundError
            If the cache directory does not exist
        RuntimeError
            If there is an error accessing the Hugging Face cache
        """
        from huggingface_hub import scan_cache_dir, HfApi
        from huggingface_hub.constants import HF_HUB_CACHE

        output_lines = []
        cache_dir: Path = Path(cache_dir or HF_HUB_CACHE)
        if not cache_dir.exists():
            raise FileNotFoundError(f"Cache directory '{cache_dir}' does not exist.")

        try:
            hf_api = HfApi()
            cache_info = scan_cache_dir(cache_dir)
            if not cache_info.repos:
                raise ValueError(f"No models found in the Hugging Face cache directory {cache_dir}")

            output_lines.append("\n=== Cached Hugging Face Models ===\n")
            for repo in cache_info.repos:
                try:
                    last_modified = datetime.fromtimestamp(repo.last_modified).strftime("%Y-%m-%d %H:%M:%S")
                    output_lines.append(f"Model: {repo.repo_id}")
                    if show_task:
                        model_info = hf_api.model_info(repo.repo_id, expand=["pipeline_tag"])
                        task = model_info.pipeline_tag
                        output_lines.append(f"Task: {task}")
                    output_lines.append(f"Size: {repo.size_on_disk / (1024 * 1024):.2f} MB")
                    output_lines.append(f"Last Modified: {last_modified}")
                    output_lines.append(f"Location: {repo.repo_path}")
                    model_id = self._get_model_id(repo)
                    output_lines.append(f"Eligable for model_id: {model_id}")
                    output_lines.append("-" * 50)
                except Exception as e:
                    output_lines.append(f"Error accessing model with id {repo.repo_id}: {str(e)}")

            output_lines.append(f"\nTotal Cache Size: {cache_info.size_on_disk / (1024 * 1024):.2f} MB")
            output_lines.append(f"Cache Directory: {cache_dir}")

            return "\n".join(output_lines)
        except Exception as e:
            raise RuntimeError(f"Error accessing Hugging Face cache: {str(e)}")

    def owl_press(
        self,
        sequence: List[str],
        exit_python_before_sequence: bool = True,
        time_before_sequence: float = 0.5,
        time_between_keys: float = 0.12,
    ) -> bool:
        """
        Simulate keyboard input for application control.

        Parameters
        ----------
        sequence : List[str]
            Supported key codes:
            - Arrow keys: 'L', 'R', 'U', 'D'
            - Modifiers: 'CTRL+A', 'CTRL+C', 'CTRL+V'
            - Special: 'ENTER', 'DEL', 'SLEEP:X.X'
        exit_python_before_sequence : bool, default=True
            Return to main menu before execution
        time_before_sequence : float, default=0.5
            Initial delay in seconds
        time_between_keys : float, default=0.12
            Typing interval in seconds

        Returns
        -------
        bool
            True if sequence started successfully

        Notes
        -----
        - Runs in separate process to avoid blocking
        - Timings approximate due to system scheduling

        Raises
        ------
        Exception
            If there is an error starting the subprocess
        """
        if not isinstance(sequence, list):
            raise TypeError("sequence must be a list")
        if not all(isinstance(item, str) for item in sequence):
            raise TypeError("sequence must contain only strings")

        if exit_python_before_sequence:
            sequence.insert(0, "ENTER")
            sequence.insert(0, "exit()")

        # Path to your _child_owl_press.py script
        script_path = Path(__file__).parent / "_child_process_owl_press.py"

        params = {
            "sequence": sequence,
            "time_before_sequence": time_before_sequence,
            "time_between_keys": time_between_keys,
        }

        try:
            self._start_child_process_owl_press(script_path, params)
            return True

        except Exception as e:
            current_function_name = inspect.currentframe().f_code.co_name
            print(f"Error starting subprocess from inside {current_function_name}: {e}")
            raise

    def owl_create_document_searcher(
        self,
        documents: Dict[str, str],
        sentence_transformer_model_name: str,
        sentence_transformer_kwargs: Optional[Dict[str, Any]] = None,
        percentile: float = 0.99,
        target_chunk_length: int = 400,
        device: Optional[str] = None,
        **document_searcher_kwargs,
    ) -> document_searcher_type:
        """
        Utility function to create a DocumentSearcher instance from a dictionary of documents.
        This is useful for semantic search across multiple documents using a sentence transformer model.
        A semantic text splitter is used first to split documents into smaller, semantically coherent chunks.
        Both chunking and embedding is done with the model specified by `sentence_transformer_model_name`.

        Parameters
        ----------
        documents : Dict[str, str]
            Dictionary mapping document names (keys) to their text content (values)
        sentence_transformer_model_name : str
            Name of the sentence transformer model to use for embeddings
        sentence_transformer_kwargs : Optional[Dict[str, Any]], default=None
            Keyword arguments to pass to the SentenceTransformer constructor.
            For example: `sentence_transformer_kwargs={"prompts": {"query": "query: ", "passage": "passage: "}}`
        percentile : float, default=0.99
            Percentile threshold for semantic text splitting.
            The higher the percentile, the larger the semantic distance required between adjacent embeddings, the less
            splitting is performed.
        target_chunk_length : int, default=400
            Target character length for text chunks during splitting
        device : Optional[str], default=None
            Device to use for embedding and chunking
        **document_searcher_kwargs :
            Additional keyword arguments to pass to the DocumentSearcher constructor.
            Examples are `cache_dir` and `cache_dir_suffix`.

        Returns
        -------
        DocumentSearcher
            Initialized DocumentSearcher instance ready for semantic search queries

        Notes
        -----
        - Uses SemanticTextSplitter for intelligent document chunking
        - Suitable for use with scraped web content or local documents
        - Use a (domain-specific) sentence transformer model for embeddings.
        Check https://huggingface.co/spaces/mteb/leaderboard with "Should be sentence-transformers compatible" turned on.
        - Caches DocumentSearcher instances based on input parameters for reuse

        Raises
        ------
        Exception
            If there is an error creating the DocumentSearcher instance
        """
        from owlsight.rag.core import DocumentSearcher
        from owlsight.rag.text_splitters import SemanticTextSplitter

        # Create cache key from all parameters
        cache_params = {
            "percentile": percentile,
            "target_chunk_length": target_chunk_length,
            **document_searcher_kwargs,
        }
        if sentence_transformer_kwargs is not None:
            cache_params.update(sentence_transformer_kwargs)

        cache_key = self._create_cache_key(documents, sentence_transformer_model_name, **cache_params)

        # Return cached instance if available
        if cache_key in self._document_searcher_cache:
            return self._document_searcher_cache[cache_key]

        # Create new instance if not in cache
        doc_splitter = SemanticTextSplitter(
            percentile=percentile,
            target_chunk_length=target_chunk_length,
            model_name=sentence_transformer_model_name,
            device=device,
            sentence_transformer_kwargs=sentence_transformer_kwargs,
        )

        searcher = DocumentSearcher(
            documents,
            sentence_transformer_model=sentence_transformer_model_name,
            text_splitter=doc_splitter,
            device=device,
            sentence_transformer_kwargs=sentence_transformer_kwargs,
            **document_searcher_kwargs,
        )

        # Cache the new instance
        self._document_searcher_cache[cache_key] = searcher
        return searcher

    def _get_model_id(self, repo: CachedRepoInfo) -> str:
        """
        Determine the model ID based on the repository content.

        Parameters
        ----------
        repo : Repository
            The repository object containing repo_id and repo_path

        Returns
        -------
        str or Path
            The determined model ID
        """
        repo_lower = repo.repo_id.lower()
        if "onnx" in repo_lower:
            for file in repo.repo_path.glob("**/*"):
                if file.is_dir() and any(f.endswith(".onnx") for f in os.listdir(file)):
                    return file
        elif "gguf" in repo_lower:
            for file in repo.repo_path.glob("**/*"):
                if str(file).endswith(".gguf"):
                    return file
        return repo.repo_id

    def _start_child_process_owl_press(self, script_path: Path, params: Dict) -> None:
        params_json = json.dumps(params)
        subprocess.Popen([sys.executable, str(script_path), params_json])

    def _get_document_reader(
        self, timeout: int = 5, ignore_patterns: Optional[List[str]] = None, tika_headers: Optional[Dict[str, str]] = None,
    ) -> document_reader_type:
        """
        Lazy initialization of DocumentReader to prevent overhead.
        Returns an instance of DocumentReader, creating it if it doesn't exist.
        """
        from owlsight.rag.document_reader import DocumentReader

        if self._document_reader is None:
            self._document_reader = DocumentReader(
                timeout=timeout,
                ignore_patterns=ignore_patterns,
                tika_headers=tika_headers,
            )
        return self._document_reader

    def _check_method_naming_convention(self):
        """Check if all methods in the class start with 'owl_'."""
        methods = inspect.getmembers(self, predicate=inspect.ismethod)
        methods = [method for method in methods if not method[0].startswith("_")]
        for name, _ in methods:
            if not name.startswith("owl_"):
                raise ValueError(f"Method '{name}' does not follow the 'owl_' naming convention!")

    def _create_cache_key(self, documents: Dict[str, str], model_name: str, **kwargs) -> str:
        """
        Create a cache key from DocumentSearcher parameters.

        The key is created by combining:
        1. A hash of the document contents
        2. The model name
        3. A hash of all other parameters
        """
        import hashlib

        # Hash documents
        docs_str = json.dumps(documents, sort_keys=True)
        docs_hash = hashlib.md5(docs_str.encode()).hexdigest()

        # Hash other parameters
        params_str = json.dumps(kwargs, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()

        return f"{docs_hash}_{model_name}_{params_hash}"


# Update get_url to use Django-style regex for better validation
# source: https://stackoverflow.com/questions/7160737/how-to-validate-a-url-in-python-malformed-or-not
IS_URL_PATTERN = re.compile(
    r"^(?:http|ftp)s?://"  # http:// or https://
    r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
    r"localhost|"  # localhost...
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
    r"(?::\d+)?"  # optional port
    r"(?:/?|[/?]\S+)$",
    re.IGNORECASE,
)


def is_url(url: str) -> bool:
    """
    Check if a string is a valid URL.

    Parameters
    ----------
    url : str
        The string to check.

    Returns
    -------
    bool
        True if the string is a valid URL, False otherwise.
    """
    return bool(re.match(IS_URL_PATTERN, url))
