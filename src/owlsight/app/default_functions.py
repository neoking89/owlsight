import importlib.util
import asyncio
import os
import inspect
import traceback
import re
from typing import Optional, List, Dict, Union, Iterable, Callable
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import json
import dill
import time
import random

from huggingface_hub import CachedRepoInfo

from owlsight.utils.custom_classes import GlobalPythonVarsDict
from owlsight.rag.document_reader import DocumentReader

EXCLUDE_TOOLS = [
    "owl_tools",
    "owl_show",
    "owl_save_namespace",
    "owl_load_namespace",
]


class OwlDefaultFunctions:
    """
    Define default functions that can be used in the Python interpreter.
    This provides the user with some utility functions to interact with the interpreter.
    Convention is that the functions start with 'owl_' to avoid conflicts with built-in functions.

    This class is open for extension, as possibly more useful functions can be added in the future.
    """

    def __init__(self, globals_dict: GlobalPythonVarsDict):
        # Add check to make sure every function starts with 'owl_'
        self._check_method_naming_convention()

        self.globals_dict = globals_dict
        self._document_reader = None

    def _check_method_naming_convention(self):
        """Check if all methods in the class start with 'owl_'."""
        methods = inspect.getmembers(self, predicate=inspect.ismethod)
        methods = [method for method in methods if not method[0].startswith("_")]
        for name, _ in methods:
            if not name.startswith("owl_"):
                raise ValueError(f"Method '{name}' does not follow the 'owl_' naming convention!")

    def _get_document_reader(
        self, timeout: int = 5, ignore_patterns: Optional[List[str]] = None, ocr_enabled: bool = True
    ) -> DocumentReader:
        """
        Lazy initialization of DocumentReader to prevent overhead.
        Returns an instance of DocumentReader, creating it if it doesn't exist.
        """
        if self._document_reader is None:
            self._document_reader = DocumentReader(
                ocr_enabled=ocr_enabled, timeout=timeout, ignore_patterns=ignore_patterns
            )
        return self._document_reader

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
        - Excludes itself from the returned tools to prevent recursion
        - Maintains compatibility with OpenAI's tool calling specifications
        """
        tools = self.globals_dict.get_tools(exclude_keys=EXCLUDE_TOOLS, as_json=as_json).copy()
        return tools

    def owl_read(
        self,
        path: Union[str, Path, Iterable[Union[str, Path]]],
        recursive: bool = False,
        ignore_patterns: Optional[List[str]] = None,
        ocr_enabled: bool = True,
        timeout: int = 5,
    ) -> Union[str, Dict[str, str]]:
        """
        Read LOCAL FILE CONTENTS with advanced document processing.

        Parameters
        ----------
        path : Union[str, Path, Iterable[Union[str, Path]]]
            LOCAL FILE SYSTEM PATHS ONLY. Can be:
            - Single local file
            - Directory (requires recursive=True)
            - List of local files
            DOES NOT SUPPORT WEB URLS

        Notes
        -----
        - For web content/URLs use owl_scrape() instead
        - URL inputs will return explicit error messages
        """
        if isinstance(path, (str, Path)) and is_url(path):
            return f"Error: owl_read requires local files. Use owl_scrape() for URLs like '{path}'"

        if isinstance(path, Iterable):
            for p in path:
                if is_url(p):
                    return "Error: Detected web URL in paths. Use owl_scrape() instead."

        try:
            reader = self._get_document_reader(
                timeout=timeout, ignore_patterns=ignore_patterns, ocr_enabled=ocr_enabled
            )

            # handle directory
            if isinstance(path, (str, Path)):
                path = Path(path)
                if path.is_dir():
                    results = {}
                    try:
                        for filepath, content in reader.read_directory(str(path), recursive=recursive):
                            results[filepath] = content
                        return results
                    except Exception as e:
                        print(f"DocumentReader failed to read directory {path}: {str(e)}")
                        return f"Error reading directory {path}: {str(e)}"
                else:
                    # Handle single file
                    try:
                        content = reader.read_file(str(path))
                        if content is not None:
                            return content
                    except Exception:
                        pass  # Silently fall back to basic file reading

                    # Fallback to basic file reading
                    try:
                        with open(path, "r", encoding="utf-8") as file:
                            return file.read()
                    except FileNotFoundError:
                        return f"File not found: {path}"
                    except Exception as e:
                        return f"Error reading file {path}: {str(e)}"
            else:
                # Handle iterable of files
                results = {}
                for file_path in path:
                    file_path = Path(file_path)
                    try:
                        content = reader.read_file(str(file_path))
                        if content is not None:
                            results[str(file_path)] = content
                            continue
                    except Exception:
                        pass  # Silently fall back to basic file reading

                    # Fallback to basic file reading
                    try:
                        with open(file_path, "r", encoding="utf-8") as file:
                            results[str(file_path)] = file.read()
                    except Exception as e:
                        results[str(file_path)] = f"Error reading file: {str(e)}"
                return results

        except Exception as e:
            print(f"Critical error in owl_read: {str(e)}")
            return f"Critical error: {str(e)}"

    def owl_search(self, query: str, max_results: int = 10, max_retries: int = 3) -> list:
        """
        Execute web search using DuckDuckGo's API.

        Parameters
        ----------
        query : str
            Search phrase to look up
        max_results : int, default=10
            Maximum number of results to return (1-20)
        max_retries : int, default=3
            Number of retry attempts for failed requests

        Returns
        -------
        list
            List of search result dictionaries containing:
            - title: Result title
            - href: URL
            - body: Content snippet

        Raises
        ------
        RuntimeError
            After exhausting all retry attempts without success

        Notes
        -----
        - Implements exponential backoff with jitter between retries
        - Results are limited to text-based web content
        """
        errors = []
        for attempt in range(max_retries):
            try:
                print(f"Searching for query: {query} (attempt {attempt + 1}/{max_retries})")

                from duckduckgo_search import DDGS

                with DDGS() as ddgs:
                    # Use a generator to avoid loading all results at once
                    results = []
                    for result in ddgs.text(query, max_results=max_results):
                        results.append(result)
                        if len(results) >= max_results:
                            break

                if not results:
                    print(f"No results found for query: {query}")
                    return []

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
        - Object types shown in parentheses after names
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
        Write text content to filesystem.

        Parameters
        ----------
        file_path : str
            Absolute path for output file
        content : str
            Text content to write

        Notes
        -----
        - Overwrites existing files without warning
        - Uses UTF-8 encoding
        - Limited to text-based formats
        """
        try:
            with open(file_path, "w") as file:
                file.write(content)
            print(f"Content successfully written to {file_path}")
        except Exception as e:
            print(f"Error writing to file: {e}")

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

    def owl_load_namespace(self, file_path: str):
        """
        Load namespace using dill.

        Parameters
        ----------
        file_path : str
            The path to the file to load the namespace from.
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
        except Exception as e:
            print(f"An error occurred while loading: {e}")

    def owl_scrape(
        self,
        urls: List[str],
        max_concurrent: int = 5,
    ) -> Dict[str, str]:
        """
        Scrape web content from URLs (use instead of owl_read for web resources).

        Parameters
        ----------
        urls : List[str]
            VALID HTTP/HTTPS URLS TO PROCESS
            Does not support local file paths
        max_concurrent : int, default=5
            Simultaneous requests allowed

        Returns
        -------
        dict
            Dictionary mapping URLs to their extracted content in markdown format

        Notes
        -----
        - Respects robots.txt and website rate limits
        - Extracts main article content when possible
        """
        from owlsight.app.url_processor import fetch_and_parse_urls

        content_dict = asyncio.run(fetch_and_parse_urls(urls, max_concurrent))
        content_dict = {url: content.strip() for url, content in content_dict.items() if content.strip()}
        return content_dict


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
        """
        from huggingface_hub import scan_cache_dir, HfApi
        from huggingface_hub.constants import HF_HUB_CACHE

        output_lines = []
        cache_dir: Path = Path(cache_dir or HF_HUB_CACHE)
        if not cache_dir.exists():
            return f"Cache directory '{cache_dir}' does not exist."

        try:
            hf_api = HfApi()
            cache_info = scan_cache_dir(cache_dir)
            if not cache_info.repos:
                return f"No models found in the Hugging Face cache directory {cache_dir}"

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
            return f"Error accessing Hugging Face cache: {str(e)}"

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
            return False

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
