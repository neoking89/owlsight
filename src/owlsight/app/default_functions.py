import importlib.util
import os
import builtins
import inspect
import traceback
import re
from typing import Optional, List, Dict
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import json
import dill

import requests
from bs4 import BeautifulSoup
from huggingface_hub import scan_cache_dir, CachedRepoInfo, HfApi
from huggingface_hub.constants import HF_HUB_CACHE

from owlsight.utils.custom_classes import SingletonDict


class OwlDefaultFunctions:
    """
    Define default functions that can be used in the Python interpreter.
    This provides the user with some utility functions to interact with the interpreter.
    Convention is that the functions start with 'owl_' to avoid conflicts with built-in functions.

    This class is open for extension, as possibly more useful functions can be added in the future.
    """

    def __init__(self, globals_dict: SingletonDict):
        # Add check to make sure every function starts with 'owl_'
        self.globals_dict = globals_dict
        self._check_method_naming_convention()

    def _check_method_naming_convention(self):
        """Check if all methods in the class start with 'owl_'."""
        methods = inspect.getmembers(self, predicate=inspect.ismethod)
        methods = [method for method in methods if not method[0].startswith("_")]
        for name, _ in methods:
            if not name.startswith("owl_"):
                raise ValueError(f"Method '{name}' does not follow the 'owl_' naming convention!")

    def owl_read(self, file_path: str) -> str:
        """
        Read the content of a text file.
        """
        try:
            with open(file_path, "r") as file:
                return file.read()
        except FileNotFoundError:
            return f"File not found: {file_path}"

    def owl_import(self, file_path: str):
        """
        Import a Python file and load its contents into the current namespace.

        Parameters
        ----------
        file_path : str
            The path to the Python file to import.
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

    def owl_show(self, docs: bool = True):
        """
        Show all currently active imported objects in the namespace except builtins.

        Parameters:
        -----------
        docs (bool): If True, also display the docstring of each object.
        """
        current_globals = self.globals_dict
        active_objects = {name: obj for name, obj in current_globals.items() if name not in dir(builtins)}

        brackets = "#" * 50
        print("Active imported objects:")
        print(brackets)
        for name, obj in active_objects.items():
            if not name.startswith("_"):
                obj_type = type(obj).__name__
                print(f"{name} ({obj_type})")

                # Optionally display the docstring if available
                if docs:
                    docstring = obj.__doc__
                    if docstring:
                        print(f"Doc: {docstring.strip()}")
                    else:
                        print("Doc: No documentation available")
                print(brackets)

    def owl_write(self, file_path: str, content: str):
        """
        Write content to a text file.
        """
        try:
            with open(file_path, "w") as file:
                file.write(content)
            print(f"Content successfully written to {file_path}")
        except Exception as e:
            print(f"Error writing to file: {e}")

    def owl_save_namespace(self, file_path: str):
        """
        Save the current python namespace using dill.
        NOTE: This will only save the variables that do not start with '_' or 'owl_'.
        Also, some complex objects (like from external libraries) may not be serializable.

        Parameters
        ----------
        file_path : str
            The path to the file to save the namespace to.
            the .dill extension will be automaticly added if not present.
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
        url_or_terms: str,
        trim_newlines: Optional[int] = 2,
        filter_by: Optional[Dict[str, str]] = None,
        **request_kwargs,
    ) -> str:
        """
        Scrape the text content of a webpage and return specific content based on the filter.

        Parameters
        ----------
        url_or_terms : str
            The URL of the webpage to scrape OR the search term to search Bing for.
        trim_newlines : int, optional
            The maximum number of consecutive newlines to allow in the output, default is 2.
        filter_by : dict, optional
            Dictionary specifying HTML tag and/or attributes to filter specific content.
            For example: {'tag': 'div', 'class': 'content'}
        **request_kwargs
            Additional keyword arguments to pass to the requests.get function.

        Returns
        -------
        str
            The filtered text content of the webpage.
        """
        if is_url(url_or_terms):
            url = url_or_terms
        else:
            urls = search_bing(url_or_terms, exclude_from_url=["microsoft"], **request_kwargs)
            if not urls:
                return ""
            url = urls[0]

        response = requests.get(url, **request_kwargs)
        html_content = response.text

        soup = BeautifulSoup(html_content, "html.parser")

        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        # Filter specific content if filter_by is provided
        if filter_by:
            tag = filter_by.get("tag", None)
            attrs = {key: value for key, value in filter_by.items() if key != "tag"}
            filtered_elements = soup.find_all(tag, attrs=attrs)

            # Join the filtered elements' text content
            filtered_text = "\n".join(element.get_text() for element in filtered_elements)
        else:
            filtered_text = soup.get_text()

        # Optionally trim consecutive newlines
        if trim_newlines:
            pattern = r"\n{" + str(trim_newlines + 1) + r",}"
            replacement = "\n" * trim_newlines
            return re.sub(pattern, replacement, filtered_text)

        return filtered_text

    def owl_models(self, cache_dir: Optional[str] = None, show_task: bool = False) -> str:
        """
        Returns a string with information about all Hugging Face models currently loaded in the cache directory.
        Print the output from this function to the console to get a nice overview.

        Parameters:
        -----------
        cache_dir (str, optional):
            The directory path to scan for models. If None, the default cache directory is used.
        show_task (bool, optional):
            If True, also display the tasks associated with each model.
            If used, showing models will take a while longer.

        Returns:
        --------
        str:
            A string containing information about all cached models
        """
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
                    output_lines.append(f"Size: {repo.size_on_disk / (1024*1024):.2f} MB")
                    output_lines.append(f"Last Modified: {last_modified}")
                    output_lines.append(f"Location: {repo.repo_path}")
                    model_id = self._get_model_id(repo)
                    output_lines.append(f"Eligable for model_id: {model_id}")
                    output_lines.append("-" * 50)
                except Exception as e:
                    output_lines.append(f"Error accessing model with id {repo.repo_id}: {str(e)}")

            output_lines.append(f"\nTotal Cache Size: {cache_info.size_on_disk / (1024*1024):.2f} MB")
            output_lines.append(f"Cache Directory: {cache_dir}")

            return "\n".join(output_lines)
        except Exception as e:
            return f"Error accessing Hugging Face cache: {str(e)}"

    def owl_press(
        self,
        sequence: List[str],
        exit_python_from_interpreter: bool = True,
        time_before_sequence: float = 0.5,
        time_between_keys: float = 0.12,
    ) -> bool:
        """
        Simulate typing a sequence of keys and automaticly control the menu inside the Owlsight application.

        The parameters passed to this function, are passed to another Python process that simulates the keystrokes.
        This is done to avoid blocking the interpreter while the sequence is being typed.

        Parameters
        ----------
        sequence : List[str]
            The sequence of keys to type. Case-sensitive when typing available keys.
            Available keys: 'L' (left), 'R' (right), 'U' (up), 'D' (down), 'ENTER' (ENTER), 'SLEEP:[float]' (sleep for time seconds).
            Any other character will be typed as is.
        exit_python_from_interpreter : bool, optional
            If True, type 'exit()' and press ENTER before typing the sequence, default is True.
            Assuming owl_press is called from the interpreter, this will return to the mainmenu before typing the sequence.
        time_before_sequence : float, optional
            The time to wait before executing the keysequence, default is 0.5 seconds.
        time_between_keys : float, optional
            The time to wait between typing each key, default is 0.12 seconds.

        Returns
        -------
        bool
            True if the subprocess was started successfully, False otherwise.
        """
        if not isinstance(sequence, list):
            raise TypeError("sequence must be a list")
        if not all(isinstance(item, str) for item in sequence):
            raise TypeError("sequence must contain only strings")

        if exit_python_from_interpreter:
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


def search_bing(term: str, exclude_from_url: Optional[List] = None, **request_kwargs) -> List[str]:
    """Search Bing for a term and return a list of URLs."""
    term = "+".join(term.split(" "))
    url = f"https://www.bing.com/search?q={term}"
    response = requests.get(url, **request_kwargs)
    soup = BeautifulSoup(response.text, "html.parser")
    urls = [a["href"] for a in soup.find_all("a", href=True) if a["href"].startswith("http")]
    if exclude_from_url:
        urls = [url for url in urls if not any(exclude in url for exclude in exclude_from_url)]
    return urls


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
