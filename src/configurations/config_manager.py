from typing import Any, Dict, List
import json
import os

import numpy as np

from utils.logger_manager import LoggerManager

logger = LoggerManager.get_logger(__name__)


class ConfigManager:
    """
    A singleton class which carries the configuration for the whole application.
    
    Most important to know, is that there are 2 different configurations:
    - self._config: the true configuration that is used in the application backend.
    - config_choices: the configuration that presented in the UI, where the user can toggle between choices.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance.config = {}
        return cls._instance


    def __init__(self):
        """
        Initialize the configuration manager with default values.
        """
        self._config = DottedDict(
            {
                "main": {
                    "max_retries_on_error": 3,
                    "prompt_code_execution": True,
                    "extra_index_url": ""
                },
                "model": {
                    "model_id": "",
                    "save_history": False,
                    "system_prompt": """
# ROLE:
You are an advanced problem-solving AI with expert-level knowledge in various programming languages, particularly Python.

# TASK:
- Prioritize Python solutions when appropriate.
- Present code in markdown format.
- Clearly state when non-Python solutions are necessary.
- Break down complex problems into manageable steps and think through the solution step-by-step.
- Adhere to best coding practices, including error handling and consideration of edge cases.
- Acknowledge any limitations in your solutions.
- Always aim to provide the best solution to the user's problem, whether it involves Python or not.
                    """.strip(),
                    # specific parameters for the different processors
                    # transformers
                    "transformers__device": None,
                    "transformers__quantization_bits": None,
                    "transformers__gguf_file": "",
                    # onnx
                    "onnx__tokenizer": "",
                    "onnx__verbose": False,
                    "onnx__num_threads": 1,
                },
                "generate": {
                    "stopwords": [],
                    "max_new_tokens": 1024,
                    "temperature": 0.0,
                    "generation_kwargs": {},
                },
            }
        )

    def get(self, key: str, default=None) -> Any:
        """
        Get a configuration value using dotted notation for nested keys.
        """
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dotted notation for nested keys and notify observers.
        """
        keys = key.split(".")
        d = self._config
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}  # Create the nested dictionary if it doesn't exist
            d = d[k]  # Move deeper into the nested dictionary
        d[keys[-1]] = value  # Set the final key's value

    @property
    def config_choices(self) -> Dict[str, Dict[str, str]]:
        """
        Get the configuration choices for the UI.

        If value is None, the key can only be selected, similar to pushing a button which might trigger a predefined action, based on the key.
        If value is a list, the key can be toggled between the values in the list.
        If value is a string, the user is free to enter any string.
        """
        config_choices = {
            "main": {
                "max_retries_on_error": _prepare_toggle_choices(
                    self._config["main"]["max_retries_on_error"], list(range(0, 10))
                ),
                "prompt_code_execution": _prepare_toggle_choices(
                    self._config["main"]["prompt_code_execution"], [False, True]
                ),
                "extra_index_url": self._config["main"]["extra_index_url"],
            },
            "model": {
                "model_id": self._config["model"]["model_id"],
                "save_history": _prepare_toggle_choices(
                    self._config["model"]["save_history"], [False, True]
                ),
                "system_prompt": self._config["model"]["system_prompt"],
                # transformers
                "transformers__device": _prepare_toggle_choices(
                    self._config["model"]["transformers__device"], [None, "cpu", "cuda"]
                ),
                "transformers__quantization_bits": _prepare_toggle_choices(
                    self._config["model"]["transformers__quantization_bits"], [None, 8, 4]
                ),
                "transformers__gguf_file": self._config["model"]["transformers__gguf_file"],
                # onnx
                "onnx__tokenizer": self._config["model"]["onnx__tokenizer"],
                "onnx__verbose": _prepare_toggle_choices(
                    self._config["model"]["onnx__verbose"], [False, True]
                ),
                "onnx__num_threads": self._config["model"]["onnx__num_threads"],
            },
            "generate": {
                "stopwords": str(self._config["generate"]["stopwords"]),
                "max_new_tokens": _prepare_toggle_choices(
                    self._config["generate"]["max_new_tokens"],
                    [32 * (2**i) for i in range(15)],
                ),
                "temperature": _prepare_toggle_choices(
                    self._config["generate"]["temperature"],
                    np.round(np.arange(0.0, 1.05, 0.05), 2).tolist(),
                ),
                "generation_kwargs": str(self._config["generate"]["generation_kwargs"]),
            },
        }

        return config_choices

    def save(self, path: str) -> None:
        """
        Save the configuration to a file as JSON.
        """
        err_msg = "Cannot save config."
        if not isinstance(path, str) or not path:
            logger.error(f"{err_msg} Invalid file path provided.")
            return

        # Ensure that the directory exists
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            logger.error(f"{err_msg} Directory does not exist: '{directory}'")
            return

        try:
            with open(path, "w") as f:
                json.dump(
                    self._config,
                    f,
                    indent=4,
                )
                logger.info(f"{err_msg} Configuration saved succesfully to '{path}'")
        except (IOError, OSError) as e:
            logger.error(f"{err_msg} Error writing to file '{path}': {e}")
        except TypeError as e:
            logger.error(f"{err_msg} Error serializing configuration to JSON: {e}")

    def load(self, path: str):
        """
        Load the configuration from a file as JSON.
        """
        err_msg = "Cannot load config."
        if not isinstance(path, str) or not path:
            logger.error("Invalid file path provided.")
            return

        if not os.path.exists(path):
            logger.error(f"{err_msg} Configuration file does not exist: '{path}'")
            return

        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (IOError, OSError) as e:
            logger.error(f"{err_msg} Error reading from file '{path}': {e}")
            return
        except json.JSONDecodeError as e:
            logger.error(f"{err_msg} Error parsing JSON in file '{path}': {e}")
            return

        try:
            self._config = DottedDict(data)
            logger.info(f"Configuration loaded succesfully from '{path}'")
        except Exception as e:
            logger.error(f"{err_msg} Error initializing configuration: {e}")

    def __repr__(self):
        return repr(self._config)


class DottedDict(dict):
    """A dictionary with dotted access to attributes, enforcing lowercase keys."""

    def __getattr__(self, attr):
        attr = attr.lower()
        value = self.get(attr)
        if isinstance(value, dict):
            return DottedDict(value)  # Recursively return DottedDict for nested dicts
        return value

    def __setattr__(self, attr, value):
        self[attr.lower()] = value

    def __delattr__(self, attr):
        del self[attr.lower()]


def _prepare_toggle_choices(current_val: Any, possible_vals: List[Any]) -> List[Any]:
    """
    Prepare the config_choices to be used in the UI for toggling between choices.

    Parameters
    ----------
    current_val : Any
        The current value. Can be seen as default value.
    possible_vals : List[Any]
        The possible values for the configuration parameter.
        Allow user to toggle between the values.
    """
    if current_val in possible_vals:
        index = possible_vals.index(current_val)
        possible_vals = possible_vals[index:] + possible_vals[:index]
    return possible_vals
