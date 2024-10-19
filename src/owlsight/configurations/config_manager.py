from typing import Any, Dict, List
import json
import os

from owlsight.utils.logger_manager import LoggerManager
from owlsight.utils.constants import DEFAULTS, CHOICES

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
        self._config = DottedDict(DEFAULTS)  # Use DEFAULTS from constants.py

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
        Set a configuration value using dotted notation for nested keys.
        """
        keys = key.split(".")
        d = self._config
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}  # Create the nested dictionary if it doesn't exist
            d = d[k]  # Move deeper into the nested dictionary
        d[keys[-1]] = value  # Set the final key's value

    @property
    def config_choices(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the configuration choices for the UI.

        If possible_values is None, the key can only be selected, similar to pushing a button which might trigger a predefined action, based on the key.
        If possible_values is a list, the key can be toggled between the values in the list.
        If possible_values is a string, the user is free to enter any string.
        """
        config_choices = {
                "main": {
                    "back": None,
                    "max_retries_on_error": _prepare_toggle_choices(
                        self._config["main"]["max_retries_on_error"], CHOICES["main"]["max_retries_on_error"]
                    ),
                    "prompt_code_execution": _prepare_toggle_choices(
                        self._config["main"]["prompt_code_execution"], CHOICES["main"]["prompt_code_execution"]
                    ),
                    "extra_index_url": self._config["main"]["extra_index_url"],
                },
                "model": {
                    "back": None,
                    "model_id": self._config["model"]["model_id"],
                    "save_history": _prepare_toggle_choices(
                        self._config["model"]["save_history"], CHOICES["model"]["save_history"]
                    ),
                    "system_prompt": self._config["model"]["system_prompt"],
                    "transformers__device": _prepare_toggle_choices(
                        self._config["model"]["transformers__device"],
                        CHOICES["model"]["transformers__device"],
                    ),
                    "transformers__quantization_bits": _prepare_toggle_choices(
                        self._config["model"]["transformers__quantization_bits"],
                        CHOICES["model"]["transformers__quantization_bits"],
                    ),
                    "gguf__filename": self._config["model"]["gguf__filename"],
                    "gguf__verbose": _prepare_toggle_choices(
                        self._config["model"]["gguf__verbose"], CHOICES["model"]["gguf__verbose"]
                    ),
                    "gguf__n_ctx": _prepare_toggle_choices(
                        self._config["model"]["gguf__n_ctx"],
                        CHOICES["model"]["gguf__n_ctx"],
                    ),
                    "gguf__n_gpu_layers": _prepare_toggle_choices(
                        self._config["model"]["gguf__n_gpu_layers"], CHOICES["model"]["gguf__n_gpu_layers"]),
                    "gguf__n_batch": _prepare_toggle_choices(
                        self._config["model"]["gguf__n_batch"], CHOICES["model"]["gguf__n_batch"]
                    ),
                    "gguf__n_cpu_threads": _prepare_toggle_choices(
                        self._config["model"]["gguf__n_cpu_threads"], CHOICES["model"]["gguf__n_cpu_threads"]
                    ),
                    "onnx__tokenizer": self._config["model"]["onnx__tokenizer"],
                    "onnx__verbose": _prepare_toggle_choices(
                        self._config["model"]["onnx__verbose"], CHOICES["model"]["onnx__verbose"]
                    ),
                    "onnx__num_threads": self._config["model"]["onnx__num_threads"],
                },
                "generate": {
                    "back": None,
                    "stopwords": str(self._config["generate"]["stopwords"]),
                    "max_new_tokens": _prepare_toggle_choices(
                        self._config["generate"]["max_new_tokens"],
                        CHOICES["generate"]["max_new_tokens"],
                    ),
                    "temperature": _prepare_toggle_choices(
                        self._config["generate"]["temperature"],
                        CHOICES["generate"]["temperature"],
                    ),
                    "generation_kwargs": str(self._config["generate"]["generation_kwargs"]),
                },
                "rag": {
                    "back": None,
                    "active": _prepare_toggle_choices(
                        self._config["rag"]["active"], CHOICES["rag"]["active"]
                    ),
                    "target_library": self._config["rag"]["target_library"],
                    "top_k": _prepare_toggle_choices(
                        self._config["rag"]["top_k"],
                        CHOICES["rag"]["top_k"],
                    ),
                    "search_query": self._config["rag"]["search_query"],
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
                logger.info(f"Configuration saved successfully to '{path}'")
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

        if not path.endswith(".json"):
            logger.error(f"{err_msg} Configuration file must be a JSON file.")
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
            logger.info(f"Configuration loaded successfully from '{path}'")
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
