from typing import Any, Dict, List, Tuple
import json
import os
import traceback
from copy import deepcopy

from owlsight.configurations.constants import CONFIG_DEFAULTS, CONFIG_CHOICES, CONFIG_TYPES
from owlsight.utils.constants import get_default_config_on_startup_path
from owlsight.utils.helper_functions import flatten_dict
from owlsight.utils.validations import validate_key_is_nested_one_layer
from owlsight.ui.custom_classes import OptionType
from owlsight.utils.logger import logger


class ConfigManager:
    """
    A singleton class which carries the configuration for the whole application.

    Most important to know, is that there are 2 different configurations:
    - self._config: the true configuration that is used in the application backend.
    - config_choices: the configuration that presented in the UI, where the user can toggle between choices.

    Also:
    For options to become available in the UI, they need to be defined in one of the create...choices() methods.
    """

    EXCLUDED_KEYS = {"main.default_config_on_startup"}

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
        self._defaults = deepcopy(CONFIG_DEFAULTS)
        self._choices = deepcopy(CONFIG_CHOICES)
        self._config = DottedDict(self._defaults)

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

    def _get_toggle_choice(self, section: str, key: str) -> Any:
        """Helper method to prepare toggle choices for a given section and key."""
        return _prepare_toggle_choices(self._config[section][key], self._choices[section][key])

    def _get_basic_choice(self, section: str, key: str) -> Any:
        """Helper method to return a basic configuration choice."""
        return self._config[section][key]

    def _create_main_choices(self) -> Dict[str, Any]:
        """Create the config choices for the 'main' section."""
        return {
            "back": None,
            "max_retries_on_error": self._get_toggle_choice("main", "max_retries_on_error"),
            "prompt_code_execution": self._get_toggle_choice("main", "prompt_code_execution"),
            "prompt_retry_on_error": self._get_toggle_choice("main", "prompt_retry_on_error"),
            "track_model_usage": self._get_toggle_choice("main", "track_model_usage"),
            "extra_index_url": self._get_basic_choice("main", "extra_index_url"),
            "python_compile_mode": self._get_toggle_choice("main", "python_compile_mode"),
            "dynamic_system_prompt": self._get_toggle_choice("main", "dynamic_system_prompt"),
            "default_config_on_startup": get_default_config_on_startup_path(return_cache_path=False),
        }

    def _create_model_choices(self) -> Dict[str, Any]:
        """Create the config choices for the 'model' section."""
        return {
            "back": None,
            "model_id": self._get_basic_choice("model", "model_id"),
            "apply_chat_history": self._get_toggle_choice("model", "apply_chat_history"),
            "system_prompt": self._get_basic_choice("model", "system_prompt"),
            "model_kwargs": str(self._get_basic_choice("model", "model_kwargs")),
            "transformers__device": self._get_toggle_choice("model", "transformers__device"),
            "transformers__quantization_bits": self._get_toggle_choice("model", "transformers__quantization_bits"),
            "transformers__stream": self._get_toggle_choice("model", "transformers__stream"),
            "gguf__filename": self._get_basic_choice("model", "gguf__filename"),
            "gguf__verbose": self._get_toggle_choice("model", "gguf__verbose"),
            "gguf__n_ctx": self._get_toggle_choice("model", "gguf__n_ctx"),
            "gguf__n_gpu_layers": self._get_toggle_choice("model", "gguf__n_gpu_layers"),
            "gguf__n_batch": self._get_toggle_choice("model", "gguf__n_batch"),
            "gguf__n_cpu_threads": self._get_toggle_choice("model", "gguf__n_cpu_threads"),
            "onnx__model_dir": self._get_basic_choice("model", "onnx__model_dir"),
            "onnx__verbose": self._get_toggle_choice("model", "onnx__verbose"),
            "onnx__n_cpu_threads": self._get_toggle_choice("model", "onnx__n_cpu_threads"),
        }

    def _create_generate_choices(self) -> Dict[str, Any]:
        """Create the config choices for the 'generate' section."""
        return {
            "back": None,
            "stop_words": str(self._get_basic_choice("generate", "stop_words")),
            "max_new_tokens": self._get_toggle_choice("generate", "max_new_tokens"),
            "temperature": self._get_toggle_choice("generate", "temperature"),
            "generation_kwargs": str(self._get_basic_choice("generate", "generation_kwargs")),
        }

    def _create_rag_choices(self) -> Dict[str, Any]:
        """Create the config choices for the 'rag' section."""
        return {
            "back": None,
            "active": self._get_toggle_choice("rag", "active"),
            "target_library": self._get_basic_choice("rag", "target_library"),
            "top_k": self._get_toggle_choice("rag", "top_k"),
            "sentence_transformer_weight": self._get_toggle_choice("rag", "sentence_transformer_weight"),
            "sentence_transformer_name_or_path": self._get_basic_choice("rag", "sentence_transformer_name_or_path"),
            "search": self._get_basic_choice("rag", "search"),
        }

    def create_agentic_choices(self):
        return {
            "back": None,
            "active": self._get_toggle_choice("agentic", "active"),
            "additional_information": self._get_basic_choice("agentic", "additional_information"),
            "show_available_tools": None,
            "exclude_tools": str(self._get_basic_choice("agentic", "exclude_tools")),
            "config_per_agent": str(self._get_basic_choice("agentic", "config_per_agent")),
        }

    def _create_huggingface_choices(self):
        return {
            "back": None,
            "search": self._get_basic_choice("huggingface", "search"),
            "top_k": self._get_toggle_choice("huggingface", "top_k"),
            "task": self._get_toggle_choice("huggingface", "task"),
            "select_model": self._get_toggle_choice("huggingface", "select_model"),
        }

    @property
    def config_choices(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the configuration choices for the UI.
        """
        return {
            "main": self._create_main_choices(),
            "model": self._create_model_choices(),
            "generate": self._create_generate_choices(),
            "rag": self._create_rag_choices(),
            "agentic": self.create_agentic_choices(),
            "huggingface": self._create_huggingface_choices(),
        }

    def save(self, path: str) -> bool:
        """
        Save the configuration to a file as JSON.

        Parameters
        ----------
        path : str
            The path to save the configuration file.

        Returns
        -------
        bool
            True if the configuration was saved successfully, False otherwise.
        """
        err_msg = "Cannot save config."
        if not isinstance(path, str) or not path:
            logger.error(f"{err_msg} Invalid file path provided.")
            return False

        if not path.endswith(".json"):
            logger.error(f"{err_msg} Configuration file must be a valid JSON file, ending with '.json'.")
            return False

        # Ensure that the directory exists
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            logger.error(f"{err_msg} Directory does not exist: '{directory}'")
            return False

        # filter out options that are None
        filtered_config = _remove_keys_from_config(self._config, CONFIG_TYPES)

        try:
            with open(path, "w") as f:
                json.dump(
                    filtered_config,
                    f,
                    indent=4,
                )
                logger.info(f"Configuration saved successfully to '{path}'")
                return True
        except (IOError, OSError) as e:
            logger.error(f"{err_msg} Error writing to file '{path}': {e}")
            return False
        except TypeError as e:
            logger.error(f"{err_msg} Error serializing configuration to JSON: {e}")
            return False

    def load(self, path: str) -> bool:
        """
        Load the configuration from a file as JSON.
        It gets stored in the _config attribute.

        Parameters
        ----------
        path : str
            The path to the configuration file.

        Returns
        -------
        bool
            True if the configuration was loaded successfully, False otherwise.
        """
        err_msg = "Cannot load config."
        if not isinstance(path, str) or not path:
            logger.error(f"{err_msg} Invalid file path provided: {path}")
            return False

        if not os.path.exists(path):
            logger.error(f"{err_msg} Configuration file does not exist: '{path}'")
            return False

        if not path.endswith(".json"):
            logger.error(f"{err_msg} Configuration file must be a JSON file.")
            return False

        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (IOError, OSError) as e:
            logger.error(f"{err_msg} Error reading from file '{path}': {e}")
            return False
        except json.JSONDecodeError as e:
            logger.error(f"{err_msg} Error parsing JSON in file '{path}': {e}")
            return False

        try:
            config = DottedDict(data)
            self.validate_config(config)
            self._config = config
            logger.info(f"Configuration loaded successfully from '{path}'")
            return True
        except Exception:
            logger.error(f"{err_msg} Error loading configuration from '{path}': {traceback.format_exc()}")
            return False

    def validate_config(self, config: dict):
        """
        Validate the configuration.
        """
        flattened_defaults = flatten_dict(self._defaults)
        flattened_config = flatten_dict(config)

        flattened_defaults = self._remove_excluded_keys(flattened_defaults)
        flattened_config = self._remove_excluded_keys(flattened_config)

        # check differences in sections:
        missing_sections = set(self._defaults.keys()) - set(config.keys())
        if missing_sections:
            raise KeyError(f"Config misses the following sections: {missing_sections}")

        invalid_sections = set(config.keys()) - set(self._defaults.keys())
        if invalid_sections:
            raise KeyError(f"Config has the following sections, which are not valid in owlsight: {invalid_sections}")

        # check differences in keys:
        missing_keys = set(flattened_defaults.keys()) - set(flattened_config.keys())
        if missing_keys:
            raise KeyError(f"Config misses the following keys: {missing_keys}")

        invalid_keys = set(flattened_config.keys()) - set(flattened_defaults.keys())
        invalid_keys = {
            key for key in invalid_keys if validate_key_is_nested_one_layer(key)
        }  # only consider keys with nested keys
        if invalid_keys:
            raise KeyError(f"Config has the following keys, which are not valid in owlsight: {invalid_keys}")

        # check if values are valid
        flattened_choices = flatten_dict(self._choices)
        for key, value in flattened_config.items():
            if not validate_key_is_nested_one_layer(key):
                logger.warning(f"Key '{key}' is not nested one layer deep. Skipping validation.")
                continue
            choices = flattened_choices[key]
            if isinstance(choices, list) and choices != []:
                # if the value is a list, it means that the value is a togglechoice. we check if the value is in the list
                if value not in choices:
                    raise ValueError(f"Invalid value {value} for key '{key}'. Possible values: {choices}")
            else:
                # if the value is not a list, it means that the value is a basic choice. we check if the type is correct
                if not isinstance(value, type(choices)):
                    raise TypeError(
                        f"Invalid type {type(value).__name__} for key '{key}'. Expected type: {type(choices).__name__}"
                    )

    def __repr__(self):
        return repr(self._config)

    def _remove_excluded_keys(self, config: dict) -> dict:
        return {key: value for key, value in config.items() if key not in self.EXCLUDED_KEYS}


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


def _dynamicly_initialize_toggle_choices(current_val: List[str], possible_vals: List[Any]) -> Tuple[str, List[Any]]:
    """
    In some cases, the possible values are not provided yet.
    This is if the possible values depend on some action which needs to be undertaken first, before the possible values can be determined.
    In this case, we dynamically initialize the possible values based on the current value.
    """
    first_item = current_val[0] if current_val else ""
    possible_vals = current_val
    current_val: str = first_item

    return current_val, possible_vals


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
    # If the possible values are not provided, we assume that the current value is a list of possible values
    if not possible_vals and isinstance(current_val, list):
        current_val, possible_vals = _dynamicly_initialize_toggle_choices(current_val, possible_vals)

    if current_val in possible_vals:
        index = possible_vals.index(current_val)
        possible_vals = possible_vals[index:] + possible_vals[:index]

    return possible_vals


def _remove_keys_from_config(config: dict, config_types: dict) -> dict:
    """
    Remove keys from the config dictionary if their type in config_types is OptionType.ACTION.
    Only keeps keys that are NOT in EXCLUDED_KEYS and are not of ACTION Optiontype.
    """
    filtered_config = {}
    for outer_key in config_types.keys():
        filtered_config[outer_key] = {}
        # Only process keys that exist in both config and config_types
        for inner_key, inner_value in config.get(outer_key, {}).items():
            # Only keep the key if it's defined in config_types and is not an ACTION type
            if (
                inner_key in config_types.get(outer_key, {})
                and config_types[outer_key][inner_key] != OptionType.ACTION
                and f"{outer_key}.{inner_key}" not in ConfigManager.EXCLUDED_KEYS
            ):
                filtered_config[outer_key][inner_key] = inner_value
    return filtered_config
