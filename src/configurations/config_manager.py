from typing import Any, Dict, List


class ConfigManager:
    def __init__(self):
        """
        Configuration that notifies observers when a value is changed.
        """

        self._config = DottedDict(
            {
                "main": {
                    "max_retries_on_error": 3,
                    "prompt_code_execution": True,
                },
                "model": {
                    "model_id": "",
                    "tokenizer": "",
                    "save_history": False,
                    "system_prompt": "",
                    # specific parameters for the different processors
                    # transformers
                    "device": None,
                    "quantization_bits": None,
                    # onnx
                    "verbose": False,
                    "num_threads": 1,
                },
                "generate": {
                    "stopwords": [],
                    "max_new_tokens": 1024,
                    "temperature": 0.0,
                    "generation_kwargs": {},
                },
            }
        )
        # self._observers: List = []

    def get(self, key: str, default=None):
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

    def set(self, key: str, value: Any):
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
        # self._notify_observers(key, value)

    # def subscribe(self, observer):
    #     """
    #     Register an observer to be notified when config changes.
    #     """
    #     self._observers.append(observer)

    # def unsubscribe(self, observer):
    #     """
    #     Unregister an observer.
    #     """
    #     self._observers.remove(observer)

    # def _notify_observers(self, key: str, value: Any):
    #     """
    #     Notify all registered observers of the config change.
    #     """
    #     for observer in self._observers:
    #         observer.update_config(key, value)

    @property
    def config_choices(self) -> Dict[str, Dict[str, str]]:
        return {
            "main": {
                "max_retries_on_error": self._config["main"]["max_retries_on_error"],
                "prompt_code_execution": get_list_bools(
                    self._config["main"]["prompt_code_execution"]
                ),
            },
            "model": {
                "model_id": "",
                "tokenizer": "",
                "save_history": get_list_bools(self._config["model"]["save_history"]),
                "system_prompt": "",
                # specific parameters for the different processors
                # transformers
                "device": [None, "cpu", "cuda"],
                "quantization_bits": [None, 8, 4],
                # onnx
                "verbose": get_list_bools(self._config["model"]["verbose"]),
                "num_threads": self._config["model"]["num_threads"],
            },
            "generate": {
                "stopwords": [],
                "max_new_tokens": self._config["generate"]["max_new_tokens"],
                "temperature": self._config["generate"]["temperature"],
                "generation_kwargs": {},
            },
        }

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


def get_list_bools(b: bool):
    return [b, not b]
