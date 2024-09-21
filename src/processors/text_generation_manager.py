from typing import Type, Any, Dict

from src.processors.text_generation_processor import TextGenerationProcessor
from src.configurations.config_manager import ConfigManager
from src.utils.helper_functions import convert_to_real_type

from src.utils.logger_manager import LoggerManager

logger = LoggerManager.get_logger(__name__)


class TextGenerationManager:
    def __init__(
        self, processor: Type[TextGenerationProcessor], config_manager: ConfigManager
    ):
        """
        Manage the lifecycle of a TextGenerationProcessor and its interaction with the configuration.

        Parameters
        ----------
        processor : TextGenerationProcessor
            An instance of the processor (either Transformers or Onnx).
        config_manager : ConfigManager
            Configuration dictionary to manage settings for the processor.
        """
        self.processor = processor(**config_manager.get("model", {}))
        self.config_manager = config_manager

    def generate(self, input_text: str):
        """
        Generate text using the processor.
        """
        generated_text = self.processor.generate(
            input_text, **self.config_manager.get("generate", {})
        )
        return generated_text

    def update_config(self, key: str, value: Any):
        """
        Update the configuration dynamically. If 'model_id' is updated, reload the processor.
        """
        # if not isinstance(value, str):
        #     raise ValueError(f"Value must be a string. Got {type(value)} instead.")

        # Convert the value to its real type if possible
        value = convert_to_real_type(value)
        self.config_manager.set(key, value)

        # If 'model_id' is updated, reload the processor
        if key == "model_id":
            logger.info(f"Model ID updated to {value}. Reloading the processor.")
            self.reload_processor()

    def reload_processor(self):
        """
        Reload the processor with a new 'model_id' while keeping other configurations.
        """
        model_id = self.config_manager.get("model.model_id")
        logger.info(f"Reloading processor with new model_id: {model_id}")

        # Save the history from the old processor
        old_history = self.processor.history

        # Inmediately overwrite the processor with a new instance to save memory
        self.processor = self.processor.__class__(**self.config_manager.get("model", {}))
        self.processor.history = old_history

        logger.info(f"Processor reloaded with model_id: {model_id}")

    def get_processor(self) -> TextGenerationProcessor:
        """
        Return the current processor instance.
        """
        return self.processor

    def get_config(self) -> dict:
        """
        Return the current configuration as dictionary.
        """
        return self.config_manager._config

    def get_config_choices(self, convert_keys_to_string=False) -> dict:
        """
        Return the available configuration choices.

        Parameters
        ----------
        convert_keys_to_string : bool, optional
            Convert keys to string if True, by default False
            All empty lists, empty dicts, are converted to an empty string.
        """
        if convert_keys_to_string:
            return convert_to_string(self.config_manager.config_choices)

        return self.config_manager.config_choices


def convert_to_string(d: Dict[str, Any]) -> Dict[str, str]:
    """
    Recursively converts all elements of a dictionary to strings.
    If an element is an empty collection (list, string, or dictionary), it converts it to an empty string.

    Parameters
    ----------
    d : dict
        Input dictionary with mixed types.

    Returns
    -------
    dict
        Dictionary with all values converted to strings and empty collections as "".
    """

    def handle_value(value: Any) -> str:
        if isinstance(value, dict):
            return convert_to_string(value)  # Recurse for nested dictionaries
        if isinstance(value, (list, dict, str)) and not value:
            return ""  # Convert empty collections to an empty string
        return str(value)  # Convert everything else to string

    return {key: handle_value(value) for key, value in d.items()}
