from typing import Any, Optional, Dict, Union
import traceback
import pkgutil
import ast

from owlsight.configurations.constants import CONFIG_DEFAULTS
from owlsight.processors.base import TextGenerationProcessor, MultiModalTextGenerationProcessor
from owlsight.processors.helper_functions import (
    select_processor_type, warn_processor_not_loaded
)
from owlsight.ui.console import get_user_choice
from owlsight.configurations.config_manager import ConfigManager
from owlsight.rag.python_lib_search import search_python_libs
from owlsight.hugging_face.core import show_and_return_model_data
from owlsight.hugging_face.constants import HUGGINGFACE_MEDIA_TASKS
from owlsight.utils.helper_functions import convert_to_real_type
from owlsight.utils.deep_learning import free_memory, track_measure_usage
from owlsight.utils.constants import get_pickle_cache
from owlsight.utils.logger import logger


class TextGenerationManager:
    def __init__(self, config_manager: ConfigManager):
        """
        Manage the lifecycle of a TextGenerationProcessor and its interaction with the configuration during runtime of the CLI app.

        Parameters
        ----------
        config_manager : ConfigManager
            Configuration dictionary to manage settings for the processor.
        """
        self.config_manager = config_manager
        self.processor: Optional[TextGenerationProcessor] = None
        self._original_generate_method = None

    def _wrap_with_usage_tracking(self):
        """Wrap the processor's generate method with the track_measure_usage decorator."""
        if not self._original_generate_method:
            # Save the original method
            self._original_generate_method = self.processor.generate

        if not getattr(self.processor.generate, "_is_tracked", False):
            # Wrap the method if not already wrapped
            self.processor.generate = track_measure_usage(self._original_generate_method, polling_time=0.5)(
                self._original_generate_method
            )
            self.processor.generate._is_tracked = True

    def _restore_original_method(self):
        """Restore the processor's original generate method if it was modified."""
        if self._original_generate_method:
            self.processor.generate = self._original_generate_method

    def generate(self, input_data: str, media_objects: Optional[Dict[str, dict]] = None) -> str:
        """
        Generate text using the processor.
        """
        generated_text = ""
        task = self.config_manager.get("huggingface.task")
        kwargs = self.config_manager.get("generate", {})

        track_model_usage = self.config_manager.get("main.track_model_usage", False)
        if track_model_usage:
            logger.info("Tracking memory usage during generation.")
            self._wrap_with_usage_tracking()
        else:
            self._restore_original_method()

        if media_objects or task in HUGGINGFACE_MEDIA_TASKS:
            if not isinstance(self.processor, MultiModalTextGenerationProcessor):
                logger.error("Processor is not a MultiModalTextGenerationProcessor, but media objects were provided.")
                logger.error(
                    f"Please select a model that supports multimodal generation through one of the following tasks: {HUGGINGFACE_MEDIA_TASKS}"
                )
                return generated_text

            generated_text = self.processor.generate(input_data, media_objects=media_objects, **kwargs)
        else:
            generated_text = self.processor.generate(input_data, **kwargs)

        if task in HUGGINGFACE_MEDIA_TASKS:
            try:
                result = ast.literal_eval(generated_text)
            except Exception:
                logger.error(f"Error evaluating generated text: {traceback.format_exc()}")
            if not result:
                logger.warning(f"No text generated for media task '{task}'.")
                logger.warning("Use double-square brackets '[[]]' syntax to pass media objects to the model.")
                for mediatype in ["image", "audio", "video"]:
                    logger.warning(f"For example: '[[{mediatype}:path/to/{mediatype}]]'")
                logger.warning("Or for QA: 'What color is the car? [[image:path/to/image.jpg]]'")

        return generated_text

    def update_config(self, key: str, value: Any):
        """
        Update the configuration dynamically. If 'model_id' is updated, reload the processor.
        """
        if key.endswith(".back"):
            return  # Do not set the "back" key
        try:
            value = convert_to_real_type(value)
            self.config_manager.set(key, value)
            logger.info(f"Configuration updated: {key} = {value}")
        except Exception:
            logger.error(f"Error updating configuration for key '{key}': {traceback.format_exc()}")
            return

        outer_key, inner_key = key.split(".", 1)
        if outer_key == "model":
            if inner_key == "model_id":
                self.load_model_processor(reload=self.processor is not None)
            else:
                if self.processor is None:
                    warn_processor_not_loaded()
                    return
                if hasattr(self.processor, inner_key):
                    setattr(self.processor, inner_key, value)
                    logger.info(f"Processor updated: {inner_key} = {value}")
                else:
                    logger.warning(f"'{inner_key}' not found in self.processor, meaning it was not updated.")
                    logger.warning(
                        "It is possible that this value is only set during initialization of self.processor."
                    )
                    logger.warning("Consider loading the model from a config file to update this value.")
        elif outer_key == "rag":
            rag_is_active = self.config_manager.get("rag.active", False)
            if rag_is_active:
                library = self.config_manager.get("rag.target_library", "")
                if not library:
                    logger.error("No library provided. Please set 'target_library' in the configuration.")
                    return

                # get all libs without the _ prefix and in sorted order
                available_libraries = [
                    module.name for module in pkgutil.iter_modules() if not module.name.startswith("_")
                ]
                if library not in available_libraries:
                    logger.error(f"Library '{library}' not found in the current Python session.")
                    logger.error(f"available libraries: {sorted(available_libraries)}")
                    return
                elif inner_key == "search":
                    search = self.config_manager.get("rag.search", "")
                    if not search:
                        logger.error("No prompt provided. Please provide a prompt in the 'search' field.")
                        return
                    top_k = self.config_manager.get("rag.top_k", CONFIG_DEFAULTS[outer_key]["top_k"])
                    context = search_python_libs(library, search, top_k, cache_dir=get_pickle_cache())
                    print(f"Context for library '{library}' with top_k={top_k}:\n{context}")
        elif outer_key == "huggingface":
            if inner_key == "search":
                # search models from huggingface
                model_search = self.config_manager.get("huggingface.search", CONFIG_DEFAULTS["huggingface"]["search"])
                top_k = self.config_manager.get("huggingface.top_k", CONFIG_DEFAULTS["huggingface"]["top_k"])
                task = self.config_manager.get("huggingface.task", CONFIG_DEFAULTS["huggingface"]["task"])
                model_dict = show_and_return_model_data(model_search, top_n_models=top_k, task=task)
                if not model_dict:
                    logger.error("No models found. Please try a different search query.")
                    return
                # set list of models from model_dict to select_model
                self.config_manager.set("huggingface.select_model", list(model_dict.keys()))
            elif inner_key == "select_model":
                select_model = self.config_manager.get(
                    "huggingface.select_model", CONFIG_DEFAULTS["huggingface"]["select_model"]
                )
                if not select_model:
                    logger.error("No model provided. Please set a model in the configuration.")
                    return
                if not isinstance(select_model, str):
                    logger.error("Model must be a string. Please set a model in the configuration.")
                    return
                # select and load a model from huggingface
                self.config_manager.set("model.model_id", select_model)
                task = self.config_manager.get("huggingface.task", CONFIG_DEFAULTS["huggingface"]["task"])
                exc = self.load_model_processor(reload=self.processor is not None)
                if exc and select_model.lower().endswith("gguf"):
                    gguf_list = str(exc).split("Available Files:")[1].strip()
                    if gguf_list:
                        gguf_list = [file for file in ast.literal_eval(gguf_list) if file.endswith("gguf")]
                        gguf_menu = {
                            "back": None,
                            "Choose a GGUF model": gguf_list,
                        }
                        gguf__filename = get_user_choice(gguf_menu)
                        if gguf__filename:
                            self.config_manager.set("model.gguf__filename", gguf__filename)
                            self.load_model_processor(reload=self.processor is not None)
                    else:
                        logger.warning("No gguf-list could be inferred")
            elif inner_key == "task":
                task = self.config_manager.get("huggingface.task", CONFIG_DEFAULTS["huggingface"]["task"])
                self.config_manager.set("huggingface.task", task)

    def save_config(self, path: str):
        """
        Save the configuration to a file.
        """
        self.config_manager.save(path)

    def load_config(self, path: str):
        """
        Load the configuration from a file.
        """
        loading_succesful = self.config_manager.load(path)
        if loading_succesful:
            self.load_model_processor(reload=self.processor is not None)

    def load_model_processor(self, reload=False) -> Union[None, Exception]:
        """
        Load the model processor with a 'model_id', to load the correct model and tokenizer.

        Parameters
        ----------
        reload : bool, optional
            If True, reload the processor with the same model_id, by default False.
            Assumes that the processor is already initialized with another model.

        Returns
        -------
        Union[None, Exception]
            None if successful, otherwise an exception is returned.
        """
        model_kwargs = self.config_manager.get("model", {})
        task = self.config_manager.get("huggingface.task", CONFIG_DEFAULTS["huggingface"]["task"])
        processor_kwargs = {"task": task, **model_kwargs}

        model_id = self.config_manager.get("model.model_id", "")
        if not model_id:
            logger.error("No model_id provided. Please set a model_id in the configuration.")
            return

        logger.info(f"Loading processor with new model_id: {model_id}")
        processor_type = select_processor_type(model_id, task=task)

        try:
            if reload:
                if self.processor is None:
                    raise ValueError("Processor is not initialized yet. Cannot reload.")
                # Save the history from the old processor
                old_chat_history = self.processor.chat_history

                # Inmediately overwrite the processor with a new instance to save memory
                self.processor = None
                free_memory()

                self.processor = processor_type(**processor_kwargs)
                self.processor.chat_history = old_chat_history
            else:
                self.processor = processor_type(**processor_kwargs)
        except Exception as e:
            logger.error(f"Error loading model_processor: {traceback.format_exc()}")
            return e

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

    def get_config_choices(self) -> dict:
        """
        Return the available configuration choices.

        Returns
        -------
        dict
            Dictionary with the available configuration choices.
        """
        return self.config_manager.config_choices

    def get_config_key(self, key: str, default: Any = None) -> Any:
        """
        Get the value of a key in the configuration.
        """
        return self.config_manager.get(key, default)
