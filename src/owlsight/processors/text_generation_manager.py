import ast
import os
import pkgutil
import traceback
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from owlsight.app.default_functions import EXCLUDE_TOOLS, OwlDefaultFunctions
from owlsight.agentic.constants import AGENT_INFORMATION
from owlsight.configurations.config_manager import ConfigManager
from owlsight.configurations.constants import CONFIG_DEFAULTS
from owlsight.huggingface.constants import HUGGINGFACE_MEDIA_TASKS
from owlsight.huggingface.core import show_and_return_model_data
from owlsight.processors.base import TextGenerationProcessor
from owlsight.processors.helper_functions import select_processor_type, warn_processor_not_loaded
from owlsight.processors.multimodal_processors import MultiModalProcessor
from owlsight.rag.constants import SENTENCETRANSFORMER_DEFAULT_MODEL
from owlsight.rag.python_lib_search import PythonLibSearcher
from owlsight.ui.console import get_user_choice
from owlsight.ui.custom_classes import AppDTO
from owlsight.utils.constants import get_default_config_on_startup_path, get_pickle_cache
from owlsight.utils.custom_classes import GlobalPythonVarsDict
from owlsight.utils.deep_learning import free_cuda_memory, track_measure_usage
from owlsight.utils.helper_functions import (
    convert_to_real_type,
    function_call_to_python_code,
    parse_function_call,
    parse_python_placeholders,
)
from owlsight.utils.logger import logger


class TextGenerationManager:
    _instance = None

    def __new__(cls, config_manager: ConfigManager):
        if cls._instance is not None:
            raise RuntimeError("Only one instance of TextGenerationManager can exist at the same time.")
        cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def _reset_instance(cls):
        """Reset the singleton instance. This should only be used in tests."""
        cls._instance = None

    def __init__(self, config_manager: ConfigManager):
        """
        Manage the lifecycle of a TextGenerationProcessor and its interaction with the configuration during runtime of the CLI app.
        This is a singleton class - only one instance can exist at a time.

        Parameters
        ----------
        config_manager : ConfigManager
            Configuration dictionary to manage settings for the processor.

        Raises
        ------
        RuntimeError
            If attempting to create a second instance of TextGenerationManager
        """
        # Skip initialization if this instance has already been initialized
        if hasattr(self, "config_manager"):
            return

        self.config_manager = config_manager
        self.processor: Optional[TextGenerationProcessor] = None

        self._original_generate_method = None
        self._last_loaded_config: Optional[str] = None
        self._tool_history: set[str] = set()
        self._init_excluded_tools = EXCLUDE_TOOLS.copy()

    @property
    def tool_history(self) -> List[Dict[str, str]]:
        tool_history = []
        for key in self._tool_history:
            try:
                tool_history.append(ast.literal_eval(key))
            except Exception as e:
                logger.error(f"Failed to parse tool history item: {key}")
                logger.error(e)
        return tool_history

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
            if not isinstance(self.processor, MultiModalProcessor):
                logger.error("Processor is not a MultiModalProcessor, but media objects were provided.")
                logger.error(
                    f"Please select a model that supports multimodal generation through one of the following tasks: {HUGGINGFACE_MEDIA_TASKS}"
                )
                return generated_text

            generated_text = self.processor.generate(input_data, media_objects=media_objects, **kwargs)
        else:
            generated_text = self.processor.generate(input_data, **kwargs)

        # if media objects or task is a media task, try to parse the generated text
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
        else:
            # else try to parse the generated text if it is a function call. if no function call, return the generated text as is
            func_name, arguments = parse_function_call(generated_text)
            # if a function call is found and arguments are valid, try to process it
            if func_name is not None and arguments is not None:
                if not self.process_tool_call(func_name, arguments):
                    error_message_for_model = f"Error: You tried to call tool '{func_name}' with arguments '{arguments}'. This tool was already used with these arguments. Do NOT suggest the following tools: {self.tool_history}. Please try again."
                    logger.warning("Correcting Tool Call, as model tried to call a tool it already used")
                    generated_text = self.processor.generate(error_message_for_model, **kwargs)
                    # Parse the new response and verify it's valid before continuing
                    new_func_name, new_arguments = parse_function_call(generated_text)
                    if new_func_name is not None and new_arguments is not None:
                        func_name, arguments = new_func_name, new_arguments
                    else:
                        return "```python\n# Error: Invalid function call after duplicate tool usage\n```"
                self._update_tool_history(func_name, arguments)
                generated_text = function_call_to_python_code(func_name, arguments)

        return generated_text

    def update_config(self, key: str, value: Any):
        """
        Update the configuration dynamically. If 'model_id' is updated, reload the processor.
        This function contains all logic to update the configuration.
        """
        # Skip "back" keys and preprocess value
        if key.endswith(".back"):
            return
        value = self._parse_python_placeholders(value)

        # Update configuration store
        try:
            value = convert_to_real_type(value)
            outer_key, inner_key = key.split(".", 1)

            # handle special cases which need to rollback if config update fails
            if outer_key == "agentic" and inner_key == "config_per_agent":
                self._update_agentic_config_per_agent(inner_key, value)
                return

            self.config_manager.set(key, value)
            logger.debug(f"Configuration updated: {key} = {value}")
        except Exception:
            logger.error(f"Error updating configuration for key '{key}': {traceback.format_exc()}")
            return

        # Handle specific configuration sections
        if outer_key == "main":
            self._update_main_config(inner_key, value)
        elif outer_key == "model":
            self._update_model_config(inner_key, value)
        elif outer_key == "rag":
            self._update_rag_config(inner_key)
        elif outer_key == "agentic":
            self._update_agentic_config(inner_key, value)
        elif outer_key == "huggingface":
            self._update_huggingface_config(inner_key)

    def _update_tool_history(self, func_name: str, arguments: Dict[str, str]) -> None:
        key = str({"name": func_name, "arguments": arguments})
        self._tool_history.add(key)

    def _update_main_config(self, inner_key: str, value: Any):
        """Handle updates to main configuration."""
        if inner_key == "default_config_on_startup":
            if value:
                if not value.endswith(".json"):
                    raise ValueError("Default config file must be a JSON file.")
                if not os.path.exists(value):
                    raise FileNotFoundError(f"Default config file '{value}' not found.")
            with open(get_default_config_on_startup_path(return_cache_path=True), "w") as f:
                f.write(value)

    def _update_model_config(self, inner_key: str, value: Any):
        """Handle updates to model-related configuration."""
        if inner_key == "model_id":
            self.load_model_processor(reload=self.processor is not None)
            return

        if self.processor is None:
            warn_processor_not_loaded()
            return

        elif hasattr(self.processor, inner_key):
            setattr(self.processor, inner_key, value)
            logger.debug(f"Processor updated: {inner_key} = {value}")
        else:
            logger.warning(f"'{inner_key}' not found in self.processor, meaning it was not updated.")
            logger.warning("It is possible that this value is only set during initialization of self.processor.")
            logger.warning("Consider loading the model from a config file to update this value.")

    def _update_rag_config(self, inner_key: str):
        """Handle updates to RAG-related configuration."""
        rag_is_active = self.config_manager.get("rag.active", False)
        if not rag_is_active:
            return

        library = self.config_manager.get("rag.target_library", "")
        if not library:
            logger.error("No library provided. Please set 'target_library' in the configuration.")
            return

        # Get all libs without the _ prefix and in sorted order
        available_libraries = [module.name for module in pkgutil.iter_modules() if not module.name.startswith("_")]
        if library not in available_libraries:
            logger.error(f"Library '{library}' not found in the current Python session.")
            logger.error(f"available libraries: {sorted(available_libraries)}")
            return

        if inner_key == "search":
            self._perform_rag_search(library)

    def _perform_rag_search(self, library: str):
        """Perform RAG search with current configuration settings."""
        search = self.config_manager.get("rag.search", "")
        if not search:
            logger.error("No prompt provided. Please provide a prompt in the 'search' field.")
            return

        top_k = self.config_manager.get("rag.top_k", CONFIG_DEFAULTS["rag"]["top_k"])
        sentence_transformer_weight = self.config_manager.get("rag.sentence_transformer_weight", 0.0)
        sentence_transformer_name_or_path = self.config_manager.get(
            "rag.sentence_transformer_name_or_path", SENTENCETRANSFORMER_DEFAULT_MODEL
        )

        if sentence_transformer_weight > 0.0:
            if not sentence_transformer_name_or_path:
                logger.error(
                    "No sentence transformer provided. Please provide a valid name or path to a sentence transformer in the 'sentence_transformer_name_or_path' field."
                )
                return
            logger.warning(
                "Using sentence transformer for semantic search. Creating embeddings for the library can take some time!"
            )

        tfidf_weight = 1 - sentence_transformer_weight
        logger.info(
            f"Using weights for search: TFIDF weight = {tfidf_weight:.2f}, Sentence Transformer weight = {sentence_transformer_weight:.2f}."
        )

        searcher = PythonLibSearcher()
        context = searcher.search(
            library,
            search,
            top_k,
            cache_dir=get_pickle_cache(),
            tfidf_weight=tfidf_weight,
            sentence_transformer_weight=sentence_transformer_weight,
            sentence_transformer_model=sentence_transformer_name_or_path,
        )
        print(f"Context for library '{library}' with top_k={top_k}:\n{context}")

    def _update_agentic_config(self, inner_key: str, value: Any):
        if inner_key == "active":
            if self.processor is None:
                warn_processor_not_loaded()
                return
            self.processor.apply_tools = self._update_agentic_active(value)
        elif inner_key == "show_available_tools":
            sep = "#" * 50
            available_tools = f"\n{sep}\n".join(
                str(obj) for obj in OwlDefaultFunctions(GlobalPythonVarsDict()).owl_tools(as_json=True)
            )
            print(f"Available tools:\n{sep}\n{available_tools}")
        elif inner_key == "exclude_tools":
            available_tool_names = [
                getattr(obj, "__name__")
                for obj in OwlDefaultFunctions(GlobalPythonVarsDict()).owl_tools(as_json=False)
                if hasattr(obj, "__name__")
            ]

            # Reset EXCLUDE_TOOLS to initial state before updating
            EXCLUDE_TOOLS.clear()
            EXCLUDE_TOOLS.extend(self._init_excluded_tools)

            # Add new tools to exclude
            for tool in value:
                if tool not in available_tool_names:
                    logger.error(f"Tool '{tool}' not found in available tools. Skipping.")
                    continue
                if tool not in EXCLUDE_TOOLS:
                    logger.info(f"Adding tool to exclude list: {tool}")
                    EXCLUDE_TOOLS.append(tool)
        elif inner_key == "config_per_agent":
            self._update_agentic_config_per_agent(inner_key, value)

    def _update_agentic_config_per_agent(self, inner_key: str, value: Any) -> None:
        """
        Update agentic.config_per_agent
        """
        config_key = f"agentic.{inner_key}"

        # Parse the incoming value into a dictionary
        if isinstance(value, dict):
            config_per_agent = value
        elif isinstance(value, str):
            config_per_agent = self._parse_config_string(value)
            if config_per_agent is None:  # parsing failed
                return
        else:
            logger.error(
                "agentic.%s must be a dict or str, got %s. Skipping.",
                inner_key,
                type(value).__name__,
            )
            return

        # Type-check result
        if not isinstance(config_per_agent, dict):
            logger.error("agentic.%s must resolve to a dictionary. Skipping.", inner_key)
            return

        # Validate agent names
        invalid_agents = [a for a in config_per_agent if a not in AGENT_INFORMATION]
        if invalid_agents:
            logger.error(
                "agentic.%s: invalid agent names: '%s'. Valid agent names: %s. Skipping.",
                inner_key,
                ", ".join(invalid_agents),
                ", ".join(AGENT_INFORMATION.keys()),
            )
            return

        # Validate (and normalise) paths
        missing_for_agents: list[str] = []
        for agent, cfg_path in config_per_agent.items():
            if not cfg_path:  # empty string means "no config for this agent"
                continue
            if not isinstance(cfg_path, str):
                logger.error(
                    "agentic.%s: expected a string path for agent %s, got %s. Skipping.",
                    inner_key,
                    agent,
                    type(cfg_path).__name__,
                )
                return
            
            if not cfg_path.endswith(".json"):
                logger.error(
                    "agentic.%s: expected a JSON file path for agent %s, got: '%s'. Skipping.",
                    inner_key,
                    agent,
                    cfg_path,
                )
                return
            
            path = Path(cfg_path).expanduser()
            resolved = path if path.is_absolute() else Path.cwd() / path
            if not resolved.exists():
                missing_for_agents.append(agent)

        if missing_for_agents:
            logger.error(
                "agentic.%s: config file not found for agents: %s. Skipping.",
                inner_key,
                ", ".join(missing_for_agents),
            )
            return

        self.config_manager.set(config_key, config_per_agent)
        logger.debug(f"Configuration updated: {config_key} = {config_per_agent}")
        
    def _update_huggingface_config(self, inner_key: str):
        """Handle updates to Hugging Face-related configuration."""
        if inner_key == "search":
            self._perform_huggingface_search()
        elif inner_key == "select_model":
            self._handle_model_selection()
        elif inner_key == "task":
            task = self.config_manager.get("huggingface.task", CONFIG_DEFAULTS["huggingface"]["task"])
            self.config_manager.set("huggingface.task", task)

    def _perform_huggingface_search(self):
        """Perform Hugging Face model search with current configuration settings."""
        model_search = self.config_manager.get("huggingface.search", CONFIG_DEFAULTS["huggingface"]["search"])
        top_k = self.config_manager.get("huggingface.top_k", CONFIG_DEFAULTS["huggingface"]["top_k"])
        task = self.config_manager.get("huggingface.task", CONFIG_DEFAULTS["huggingface"]["task"])
        model_dict = show_and_return_model_data(model_search, top_n_models=top_k, task=task)

        if not model_dict:
            logger.error("No models found. Please try a different search query.")
            return

        self.config_manager.set("huggingface.select_model", list(model_dict.keys()))

    def _handle_model_selection(self):
        """Handle model selection and loading process through Hugging Face"""
        select_model = self.config_manager.get(
            "huggingface.select_model", CONFIG_DEFAULTS["huggingface"]["select_model"]
        )

        if not select_model:
            logger.error("No model provided. Please set a model in the configuration.")
            return

        if not isinstance(select_model, str):
            logger.error("Model must be a string. Please set a model in the configuration.")
            return

        # Select and load model from huggingface
        self.config_manager.set("model.model_id", select_model)
        exc = self.load_model_processor(reload=self.processor is not None)

        if exc and select_model.lower().endswith("gguf"):
            self._handle_gguf_model_selection(exc)

    def _handle_gguf_model_selection(self, exc: Exception):
        """Handle GGUF model selection process."""
        try:
            gguf_list = str(exc).split("Available Files:")[1].strip()
            if not gguf_list:
                logger.warning("No gguf-list could be inferred")
                return

            gguf_list = [file for file in ast.literal_eval(gguf_list) if file.endswith("gguf")]
            gguf_menu = {
                "back": None,
                "Choose a GGUF model": gguf_list,
            }
            gguf_filename = get_user_choice(gguf_menu, app_dto=AppDTO(return_value_only=True))

            if gguf_filename:
                self.config_manager.set("model.gguf__filename", gguf_filename)
                self.load_model_processor(reload=self.processor is not None)
        except Exception:
            logger.warning("Error processing GGUF model selection")

    def save_config(self, path: str):
        """
        Save the configuration to a file.
        """
        # set all the values to legimitate default values before saving the config
        self.config_manager.set("huggingface.select_model", "")
        self.config_manager.save(path)

    def load_config(self, path: str, execute_sequence_on_loading: bool = True) -> bool:
        """
        Load the configuration from a file.

        Parameters
        ----------
        path : str
            Path to the configuration file to load.
        execute_sequence_on_loading : bool, optional
            If True, execute the sequence of actions defined in the configuration file after loading.
            Defaults to True.

        Returns
        -------
        bool
            True if the configuration was loaded successfully, False otherwise.
        """
        self._last_loaded_config = path
        config_successfully_loaded = self.config_manager.load(path)
        if config_successfully_loaded:
            self.load_model_processor(reload=self.processor is not None)
            if execute_sequence_on_loading:
                self._execute_sequence_on_loading()

        return config_successfully_loaded

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
        agentic_active = self.config_manager.get("agentic.active", False)
        processor_kwargs["apply_tools"] = self._update_agentic_active(agentic_active)

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
                free_cuda_memory()

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

    def process_tool_call(self, tool_name: str, arguments: Dict) -> bool:
        """Returns True if tool can be used, False if duplicate"""
        key = {"name": tool_name, "arguments": arguments}
        key_str = str(key)
        if key_str in self._tool_history:
            return False
        self._update_tool_history(tool_name, arguments)
        return True

    def _execute_sequence_on_loading(self):
        """
        Execute the keystrokes from sequence_on_loading if it is not an empty list in the configuration.
        """
        sequence = self.config_manager.get("main.sequence_on_loading", None)
        if sequence and isinstance(sequence, list):
            try:
                OwlDefaultFunctions({}).owl_press(sequence, exit_python_before_sequence=False)
            except Exception as e:
                logger.error(f"Error executing main.sequence_on_loading: {e}")

    def _parse_python_placeholders(self, value: Any):
        """
        Parse python placeholders in the value.
        """
        try:
            return parse_python_placeholders(value, GlobalPythonVarsDict())
        except Exception:
            return value

    def _update_agentic_active(self, value: bool) -> Optional[Dict[str, Any]]:
        if value:
            global_vars_dict = GlobalPythonVarsDict()
            active = OwlDefaultFunctions(global_vars_dict).owl_tools(as_json=True)
            return active
        else:
            return None

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

    @staticmethod
    def _parse_config_string(raw: str) -> Union[dict[str, Any], None]:
        """
        Convert raw string into a dict.

        1. First attempt: strict ``json.loads`` (expects double quotes).
        2. Second attempt: ``ast.literal_eval`` after doubling backslashes.
           This handles single-quoted dict literals and Windows paths.

        Returns
        -------
        Union[dict[str, Any], None]
            Parsed mapping, or *None* if parsing failed.
        """
        # Strict JSON first
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Fallback: safe Python literal
        safe_raw = raw.replace("\\", "\\\\")  # prevent \Uxxxx issues
        try:
            return ast.literal_eval(safe_raw)
        except (SyntaxError, ValueError):
            logger.error(
                "Invalid format for config_per_agent string. Expected valid JSON or Python dict literal. Skipping."
            )
            return None
