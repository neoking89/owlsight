import pytest
from typing import List, Optional, Dict, Any, Union

from owlsight.processors.base import TextGenerationProcessor
from owlsight.processors.text_generation_manager import TextGenerationManager
from owlsight.configurations.config_manager import ConfigManager
from owlsight.utils.custom_classes import GlobalPythonVarsDict
from owlsight.app.default_functions import OwlDefaultFunctions


class MockTextGenerationProcessor(TextGenerationProcessor):
    def __init__(
        self,
        model_id: str,
        apply_chat_history: bool = False,
        mock_responses: Union[str, List[str]] = "Default mock response",
    ):
        super().__init__(model_id, apply_chat_history, system_prompt=None)
        self.mock_responses = [mock_responses] if isinstance(mock_responses, str) else mock_responses
        self.response_index = 0

    def generate(
        self,
        input_data: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        stop_words: Optional[List[str]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        response = self.mock_responses[self.response_index % len(self.mock_responses)]
        self.response_index += 1

        if self.apply_chat_history:
            self.chat_history.append((input_data, response))
        return response


@pytest.fixture
def config_manager():
    return ConfigManager()


@pytest.fixture
def text_generation_manager(config_manager):
    # Reset singleton state before creating new instance
    TextGenerationManager._reset_instance()
    manager = TextGenerationManager(config_manager=config_manager)
    yield manager
    # Reset singleton state after test is done
    TextGenerationManager._reset_instance()


@pytest.fixture
def media_model_mappings():
    return {
        "image-to-text": "Salesforce/blip-image-captioning-base",
        "visual-question-answering": "dandelin/vilt-b32-finetuned-vqa",
        "automatic-speech-recognition": "facebook/wav2vec2-base-960h",
        "document-question-answering": "microsoft/layoutlm-base-uncased",
    }


@pytest.fixture
def owl_instance():
    globals_dict = GlobalPythonVarsDict()
    return OwlDefaultFunctions(globals_dict)
