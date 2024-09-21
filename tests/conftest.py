import pytest
from typing import List, Optional, Dict, Any, Union

from src.processors.text_generation_processor import TextGenerationProcessor
from src.processors.text_generation_manager import TextGenerationManager
from src.configurations.config_manager import ConfigManager


class MockTextGenerationProcessor(TextGenerationProcessor):
    def __init__(
        self,
        model_id: str,
        save_history: bool = False,
        mock_responses: Union[str, List[str]] = "Default mock response",
    ):
        super().__init__(model_id, save_history, system_prompt=None)
        self.mock_responses = (
            [mock_responses] if isinstance(mock_responses, str) else mock_responses
        )
        self.response_index = 0

    def generate(
        self,
        input_text: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        stopwords: Optional[List[str]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        response = self.mock_responses[self.response_index % len(self.mock_responses)]
        self.response_index += 1

        if self.save_history:
            self.history.append((input_text, response))
        return response


@pytest.fixture
def mock_text_generator(request):
    mock_responses = getattr(request, "param", "Default mock response")
    if isinstance(mock_responses, list) and len(mock_responses) == 1:
        mock_responses = mock_responses[0]
    return MockTextGenerationProcessor(
        "mock-model", save_history=True, mock_responses=mock_responses
    )


@pytest.fixture
def config_manager():
    return ConfigManager()


@pytest.fixture
def text_generation_manager(config_manager):
    return TextGenerationManager(config_manager=config_manager)
