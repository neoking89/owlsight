import pytest
from typing import Any, Dict, List, Optional

from owlsight.processors.base import TextGenerationProcessor

class MockOpenAICompatibleTextGenerationProcessor(TextGenerationProcessor):
    """
    Mock implementation of TextGenerationProcessor that simulates OpenAI compatible behavior.
    Specifically for testing generate_openai_comp method.
    """
    def __init__(self, model_id: str = "mock_model", apply_chat_history: bool = True, system_prompt: str = "Default system prompt"):
        super().__init__(model_id, apply_chat_history, system_prompt)
        self.last_generate_input_data: Optional[str] = None
        self.last_generate_system_prompt_used: Optional[str] = None
        self.last_generate_chat_history_used: Optional[List[Dict[str, str]]] = None
        self.last_generate_kwargs_passed: Optional[Dict[str, Any]] = None
        self.generate_return_value: str = "mocked_response"

    def generate(
        self,
        input_data: str,
        max_new_tokens: int,
        temperature: float,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        self.last_generate_input_data = input_data
        self.last_generate_system_prompt_used = self.system_prompt
        self.last_generate_chat_history_used = list(self.chat_history) # Capture a copy
        
        # Consolidate all keyword arguments passed to generate
        all_passed_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "generation_kwargs": generation_kwargs,
            **kwargs
        }
        self.last_generate_kwargs_passed = all_passed_kwargs
        
        return self.generate_return_value

    def get_max_context_length(self) -> int:
        return 4096 # Mock value

# --- Test Data ---
EXAMPLE_MESSAGES_SIMPLE = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather like in Paris?"}
]

EXAMPLE_MESSAGES_NO_SYSTEM_IN_LIST = [
    {"role": "user", "content": "Tell me a joke."}
]

EXAMPLE_MESSAGES_MULTI_TURN = [
    {"role": "system", "content": "You are a coding assistant."},
    {"role": "user", "content": "How do I write a Python function to reverse a string?"},
    {"role": "assistant", "content": "You can use slicing:\n\ndef reverse_string(s):\n    return s[::-1]"},
    {"role": "user", "content": "Can you explain how slicing works?"}
]

DEFAULT_PROCESSOR_SYSTEM_PROMPT = "Initial Mock System Prompt"
DEFAULT_PROCESSOR_CHAT_HISTORY = [{"role": "user", "content": "Original history item"}]

# --- Pytest Fixture ---
@pytest.fixture
def mock_openai_compatible_processor() -> MockOpenAICompatibleTextGenerationProcessor:
    processor = MockOpenAICompatibleTextGenerationProcessor(
        system_prompt=DEFAULT_PROCESSOR_SYSTEM_PROMPT
    )
    processor.chat_history = list(DEFAULT_PROCESSOR_CHAT_HISTORY) # Set initial state
    return processor

# --- Test Cases ---
def test_simple_conversation(mock_openai_compatible_processor: MockOpenAICompatibleTextGenerationProcessor):
    messages = list(EXAMPLE_MESSAGES_SIMPLE) # Use a copy
    gen_kwargs = {"max_new_tokens": 50, "temperature": 0.7, "extra_param": "test"}
    
    response = mock_openai_compatible_processor.generate_openai_comp(messages, **gen_kwargs)

    assert response == mock_openai_compatible_processor.generate_return_value
    assert mock_openai_compatible_processor.last_generate_system_prompt_used == "You are a helpful assistant."
    assert mock_openai_compatible_processor.last_generate_input_data == "What's the weather like in Paris?"
    assert mock_openai_compatible_processor.last_generate_chat_history_used == []
    assert mock_openai_compatible_processor.last_generate_kwargs_passed is not None
    assert mock_openai_compatible_processor.last_generate_kwargs_passed.get("max_new_tokens") == 50
    assert mock_openai_compatible_processor.last_generate_kwargs_passed.get("temperature") == 0.7
    assert mock_openai_compatible_processor.last_generate_kwargs_passed.get("extra_param") == "test"

    # Verify restoration of processor's state
    assert mock_openai_compatible_processor.system_prompt == DEFAULT_PROCESSOR_SYSTEM_PROMPT
    assert mock_openai_compatible_processor.chat_history == DEFAULT_PROCESSOR_CHAT_HISTORY

def test_no_system_prompt_in_messages(mock_openai_compatible_processor: MockOpenAICompatibleTextGenerationProcessor):
    messages = list(EXAMPLE_MESSAGES_NO_SYSTEM_IN_LIST)
    gen_kwargs = {"max_new_tokens": 60, "temperature": 0.1}

    response = mock_openai_compatible_processor.generate_openai_comp(messages, **gen_kwargs)

    assert response == mock_openai_compatible_processor.generate_return_value
    assert mock_openai_compatible_processor.last_generate_system_prompt_used == DEFAULT_PROCESSOR_SYSTEM_PROMPT
    assert mock_openai_compatible_processor.last_generate_input_data == "Tell me a joke."
    assert mock_openai_compatible_processor.last_generate_chat_history_used == []
    assert mock_openai_compatible_processor.last_generate_kwargs_passed is not None
    assert mock_openai_compatible_processor.last_generate_kwargs_passed.get("max_new_tokens") == 60

    assert mock_openai_compatible_processor.system_prompt == DEFAULT_PROCESSOR_SYSTEM_PROMPT
    assert mock_openai_compatible_processor.chat_history == DEFAULT_PROCESSOR_CHAT_HISTORY

def test_multi_turn_conversation(mock_openai_compatible_processor: MockOpenAICompatibleTextGenerationProcessor):
    messages = list(EXAMPLE_MESSAGES_MULTI_TURN)
    gen_kwargs = {"max_new_tokens": 100, "temperature": 0.5}

    response = mock_openai_compatible_processor.generate_openai_comp(messages, **gen_kwargs)

    assert response == mock_openai_compatible_processor.generate_return_value
    assert mock_openai_compatible_processor.last_generate_system_prompt_used == "You are a coding assistant."
    assert mock_openai_compatible_processor.last_generate_input_data == "Can you explain how slicing works?"
    expected_history_for_generate = [
        {"role": "user", "content": "How do I write a Python function to reverse a string?"},
        {"role": "assistant", "content": "You can use slicing:\n\ndef reverse_string(s):\n    return s[::-1]"}
    ]
    assert mock_openai_compatible_processor.last_generate_chat_history_used == expected_history_for_generate
    assert mock_openai_compatible_processor.last_generate_kwargs_passed is not None
    assert mock_openai_compatible_processor.last_generate_kwargs_passed.get("temperature") == 0.5

    assert mock_openai_compatible_processor.system_prompt == DEFAULT_PROCESSOR_SYSTEM_PROMPT
    assert mock_openai_compatible_processor.chat_history == DEFAULT_PROCESSOR_CHAT_HISTORY

def test_empty_messages_list(mock_openai_compatible_processor: MockOpenAICompatibleTextGenerationProcessor):
    response = mock_openai_compatible_processor.generate_openai_comp([])
    assert response == ""
    assert mock_openai_compatible_processor.last_generate_input_data is None # generate should not have been called
    assert mock_openai_compatible_processor.system_prompt == DEFAULT_PROCESSOR_SYSTEM_PROMPT
    assert mock_openai_compatible_processor.chat_history == DEFAULT_PROCESSOR_CHAT_HISTORY

def test_messages_list_with_only_system_prompt(mock_openai_compatible_processor: MockOpenAICompatibleTextGenerationProcessor):
    messages = [{"role": "system", "content": "Test System Only"}]
    response = mock_openai_compatible_processor.generate_openai_comp(messages)
    assert response == ""
    assert mock_openai_compatible_processor.last_generate_input_data is None # generate should not have been called
    assert mock_openai_compatible_processor.system_prompt == DEFAULT_PROCESSOR_SYSTEM_PROMPT
    assert mock_openai_compatible_processor.chat_history == DEFAULT_PROCESSOR_CHAT_HISTORY
