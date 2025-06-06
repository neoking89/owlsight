"""
Unit-tests for the OpenAI-compatible generate_openai_comp wrapper.
"""

from typing import Any, Dict, List, Optional

import pytest

from owlsight.processors.base import TextGenerationProcessor


# --------------------------------------------------------------------------- #
#                                   Mocks                                     #
# --------------------------------------------------------------------------- #
class MockOpenAICompatibleTextGenerationProcessor(TextGenerationProcessor):
    """
    Minimal mock of a concrete processor – only the pieces needed for the unit-tests.
    """

    def __init__(
        self,
        model_id: str = "mock_model",
        apply_chat_history: bool = True,
        system_prompt: str = "Default system prompt",
    ):
        super().__init__(model_id, apply_chat_history, system_prompt)
        self.last_generate_input_data: Optional[str] = None
        self.last_generate_system_prompt_used: Optional[str] = None
        self.last_generate_chat_history_used: Optional[List[Dict[str, str]]] = None
        self.last_generate_kwargs_passed: Optional[Dict[str, Any]] = None
        self.generate_return_value: str = "mocked_response"

    # ---------- TextGenerationProcessor API ---------- #
    def generate(  # type: ignore[override]
        self,
        input_data: str,
        max_new_tokens: int,
        temperature: float,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Just record the parameters so the tests can inspect them,
        then return the canned response.
        """
        self.last_generate_input_data = input_data
        self.last_generate_system_prompt_used = self.system_prompt
        # copy – we don’t want the original list mutated elsewhere
        self.last_generate_chat_history_used = list(self.chat_history)

        # Consolidate everything the real generate() saw
        self.last_generate_kwargs_passed = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "generation_kwargs": generation_kwargs,
            **kwargs,
        }
        return self.generate_return_value

    def generate_stream(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError("streaming not required for these tests")

    def get_max_context_length(self) -> int:  # type: ignore[override]
        return 4096  # mock value


# --------------------------------------------------------------------------- #
#                              Example payloads                               #
# --------------------------------------------------------------------------- #
EXAMPLE_MESSAGES_SIMPLE = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather like in Paris?"},
]

EXAMPLE_MESSAGES_NO_SYSTEM_IN_LIST = [
    {"role": "user", "content": "Tell me a joke."},
]

EXAMPLE_MESSAGES_MULTI_TURN = [
    {"role": "system", "content": "You are a coding assistant."},
    {"role": "user", "content": "How do I write a Python function to reverse a string?"},
    {
        "role": "assistant",
        "content": "You can use slicing:\n\ndef reverse_string(s):\n    return s[::-1]",
    },
    {"role": "user", "content": "Can you explain how slicing works?"},
]

DEFAULT_PROCESSOR_SYSTEM_PROMPT = "Initial Mock System Prompt"
DEFAULT_PROCESSOR_CHAT_HISTORY = [{"role": "user", "content": "Original history item"}]


# --------------------------------------------------------------------------- #
#                               Test fixtures                                 #
# --------------------------------------------------------------------------- #
@pytest.fixture
def mock_openai_compatible_processor() -> MockOpenAICompatibleTextGenerationProcessor:
    processor = MockOpenAICompatibleTextGenerationProcessor(
        system_prompt=DEFAULT_PROCESSOR_SYSTEM_PROMPT
    )
    # simulate pre-existing conversation history
    processor.chat_history = list(DEFAULT_PROCESSOR_CHAT_HISTORY)
    return processor


# --------------------------------------------------------------------------- #
#                                 Test cases                                  #
# --------------------------------------------------------------------------- #
def _assert_openai_response_structure(resp: Dict[str, Any], expected_content: str) -> None:
    """Common assertions for the OpenAI ChatCompletion shape."""
    assert resp["object"] == "chat.completion"
    assert isinstance(resp["id"], str)
    assert resp["choices"][0]["message"]["content"] == expected_content
    assert resp["choices"][0]["finish_reason"] == "stop"


def test_simple_conversation(
    mock_openai_compatible_processor: MockOpenAICompatibleTextGenerationProcessor,
):
    messages = list(EXAMPLE_MESSAGES_SIMPLE)  # copy
    gen_kwargs = {"max_new_tokens": 50, "temperature": 0.7, "extra_param": "test"}

    response = mock_openai_compatible_processor.generate_openai_comp(messages, **gen_kwargs)

    # ----------  response shape  ---------- #
    assert isinstance(response, dict)
    _assert_openai_response_structure(response, mock_openai_compatible_processor.generate_return_value)

    # ----------  internals  ---------- #
    assert mock_openai_compatible_processor.last_generate_system_prompt_used == "You are a helpful assistant."
    assert mock_openai_compatible_processor.last_generate_input_data == "What's the weather like in Paris?"
    assert mock_openai_compatible_processor.last_generate_chat_history_used == []
    assert mock_openai_compatible_processor.last_generate_kwargs_passed["max_new_tokens"] == 50
    assert mock_openai_compatible_processor.last_generate_kwargs_passed["temperature"] == 0.7
    assert mock_openai_compatible_processor.last_generate_kwargs_passed["generation_kwargs"]["extra_param"] == "test"

    # ----------  state restoration  ---------- #
    assert mock_openai_compatible_processor.system_prompt == DEFAULT_PROCESSOR_SYSTEM_PROMPT
    assert mock_openai_compatible_processor.chat_history == DEFAULT_PROCESSOR_CHAT_HISTORY


def test_no_system_prompt_in_messages(
    mock_openai_compatible_processor: MockOpenAICompatibleTextGenerationProcessor,
):
    messages = list(EXAMPLE_MESSAGES_NO_SYSTEM_IN_LIST)
    gen_kwargs = {"max_new_tokens": 60, "temperature": 0.1}

    response = mock_openai_compatible_processor.generate_openai_comp(messages, **gen_kwargs)

    assert isinstance(response, dict)
    _assert_openai_response_structure(response, mock_openai_compatible_processor.generate_return_value)

    assert mock_openai_compatible_processor.last_generate_system_prompt_used == DEFAULT_PROCESSOR_SYSTEM_PROMPT
    assert mock_openai_compatible_processor.last_generate_input_data == "Tell me a joke."
    assert mock_openai_compatible_processor.last_generate_chat_history_used == []
    assert mock_openai_compatible_processor.last_generate_kwargs_passed["max_new_tokens"] == 60

    assert mock_openai_compatible_processor.system_prompt == DEFAULT_PROCESSOR_SYSTEM_PROMPT
    assert mock_openai_compatible_processor.chat_history == DEFAULT_PROCESSOR_CHAT_HISTORY


def test_multi_turn_conversation(
    mock_openai_compatible_processor: MockOpenAICompatibleTextGenerationProcessor,
):
    messages = list(EXAMPLE_MESSAGES_MULTI_TURN)
    gen_kwargs = {"max_new_tokens": 100, "temperature": 0.5}

    response = mock_openai_compatible_processor.generate_openai_comp(messages, **gen_kwargs)

    assert isinstance(response, dict)
    _assert_openai_response_structure(response, mock_openai_compatible_processor.generate_return_value)

    assert mock_openai_compatible_processor.last_generate_system_prompt_used == "You are a coding assistant."
    assert mock_openai_compatible_processor.last_generate_input_data == "Can you explain how slicing works?"
    expected_history_for_generate = [
        {
            "role": "user",
            "content": "How do I write a Python function to reverse a string?",
        },
        {
            "role": "assistant",
            "content": "You can use slicing:\n\ndef reverse_string(s):\n    return s[::-1]",
        },
    ]
    assert (
        mock_openai_compatible_processor.last_generate_chat_history_used
        == expected_history_for_generate
    )
    assert mock_openai_compatible_processor.last_generate_kwargs_passed["temperature"] == 0.5

    assert mock_openai_compatible_processor.system_prompt == DEFAULT_PROCESSOR_SYSTEM_PROMPT
    assert mock_openai_compatible_processor.chat_history == DEFAULT_PROCESSOR_CHAT_HISTORY


def test_empty_messages_list(
    mock_openai_compatible_processor: MockOpenAICompatibleTextGenerationProcessor,
):
    with pytest.raises(ValueError, match=r"messages.*empty"):
        mock_openai_compatible_processor.generate_openai_comp([])

    # generate() must not have been called
    assert mock_openai_compatible_processor.last_generate_input_data is None
    assert mock_openai_compatible_processor.system_prompt == DEFAULT_PROCESSOR_SYSTEM_PROMPT
    assert mock_openai_compatible_processor.chat_history == DEFAULT_PROCESSOR_CHAT_HISTORY


def test_messages_list_with_only_system_prompt(
    mock_openai_compatible_processor: MockOpenAICompatibleTextGenerationProcessor,
):
    messages = [{"role": "system", "content": "Test System Only"}]

    with pytest.raises(ValueError, match=r"No user message"):
        mock_openai_compatible_processor.generate_openai_comp(messages)

    assert mock_openai_compatible_processor.last_generate_input_data is None
    assert mock_openai_compatible_processor.system_prompt == DEFAULT_PROCESSOR_SYSTEM_PROMPT
    assert mock_openai_compatible_processor.chat_history == DEFAULT_PROCESSOR_CHAT_HISTORY
