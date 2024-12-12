import time
import sys

sys.path.append("src")

import pytest
from transformers import AutoTokenizer
from owlsight.processors.text_generation_processors import (
    TextGenerationProcessorTransformers,
)


@pytest.fixture
def setup_processor():
    """Fixture to set up the text generation processor and tokenizer."""
    model_id = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    processor = TextGenerationProcessorTransformers(model_id)
    return processor, tokenizer


def test_generate_response(setup_processor):
    """Test that the processor generates a valid response."""
    processor, _ = setup_processor
    prompt = "test prompt"
    max_new_tokens = 128

    response = processor.generate(prompt, max_new_tokens=max_new_tokens)
    assert isinstance(response, str), "Generated response should be a string."


def test_token_count_within_tolerance(setup_processor):
    """Test that the generated token count is within the acceptable range."""
    processor, tokenizer = setup_processor
    prompt = "test prompt"
    max_new_tokens = 128
    tolerance_fraction = 0.5

    response = processor.generate(prompt, max_new_tokens=max_new_tokens)

    # Tokenize response and calculate new tokens
    response_tokens = tokenizer.tokenize(response)
    prompt_tokens = tokenizer.tokenize(prompt)
    new_tokens = response_tokens[len(prompt_tokens) :]

    # Calculate acceptable range
    lower_bound = max_new_tokens - (max_new_tokens * tolerance_fraction)
    upper_bound = max_new_tokens + (max_new_tokens * tolerance_fraction)

    assert lower_bound <= len(new_tokens) <= upper_bound, (
        f"Expected approximately {max_new_tokens} tokens "
        f"(range: {lower_bound:.2f} - {upper_bound:.2f}), got {len(new_tokens)}"
    )


def test_prompt_tokens_exclusion(setup_processor):
    """Test that the response excludes prompt tokens when counting new tokens."""
    processor, tokenizer = setup_processor
    prompt = "test prompt"
    max_new_tokens = 128

    response = processor.generate(prompt, max_new_tokens=max_new_tokens)

    # Tokenize response and calculate new tokens
    response_tokens = tokenizer.tokenize(response)
    prompt_tokens = tokenizer.tokenize(prompt)

    assert len(response_tokens) > len(prompt_tokens), "Response tokens should exceed prompt tokens."


def test_invalid_generation_kwargs(setup_processor):
    """Test that the processor handles invalid kwargs without freezing."""
    processor, _ = setup_processor
    prompt = "test prompt"
    max_new_tokens = 128
    generation_kwargs = {"non_existing_kwarg": "value"}

    # Record start time
    start_time = time.time()

    try:
        # Call generate method and ensure it returns within a reasonable time
        response = processor.generate(prompt, max_new_tokens=max_new_tokens, generation_kwargs=generation_kwargs)
    except Exception as exc:
        # If an exception is raised, ensure it's not due to a freeze
        duration = time.time() - start_time
        assert duration < 5, "The generate method froze."
        assert "non_existing_kwarg" in str(exc), "Error message should contain the invalid keyword."
        return  # Test passes as the exception was correctly raised

    # If no exception is raised, ensure the response is as expected
    duration = time.time() - start_time
    assert duration < 5, "The generate method froze."
    assert response == "", "Response should be empty for invalid generation kwargs."


def test_num_beams_with_streaming_raises_error(setup_processor):
    """Test that using num_beams with streaming produces appropriate error output."""
    processor, _ = setup_processor
    prompt = "test prompt"
    max_new_tokens = 128
    num_beams = 4

    # Generate with num_beams (will print error in thread)
    with pytest.raises(ValueError) as exc:
        processor.generate(prompt, max_new_tokens=max_new_tokens, generation_kwargs={"num_beams": num_beams})
        assert "num_beams" in str(exc.value), "Error message should mention num_beams parameter"


def test_get_max_context_length(setup_processor):
    """Test that the processor returns the correct max context length."""
    processor, _ = setup_processor
    max_context_length = processor.get_max_context_length()
    assert isinstance(max_context_length, int), "Max context length should be an integer."
    assert max_context_length > 0, "Max context length should be greater than zero."

if __name__ == "__main__":
    pytest.main([__file__])
