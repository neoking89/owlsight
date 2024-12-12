import os
import sys
import logging

from pathlib import Path
import subprocess
import pytest
from typing import Tuple
from transformers import AutoTokenizer

sys.path.append("src")
from owlsight.processors.text_generation_processors import TextGenerationProcessorOnnx

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_model(repo_url: str, destination: Path) -> bool:
    """
    Download model from HuggingFace using git, with proper error handling.

    Args:
        repo_url: HuggingFace repository URL
        destination: Local destination path

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if destination.exists():
            logger.info("Model directory already exists, skipping download")
            return True

        logger.info(f"Downloading model from {repo_url}")
        result = subprocess.run(
            ["git", "clone", repo_url, str(destination)], capture_output=True, text=True, check=True
        )
        logger.info("Model downloaded successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone repository: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during model download: {e}")
        return False


@pytest.fixture
def setup_processor() -> Tuple[TextGenerationProcessorOnnx, AutoTokenizer]:
    """
    Fixture to set up the GGUF processor and tokenizer with proper error handling.

    Returns:
        Tuple[TextGenerationProcessorOnnx, AutoTokenizer]: Processor and tokenizer
    """
    model_repo = "https://huggingface.co/llmware/tiny-llama-chat-onnx"
    destination_dir = Path("tests/data/tiny-llama-chat-onnx")
    tokenizer_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # Ensure parent directory exists
    destination_dir.parent.mkdir(parents=True, exist_ok=True)

    # Download model if needed
    if not download_model(model_repo, destination_dir):
        pytest.skip("Failed to download model")

    try:
        logger.info("Initializing processor and tokenizer...")
        processor = TextGenerationProcessorOnnx(
            model_id=str(destination_dir),
            onnx__tokenizer=tokenizer_id,
        )
        logger.info("Processor initialized successfully")
        return processor, processor.transformers_tokenizer
    except Exception as e:
        logger.error(f"Failed to initialize processor: {e}")
        pytest.skip(f"Processor initialization failed: {e}")


@pytest.mark.slow  # Mark as slow test
def test_gguf_generate_response(setup_processor):
    """Test that the GGUF processor generates a valid response."""
    processor, _ = setup_processor
    prompt = "test prompt"
    max_new_tokens = 128

    logger.info("Generating response...")
    response = processor.generate(prompt, max_new_tokens=max_new_tokens)
    logger.info("Response generated successfully")

    assert isinstance(response, str), "Generated response should be a string."
    assert len(response) > 0, "Generated response should not be empty."


@pytest.mark.slow
def test_gguf_token_count_within_tolerance(setup_processor):
    """Test that the GGUF generated token count is within the acceptable range."""
    processor, tokenizer = setup_processor
    prompt = "test prompt"
    max_new_tokens = 128
    tolerance_fraction = 0.5

    logger.info("Testing token count...")
    response = processor.generate(prompt, max_new_tokens=max_new_tokens)

    response_tokens = tokenizer.tokenize(response)
    prompt_tokens = tokenizer.tokenize(prompt)
    new_tokens = response_tokens[len(prompt_tokens) :]

    lower_bound = max_new_tokens - (max_new_tokens * tolerance_fraction)
    upper_bound = max_new_tokens + (max_new_tokens * tolerance_fraction)

    logger.info(f"New tokens generated: {len(new_tokens)}")
    logger.info(f"Expected range: {lower_bound:.2f} - {upper_bound:.2f}")

    assert lower_bound <= len(new_tokens) <= upper_bound, (
        f"Expected approximately {max_new_tokens} tokens "
        f"(range: {lower_bound:.2f} - {upper_bound:.2f}), got {len(new_tokens)}"
    )


@pytest.mark.slow
def test_gguf_prompt_tokens_exclusion(setup_processor):
    """Test that the GGUF response excludes prompt tokens when counting new tokens."""
    processor, tokenizer = setup_processor
    prompt = "test prompt"
    max_new_tokens = 128

    logger.info("Testing prompt token exclusion...")
    response = processor.generate(prompt, max_new_tokens=max_new_tokens)

    response_tokens = tokenizer.tokenize(response)
    prompt_tokens = tokenizer.tokenize(prompt)

    logger.info(f"Response tokens: {len(response_tokens)}")
    logger.info(f"Prompt tokens: {len(prompt_tokens)}")

    assert len(response_tokens) > len(prompt_tokens), "Response tokens should exceed prompt tokens."


def test_get_max_context_length(setup_processor):
    """Test that the GGUF processor can retrieve the maximum context length."""
    processor, _ = setup_processor
    max_context_length = processor.get_max_context_length()

    assert isinstance(max_context_length, int), "Max context length should be an integer."
    assert max_context_length > 0, "Max context length should be greater than zero."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
