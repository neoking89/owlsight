import psutil
import os
import time

from owlsight.processors.memory_context import ProcessorMemoryContext
from owlsight.processors.text_generation_processors import (
    TextGenerationProcessorTransformers,
    TextGenerationProcessorOnnx,
    TextGenerationProcessorGGUF,
)

# Small test models for each processor type
TRANSFORMER_TEST_MODEL = "sshleifer/tiny-gpt2"
ONNX_TEST_MODEL = "llmware/tiny-llama-chat-onnx"
GGUF_TEST_MODEL = "TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF"
GGUF_TEST_FILE = "tinyllama-1.1b-chat-v0.3.Q2_K.gguf"


def get_process_memory():
    """Get current process memory usage in bytes"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def test_transformers_memory_management():
    """Test memory management for TransformersProcessor"""
    initial_mem = get_process_memory()

    processor = TextGenerationProcessorTransformers(
        model_id=TRANSFORMER_TEST_MODEL,
        transformers__device="cpu",  # Force CPU
        task="text-generation",
    )

    with ProcessorMemoryContext(processor) as managed_processor:
        # Generate some text to ensure model is loaded
        _ = managed_processor.generate("Test input", max_new_tokens=20)

    # Allow time for memory to be reclaimed
    time.sleep(0.5)
    final_mem = get_process_memory()

    # Allow for memory fragmentation and allocator behavior
    assert final_mem <= initial_mem * 1.25  # 25% overhead allowance


def test_onnx_memory_management():
    """Test memory management for ONNXProcessor"""
    initial_mem = get_process_memory()

    processor = TextGenerationProcessorOnnx(
        model_id=ONNX_TEST_MODEL,
        onnx__n_gpu_layers=0,  # CPU only
        task="text-generation",
    )

    with ProcessorMemoryContext(processor) as managed_processor:
        # Generate some text to ensure model is loaded
        _ = managed_processor.generate("Test input", max_new_tokens=20)

    # Allow time for memory to be reclaimed
    time.sleep(0.5)
    final_mem = get_process_memory()

    # Allow for memory fragmentation and allocator behavior
    assert final_mem <= initial_mem * 1.25  # 25% overhead allowance


def test_gguf_memory_management():
    """Test memory management for GGUFProcessor"""
    initial_mem = get_process_memory()

    processor = TextGenerationProcessorGGUF(
        model_id=GGUF_TEST_MODEL,
        gguf__filename=GGUF_TEST_FILE,
        gguf__n_gpu_layers=0,  # CPU only
    )

    with ProcessorMemoryContext(processor) as managed_processor:
        # Generate some text to ensure model is loaded
        _ = managed_processor.generate("Test input", max_new_tokens=10)

    # Allow time for memory to be reclaimed
    time.sleep(0.5)
    final_mem = get_process_memory()

    # Allow for memory fragmentation and allocator behavior
    assert final_mem <= initial_mem * 1.25  # 25% overhead allowance


def test_processor_cleanup_on_exception():
    """Test memory management when an exception occurs"""
    initial_mem = get_process_memory()

    processor = TextGenerationProcessorTransformers(
        model_id=TRANSFORMER_TEST_MODEL,
        transformers__device="cpu",  # Force CPU
        task="text-generation",
    )

    try:
        with ProcessorMemoryContext(processor) as managed_processor:
            # Generate some text to ensure model is loaded
            _ = managed_processor.generate("Test input", max_new_tokens=10)
            raise ValueError("Test exception")
    except ValueError:
        pass

    # Verify memory was freed even after exception
    final_mem = get_process_memory()
    assert final_mem <= initial_mem * 1.25  # 25% overhead allowance
