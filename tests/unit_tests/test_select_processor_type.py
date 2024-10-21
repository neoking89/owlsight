import pytest
from owlsight import (
    select_processor_type,
    TextGenerationProcessorTransformers,
    TextGenerationProcessorOnnx,
    TextGenerationProcessorGGUF,
)


@pytest.mark.parametrize(
    "model_path, expected_result",
    [
        (r"nvidia/NVLM-D-72B", TextGenerationProcessorTransformers),  # Replace with the expected result
        (
            r"C:\Users\Test\.cache\huggingface\hub\models--microsoft--Phi-3-mini-4k-instruct-gguf",
            TextGenerationProcessorGGUF,
        ),
        (r"EleutherAI/gpt-neo-2.7B", TextGenerationProcessorTransformers),
        (
            r"C:\Users\Test\.cache\lm-studio\models\hugging-quants\Llama-3.2-1B-Instruct-Q8_0-GGUF",
            TextGenerationProcessorGGUF,
        ),
        (r"C:\Users\Test\.cache\huggingface\hub\models--yhavinga--t5-base-dutch", TextGenerationProcessorTransformers),
        (r"microsoft/Phi-3.5-mini-instruct-onnx", TextGenerationProcessorOnnx),
    ],
)
def test_select_processor_type(model_path, expected_result):
    assert select_processor_type(model_path) == expected_result
