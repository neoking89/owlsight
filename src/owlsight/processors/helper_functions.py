from typing import Optional, Union, Type
import os

from owlsight.hugging_face.constants import HUGGINGFACE_MEDIA_TASKS
from owlsight.processors.base import TextGenerationProcessor
from owlsight.processors.multimodal_processors import TextGenerationProcessorWithMedia
from owlsight.processors.text_generation_processors import (
    TextGenerationProcessorGGUF,
    TextGenerationProcessorOnnx,
    TextGenerationProcessorTransformers,
)


def _select_transformers_processor_type_on_task(
    task: Optional[str],
) -> Union[
    Type["TextGenerationProcessorTransformers"],
    Type["TextGenerationProcessorWithMedia"],
]:
    if task and task in HUGGINGFACE_MEDIA_TASKS:
        return TextGenerationProcessorWithMedia

    return TextGenerationProcessorTransformers


def select_processor_type(model_id: str, task: Optional[str] = None) -> Type["TextGenerationProcessor"]:
    """
    Utilityfunction which selects the appropriate TextGenerationProcessor class based on the model ID or directory.

    If the model_id is a directory, the function will inspect the contents of the directory
    to decide the processor type. Otherwise, it will use the model_id string to make the decision.
    """
    # Check if the model_id is a directory
    if os.path.isdir(model_id):
        # Check if any file in the directory ends with .onnx
        if any(f.endswith("onnx") for f in os.listdir(model_id)):
            return TextGenerationProcessorOnnx
        elif model_id.lower().endswith("gguf") or any(f.endswith("gguf") for f in os.listdir(model_id)):
            return TextGenerationProcessorGGUF
        else:
            return _select_transformers_processor_type_on_task(task)
    else:
        # If model_id is not a directory, use the model_id string
        if model_id.lower().endswith("gguf"):
            return TextGenerationProcessorGGUF
        elif "onnx" in model_id.lower():
            return TextGenerationProcessorOnnx
        else:
            return _select_transformers_processor_type_on_task(task)
