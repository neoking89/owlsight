# __init__.py

# Importing functions to make them accessible from the package's root
from .processors.helper_functions import select_processor_type
from .multimodal.tesseract import setup_tesseract
from .rag.python_lib_search import search_python_libs
from .utils.deep_learning import (
    get_best_device,
    check_gpu_and_cuda,
    calculate_max_parameters_per_dtype,
    calculate_memory_for_model,
    calculate_available_vram,
    check_onnx_device,
)
from .processors.text_generation_processors import (
    TextGenerationProcessorOnnx,
    TextGenerationProcessorTransformers,
    TextGenerationProcessorGGUF,
)
from .processors.multimodal_processors import MultiModalProcessorTransformers
from .rag.core import (
    search_documents,
    HashingVectorizerSearch,
    TfidfSearch,
    SentenceTransformerSearch,
)
from .app.default_functions import OwlDefaultFunctions, search_bing, is_url
from .hugging_face.core import get_model_data

__all__ = [
    "setup_tesseract",
    "get_best_device",
    "check_onnx_device",
    "check_gpu_and_cuda",
    "calculate_max_parameters_per_dtype",
    "calculate_memory_for_model",
    "calculate_available_vram",
    "select_processor_type",
    "TextGenerationProcessorOnnx",
    "TextGenerationProcessorTransformers",
    "TextGenerationProcessorGGUF",
    "MultiModalProcessorTransformers",
    "search_python_libs",
    "search_documents",
    "HashingVectorizerSearch",
    "TfidfSearch",
    "SentenceTransformerSearch",
    "OwlDefaultFunctions",
    "search_bing",
    "is_url",
    "get_model_data",
]
