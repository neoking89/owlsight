# __init__.py

# Importing functions to make them accessible from the package's root
from .rag.python_lib_search import search_python_libs
from .utils.deep_learning import (
    get_best_device,
    check_gpu_and_cuda,
    calculate_max_parameters_per_dtype,
    calculate_memory_for_model,
    calculate_available_vram,
)
from .processors.text_generation_processor import (
    select_processor_type,
    TextGenerationProcessorOnnx,
    TextGenerationProcessorTransformers,
    TextGenerationProcessorGGUF,
)
from .rag.core import (
    search_documents,
    HashingVectorizerSearch,
    TfidfSearch,
    SentenceTransformerSearch,
)
from .app.default_functions import OwlDefaultFunctions, search_bing, is_url

__all__ = [
    "get_best_device",
    "check_gpu_and_cuda",
    "calculate_max_parameters_per_dtype",
    "calculate_memory_for_model",
    "calculate_available_vram",
    "select_processor_type",
    "TextGenerationProcessorOnnx",
    "TextGenerationProcessorTransformers",
    "TextGenerationProcessorGGUF",
    "search_python_libs",
    "search_documents",
    "HashingVectorizerSearch",
    "TfidfSearch",
    "SentenceTransformerSearch",
    "OwlDefaultFunctions",
    "search_bing",
    "is_url",
]
