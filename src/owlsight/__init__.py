# __init__.py

# Importing functions to make them accessible from the package's root
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
from .rag.search import (
    get_context_for_library,
    CosineSimilaritySearch,
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
    "get_context_for_library",
    "CosineSimilaritySearch",
    "SentenceTransformerSearch",
    "OwlDefaultFunctions",
    "search_bing",
    "is_url",
]
