# __init__.py

from .processors.helper_functions import select_processor_type
from .multimodal.tesseract import setup_tesseract
from .rag.python_lib_search import PythonLibSearcher
from .utils.deep_learning import (
    get_best_device,
    check_gpu_and_cuda,
    llama_supports_gpu_offload,
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
    DocumentSearcher,
    HashingVectorizerSearchEngine,
    TFIDFSearchEngine,
    SentenceTransformerSearchEngine,
)
from .rag.text_splitters import SemanticTextSplitter, SentenceTextSplitter
from .rag.document_reader import DocumentReader
from .app.default_functions import OwlDefaultFunctions, is_url
from .huggingface.core import get_model_data
from .prompts.system_prompts import ExpertPrompts, AgentPrompts, PromptWriter
from .prompts.helper_functions import function_to_json_for_tool_calling
from .voice.voice_control import VoiceControl

__all__ = [
    "setup_tesseract",
    "get_best_device",
    "check_onnx_device",
    "check_gpu_and_cuda",
    "llama_supports_gpu_offload",
    "calculate_max_parameters_per_dtype",
    "calculate_memory_for_model",
    "calculate_available_vram",
    "select_processor_type",
    "TextGenerationProcessorOnnx",
    "TextGenerationProcessorTransformers",
    "TextGenerationProcessorGGUF",
    "MultiModalProcessorTransformers",
    "PythonLibSearcher",
    "SentenceTextSplitter",
    'SemanticTextSplitter',
    "DocumentSearcher",
    "DocumentReader",
    "HashingVectorizerSearchEngine",
    "TFIDFSearchEngine",
    "SentenceTransformerSearchEngine",
    "OwlDefaultFunctions",
    "is_url",
    "get_model_data",
    "ExpertPrompts",
    "AgentPrompts",
    "PromptWriter",
    "function_to_json_for_tool_calling",
    "VoiceControl"
]
