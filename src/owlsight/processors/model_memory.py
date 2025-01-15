from owlsight.utils.deep_learning import free_cuda_memory
from owlsight.processors.base import TextGenerationProcessor


class ProcessorMemoryManager:
    def __init__(self, processor: TextGenerationProcessor):
        """Memory manager that wraps text generation processors to ensure proper cleanup.

        Parameters
        ----------
        processor : TextGenerationProcessor
            The text generation processor to manage (TextGenerationProcessorTransformers,
            TextGenerationProcessorOnnx, or TextGenerationProcessorGGUF)
        """
        if not isinstance(processor, TextGenerationProcessor):
            raise TypeError(f"Processor must be an instance of TextGenerationProcessor, not {type(processor)}")

        self.processor = processor

    def __enter__(self):
        return self.processor

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear_memory()

    def clear_memory(self):
        """Clear all processor and model memory"""
        # Clear model memory if it exists
        if hasattr(self.processor, "model"):
            if hasattr(self.processor.model, "cpu"):
                self.processor.model.cpu()
            del self.processor.model

        # Clear ONNX specific memory
        if hasattr(self.processor, "_model"):
            del self.processor._model

        # Clear GGUF specific memory
        if hasattr(self.processor, "llm"):
            del self.processor.llm

        # Clear tokenizer
        if hasattr(self.processor, "tokenizer"):
            del self.processor.tokenizer

        # Clear pipeline
        if hasattr(self.processor, "pipeline"):
            del self.processor.pipeline

        free_cuda_memory()

        # Clear processor reference
        del self.processor
