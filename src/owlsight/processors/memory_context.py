from owlsight.utils.deep_learning import free_cuda_memory
from owlsight.processors.base import TextGenerationProcessor


class ProcessorMemoryContext:
    def __init__(self, processor: TextGenerationProcessor):
        """Context that wraps text generation processors to clean memory and ensure proper cleanup.

        Parameters
        ----------
        processor : TextGenerationProcessor
            The text generation processor to manage memory for.

        Examples
        --------
        >>> from owlsight.processors import TextGenerationProcessor
        >>> from owlsight.processors.memory_context import ProcessorMemoryContext
        >>> processor = TextGenerationProcessor(model_id="gpt2", task="text-generation")
        >>> with ProcessorMemoryContext(processor) as managed_processor:
        ...     # Generate some text to ensure model is loaded
        ...     _ = managed_processor.generate("Test input", max_new_tokens=20)
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
