import gc
import logging
import time
from owlsight.utils.deep_learning import free_cuda_memory
from owlsight.processors.base import TextGenerationProcessor

logger = logging.getLogger(__name__)

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
        try:
            self.clear_memory()
        except Exception as e:
            logger.error(f"Memory cleanup error: {str(e)}")

    def clear_memory(self):
        """Clear all processor and model memory using proven methods"""
        try:
            if hasattr(self.processor, 'pipeline'):
                if hasattr(self.processor.pipeline, 'device'):
                    self.processor.pipeline.device = None
                if hasattr(self.processor.pipeline, 'model'):
                    self.processor.pipeline.model = None
                del self.processor.pipeline
                gc.collect()

            # Clear model memory if it exists
            if hasattr(self.processor, "model"):
                if hasattr(self.processor.model, "cpu"):
                    self.processor.model.cpu()
                if hasattr(self.processor.model, "to"):
                    self.processor.model.to('cpu')
                del self.processor.model
                gc.collect()

            # Clear ONNX components
            if hasattr(self.processor, '_model'):
                # Release ONNX session resources
                if hasattr(self.processor._model, 'end_profiling'):
                    self.processor._model.end_profiling()
                if hasattr(self.processor._model, 'close'):
                    self.processor._model.close()
                del self.processor._model
                gc.collect()

            # Clear tokenizer and related components
            for attr in ['tokenizer', 'tokenizer_stream', 'transformers_tokenizer']:
                if hasattr(self.processor, attr):
                    delattr(self.processor, attr)
            gc.collect()

            # Clear GGUF specific memory
            if hasattr(self.processor, "llm"):
                del self.processor.llm
                gc.collect()

            # Clear any remaining components
            for attr in ['generator', 'params']:
                if hasattr(self.processor, attr):
                    delattr(self.processor, attr)

            # Force memory release
            free_cuda_memory()
            
            # Triple GC for conservative cleanup with small delays
            for _ in range(3):
                gc.collect()
                time.sleep(0.1)  # Small delay to allow OS to reclaim memory

        except Exception as e:
            logger.error(f"Memory cleanup error: {str(e)}")
            raise
