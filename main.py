from src.main_logic.main_loop import main
from src.utils.deep_learning import check_gpu_and_cuda
from src.processors.text_generation import (
    TextGenerationProcessorOnnx,
    TextGenerationProcessorTransformers,
)
from src.utils.logger_manager import LoggerManager

logger = LoggerManager.get_logger(__name__)


if __name__ == "__main__":
    check_gpu_and_cuda()

    model_path = r"models\small\cuda\cuda-int4-rtn-block-32"
    model_hf_id = "microsoft/Phi-3-mini-4k-instruct"

    # Initialize processor (uncomment if Transformers processor is needed)
    # processor = TextGenerationProcessorTransformers(
    #     model_id=model_hf_id,
    #     quantization_bits=4,
    #     save_history=True,
    # )

    processor = TextGenerationProcessorOnnx(
        model_id=model_path,
        tokenizer=model_hf_id,
        verbose=True,
        save_history=True,
    )

    main(processor, max_retries=3, max_new_tokens=1024, prompt_code_execution=True)
