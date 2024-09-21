from src.main_logic.main_loop import main
from src.utils.deep_learning import check_gpu_and_cuda
from src.processors.text_generation_processor import (
    TextGenerationProcessorOnnx,
    TextGenerationProcessorTransformers,
)
from src.processors.text_generation_manager import TextGenerationManager
from src.ui.logo import print_logo
from src.configurations.config_manager import ConfigManager


from src.utils.logger_manager import LoggerManager
logger = LoggerManager.get_logger(__name__)


if __name__ == "__main__":
    print_logo()
    check_gpu_and_cuda()

    model_path = r"models\small\cuda\cuda-int4-rtn-block-32"
    tokenizer = "microsoft/Phi-3-mini-4k-instruct"

    config_manager = ConfigManager()
    config_manager.set("model.model_id", model_path)
    config_manager.set("model.tokenizer", tokenizer)

    text_generation_manager = TextGenerationManager(
        processor=TextGenerationProcessorOnnx,
        config_manager=config_manager,
    )

    # initialize agent
    main(text_generation_manager)
