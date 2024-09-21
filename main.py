from src.main_logic.main_loop import main
from src.utils.deep_learning import check_gpu_and_cuda
from src.processors.text_generation_manager import TextGenerationManager
from src.ui.logo import print_logo
from src.configurations.config_manager import ConfigManager


from src.utils.logger_manager import LoggerManager

logger = LoggerManager.get_logger(__name__)


if __name__ == "__main__":
    print_logo()
    check_gpu_and_cuda()

    config_manager = ConfigManager()
    text_generation_manager = TextGenerationManager(
        config_manager=config_manager,
    )

    # initialize agent
    main(text_generation_manager)
