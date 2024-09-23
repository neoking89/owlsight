from app.run_app import run
from utils.deep_learning import check_gpu_and_cuda
from processors.text_generation_manager import TextGenerationManager
from ui.logo import print_logo
from configurations.config_manager import ConfigManager
from utils.logger_manager import LoggerManager

logger = LoggerManager.get_logger(__name__)


def main():
    print_logo()
    check_gpu_and_cuda()

    config_manager = ConfigManager()
    text_generation_manager = TextGenerationManager(
        config_manager=config_manager,
    )

    # initialize agent
    run(text_generation_manager)


if __name__ == "__main__":
    main()
