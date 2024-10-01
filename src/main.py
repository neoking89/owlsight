from app.run_app import run
from processors.text_generation_manager import TextGenerationManager
from ui.logo import print_logo
from configurations.config_manager import ConfigManager
from utils.deep_learning import check_gpu_and_cuda, calculate_max_parameters_per_dtype
from utils.logger_manager import LoggerManager

logger = LoggerManager.get_logger(__name__)


def main():
    print_logo()
    check_gpu_and_cuda()
    calculate_max_parameters_per_dtype()

    config_manager = ConfigManager()
    text_generation_manager = TextGenerationManager(
        config_manager=config_manager,
    )

    # initialize agent
    run(text_generation_manager)


if __name__ == "__main__":
    main()
