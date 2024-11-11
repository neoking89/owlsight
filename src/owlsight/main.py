from owlsight.app.run_app import run
from owlsight.processors.text_generation_manager import TextGenerationManager
from owlsight.ui.logo import print_logo
from owlsight.configurations.config_manager import ConfigManager
from owlsight.utils.deep_learning import check_gpu_and_cuda, calculate_max_parameters_per_dtype


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
