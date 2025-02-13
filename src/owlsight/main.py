from owlsight.app.run_app import run
from owlsight.processors.text_generation_manager import TextGenerationManager
from owlsight.ui.logo import print_logo
from owlsight.configurations.config_manager import ConfigManager
from owlsight.utils.deep_learning import check_gpu_and_cuda, calculate_max_parameters_per_dtype
from owlsight.utils.logger import logger

import logging
import argparse

def setup_logging():
    default_log_level = 'info'
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', help='Log file to write to')
    parser.add_argument('--log_level', 
                       choices=['debug', 'info', 'warning', 'error', 'critical'],
                       default=default_log_level,
                       help='Set the logging level')
    args = parser.parse_args()
    
    # Set log level
    level = getattr(logging, args.log_level.upper())
    logger.setLevel(level)
    
    # Add file logging if specified
    if args.log:
        logger.configure_file_logging(args.log, level=level)

def main():
    setup_logging()
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
