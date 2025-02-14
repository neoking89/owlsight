import logging
import argparse
from typing import Optional

from owlsight.app.run_app import run
from owlsight.processors.text_generation_manager import TextGenerationManager
from owlsight.ui.logo import print_logo
from owlsight.configurations.config_manager import ConfigManager
from owlsight.utils.deep_learning import check_gpu_and_cuda, calculate_max_parameters_per_dtype
from owlsight.utils.logger import logger



def setup_logging(log_level='info', log_path: Optional[str] = None):
    """
    Set up logging
    
    Parameters
    ----------
    log_level : str, optional
        Default log level, by default 'info'
        Options: debug, info, warning, error, critical
    log_path : Optional[str], optional
        Path to the log file, by default None
        If a path is specified, all logging will be written to the file next to the console
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', help='Log file to write to')
    parser.add_argument('--log_level', 
                       choices=['debug', 'info', 'warning', 'error', 'critical'],
                       default=log_level,
                       help='Set the logging level')
    args = parser.parse_args()
    
    # Set log level
    level = getattr(logging, args.log_level.upper())
    logger.setLevel(level)
    
    log_path = args.log or log_path
    if log_path:
        logger.configure_file_logging(log_path, level=level)

def main(log_level='info', log_path: Optional[str] = None):
    """
    Main entry point for the application
    
    Parameters
    ----------
    log_level : str, optional
        Log level, by default 'info'
        Options: debug, info, warning, error, critical
    log_path : Optional[str], optional
        Path to the log file, by default None
        If a path is specified, all logging will be written to the file next to the console
    """
    setup_logging(log_level, log_path)
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
