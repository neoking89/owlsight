import logging
import argparse
from tkinter import W
from typing import Optional

from owlsight.app.run_app import run
from owlsight.processors.text_generation_manager import TextGenerationManager
from owlsight.ui.logo import print_logo
from owlsight.configurations.config_manager import ConfigManager
from owlsight.utils.deep_learning import check_gpu_and_cuda, calculate_max_parameters_per_dtype
from owlsight.utils.logger import logger
from owlsight.voice.voice_control import VoiceControl


def parse_arguments(log_level="info"):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Owlsight Application")
    parser.add_argument("--log", help="Log file to write to")
    parser.add_argument(
        "--log_level",
        choices=["debug", "info", "warning", "error", "critical"],
        default=log_level,
        help="Set the logging level",
    )
    parser.add_argument("--voice", action="store_true", help="Activate voice control functionality", default=False)
    return parser.parse_args()


def setup_logging(args, log_path: Optional[str] = None):
    """
    Set up logging

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments
    log_level : str, optional
        Default log level, by default 'info'
    log_path : Optional[str], optional
        Path to the log file, by default None
        If a path is specified, all logging will be written to the file next to the console
    """
    # Set log level
    level = getattr(logging, args.log_level.upper())
    logger.setLevel(level)

    # Use either command line log path or function parameter
    log_path = args.log or log_path
    if log_path:
        logger.configure_file_logging(log_path, level=level)


def main(default_log_level="info", log_path: Optional[str] = None, voice_control: bool = False):
    """
    Main entry point for the application

    Parameters
    ----------
    default_log_level : str, optional
        Log level, by default 'info'
        Options: debug, info, warning, error, critical
    log_path : Optional[str], optional
        Path to the log file, by default None
        If a path is specified, all logging will be written to the file next to the console
    voice_control : bool, optional
        Whether to enable voice control, by default False
    """
    args = parse_arguments(default_log_level)
    setup_logging(args, log_path)

    print_logo()
    check_gpu_and_cuda()
    calculate_max_parameters_per_dtype()

    config_manager = ConfigManager()
    text_generation_manager = TextGenerationManager(
        config_manager=config_manager,
    )

    if args.voice or voice_control:
        logger.info("Voice control enabled")
        # Example command mapping
        WORD_TO_KEY_MAP = {"boom": "left", 
        "bang": "right", 
        "ding": "up", 
        "dong": "down", 
        "sick": "enter",
        "select all": ["ctrl", "a"],
        "copy": ["ctrl", "c"],
        "paste": ["ctrl", "y"],
        "delete": "delete",
        }
        WORD_TO_WORD_MAP = {
            "Exit": "exit()",
            "exit": "exit()",
        }

        # Create and start voice control
        vc = VoiceControl(
            word_to_key_map=WORD_TO_KEY_MAP,
            word_to_word_map=WORD_TO_WORD_MAP,
            debug=False,
        )
        vc.start()  # This now runs in background

    # initialize agent
    run(text_generation_manager)

    # Cleanup voice control if it was started
    if 'vc' in locals() and vc.is_running:
        vc.stop()


if __name__ == "__main__":
    main()
