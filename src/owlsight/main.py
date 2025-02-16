import logging
import argparse
from typing import Optional
import json

from owlsight.app.run_app import run
from owlsight.processors.text_generation_manager import TextGenerationManager
from owlsight.ui.logo import print_logo
from owlsight.configurations.config_manager import ConfigManager
from owlsight.utils.deep_learning import check_gpu_and_cuda, calculate_max_parameters_per_dtype
from owlsight.voice.constants import WORD_TO_KEY_MAP, WORD_TO_WORD_MAP
from owlsight.voice.voice_control import VoiceControl
from owlsight.utils.logger import logger


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
    parser.add_argument(
        "--word-to-key",
        type=str,
        help="""JSON string containing voice command to keyboard key mappings.
        Maps spoken words to keyboard actions. Supports single keys and key combinations.
        
        Examples:
        - Single keys:
          '{"backward": "left", "forward": "right"}'
        
        - Key combinations:
          '{"save": ["ctrl", "s"], "undo": ["ctrl", "z"]}'
        
        - Mixed mappings:
          {
            "backward": "left",
            "forward": "right",
            "save": ["ctrl", "s"],
            "select all": ["ctrl", "a"]
          }
        
        Common key names:
        - Navigation: up, down, left, right, home, end, pageup, pagedown
        - Editing: backspace, delete, enter, tab, space
        - Function keys: f1, f2, ..., f12
        - Modifiers: ctrl, alt, shift
        
        Default mappings include: left, right, up, down, enter, delete,
        and combinations like 'select all' (ctrl+a), 'copy' (ctrl+c), 'paste' (ctrl+v)""",
    )
    parser.add_argument(
        "--word-to-word",
        type=str,
        help="""JSON string containing voice command to text substitution mappings.
        Maps spoken phrases to text that will be typed out.
        
        Examples:
        - Code snippets:
          {
            "print": "print()",
            "function": "def my_function():",
            "class": "class MyClass:"
          }
        
        - Common phrases:
          {
            "greeting": "Hello World",
            "bye": "Goodbye!",
            "thanks": "Thank you very much"
          }
        
        - Shell commands:
          {
            "files": "ls -la",
            "clear": "clear()",
            "exit": "exit()"
          }
        
        Default mappings include: {"exit": "exit()"}""",
    )
    parser.add_argument(
        "--voice-control-kwargs",
        type=str,
        help="""JSON string containing keyword arguments for VoiceControl and AudioToTextRecorder.
        
        Examples:
        - Basic configuration:
          {
            "cmd_cooldown": 0.5,
            "debug": true,
            "language": "en"
          }
        
        - Advanced configuration with recorder options:
          {
            "cmd_cooldown": 0.5,
            "model": "base.en",
            "key_press_interval": 0.1,
            "typing_interval": 0.05
          }
        
        Available VoiceControl parameters:
        - cmd_cooldown (float): Cooldown between commands (default: 1.0)
        - debug (bool): Enable debug mode (default: false)
        - language (str): Language code (default: "en")
        - model (str): Model name (default: "small.en")
        - key_press_interval (float): Interval between key presses (default: 0.05)
        - typing_interval (float): Interval between typing (default: 0.03)
        
        AudioToTextRecorder parameters are also supported and will be passed through.""",
    )
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


def main(
    default_log_level="info",
    log_path: Optional[str] = None,
    voice_control: bool = False,
    voice_control_kwargs: Optional[dict] = None,
):
    """
    Main entry point for the application

    Parameters
    ----------
    default_log_level : str, optional
        Log level, by default 'info'
        Set to 'debug' for more detailed logs
        Options: debug, info, warning, error, critical
    log_path : Optional[str], optional
        Path to the log file, by default None
        If a path is specified, all logging will be written to the file next to the console
    voice_control : bool, optional
        Whether to enable voice control, by default False
    voice_control_kwargs : Optional[dict], optional
        Keyword arguments for VoiceControl and AudioToTextRecorder, by default None\n
        For example:\n
        voice_control_kwargs = {
            "word_to_key_map": {
                "hello": "hello",
                "world": "world"
            },
            "word_to_word_map": {
                "hello": "world"
            },
            "language": "en"
        }
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

        # Initialize with default mappings
        custom_word_to_key = dict(WORD_TO_KEY_MAP)
        custom_word_to_word = dict(WORD_TO_WORD_MAP)

        # Parse word-to-key mappings if provided
        if args.word_to_key:
            try:
                key_mappings = json.loads(args.word_to_key)
                if not isinstance(key_mappings, dict):
                    raise ValueError("word-to-key must be a JSON object")
                custom_word_to_key.update(key_mappings)
                logger.info(f"Added custom word-to-key mappings: {key_mappings}")
            except json.JSONDecodeError:
                logger.warning("Invalid JSON format for word-to-key. Using defaults.")
            except Exception as e:
                logger.warning(f"Error parsing word-to-key: {e}. Using defaults.")

        # Parse word-to-word mappings if provided
        if args.word_to_word:
            try:
                word_mappings = json.loads(args.word_to_word)
                if not isinstance(word_mappings, dict):
                    raise ValueError("word-to-word must be a JSON object")
                custom_word_to_word.update(word_mappings)
                logger.info(f"Added custom word-to-word mappings: {word_mappings}")
            except json.JSONDecodeError:
                logger.warning("Invalid JSON format for word-to-word. Using defaults.")
            except Exception as e:
                logger.warning(f"Error parsing word-to-word: {e}. Using defaults.")

        # Parse voice control kwargs if provided
        voice_kwargs = {} or voice_control_kwargs
        if args.voice_control_kwargs:
            try:
                voice_kwargs = json.loads(args.voice_control_kwargs)
                if not isinstance(voice_kwargs, dict):
                    raise ValueError("voicecontrol-kwargs must be a JSON object")
                logger.info(f"Using custom voice control configuration: {voice_kwargs}")
            except json.JSONDecodeError:
                logger.warning("Invalid JSON format for voicecontrol-kwargs. Using defaults.")
            except Exception as e:
                logger.warning(f"Error parsing voicecontrol-kwargs: {e}. Using defaults.")

        # Create and start voice control with custom mappings and kwargs
        vc = VoiceControl(
            word_to_key_map=custom_word_to_key, word_to_word_map=custom_word_to_word, debug=False, **voice_kwargs
        )

        # If voice control is not available, VoiceControl will be the DummyVoiceControl class
        # which will log a warning about missing dependencies
        vc.start()  # This now runs in background

    # initialize agent
    run(text_generation_manager)

    # Cleanup voice control if it was started
    if "vc" in locals() and vc.is_running:
        vc.stop()


if __name__ == "__main__":
    main()
