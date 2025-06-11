"""
Entry point for the Owlsight OpenAI-Compatible Server.

This script parses command-line arguments, initializes the specified
Owlsight text generation processor, and starts a Uvicorn server
to serve the FastAPI application defined in app.py.
"""
import argparse
import logging
import uvicorn
import time
from typing import Type

from owlsight import select_processor_type
from owlsight.processors.text_generation_processors import TextGenerationProcessor
from .app import app # FastAPI instance
from . import app as server_app_module # For accessing/modifying app.py's module-level variables

# Configure logging for the server startup script
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    """Parse CLI flags, initialise the processor, and run the server."""
    parser = argparse.ArgumentParser(
        description=(
            "Run any owlsight text-generation processor behind an OpenAI-compatible API."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Identifier of the model to load (e.g., 'gguf/model_name', 'hf/org/repo').",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server to."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output including token-by-token printing during streaming."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level."
    )
    # Capture all other arguments for the processor
    args, unknown_args = parser.parse_known_args()

    # Process unknown arguments into a dictionary for the processor
    # (e.g. --gguf__filename model.gguf -> {"gguf__filename": "model.gguf"})
    processor_kwargs = {}
    # Default generation parameters that can be overridden by CLI
    cli_default_params = {}

    # This logic is adapted from examples/openai_compatible.py
    idx = 0
    while idx < len(unknown_args):
        arg_name = unknown_args[idx]
        if not arg_name.startswith("--"):
            logger.warning(f"Skipping malformed argument: {arg_name}")
            idx += 1
            continue

        key = arg_name[2:]  # Remove "--"

        # Check if it's a flag (no value follows or next arg is another --option)
        if idx + 1 >= len(unknown_args) or unknown_args[idx + 1].startswith("--"):
            processor_kwargs[key] = True
            idx += 1
        else:
            value_str = unknown_args[idx + 1]
            # Attempt to parse value as int, float, bool, or keep as str
            try:
                value = int(value_str)
            except ValueError:
                try:
                    value = float(value_str)
                except ValueError:
                    if value_str.lower() == "true":
                        value = True
                    elif value_str.lower() == "false":
                        value = False
                    else:
                        value = value_str # Keep as string
            
            # Check if this is a known default generation parameter
            if key in ["max_new_tokens", "temperature", "top_p"]:
                 cli_default_params[key] = value
            else:
                processor_kwargs[key] = value
            idx += 2

    logger.info(f"Selected model: {args.model}")
    logger.info(f"Processor-specific arguments: {processor_kwargs}")
    logger.info(f"Default generation parameters from CLI: {cli_default_params}")

    # server_app_module is now imported at the top level
    # No global statement needed as we are modifying attributes of the imported module
    server_app_module.START_TIME = int(time.time())
    server_app_module.SERVER_MODEL_ID = args.model

    # Initialize the processor
    try:
        processor_type: Type[TextGenerationProcessor] = select_processor_type(args.model)
        logger.info(f"Instantiating processor of type: {processor_type.__name__}")
        server_app_module.processor = processor_type(model_id=args.model, **processor_kwargs)
        logger.info(f"Successfully instantiated processor for model: {args.model}")
    except Exception as e:
        logger.error(f"Failed to initialize processor for model '{args.model}': {e}", exc_info=True)
        # Exit if processor fails to load, as the server would be useless.
        return

    # Set default generation parameters (CLI overrides code defaults)
    server_app_module.default_params.update(cli_default_params)
    logger.info(f"Final default generation parameters: {server_app_module.default_params}")

    # Set the log level based on command line argument
    log_level = getattr(logging, args.log_level)
    logging.getLogger().setLevel(log_level)
    
    # Set verbose flag for app module
    server_app_module.VERBOSE = args.verbose
    logger.info(f"Verbose mode: {'enabled' if args.verbose else 'disabled'}")
    display_host = "localhost" if args.host == "0.0.0.0" else args.host
    logging.info("Swagger UI available at http://%s:%d/docs", display_host, args.port)
    logger.info(f"Starting Uvicorn server on {display_host}:{args.port} for model '{args.model}'")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()