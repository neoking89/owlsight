import argparse
import json
import logging
from typing import Any, Dict, List, Optional, Type

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from owlsight import select_processor_type
from owlsight.processors.text_generation_processors import TextGenerationProcessor

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

app = FastAPI(title="Chat Completion API", version="1.0.0")

# Globals (to be populated from CLI)
DEFAULT_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
default_params: Dict[str, Any] = {}
processor: Optional[TextGenerationProcessor] = None


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    model: str  # Required parameter - no default value
    stream: bool = False
    max_new_tokens: int = 512

    # Allow clients to pass any other OpenAI-like parameters (e.g., temperature, top_p, etc.)
    model_config = {"extra": "allow"}


@app.post("/v1/chat/completions")
async def completions(request: ChatCompletionRequest):
    """
    Handle chat completion requests, either streaming or single-response.
    Accepts OpenAI-compatible parameters (e.g., temperature, top_p, etc.) as extras.
    """
    try:
        global processor
        
        # Get model ID from the required model parameter
        model_id = request.model
        
        # Check if we need to initialize or change the processor
        if processor is None or getattr(processor, 'model_id', None) != model_id:
            # Select and initialize the appropriate processor
            logging.info(f"Initializing processor for model: {model_id}")
            processor_type = select_processor_type(model_id)
            
            # Get additional model-specific parameters
            model_init_params = {k: v for k, v in request.model_dump().items() 
                               if k.startswith(('gguf__', 'transformers__'))}
            
            # Initialize the processor with model-specific parameters
            processor = processor_type(model_id=model_id, **model_init_params)
            logging.info(f"Processor initialized: {processor.__class__.__name__}")
        else:
            logging.info(f"Using existing processor for model: {model_id}")
            
        # Make sure a processor is available
        if processor is None:
            raise HTTPException(
                status_code=500, detail="Failed to initialize model processor."
            )

        # Convert the pydantic model into a dict, preserving extra fields
        request_dict = request.model_dump()

        # Convert each Message object into a dict
        messages_payload = [msg.model_dump() for msg in request.messages]

        # Known parameters that the underlying processor supports
        known_params = {
            "model_id",  # Changed from 'model' to match TextGenerationProcessor parameter
            "max_new_tokens",
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "stream",
            "n",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "logprobs",
            "echo",
            "best_of",
            "seed",
        }

        # Filter out only the known parameters (excluding "messages")
        filtered_params: Dict[str, Any] = {
            key: value
            for key, value in request_dict.items()
            if key != "messages" and key in known_params and value is not None
        }

        # Fill in defaults from CLI if not provided in the request
        for param_key, param_value in default_params.items():
            if param_key not in filtered_params and param_key != "model_id":
                filtered_params[param_key] = param_value

        # Remove model and model_id from filtered params to avoid duplicates and errors
        # IMPORTANT: model_id must NOT be included in filtered_params
        # as it will cause an error in the transformers generate method
        if "model" in filtered_params:
            del filtered_params["model"]
        
        if "model_id" in filtered_params:
            del filtered_params["model_id"]

        # If streaming is requested, return an SSE stream
        if request.stream:
            logging.info("Starting streaming response with parameters: %s", filtered_params)
            generator = processor.generate_openai_comp(messages_payload, **filtered_params)

            # Wrap each chunk into SSE "data: ..." format
            async def event_generator():
                try:
                    for chunk in generator:
                        yield f"data: {json.dumps(chunk)}\n\n"
                except Exception as gen_exc:
                    logging.exception("Error during streaming generation:")
                    # In case of an error mid-stream, send an error event
                    yield f"data: {{\"error\": \"{str(gen_exc)}\"}}\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        # Non-streaming: single JSON response
        else:
            logging.info("Generating single response with parameters: %s", filtered_params)
            output = processor.generate_openai_comp(messages_payload, **filtered_params)
            return JSONResponse(content=output)

    except HTTPException:
        raise
    except Exception as e:
        # Log the full stack trace for debugging
        logging.exception("Unhandled exception in /v1/chat/completions:")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")


def _parse_stop_list(raw: Optional[str]) -> Optional[List[str]]:
    """
    Helper to convert a comma-separated stop-list string into a Python list.
    If raw is None or empty, returns None.
    """
    if not raw:
        return None
    # Split on commas, strip whitespace
    return [token.strip() for token in raw.split(",") if token.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Start the FastAPI chat-completion server with default parameters."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"Model identifier (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature for generation (e.g., 0.7).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        dest="top_p",
        default=None,
        help="Top-p (nucleus) sampling parameter (e.g., 0.9).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        dest="top_k",
        default=None,
        help="Top-k sampling parameter (e.g., 50).",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        dest="repetition_penalty",
        default=None,
        help="Repetition penalty for generation (e.g., 1.2).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        dest="max_new_tokens",
        default=None,
        help="Maximum number of new tokens to generate if client does not specify (e.g., 256).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Number of completions to generate (e.g., 1).",
    )
    parser.add_argument(
        "--frequency-penalty",
        type=float,
        dest="frequency_penalty",
        default=None,
        help="Frequency penalty for generation (e.g., 0.5).",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        dest="presence_penalty",
        default=None,
        help="Presence penalty for generation (e.g., 0.0).",
    )
    parser.add_argument(
        "--stop",
        type=str,
        default=None,
        help="Comma-separated list of stop tokens (e.g., \"\\n, Stop\").",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (e.g., 42).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host interface to bind the server (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server (default: 8000).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Uvicorn log level (default: info).",
    )

    args = parser.parse_args()

    # Populate default_params from CLI args
    # model is always available since we set a default
    default_params["model"] = args.model

    if args.temperature is not None:
        default_params["temperature"] = args.temperature
    if args.top_p is not None:
        default_params["top_p"] = args.top_p
    if args.top_k is not None:
        default_params["top_k"] = args.top_k
    if args.repetition_penalty is not None:
        default_params["repetition_penalty"] = args.repetition_penalty
    if args.max_new_tokens is not None:
        default_params["max_new_tokens"] = args.max_new_tokens
    if args.n is not None:
        default_params["n"] = args.n
    if args.frequency_penalty is not None:
        default_params["frequency_penalty"] = args.frequency_penalty
    if args.presence_penalty is not None:
        default_params["presence_penalty"] = args.presence_penalty
    if args.seed is not None:
        default_params["seed"] = args.seed
    stop_list = _parse_stop_list(args.stop)
    if stop_list is not None:
        default_params["stop"] = stop_list

    # Initialize the processor with the chosen model
    model_id = args.model  # Use the model from args directly
    logging.info("Initializing TextGenerationProcessor with model_id: %s", model_id)
    processor_type: Type[TextGenerationProcessor] = select_processor_type(model_id)
    processor = processor_type(model_id)

    # Display summary of defaults for transparency
    logging.info("Default generation parameters: %s", {k: v for k, v in default_params.items() if k != "model"})

    print("Starting FastAPI server with auto-reload enabled:")
    print("  • Swagger UI: http://localhost:8000/docs")
    print("  • Endpoint: POST http://localhost:8000/v1/chat/completions")

    # Use the app object directly instead of an import string when running the script directly
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=True,
    )
