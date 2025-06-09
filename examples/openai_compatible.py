#!/usr/bin/env python3
"""
openai_compatible.py
--------------------

Run **any** OwlSight text‑generation processor behind a FastAPI server that
behaves like the real OpenAI **Chat Completions** API – including streaming –
so that tools such as **Aider**, **LiteLLM**, LangChain, etc. can talk to your
_local_ model without changes.

Usage (GGUF example):

    python openai_compatible.py \
        --model unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF \
        --gguf__filename DeepSeek-R1-0528-Qwen3-8B-Q4_PK_M.gguf \
        --port 8000

The server will be reachable at:

    http://localhost:8000/v1/chat/completions
"""

# --------------------------------------------------------------------------- #
#  Imports                                                                    #
# --------------------------------------------------------------------------- #
import argparse
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Type

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from owlsight import select_processor_type
from owlsight.processors.text_generation_processors import TextGenerationProcessor

# --------------------------------------------------------------------------- #
#  Logging                                                                    #
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# --------------------------------------------------------------------------- #
#  FastAPI app                                                                #
# --------------------------------------------------------------------------- #
app = FastAPI(title="Chat Completion API", version="1.0.0")

# --------------------------------------------------------------------------- #
#  Globals                                                                    #
# --------------------------------------------------------------------------- #
SERVER_MODEL_ID: Optional[str] = None         # Filled from --model CLI arg
default_params: Dict[str, Any] = {}           # CLI‑specified generation defaults
processor: Optional[TextGenerationProcessor] = None  # Initialised at startup

# --------------------------------------------------------------------------- #
#  OpenAI‑compatibility helpers                                               #
# --------------------------------------------------------------------------- #
def _wrap_openai_chunk(
    content: str = "",
    *,
    model: str,
    index: int = 0,
    finish: bool = False,
) -> Dict[str, Any]:
    """Return **one** SSE chunk in strict OpenAI format."""
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "delta": {"content": content} if not finish else {},
                "index": index,
                "finish_reason": "stop" if finish else None,
            }
        ],
    }


def _wrap_openai_full(
    message: str,
    *,
    model: str,
    usage: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Wrap a **non‑streaming** answer in the OpenAI response envelope."""
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "message": {"role": "assistant", "content": message},
                "index": 0,
                "finish_reason": "stop",
            }
        ],
        "usage": usage
        or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _wrap_openai_error(message: str, status_code: int) -> Dict[str, Any]:
    """Generate the OpenAI error envelope."""
    return {
        "error": {
            "message": message,
            "type": "server_error" if status_code >= 500 else "invalid_request_error",
            "code": status_code,
        }
    }


# --------------------------------------------------------------------------- #
#  Pydantic models for the request body                                       #
# --------------------------------------------------------------------------- #
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    model: str                 # must match SERVER_MODEL_ID
    stream: bool = False
    max_new_tokens: int = 512

    # Allow any other OpenAI‑style parameters
    model_config = {"extra": "allow"}


# --------------------------------------------------------------------------- #
#  Main endpoint                                                              #
# --------------------------------------------------------------------------- #
@app.post("/v1/chat/completions")
async def completions(
    body: ChatCompletionRequest, raw_request: Request
):  # two params: parsed body and raw request
    """
    OpenAI‑compatible **chat completions** endpoint.

    • Supports streaming (text/event‑stream) and non‑stream replies.  
    • Accepts *all* extra parameters in the request body.  
    • Ignores but tolerates the `Authorization` header expected by clients.
    """
    try:
        global processor, SERVER_MODEL_ID

        # ------------------------------------------------------------------ #
        #  Basic validation                                                   #
        # ------------------------------------------------------------------ #
        if body.model != SERVER_MODEL_ID:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid model: '{body.model}'. "
                    f"Server is configured for model: '{SERVER_MODEL_ID}'."
                ),
            )

        if processor is None:  # should never happen after successful startup
            logging.error("Processor not initialised – startup failure.")
            raise HTTPException(
                status_code=500,
                detail="Model processor not available. Server configuration error.",
            )

        # ------------------------------------------------------------------ #
        #  OpenAI auth header (optional)                                      #
        # ------------------------------------------------------------------ #
        _ = raw_request.headers.get("authorization")  # accept & ignore

        # ------------------------------------------------------------------ #
        #  Convert request body                                              #
        # ------------------------------------------------------------------ #
        request_dict = body.model_dump()
        messages_payload = [msg.model_dump() for msg in body.messages]

        known_params = {
            "model_id",              # internal name for underlying processor
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

        # Keep only recognised generation args
        filtered_params: Dict[str, Any] = {
            k: v
            for k, v in request_dict.items()
            if k != "messages" and k in known_params and v is not None
        }

        # Fill defaults from CLI
        for k, v in default_params.items():
            if k not in filtered_params and k != "model_id":
                filtered_params[k] = v

        # Ensure we do **not** pass model/model_id twice
        filtered_params.pop("model", None)
        filtered_params.pop("model_id", None)

        # ------------------------------------------------------------------ #
        #  STREAMING mode                                                    #
        # ------------------------------------------------------------------ #
        if body.stream:
            logging.info("Streaming generation with params: %s", filtered_params)
            generator = processor.generate_openai_comp(messages_payload, **filtered_params)

            async def event_generator():
                try:
                    for token in generator:
                        payload = _wrap_openai_chunk(token, model=SERVER_MODEL_ID)
                        yield f"data: {json.dumps(payload)}\n\n"

                    # final stop chunk
                    stop_payload = _wrap_openai_chunk(model=SERVER_MODEL_ID, finish=True)
                    yield f"data: {json.dumps(stop_payload)}\n\n"

                    # OpenAI sentinel
                    yield "data: [DONE]\n\n"
                except Exception as gen_exc:
                    logging.exception("Error during streaming generation:")
                    error_event = _wrap_openai_error(str(gen_exc), 500)
                    yield f"data: {json.dumps(error_event)}\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        # ------------------------------------------------------------------ #
        #  NON‑streaming mode                                                #
        # ------------------------------------------------------------------ #
        logging.info("Generating single response with params: %s", filtered_params)
        text_out = processor.generate_openai_comp(messages_payload, **filtered_params)
        wrapped = _wrap_openai_full(text_out, model=SERVER_MODEL_ID)
        return JSONResponse(content=wrapped)

    except HTTPException:
        raise  # handled by our custom handler below
    except Exception as exc:
        # Log full traceback
        logging.exception("Unhandled exception in /v1/chat/completions:")
        # Re‑raise as HTTPException so FastAPI handler can wrap it
        raise HTTPException(
            status_code=500, detail=f"An internal error occurred: {str(exc)}"
        )


# --------------------------------------------------------------------------- #
#  Global OpenAI‑style error handler                                          #
# --------------------------------------------------------------------------- #
@app.exception_handler(HTTPException)
async def openai_http_exception_handler(request: Request, exc: HTTPException):
    """Return errors using the official OpenAI error envelope."""
    return JSONResponse(
        status_code=exc.status_code,
        content=_wrap_openai_error(str(exc.detail), exc.status_code),
    )


# --------------------------------------------------------------------------- #
#  Helper: parse comma‑separated stop list                                    #
# --------------------------------------------------------------------------- #
def _parse_stop_list(raw: Optional[str]) -> Optional[List[str]]:
    return [tok.strip() for tok in raw.split(",") if tok.strip()] if raw else None


# --------------------------------------------------------------------------- #
#  Main (CLI)                                                                 #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Start the FastAPI chat‑completion server.\n"
            "Example GGUF launch:\n"
            "  python openai_compatible.py --model unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF "
            "--gguf__filename DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf --port 8000"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # --- required --------------------------------------------------------- #
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model identifier (HF hub name, local path, etc.).",
    )

    # --- optional generation defaults ------------------------------------- #
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", dest="top_p", type=float, default=None)
    parser.add_argument("--top-k", dest="top_k", type=int, default=None)
    parser.add_argument("--repetition-penalty", dest="repetition_penalty", type=float, default=None)
    parser.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--frequency-penalty", dest="frequency_penalty", type=float, default=None)
    parser.add_argument("--presence-penalty", dest="presence_penalty", type=float, default=None)
    parser.add_argument("--stop", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)

    # --- server params ----------------------------------------------------- #
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
    )

    # Parse known vs unknown CLI args
    args, unknown_args = parser.parse_known_args()

    # --------------------------------------------------------------------- #
    #  Build default generation params from CLI                            #
    # --------------------------------------------------------------------- #
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

    # --------------------------------------------------------------------- #
    #  Collect unknown args for model initialisation                       #
    # --------------------------------------------------------------------- #
    def _parse_arg_value(val: str) -> Any:
        lower = val.lower()
        if lower == "true":
            return True
        if lower == "false":
            return False
        try:
            return int(val)
        except ValueError:
            try:
                return float(val)
            except ValueError:
                return val

    model_init_cli_params: Dict[str, Any] = {}
    idx, num_args = 0, len(unknown_args)

    while idx < num_args:
        token = unknown_args[idx]
        if not token.startswith("--"):
            logging.warning(f"Ignoring unrecognised token: {token}")
            idx += 1
            continue

        key = token[2:]
        if idx + 1 < num_args and not unknown_args[idx + 1].startswith("--"):
            model_init_cli_params[key] = _parse_arg_value(unknown_args[idx + 1])
            idx += 2
        else:
            # flag without value
            model_init_cli_params[key] = True
            idx += 1

    # --------------------------------------------------------------------- #
    #  Initialise processor                                                 #
    # --------------------------------------------------------------------- #
    SERVER_MODEL_ID = args.model
    logging.info(
        "Initialising TextGenerationProcessor for model '%s' with params %s",
        SERVER_MODEL_ID,
        model_init_cli_params,
    )
    try:
        processor_type: Type[TextGenerationProcessor] = select_processor_type(SERVER_MODEL_ID)
        processor = processor_type(model_id=SERVER_MODEL_ID, **model_init_cli_params)
        logging.info("Processor '%s' initialised successfully.", processor.__class__.__name__)
    except Exception as exc:
        logging.error("Fatal: Failed to initialise processor.", exc_info=True)
        print("\nError: Could not initialise the model processor.")
        print(f"Model ID: {SERVER_MODEL_ID}")
        print(f"Init params: {model_init_cli_params}")
        print(f"Details: {exc}\n")
        exit(1)

    # --------------------------------------------------------------------- #
    #  Startup banner                                                       #
    # --------------------------------------------------------------------- #
    logging.info("Server configured for model: %s", SERVER_MODEL_ID)
    if default_params:
        logging.info("Default generation parameters: %s", default_params)

    print("\n🚀  Starting FastAPI Chat‑Completion server")
    print(f"   • Model: {SERVER_MODEL_ID}")
    print(f"   • Listen: http://{args.host}:{args.port}")
    print(f"   • Swagger: http://{args.host}:{args.port}/docs")
    print("   • Endpoint: POST /v1/chat/completions\n")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
    )
