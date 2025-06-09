
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

# ---------------------------------------------------------------------- #
#  Imports                                                               #
# ---------------------------------------------------------------------- #
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

# ---------------------------------------------------------------------- #
#  Logging                                                               #
# ---------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

app = FastAPI(title="Chat Completion API", version="1.0.0")

SERVER_MODEL_ID: Optional[str] = None
processor: Optional[TextGenerationProcessor] = None
default_params: Dict[str, Any] = {}

# ---------------------------------------------------------------------- #
#  Helper: streaming chunk builder                                       #
# ---------------------------------------------------------------------- #
def _wrap_openai_chunk(
    content: str = "",
    *,
    model: str,
    index: int = 0,
    finish: bool = False,
    first: bool = False,
) -> Dict[str, Any]:
    """Wrap a chunk of text into an OpenAI-compatible chunk."""
    delta: Dict[str, str] = {}
    if first:
        delta["role"] = "assistant"
    if content:
        delta["content"] = content
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "delta": delta,
                "index": index,
                "finish_reason": "stop" if finish else None,
            }
        ],
    }

# ---------------------------------------------------------------------- #
#  Error wrapper                                                         #
# ---------------------------------------------------------------------- #
def _wrap_openai_error(message: str, status_code: int) -> Dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": "server_error" if status_code >= 500 else "invalid_request_error",
            "code": status_code,
        }
    }

# ---------------------------------------------------------------------- #
#  Pydantic request schema                                               #
# ---------------------------------------------------------------------- #
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    model: str
    stream: bool = False
    max_new_tokens: int = 2048

    model_config = {"extra": "allow"}

# ---------------------------------------------------------------------- #
#  Main endpoint                                                         #
# ---------------------------------------------------------------------- #
@app.post("/v1/chat/completions")
async def completions(body: ChatCompletionRequest, raw_request: Request):
    global processor, SERVER_MODEL_ID

    if body.model != SERVER_MODEL_ID:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{body.model}'. Server is configured for '{SERVER_MODEL_ID}'.",
        )

    if processor is None:
        raise HTTPException(status_code=500, detail="Model processor not available.")

    # Tolerate but ignore Authorization header
    _ = raw_request.headers.get("authorization")

    # Collect generation parameters
    request_dict = body.model_dump()
    messages_payload = [m.model_dump() for m in body.messages]

    known_params = {
        "max_new_tokens", "temperature", "top_p", "top_k",
        "repetition_penalty", "n", "frequency_penalty",
        "presence_penalty", "seed", "stream",
    }

    filtered_params: Dict[str, Any] = {
        k: v for k, v in request_dict.items()
        if k in known_params and v is not None
    }
    filtered_params.update({k: v for k, v in default_params.items() if k not in filtered_params})

    # ------------------------------------------------------------------ #
    #  Streaming                                                         #
    # ------------------------------------------------------------------ #
    if body.stream:
        logging.info("[stream] params=%s", filtered_params)
        token_gen = processor.generate_openai_comp(messages_payload, **filtered_params)

        async def event_generator():
            try:
                # opener
                yield f"data: {json.dumps(_wrap_openai_chunk(model=SERVER_MODEL_ID, first=True))}\n\n"
                # tokens
                for token in token_gen:   # synchronous generator
                    yield f"data: {json.dumps(_wrap_openai_chunk(token, model=SERVER_MODEL_ID))}\n\n"
                # terminator
                yield f"data: {json.dumps(_wrap_openai_chunk(model=SERVER_MODEL_ID, finish=True))}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:
                logging.exception("streaming error:")
                if not await raw_request.is_disconnected():
                    err = _wrap_openai_error(str(exc), 500)
                    yield f"data: {json.dumps(err)}\n\n"
                    yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # ------------------------------------------------------------------ #
    #  Non‑streaming                                                     #
    # ------------------------------------------------------------------ #
    logging.info("[sync] params=%s", filtered_params)
    response_json = processor.generate_openai_comp(messages_payload, **filtered_params)
    return JSONResponse(content=response_json)

# ---------------------------------------------------------------------- #
#  Error handler                                                         #
# ---------------------------------------------------------------------- #
@app.exception_handler(HTTPException)
async def openai_http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=_wrap_openai_error(str(exc.detail), exc.status_code),
    )

# ---------------------------------------------------------------------- #
#  CLI bootstrap                                                         #
# ---------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--log-level", default="info")
    # Optional default generation params
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", dest="top_p", type=float)
    parser.add_argument("--top-k", dest="top_k", type=int)
    parser.add_argument("--max-new-tokens", dest="max_new_tokens", type=int)
    parser.add_argument("--seed", type=int)
    args, unknown = parser.parse_known_args()

    global SERVER_MODEL_ID, processor, default_params
    SERVER_MODEL_ID = args.model

    # build default params
    for k in ("temperature", "top_p", "top_k", "max_new_tokens", "seed"):
        v = getattr(args, k)
        if v is not None:
            default_params[k] = v

    # unknown args are forwarded to processor init
    def _parse(val: str):
        if val.lower() == "true": return True
        if val.lower() == "false": return False
        try:
            return int(val)
        except ValueError:
            try: return float(val)
            except ValueError: return val

    init_kwargs: Dict[str, Any] = {}
    it = iter(unknown)
    for tok in it:
        if tok.startswith("--"):
            key = tok[2:]
            nxt = next(it, None)
            if nxt and not nxt.startswith("--"):
                init_kwargs[key] = _parse(nxt)
            else:
                init_kwargs[key] = True
                if nxt: it = (v for v in [nxt] + list(it))  # put back

    logging.info("initialising processor '%s' with %s", SERVER_MODEL_ID, init_kwargs)
    proc_cls: Type[TextGenerationProcessor] = select_processor_type(SERVER_MODEL_ID)
    processor = proc_cls(model_id=SERVER_MODEL_ID, **init_kwargs)

    logging.info("Server ready on http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())

if __name__ == "__main__":
    main()
