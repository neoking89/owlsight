# src/owlsight/server/app.py
"""
FastAPI application for the OpenAI-compatible server.
"""
import logging
import time
import uuid
import json
from typing import Any, Dict, List, Optional, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from owlsight.processors.text_generation_processors import TextGenerationProcessor
from .api_models import ChatCompletionRequest, ModelCard, ModelList, ModelPermission

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Owlsight OpenAI-Compatible Chat Completion API",
    version="1.0.0",
    description="Run any owlsight text-generation processor behind an OpenAI-compatible API."
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables to be populated by main.py
SERVER_MODEL_ID: Optional[str] = None
processor: Optional[TextGenerationProcessor] = None
default_params: Dict[str, Any] = {}
START_TIME: int = int(time.time())
# Configuration for verbose output
VERBOSE: bool = False

# Helper functions (adapted from examples/openai_compatible.py)
def _wrap_openai_chunk(
    content: str = "",
    *,
    model: str,
    index: int = 0,
    finish: bool = False,
    first: bool = False,
) -> Dict[str, Any]:
    delta: Dict[str, str] = {}
    if first:
        delta["role"] = "assistant"
    if content:
        delta["content"] = content
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
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

def _wrap_openai_error(message: str, status_code: int, type: str = "invalid_request_error") -> Dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": type,
            "param": None,
            "code": None,
        }
    }

@app.exception_handler(HTTPException)
async def openai_http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=_wrap_openai_error(exc.detail, exc.status_code),
    )

# API Endpoints
@app.post("/v1/chat/completions")
async def completions(body: ChatCompletionRequest, raw_request: Request):
    global SERVER_MODEL_ID, processor, default_params

    if processor is None or SERVER_MODEL_ID is None:
        logger.error("Server not properly initialized. Processor or SERVER_MODEL_ID is None.")
        raise HTTPException(status_code=500, detail="Server not initialized")

    if body.model != SERVER_MODEL_ID:
        # Also allow if the model is just the last part of SERVER_MODEL_ID (e.g. user/model-name vs model-name)
        if not SERVER_MODEL_ID.endswith(f"/{body.model}"):
            logger.warning(f"Request model '{body.model}' does not match server model '{SERVER_MODEL_ID}'.")
            raise HTTPException(
                status_code=400,
                detail=f"This server is configured to use the model '{SERVER_MODEL_ID}', "
                       f"but the request specified '{body.model}'.",
            )

    # Prepare input for owlsight processor
    # The example script uses a simple concatenation of messages.
    # For a more robust solution, this should align with how the specific
    # owlsight processor expects its input (e.g., using its template).
    # For now, let's keep it simple as in the example.
    prompt_parts = []
    for msg in body.messages:
        prompt_parts.append(f"{msg.role}: {msg.content}")
    input_text = "\n".join(prompt_parts) + "\nassistant:"

    # Consolidate generation parameters
    gen_params = default_params.copy()
    if body.max_new_tokens is not None:
        gen_params["max_new_tokens"] = body.max_new_tokens
    if body.temperature is not None:
        gen_params["temperature"] = body.temperature
    if body.top_p is not None:
        gen_params["top_p"] = body.top_p
    if body.stop is not None:
        gen_params["stop"] = body.stop
    
    # Add any extra parameters from the request
    extra_request_params = body.model_dump(exclude_unset=True, by_alias=True)
    for key in ["messages", "model", "stream", "max_tokens", "stop", "temperature", "top_p"]:
        extra_request_params.pop(key, None)
    gen_params.update(extra_request_params)

    logger.info(f"Processing request for model '{body.model}' with params: {gen_params}")
    
    # In verbose mode, log the full prompt
    if VERBOSE:
        logger.debug("\nINPUT PROMPT:")
        logger.debug("-" * 40)
        logger.debug(input_text)
        logger.debug("-" * 40)

    if body.stream:
        async def stream_generator() -> AsyncGenerator[str, None]:
            try:
                first_chunk = True
                full_response_content = ""
                if VERBOSE:
                    print(f"\n{SERVER_MODEL_ID} response (stream): ", end="", flush=True)
                    
                for chunk in processor.generate_stream(input_text, **gen_params):
                    if await raw_request.is_disconnected():
                        logger.info("Client disconnected during streaming.")
                        break 
                    openai_chunk = _wrap_openai_chunk(
                        content=chunk,
                        model=SERVER_MODEL_ID, # Use the server's model ID
                        first=first_chunk,
                    )
                    yield f"data: {json.dumps(openai_chunk)}\n\n"
                    full_response_content += chunk
                    
                    # Print token by token in verbose mode
                    if VERBOSE:
                        print(chunk, end="", flush=True)
                        
                    if first_chunk:
                        first_chunk = False
                
                # Send final chunk with finish_reason
                final_chunk_data = _wrap_openai_chunk(model=SERVER_MODEL_ID, finish=True)
                yield f"data: {json.dumps(final_chunk_data)}\n\n"
                yield "data: [DONE]\n\n"
                
                if VERBOSE:
                    print("\n\nFull response:") 
                    print("-" * 40)
                    print(full_response_content)
                    print("-" * 40)
                    
                # Always log completion, but limit size in non-verbose mode
                if VERBOSE:
                    logger.info("Stream completed with full response logged above.")
                else:
                    logger.info(f"Stream completed. Full response: {full_response_content[:200]}...")
            except Exception as e:
                logger.error(f"Error during streaming: {e}", exc_info=True)
                error_chunk = _wrap_openai_chunk(model=SERVER_MODEL_ID, finish=True) # Indicate finish
                # How to send error in SSE? OpenAI just closes connection or sends [DONE]
                # For now, log and send a final chunk. Client might need to handle this.
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        try:
            response_text = processor.generate(input_text, **gen_params)
            
            # Log the full response in verbose mode
            if VERBOSE:
                print("\n\nFull model response:")
                print("-" * 40)
                print(response_text)
                print("-" * 40)
                logger.info("Non-streamed generation completed with full response logged above.")
            else:
                logger.info(f"Generated non-streamed response: {response_text[:200]}...")
            completion_data = {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": SERVER_MODEL_ID, # Use the server's model ID
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_text,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    # Token counts are not easily available from the generic processor
                    # and would require model-specific logic or tokenizer access.
                    "prompt_tokens": 0, # Placeholder
                    "completion_tokens": 0, # Placeholder
                    "total_tokens": 0, # Placeholder
                },
            }
            return JSONResponse(content=completion_data)
        except Exception as e:
            logger.error(f"Error during non-streaming generation: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models", response_model=ModelList)
async def list_models():
    global SERVER_MODEL_ID, START_TIME
    if SERVER_MODEL_ID is None:
        # This case should ideally not happen if main.py initializes correctly
        logger.error("SERVER_MODEL_ID not set in list_models")
        return ModelList(data=[]) 

    model_card = ModelCard(
        id=SERVER_MODEL_ID,
        created=START_TIME,
        permission=[
            ModelPermission(id=f"modelperm-{uuid.uuid4().hex}", created=START_TIME)
        ]
    )
    return ModelList(data=[model_card])

@app.get("/v1/models/{model_id}", response_model=ModelCard)
async def retrieve_model(model_id: str):
    global SERVER_MODEL_ID, START_TIME
    if SERVER_MODEL_ID is None:
        logger.error("SERVER_MODEL_ID not set in retrieve_model")
        raise HTTPException(status_code=500, detail="Server model not configured")

    # Allow if the requested model_id is the server's model or just the name part
    is_match = (model_id == SERVER_MODEL_ID) or (SERVER_MODEL_ID.endswith(f"/{model_id}"))

    if not is_match:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found. This server serves: '{SERVER_MODEL_ID}'."
        )
    
    return ModelCard(
        id=SERVER_MODEL_ID,
        created=START_TIME,
        permission=[
            ModelPermission(id=f"modelperm-{uuid.uuid4().hex}", created=START_TIME)
        ]
    )
