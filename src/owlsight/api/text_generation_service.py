"""
FastAPI Text Generation API with optimized resource management and type safety.
This module provides a REST API for text generation using various model backends with improved:
- Resource management through model caching
- Type safety with Pydantic models
- Error handling with middleware
- Configuration management
- Memory optimization
"""

from typing import Optional, List, Dict, Any, Union, AsyncGenerator
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys

sys.path.append("src")
from owlsight.processors.base import TextGenerationProcessor
from owlsight.configurations.constants import CONFIG_DEFAULTS, CONFIG_DESCRIPTIONS
from owlsight.processors.helper_functions import select_processor_type


class GenerationRequest(BaseModel):
    """Request model for text generation with validation."""

    input_data: str = Field(..., description="Input text for generation")
    max_new_tokens: int = Field(
        default=CONFIG_DEFAULTS["generate"]["max_new_tokens"],
        description=CONFIG_DESCRIPTIONS["generate"]["max_new_tokens"],
    )
    temperature: float = Field(
        default=CONFIG_DEFAULTS["generate"]["temperature"], description=CONFIG_DESCRIPTIONS["generate"]["temperature"]
    )
    stopwords: Optional[List[str]] = Field(
        default=CONFIG_DEFAULTS["generate"]["stopwords"], description=CONFIG_DESCRIPTIONS["generate"]["stopwords"]
    )
    # stream: bool = Field(
    #     default=False, description="Stream generation output"
    # )
    generation_kwargs: Optional[Dict[str, Any]] = Field(
        default=CONFIG_DEFAULTS["generate"]["generation_kwargs"],
        description=CONFIG_DESCRIPTIONS["generate"]["generation_kwargs"],
    )


class ModelConfig(BaseModel):
    """Configuration model for loading models."""

    model_id: str = Field(..., description="Model ID or path")
    model_kwargs: Optional[Dict[str, Any]] = Field(default=None, description="Additional model configuration")


class GenerationResponse(BaseModel):
    """Response model for text generation."""

    generated_text: str


class ModelLoadResponse(BaseModel):
    """Response model for model loading."""

    status: str
    model_key: str


class ModelCache:
    """
    Cache manager for loaded models.
    Implements LRU-style caching and manages model lifecycle.
    """

    def __init__(self, max_models: int = 5):
        self.models: Dict[str, TextGenerationProcessor] = {}
        self.last_used: Dict[str, datetime] = {}
        self.max_models = max_models

    def add_model(self, key: str, model: TextGenerationProcessor) -> None:
        """Add model to cache with LRU policy."""
        if len(self.models) >= self.max_models:
            self._remove_least_used_model()
        self.models[key] = model
        self.last_used[key] = datetime.now()

    def get_model(self, key: str) -> Optional[TextGenerationProcessor]:
        """Get model from cache and update last used time."""
        if key in self.models:
            self.last_used[key] = datetime.now()
            return self.models[key]
        return None

    def remove_model(self, key: str) -> None:
        """Remove a model from the cache."""
        if key in self.models:
            del self.models[key]
            del self.last_used[key]

    def _remove_least_used_model(self) -> None:
        """Remove least recently used model to free up memory."""
        if not self.last_used:
            return
        oldest_key = min(self.last_used.items(), key=lambda x: x[1])[0]
        self.remove_model(oldest_key)


class ModelFactory:
    """Factory for creating model instances."""

    @staticmethod
    def create_model(config: ModelConfig) -> TextGenerationProcessor:
        """Create and return appropriate model instance based on config."""
        model_class = select_processor_type(config.model_id)
        return model_class(model_id=config.model_id, **(config.model_kwargs or {}))


# Initialize FastAPI app
app = FastAPI(
    title="OwlSight Text Generation API",
    description="REST API for text generation using various model backends",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model cache
model_cache = ModelCache()


@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    """Global error handling middleware."""
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})


async def get_model(model_key: str) -> TextGenerationProcessor:
    """Dependency for model retrieval."""
    model = model_cache.get_model(model_key)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_key} not found")
    return model


# async def generate_stream(model: TextGenerationProcessor, request: GenerationRequest) -> StreamingResponse:
#     """Helper function for streaming generation."""

#     async def stream() -> AsyncGenerator[str, None]:
#         try:
#             for token in model.generate_stream(
#                 input_data=request.input_data,
#                 max_new_tokens=request.max_new_tokens,
#                 temperature=request.temperature,
#                 **(request.generation_kwargs or {}),
#             ):
#                 yield f"data: {token}\n\n"
#         except Exception as e:
#             yield f"data: error: {str(e)}\n\n"

#     return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/models/load", response_model=ModelLoadResponse)
async def load_model(config: ModelConfig) -> ModelLoadResponse:
    """Load a new model into memory."""
    try:
        model = ModelFactory.create_model(config)
        model_key = config.model_id
        model_cache.add_model(model_key, model)
        return {"status": "success", "model_key": model_key}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models() -> Dict[str, List[str]]:
    """List all loaded models."""
    return {"models": list(model_cache.models.keys())}


@app.post("/generate/{model_key}", response_model=GenerationResponse)
async def generate(
    model_key: str, request: GenerationRequest, model: TextGenerationProcessor = Depends(get_model)
) -> Union[GenerationResponse, StreamingResponse]:
    """Generate text using a loaded model."""
    try:
        # if request.stream:
        #     return await generate_stream(model, request)

        response = model.generate(
            input_data=request.input_data,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            stopwords=request.stopwords,
            **(request.generation_kwargs or {}),
        )
        return {"generated_text": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/models/{model_key}")
async def unload_model(model_key: str, background_tasks: BackgroundTasks) -> Dict[str, str]:
    """Unload a model from memory."""
    if model_key not in model_cache.models:
        raise HTTPException(status_code=404, detail=f"Model {model_key} not found")

    try:
        background_tasks.add_task(model_cache.remove_model, model_key)
        return {"status": "success", "message": f"Model {model_key} unloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
