# src/owlsight/server/api_models.py
"""
Defines the Pydantic models for the OpenAI-compatible API.
"""
from typing import Any, List, Optional
from pydantic import BaseModel, Field

class Message(BaseModel):
    """Single message in the chat conversation."""
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    """Subset of the OpenAI *Chat Completions* request payload.

    Only the fields required by this lightweight proxy are enforced. All
    additional keys are accepted and forwarded to the underlying processor
    thanks to Pydantic's ``model_config = {"extra": "allow"}``.

    Attributes
    ----------
    messages : list[Message]
        Full conversation history (system, user, assistant).
    model : str
        Name of the model. *Must* match the model ID configured at server start-up.
    stream : bool, default ``False``
        If *True* the response is returned as an HTTP *event-stream* (Server Sent
        Events) compatible with the OpenAI SDK's streaming mode.
    stop : str | list[str] | None
        Generation will halt if the model emits any of the stop sequences.
    max_new_tokens : int, default 2048
        Maximum number of tokens to sample.
    temperature : float | None, default None
        Sampling temperature.
    top_p : float | None, default None
        Nucleus sampling parameter.
    """
    messages: List[Message]
    model: str
    stream: bool = False
    max_new_tokens: int = Field(default=2048, alias="max_tokens") # OpenAI uses max_tokens
    stop: Optional[Any] = None  # Using Any to match example, can be List[str] or str
    temperature: Optional[float] = None
    top_p: Optional[float] = None

    model_config = {"extra": "allow", "populate_by_name": True}

class ModelPermission(BaseModel):
    id: str
    object: str = "model_permission"
    created: int
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: bool = False

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "owlsight"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: List[ModelPermission] = Field(default_factory=list)

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)
