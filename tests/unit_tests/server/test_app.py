# tests/unit_tests/server/test_app.py
"""
Unit tests for the FastAPI application in src/owlsight/server/app.py.
"""
from unittest import mock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

# The app object needs to be imported from the source file
from src.owlsight.server.app import app, _wrap_openai_chunk

# Use FastAPI's TestClient to make requests to the app
client = TestClient(app)

# --- Test for _wrap_openai_chunk helper --- 
def test_wrap_openai_chunk_first():
    """Test _wrap_openai_chunk for the first chunk with role."""
    chunk = _wrap_openai_chunk(content="Hello", model="test-model", first=True)
    assert chunk["choices"][0]["delta"]["role"] == "assistant"
    assert chunk["choices"][0]["delta"]["content"] == "Hello"
    assert chunk["model"] == "test-model"
    assert chunk["object"] == "chat.completion.chunk"

def test_wrap_openai_chunk_content():
    """Test _wrap_openai_chunk for a subsequent content chunk."""
    chunk = _wrap_openai_chunk(content=" World", model="test-model")
    assert "role" not in chunk["choices"][0]["delta"]
    assert chunk["choices"][0]["delta"]["content"] == " World"
    assert chunk["choices"][0]["finish_reason"] is None

def test_wrap_openai_chunk_finish():
    """Test _wrap_openai_chunk for the final chunk with finish_reason."""
    chunk = _wrap_openai_chunk(model="test-model", finish=True)
    assert "content" not in chunk["choices"][0]["delta"]
    assert chunk["choices"][0]["finish_reason"] == "stop"

# --- Tests for /v1/models endpoint --- 
@mock.patch("src.owlsight.server.app.SERVER_MODEL_ID", "test-server-model")
@mock.patch("src.owlsight.server.app.START_TIME", 1234567890)
def test_list_models_success():
    """Test GET /v1/models when SERVER_MODEL_ID is set."""
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    model_card = data["data"][0]
    assert model_card["id"] == "test-server-model"
    assert model_card["object"] == "model"
    assert model_card["created"] == 1234567890
    assert model_card["owned_by"] == "owlsight"
    assert len(model_card["permission"]) == 1
    assert model_card["permission"][0]["created"] == 1234567890

@mock.patch("src.owlsight.server.app.SERVER_MODEL_ID", None)
def test_list_models_server_id_none():
    """Test GET /v1/models when SERVER_MODEL_ID is None."""
    response = client.get("/v1/models")
    assert response.status_code == 200 # Current behavior returns empty list
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 0

# --- Tests for /v1/models/{model_id} endpoint --- 
@mock.patch("src.owlsight.server.app.SERVER_MODEL_ID", "test-server-model")
@mock.patch("src.owlsight.server.app.START_TIME", 1234567890)
def test_retrieve_model_success():
    """Test GET /v1/models/{model_id} with a matching ID."""
    response = client.get("/v1/models/test-server-model")
    assert response.status_code == 200
    model_card = response.json()
    assert model_card["id"] == "test-server-model"
    assert model_card["object"] == "model"
    assert model_card["created"] == 1234567890

@mock.patch("src.owlsight.server.app.SERVER_MODEL_ID", "test-server-model/variant")
@mock.patch("src.owlsight.server.app.START_TIME", 1234567890)
def test_retrieve_model_suffix_match_success():
    """Test GET /v1/models/{model_id} with a matching suffix ID."""
    response = client.get("/v1/models/variant") # Requesting just the suffix
    assert response.status_code == 200
    model_card = response.json()
    assert model_card["id"] == "test-server-model/variant"

@mock.patch("src.owlsight.server.app.SERVER_MODEL_ID", "test-server-model")
def test_retrieve_model_not_found():
    """Test GET /v1/models/{model_id} with a non-matching ID."""
    response = client.get("/v1/models/wrong-model-id")
    assert response.status_code == 404
    error_data = response.json()["error"]
    assert error_data["message"] == "Model 'wrong-model-id' not found. This server serves: 'test-server-model'."

@mock.patch("src.owlsight.server.app.SERVER_MODEL_ID", None)
def test_retrieve_model_server_id_none():
    """Test GET /v1/models/{model_id} when SERVER_MODEL_ID is None."""
    response = client.get("/v1/models/any-model")
    assert response.status_code == 500
    error_data = response.json()["error"]
    assert error_data["message"] == "Server model not configured"

# --- Test for the existing openai_http_exception_handler --- 
# We need a temporary route that raises an HTTPException for this test.
@app.get("/_test-exception-route") # Changed path to avoid potential clashes
async def route_that_raises_exception():
    """A dummy route that always raises an HTTPException."""
    raise HTTPException(status_code=418, detail="I'm a teapot")

def test_openai_http_exception_handler():
    """
    Verify that when an HTTPException is raised, the custom handler:
    1. Catches the exception.
    2. Returns the correct status code.
    3. Formats the response body into the OpenAI error schema.
    """
    response = client.get("/_test-exception-route")
    assert response.status_code == 418
    expected_json = {
        "error": {
            "message": "I'm a teapot",
            "type": "invalid_request_error",
            "param": None,
            "code": None,
        }
    }
    assert response.json() == expected_json

@pytest.fixture(scope="module", autouse=True)
def manage_test_exception_route():
    """
    Manages the lifecycle of the '/_test-exception-route' for this test module.
    The route is added globally at module import time via @app.get().
    This fixture ensures it's removed once after all tests in this module complete.
    """
    test_route_path = "/_test-exception-route"
    
    # The route is already added by the @app.get decorator when the module loads.
    # We just need to ensure it's cleaned up afterwards.
    
    yield # All tests in the module run here

    # After all tests in the module have completed, remove the test route
    # by filtering app.router.routes.
    app.router.routes = [
        route for route in app.router.routes 
        if getattr(route, 'path', '') != test_route_path
    ]

