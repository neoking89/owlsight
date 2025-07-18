import pytest
import importlib.util

# Check if fastapi and uvicorn are installed
fastapi_installed = importlib.util.find_spec("fastapi") is not None
uvicorn_installed = importlib.util.find_spec("uvicorn") is not None

# If either dependency is missing, mark the entire module to be skipped
if not (fastapi_installed and uvicorn_installed):
    missing_deps = []
    if not fastapi_installed:
        missing_deps.append("fastapi")
    if not uvicorn_installed:
        missing_deps.append("uvicorn")
    
    # Create a reason string
    reason = f"Skipping server tests because {', '.join(missing_deps)} {'is' if len(missing_deps) == 1 else 'are'} not installed."
    
    # Apply the skip to the entire module at import time
    pytest.skip(reason, allow_module_level=True)
