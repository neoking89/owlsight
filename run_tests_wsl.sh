#!/bin/bash

# Enable debug mode to see what commands are being executed
set -x

# Exit on any error
set -e

# Convert activate script to Unix format if needed
dos2unix .venv/Scripts/activate 2>/dev/null || true

# Activate virtual environment
source .venv/Scripts/activate

# Install all dependencies without the package itself
pip install .[all]
pip install tf-keras

# Run tests
python3 -m pytest -v