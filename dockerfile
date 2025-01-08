# Dockerfile for a CLI application
# build: docker build -t owlsight .
# run: docker run -it --rm owlsight


FROM python:3.8 AS python38

WORKDIR /app
COPY . /app

# Pre-install setuptools-scm so the environment variable is recognized
RUN pip install --no-cache-dir setuptools-scm

# Provide a fallback version, e.g., "0.1.0"
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0

RUN pip install --no-cache-dir .[all]

# Define the default command for the CLI application
CMD ["owlsight"]

# FROM python:3.10-slim as python310
# WORKDIR /app
# COPY pyproject.toml .
# RUN pip install --no-cache-dir .[all]
# COPY . .

# FROM python:3.9-slim as python39
# WORKDIR /app
# COPY pyproject.toml .
# RUN pip install --no-cache-dir .[all]
# COPY . .

# FROM python:3.8-slim as python38
# WORKDIR /app
# COPY pyproject.toml .
# RUN pip install --no-cache-dir .[all]
# COPY . .