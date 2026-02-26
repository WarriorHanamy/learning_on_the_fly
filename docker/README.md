# Docker Setup for LOTF

This directory contains Docker configuration files to build and run the Learning on the Fly (LOTF) project in a containerized environment.

## Prerequisites

- Docker (>= 20.10)
- NVIDIA Docker runtime (nvidia-container-toolkit)
- CUDA-capable GPU (for CUDA support)

## Building the Image

To build the Docker image with the name `rec-lotf`:

```bash
docker build -t rec-lotf .
```

This will create a Docker image with:
- Python 3.10
- CUDA 12.4 runtime
- uv package manager
- All project dependencies (including JAX with CUDA support)

## Running the Container

### Basic Usage

To run the container with GPU support:

```bash
docker run --gpus all -it rec-lotf
```

### Interactive Shell

To access an interactive shell in the container:

```bash
docker run --gpus all -it --entrypoint /bin/bash rec-lotf
```

### Mounting Project Files

To mount your local project directory for development:

```bash
docker run --gpus all -it -v $(pwd):/app rec-lotf
```

### Running Jupyter Notebook

To run Jupyter notebooks:

```bash
docker run --gpus all -it -p 8888:8888 -v $(pwd):/app rec-lotf jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

## Running Tests

To run the project tests inside the container:

```bash
docker run --gpus all -it --entrypoint python rec-lotf -m pytest tests/
```

## Image Details

- **Base Image**: nvidia/cuda:12.4.0-runtime-ubuntu22.04
- **Python Version**: 3.10
- **CUDA Version**: 12.4
- **Package Manager**: uv
- **Working Directory**: /app
- **Default Entry Point**: python -m lotf

## Environment Variables

The container sets up the following environment variables:
- `PATH`: Includes /root/.local/bin (uv) and /app/.venv/bin (project venv)
- `DEBIAN_FRONTEND`: noninteractive (to prevent prompts during apt installs)

## Notes

- The image uses `uv sync --frozen --extra cuda12` to install dependencies, ensuring reproducible builds
- JAX with CUDA support is installed via the `cuda12` extra in pyproject.toml
- The virtual environment is located at `/app/.venv`
