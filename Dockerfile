FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Set metadata
LABEL maintainer="Michael Pan <michael.pan31415@gmail.com>"
LABEL description="Docker image for Learning on the Fly (LOTF) project"
LABEL version="0.1.0"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    python3-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY lotf/ ./lotf/
COPY setup.py ./
COPY README.md ./
COPY configs/ ./configs/

# Install dependencies with uv, including CUDA 12 extras
RUN uv sync --frozen --extra cuda12

# Make the venv accessible
ENV PATH="/app/.venv/bin:$PATH"

# Set default entry point
ENTRYPOINT ["python", "-m", "lotf"]
CMD ["--help"]
