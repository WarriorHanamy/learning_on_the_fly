# Optimized multi-stage Dockerfile for LOTF project
# Following MUST_KNOW.md guidelines with Tsinghua mirror and explicit ARG handling
# Splitting uv sync into multiple layers for better control and caching

# Stage 0: Base system dependencies (minimal, cached)
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS base

LABEL maintainer="Michael Pan <michael.pan31415@gmail.com>"
LABEL description="Optimized Docker image for Learning on the Fly (LOTF) project"
LABEL version="0.1.0"

ENV DEBIAN_FRONTEND=noninteractive

# UV_INDEX_URL as build arg (MUST_KNOW.md rule #3)
ARG UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV UV_INDEX_URL=$UV_INDEX_URL

# Install only essential system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install uv in a separate layer for caching
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Stage 1: Core Python dependencies (cached separately)
FROM base AS core-deps

COPY pyproject.toml setup.py ./

# Step 1: Create virtual environment and install core dependencies without CUDA
RUN uv venv .venv && \
    uv pip install --index-url $UV_INDEX_URL \
    numpy==1.26.4 \
    scipy==1.13.1 \
    pandas \
    pyyaml \
    matplotlib \
    seaborn \
    tqdm

# Step 2: Install JAX and Flax core (without CUDA)
RUN uv pip install --index-url $UV_INDEX_URL \
    jax==0.6.2 \
    jaxlib==0.6.2 \
    flax==0.8.5 \
    optax==0.2.4 \
    orbax-checkpoint==0.6.4 \
    chex==0.1.90

# Step 3: Install CUDA extras (heavy packages, separate layer)
RUN uv pip install --index-url $UV_INDEX_URL \
    jax-cuda12-pjrt==0.6.2 \
    jax-cuda12-plugin==0.6.2 \
    "jax[cuda12]>=0.6.2"

# Step 4: Install remaining project dependencies
RUN uv pip install --index-url $UV_INDEX_URL \
    jax-dataclasses==1.6.3 \
    casadi==3.7.1 \
    dm-tree==0.1.8 \
    ml-dtypes==0.5.3

# Stage 2: Application code (lightweight)
FROM core-deps AS app

# Copy application files
COPY lotf/ ./lotf/
COPY configs/ ./configs/
COPY examples/ ./examples/
COPY checkpoints/ ./checkpoints/
COPY README.md ./

# Install project in editable mode
RUN uv pip install -e .

# Verify installation
RUN uv run python --version && \
    uv run python -c "import jax; print(f'JAX version: {jax.__version__}')"

WORKDIR /app

# Default command
CMD ["uv", "run", "python", "-m", "lotf", "--help"]
