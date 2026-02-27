# Docker Build Guide

This document describes the optimized Docker build process for the Learning on the Fly (LOTF) project, including cache mounting strategies, build performance optimizations, and troubleshooting guidance.

## Table of Contents

- [Overview](#overview)
- [Cache Mounting Strategy](#cache-mounting-strategy)
- [Build Performance Optimizations](#build-performance-optimizations)
- [Using build-docker.sh](#using-build-dockersh)
- [Build Stages Explained](#build-stages-explained)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

## Overview

The LOTF project uses a highly optimized multi-stage Docker build process designed for:

- **Fast incremental builds** - Only rebuild layers that changed
- **Efficient dependency caching** - Avoid re-downloading Python packages
- **Layer separation** - Split heavy dependencies (CUDA) from core dependencies
- **BuildKit integration** - Leverage Docker BuildKit's advanced caching features

### Key Components

- `Dockerfile` - Multi-stage build definition with cache mounts
- `build-docker.sh` - Build script with BuildKit configuration
- `.dockerignore` - Excludes unnecessary files from build context

## Cache Mounting Strategy

The build process uses Docker BuildKit's cache mount feature to avoid re-downloading packages on every build.

### How It Works

1. **Host Cache Directory**: `/home/rec/.cache/uv`
   - Persistent cache directory on the host machine
   - Shared across multiple builds

2. **Container Cache Directory**: `/root/.cache/uv`
   - Temporary mount point inside the container during build
   - Cache is persisted back to host directory

3. **Cache Mount Syntax** (in Dockerfile):
   ```dockerfile
   RUN --mount=type=cache,target=/root/.cache/uv uv pip install <package>
   ```

### Benefits

| Benefit | Description |
|---------|-------------|
| **Faster Builds** | Skip package downloads on subsequent builds |
| **Bandwidth Savings** | Download packages only once across builds |
| **Reproducibility** | Same package versions cached across builds |
| **Incremental Development** | Quick rebuilds during development cycle |
| **Offline Builds** | Build from cache when network is unavailable |

### Cache Layer Structure

The Dockerfile splits dependencies into multiple cached layers for optimal performance:

```
Layer 1: Core Python dependencies (numpy, scipy, pandas, etc.)
Layer 2: JAX and Flax core (without CUDA)
Layer 3: CUDA extras (heavy packages, separate layer)
Layer 4: Remaining project dependencies (casadi, dm-tree, etc.)
```

This separation allows independent caching and faster rebuilds when only specific dependencies change.

## Build Performance Optimizations

### 1. Multi-Stage Build

The Dockerfile uses three stages:

- **base**: System dependencies and uv installation
- **core-deps**: Python dependencies with cache mounts
- **app**: Application code and final image

This reduces final image size by not including build artifacts.

### 2. BuildKit Inline Cache

The build script enables inline caching:

```bash
DOCKER_BUILDKIT=1 docker build \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    ...
```

This allows Docker to use previous build layers as cache sources.

### 3. Layer Ordering

Dependencies are ordered from least likely to change to most likely:

1. System dependencies (rarely change)
2. uv installation (rarely change)
3. Core Python packages (sometimes change)
4. CUDA packages (rarely change)
5. Application code (frequently changes)

### 4. Tsinghua Mirror

The build uses the Tsinghua University PyPI mirror for faster downloads in China:

```dockerfile
ARG UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV UV_INDEX_URL=$UV_INDEX_URL
```

## Using build-docker.sh

The `build-docker.sh` script provides a convenient wrapper for building the Docker image with optimal settings.

### Prerequisites

- Docker (>= 20.10) with BuildKit support
- NVIDIA Docker runtime (nvidia-container-toolkit)
- Sufficient disk space for cache (~2-3 GB)

### Basic Usage

```bash
# Build with default settings (tag: lotf:latest)
./build-docker.sh

# Build with custom tag
IMAGE_TAG=v1.0.0 ./build-docker.sh
```

### Script Features

- **Automatic dependency checking** - Verifies Docker is installed
- **Cache directory validation** - Warns if host cache doesn't exist
- **BuildKit activation** - Enables DOCKER_BUILDKIT=1 automatically
- **Error handling** - Provides clear error messages and exit codes
- **Logging** - Color-coded output (INFO, WARN, ERROR)

### Script Configuration

The script defines these configurable variables:

```bash
IMAGE_NAME="lotf"                      # Docker image name
IMAGE_TAG="${IMAGE_TAG:-latest}"       # Docker image tag (default: latest)
HOST_CACHE_DIR="/home/rec/.cache/uv"   # Host cache directory
CONTAINER_CACHE_DIR="/root/.cache/uv"  # Container cache directory
```

### Expected Output

```
[INFO] Building Docker image: lotf:latest
[INFO] Using DOCKER_BUILDKIT=1 for optimal build performance
[INFO] Mounting host cache from: /home/rec/.cache/uv
...
[INFO] Build completed successfully: lotf:latest
```

### Build Without Script

If you need to build manually without the script:

```bash
# Enable BuildKit
DOCKER_BUILDKIT=1

# Build with inline cache
docker build \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    --tag lotf:latest \
    --file Dockerfile \
    .
```

## Build Stages Explained

### Stage 0: Base System Dependencies

```dockerfile
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS base
```

**Purpose**: Install minimal system dependencies

**Includes**:
- Python 3.10 and venv
- Development headers
- curl for downloading uv

**Caching**: System packages are cached by Docker's layer cache

### Stage 1: Core Python Dependencies

```dockerfile
FROM base AS core-deps

COPY pyproject.toml ./

# Core dependencies without CUDA
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --index-url $UV_INDEX_URL \
    numpy==1.26.4 \
    scipy==1.13.1 \
    ...
```

**Purpose**: Install Python packages with cache mount

**Caching**: Uses BuildKit cache mount for uv package cache

**Optimizations**:
- Split into 4 separate RUN commands for fine-grained caching
- Core deps (layer 1), JAX/Flax (layer 2), CUDA extras (layer 3), other deps (layer 4)

### Stage 2: Application Code

```dockerfile
FROM core-deps AS app

COPY lotf/ ./lotf/
COPY configs/ ./configs/
COPY examples/ ./examples/
COPY checkpoints/ ./checkpoints/
COPY README.md ./

RUN --mount=type=cache,target=/root/.cache/uv uv pip install -e .
```

**Purpose**: Install application code in editable mode

**Caching**: Application code changes trigger this layer rebuild

## Troubleshooting

### Cache-Related Issues

#### Issue 1: Cache directory doesn't exist

**Symptom**:
```
[WARN] Host cache directory does not exist: /home/rec/.cache/uv
[WARN] Build will continue, but caching may not be optimal
```

**Solution**:
```bash
# Create cache directory manually
mkdir -p /home/rec/.cache/uv

# Or let Docker/BuildKit create it automatically during first build
```

**Impact**: Build will succeed but will be slower on first run.

#### Issue 2: Cache not being used

**Symptom**: Build always downloads packages even on subsequent builds

**Possible Causes**:
1. BuildKit not enabled
2. Cache directory path mismatch
3. Dockerfile syntax error in cache mount

**Solutions**:

```bash
# Verify BuildKit is enabled
echo $DOCKER_BUILDKIT  # Should be "1"

# Check Dockerfile for cache mount syntax
grep "type=cache" Dockerfile

# Clear Docker build cache (if corrupted)
docker builder prune -a

# Try building without cache to isolate issue
docker build --no-cache -t lotf:test .
```

#### Issue 3: Out of disk space

**Symptom**: Build fails with "no space left on device" error

**Solution**:
```bash
# Clean up Docker resources
docker system prune -a --volumes

# Clean up uv cache specifically
rm -rf /home/rec/.cache/uv/*

# Or reduce cache usage by removing unused packages
docker builder prune
```

#### Issue 4: Cache corruption

**Symptom**: Build fails with inconsistent cache errors

**Solution**:
```bash
# Clear BuildKit cache
docker builder prune -f

# Clear uv cache
rm -rf /home/rec/.cache/uv/*

# Rebuild from scratch
docker build --no-cache -t lotf:latest .
```

### Build Issues

#### Issue 5: Build fails on package download

**Symptom**: Build fails when downloading from PyPI mirror

**Possible Causes**:
- Network connectivity issues
- Mirror is down
- Package version not available

**Solutions**:

```bash
# Check network connectivity
ping pypi.tuna.tsinghua.edu.cn

# Try with default PyPI (edit Dockerfile temporarily)
# Change: ARG UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
# To: ARG UV_INDEX_URL=https://pypi.org/simple

# Or set at build time
docker build \
    --build-arg UV_INDEX_URL=https://pypi.org/simple \
    --tag lotf:latest \
    --file Dockerfile \
    .
```

#### Issue 6: CUDA package installation fails

**Symptom**: Build fails when installing jax-cuda12 packages

**Possible Causes**:
- NVIDIA driver version incompatibility
- CUDA runtime version mismatch
- GPU not available on host

**Solutions**:

```bash
# Check NVIDIA driver version
nvidia-smi

# Verify NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.4.0-runtime-ubuntu22.04 nvidia-smi

# Build without CUDA support (CPU only)
# Comment out or modify lines 61-65 in Dockerfile

# Or use CPU-only JAX
docker build \
    --build-arg CPU_ONLY=1 \
    --tag lotf:cpu \
    --file Dockerfile \
    .
```

### Script Issues

#### Issue 7: build-docker.sh permission denied

**Symptom**: `bash: ./build-docker.sh: Permission denied`

**Solution**:
```bash
# Make script executable
chmod +x build-docker.sh

# Or run with bash directly
bash build-docker.sh
```

#### Issue 8: Docker not found

**Symptom**: `[ERROR] Docker is not installed or not in PATH`

**Solution**:
```bash
# Install Docker (Ubuntu)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER

# Log out and back in for group change to take effect
```

### Verification Issues

#### Issue 9: Build succeeds but JAX not installed

**Symptom**: Image builds but `import jax` fails

**Solution**:
```bash
# Verify JAX installation in container
docker run --rm lotf:latest uv run python -c "import jax; print(jax.__version__)"

# If fails, rebuild with clean cache
docker builder prune -f
./build-docker.sh
```

## Advanced Usage

### Custom Cache Location

To use a different cache directory:

```bash
# Edit build-docker.sh or override at runtime
HOST_CACHE_DIR="/custom/cache/path" ./build-docker.sh
```

### Multi-Architecture Builds

To build for different CPU architectures:

```bash
# Build for ARM64
docker buildx build --platform linux/arm64 --tag lotf:arm64 .

# Build for AMD64
docker buildx build --platform linux/amd64 --tag lotf:amd64 .
```

### Build Arguments

The Dockerfile accepts build arguments:

```bash
# Use custom PyPI mirror
docker build \
    --build-arg UV_INDEX_URL=https://pypi.org/simple \
    --tag lotf:latest \
    .

# Note: Changing build arguments invalidates affected layers
```

### Integration with CI/CD

Example GitHub Actions workflow:

```yaml
name: Build Docker Image

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Build Docker image
        run: |
          chmod +x build-docker.sh
          ./build-docker.sh

      - name: Verify installation
        run: |
          docker run --rm lotf:latest uv run python -c "import jax; print(jax.__version__)"
```

### Cache Management

Monitor cache usage:

```bash
# Check cache directory size
du -sh /home/rec/.cache/uv

# List cached packages
ls -lh /home/rec/.cache/uv/

# Clear Docker build cache
docker builder prune -a

# Clear specific cache
docker builder prune --filter until=24h
```

## Additional Resources

- [Docker BuildKit Documentation](https://docs.docker.com/build/buildkit/)
- [Docker Cache Mounts](https://docs.docker.com/build/building/cache/)
- [uv Package Manager](https://github.com/astral-sh/uv)
- [JAX Installation Guide](https://github.com/google/jax#installation)
- [Docker Multi-Stage Builds](https://docs.docker.com/build/building/multi-stage/)

## Related Documentation

- [Docker Setup Guide](docker/README.md) - Basic Docker usage and running containers
- [Installation Guide](docs/installation.md) - Local environment setup
- [Deployment Guide](docs/deployment.md) - Docker and ROS2 integration

## Support

For issues specific to Docker builds:

1. Check the [Troubleshooting](#troubleshooting) section above
2. Review Docker logs: `docker build --progress=plain --no-cache -t lotf:test .`
3. Open a GitHub issue with build logs and environment details
