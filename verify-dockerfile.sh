#!/bin/bash
# Quick verification script for Dockerfile

echo "=== Dockerfile Verification ==="
echo ""

# Check if Dockerfile exists
if [ -f "Dockerfile" ]; then
    echo "✓ Dockerfile exists"
else
    echo "✗ Dockerfile not found"
    exit 1
fi

# Check base image
if grep -q "FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04" Dockerfile; then
    echo "✓ Base image: nvidia/cuda:12.4.0-runtime-ubuntu22.04"
else
    echo "✗ Base image not found or incorrect"
fi

# Check Python 3.10
if grep -q "python3.10" Dockerfile; then
    echo "✓ Python 3.10 specified"
else
    echo "✗ Python 3.10 not found"
fi

# Check uv installation
if grep -q "https://astral.sh/uv/install.sh" Dockerfile; then
    echo "✓ uv installation script found"
else
    echo "✗ uv installation not found"
fi

# Check uv sync with cuda12
if grep -q "uv sync.*--extra cuda12" Dockerfile; then
    echo "✓ uv sync --extra cuda12 configured"
else
    echo "✗ uv sync --extra cuda12 not found"
fi

# Check .dockerignore exists
if [ -f ".dockerignore" ]; then
    echo "✓ .dockerignore exists"
else
    echo "✗ .dockerignore not found"
fi

# Check working directory
if grep -q "WORKDIR /app" Dockerfile; then
    echo "✓ Working directory set to /app"
else
    echo "✗ Working directory not set"
fi

# Check entry point
if grep -q "ENTRYPOINT \[\"python\", \"-m\", \"lotf\"\]" Dockerfile; then
    echo "✓ Entry point set to python -m lotf"
else
    echo "✗ Entry point not set"
fi

echo ""
echo "=== Build Instructions ==="
echo "To build the image:"
echo "  docker build -t rec-lotf ."
echo ""
echo "To run with GPU:"
echo "  docker run --gpus all -it rec-lotf"
echo ""
echo "Documentation: See docker/README.md"
