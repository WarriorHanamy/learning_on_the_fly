# MUST KNOW - LOTF Project Docker Guidelines

## Python Entry Points

### Inside Docker
```bash
# Always use this entry point for Python commands in Docker
uv run python
```

### Outside Docker (Development)
```bash
# Use project's uv-managed Python
uv run python
# or
uv python
```

**IMPORTANT**: Never use `python` or `python3` directly. Always use `uv run python` for the environment.

## Docker Build Rules

### 1. Always Check Docker Config First
Before building any Docker image, ALWAYS check the Docker configuration:
```bash
cat ~/.docker/config.json
```

Current configuration shows:
- Proxy: http://172.17.0.1:7890
- MTU: 1350
- Auth tokens for nvcr.io registry

### 2. Use Tsinghua Mirror for Python Packages
In Dockerfile, ALWAYS set UV_INDEX_URL to use Tsinghua mirror for faster downloads:
```dockerfile
ARG UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV UV_INDEX_URL=$UV_INDEX_URL
```

This ensures uv uses the Tsinghua PyPI mirror for all package downloads, significantly speeding up builds in China.

### 3. Explicitly Pass ARG Variables in Build and Run
When Dockerfile defines an ARG (like UV_INDEX_URL), ALWAYS pass it explicitly in build and run commands:

**Build Command**:
```bash
docker build --build-arg UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple -t rec-lotf .
```

**Run Command**:
```bash
docker run -e UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple --gpus all rec-lotf uv run python <script>
```

**Complete Example**:
```bash
# Build with explicit ARG
docker build --build-arg UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple -t rec-lotf .

# Run with same ARG (for consistency)
docker run -e UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple --gpus all -v $(pwd):/app rec-lotf uv run python -m lotf.scripts.train_state_hovering --config configs/state_hovering.yaml
```

**Rules**:
- Always use `--build-arg <ARG_NAME>=<value>` for build
- Always use `-e <ARG_NAME>=<value>` for run
- Match ARG values between build and run for consistency
- This prevents unexpected behavior from default values

### 4. Avoid Batch Installation with uv sync

**Problem**: `uv sync` installs all dependencies in one batch, including huge CUDA packages (500MB+), making it difficult to control and monitor.

**Solution**: Split dependency installation into multiple RUN layers:
```dockerfile
# Step 1: Core dependencies (numpy, scipy, pandas, etc.)
RUN uv pip install numpy==1.26.4 scipy==1.13.1 pandas pyyaml

# Step 2: JAX/Flax core (without CUDA)
RUN uv pip install jax==0.6.2 jaxlib==0.6.2 flax==0.8.5 optax==0.2.4

# Step 3: CUDA extras (heavy packages, separate layer)
RUN uv pip install jax-cuda12-pjrt==0.6.2 jax-cuda12-plugin==0.6.2

# Step 4: Remaining dependencies
RUN uv pip install jax-dataclasses casadi dm-tree ml-dtypes
```

**Benefits**:
- Each layer is independently cached
- Progress is visible step-by-step
- Can retry individual layers if they fail
- Build time is more predictable
- Heavy CUDA packages are isolated in their own layer

**Current Dockerfile Strategy**:
- Stage 1 (base): System dependencies + uv
- Stage 2 (core-deps): Split Python deps into 4 layers for better control
- Stage 3 (app): Copy application code

**Build Time**: Expect 6-8 minutes for first build (verified: ~7 minutes total)

## Docker Run Rules

### 1. GPU Access
For LOTF training, always include GPU access:
```bash
--gpus all
```

### 2. Example Commands
```bash
# Build
docker build --build-arg UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple -t rec-lotf .

# Run training
docker run -e UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple --gpus all -v $(pwd):/app rec-lotf uv run python -m lotf.scripts.train_state_hovering --config configs/state_hovering.yaml

# Interactive shell
docker run -e UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple --gpus all -it --rm rec-lotf /bin/bash
```

## Project-Specific Notes

### Python Version
- Target: Python 3.10
- Managed by: uv (NOT conda - environment.yml has been removed)

### Package Manager
- Use: uv for all Python package operations
- Lock file: uv.lock (contains all dependencies)
- Install command: Layered `uv pip install` instead of single `uv sync`

### JAX/CUDA
- JAX version: 0.6.2 (from uv.lock)
- CUDA version: 12
- GPU required for training
- Verify GPU in container: `docker run --gpus all rec-lotf uv run python -c "import jax; print(jax.devices())"`

### Training Commands
```bash
# State hovering training
uv run python -m lotf.scripts.train_state_hovering --config configs/state_hovering.yaml

# Residual dynamics training
uv run python -m lotf.scripts.train_residual --dataset examples/residual_dynamics/example_dataset.csv

# Trajectory tracking training
uv run python -m lotf.scripts.train_traj_tracking --config configs/traj_tracking.yaml
```

## Proxy Considerations

The Docker config shows proxy settings (http://172.17.0.1:7890). When building or running containers:

1. Docker build will automatically respect ~/.docker/config.json proxy settings
2. No need to manually set -e HTTP_PROXY/-e HTTPS_PROXY flags
3. Default network mode will use proxy from config

## Common Pitfalls

### ❌ Wrong
```bash
docker build -t rec-lotf .                          # Missing --build-arg UV_INDEX_URL
docker run --gpus all rec-lotf python script.py  # Wrong python entry point, missing -e UV_INDEX_URL
python script.py                                   # Direct Python, not uv
```

### ✅ Correct
```bash
docker build --build-arg UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple -t rec-lotf .
docker run -e UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple --gpus all rec-lotf uv run python script.py
uv run python script.py
```

## Checklist Before Any Docker Operation

- [ ] Read MUST_KNOW.md
- [ ] Check ~/.docker/config.json exists
- [ ] For build: Add `--build-arg UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`
- [ ] For run: Add `-e UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`
- [ ] Use correct Python entry point (uv run python)
- [ ] Verify GPU access if running training (--gpus all)
