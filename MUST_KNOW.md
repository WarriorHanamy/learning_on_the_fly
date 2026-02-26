# MUST KNOW - LOTF Project Docker Guidelines

## Python Entry Points

### Inside Docker
```bash
# Always use this entry point for Python commands in Docker
docker uv python
```

### Outside Docker (Development)
```bash
# Use project's uv-managed Python
uv run python
# or
uv python
```

**IMPORTANT**: Never use `python` or `python3` directly. Always use `uv run python` or the appropriate entry point for the environment.

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

### 2. Always Use Bridge Mode
When building Docker images, ALWAYS specify bridge network mode explicitly:
```bash
docker build --network=bridge -t rec-lotf .
```

Never omit `--network=bridge` - this ensures consistent network behavior and respects proxy settings.

### 3. Use Tsinghua Mirror for Python Packages
In Dockerfile, ALWAYS set UV_INDEX_URL to use Tsinghua mirror for faster downloads:
```dockerfile
ARG UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV UV_INDEX_URL=$UV_INDEX_URL
```

This ensures uv uses the Tsinghua PyPI mirror for all package downloads, significantly speeding up builds in China.

## Docker Run Rules

### 1. Always Use Bridge Mode
When running Docker containers, ALWAYS specify bridge network mode:
```bash
docker run --network=bridge --gpus all rec-lotf uv run python <script>
```

### 2. GPU Access
For LOTF training, always include GPU access:
```bash
--gpus all
```

### 3. Example Commands
```bash
# Build
docker build --network=bridge -t rec-lotf .

# Run training
docker run --network=bridge --gpus all -v $(pwd):/app rec-lotf uv run python -m lotf.scripts.train_state_hovering --config configs/state_hovering.yaml

# Interactive shell
docker run --network=bridge --gpus all -it --rm rec-lotf /bin/bash
```

## Project-Specific Notes

### Python Version
- Target: Python 3.10
- Managed by: uv (NOT conda - environment.yml has been removed)

### Package Manager
- Use: uv for all Python package operations
- Lock file: uv.lock (contains all dependencies)
- Install command: `uv sync --extra cuda12`

### JAX/CUDA
- JAX version: 0.6.2 (from uv.lock)
- CUDA version: 12
- GPU required for training
- Verify GPU in container: `docker run --network=bridge --gpus all rec-lotf uv run python -c "import jax; print(jax.devices())"`

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

1. Docker build with bridge mode will automatically respect ~/.docker/config.json
2. No need to manually set -e HTTP_PROXY/-e HTTPS_PROXY flags
3. Bridge mode ensures consistent network behavior

## Common Pitfalls

### ❌ Wrong
```bash
docker build -t rec-lotf .  # Missing --network=bridge
python script.py              # Direct Python, not uv
docker run --gpus all rec-lotf python script.py  # Wrong python entry point
```

### ✅ Correct
```bash
docker build --network=bridge -t rec-lotf .
uv run python script.py
docker run --network=bridge --gpus all rec-lotf uv run python script.py
```

## Checklist Before Any Docker Operation

- [ ] Read MUST_KNOW.md
- [ ] Check ~/.docker/config.json exists
- [ ] Verify command includes `--network=bridge`
- [ ] Use correct Python entry point (uv run python or docker uv python)
- [ ] Verify GPU access if running training (--gpus all)
