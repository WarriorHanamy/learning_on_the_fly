# Core Design Philosophy

## Introduction

The `dockrun` and `dockerfile_utils` tools are designed with a philosophy of **simplicity, separation of concerns, and extensibility**. This document outlines the core design principles that guide their architecture.

---

## Core Principles

### 1. Single Responsibility Principle (SRP)

Each tool has exactly one responsibility:

- **dockrun**: Run commands in containers
- **dockerfile_utils**: Create Dockerfiles and build images

No tool does more than one thing. This ensures:
- Easy to understand
- Easy to test
- Easy to maintain
- Easy to extend

### 2. Fixed Behavior, Flexible Interface

**dockrun** has fixed behavior (non-configurable):

- Image: `lotf:latest`
- GPU: `--gpus=all`
- Volume: `-v $(pwd):/app`
- Work directory: `-w /app`
- Auto-remove: `--rm`

But its interface is completely flexible:

```bash
dockrun --non-interactive [any command]
```

This design choice means:
- **No configuration complexity** - Users don't need to remember parameters
- **No feature bloat** - No need for flags like `--gpu`, `--mount`, `--workdir`
- **Complete generality** - LLM agents can generate any detailed command
- **Predictable behavior** - Every run has the same environment

### 3. Separation of Concerns

The tools are completely independent:

```
dockrun.py          ←→  dockerfile_utils.py
     │                           │
     │                           │
   Running                  Building
   containers                images
```

**No overlap:**
- dockrun.py cannot build images
- dockerfile_utils.py cannot run containers
- No shared code between tools
- No dependencies between tools

**Benefits:**
- Can use one without the other
- Clear ownership of responsibilities
- Easier to debug issues
- Can evolve independently

### 4. LLM Agent Compatibility

The design prioritizes **LLM agent usability**:

```python
# LLM agent generates detailed, specific commands
command = [
    "dockrun", "--non-interactive",
    "uv", "run", "python", "-m", "lotf", "residual",
    "--dataset", "examples/residual_dynamics/example_dataset.csv",
    "--config", "configs/residual_dynamics.yaml",
    "--output", "checkpoints/residual_dynamics/model_2026_03_01",
    "--num-models", "5",
    "--num-epochs", "200",
    "--batch-size", "512",
    "--learning-rate", "0.005",
    "--lambda-reg", "0.001",
    "--weight-init-scale", "1.0"
]
```

**Key features for LLM agents:**
- No interactive prompts
- No ambiguous defaults
- Full control via command-line arguments
- Clear, predictable interface
- No hidden state

### 5. Ephemeral by Default

Containers are **always ephemeral** (`--rm`):

**No persistent containers**:
- Each run creates a fresh container
- Container is deleted after completion
- No state pollution between runs
- No manual cleanup needed

**Why ephemeral?**
- Reproducibility: Each run starts clean
- Isolation: No leftover state affects subsequent runs
- Simplicity: No need for container lifecycle management
- Automation: Perfect for batch jobs and CI/CD

### 6. Convention over Configuration

The tools follow conventions rather than requiring configuration:

**dockrun conventions:**
- Always use `lotf:latest` image
- Always mount `$(pwd)` to `/app`
- Always work in `/app`
- Always use all GPUs
- Always delete container after run

**Benefits:**
- No config files needed
- No environment variables needed
- No setup process
- Just works out of the box

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   User / LLM Agent                  │
│                                                     │
│  Generates detailed, specific commands                  │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                   dockrun                          │
│  - Parse command line                            │
│  - Build docker run command                       │
│  - Execute container                             │
│  - Stream logs to stdout                         │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│               Docker Container                      │
│  - Image: lotf:latest                          │
│  - GPUs: all                                   │
│  - Mount: $(pwd):/app                          │
│  - Workdir: /app                                │
│  - Auto-remove: true                             │
│                                                     │
│  ┌─────────────────────────────────────────┐       │
│  │  Training Process                    │       │
│  │  - Load data from /app                │       │
│  │  - Train model                        │       │
│  │  - Save weights to /app/checkpoints    │       │
│  └─────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
                        │              │
                        ▼              ▼
                ┌──────────┐      ┌──────────┐
                │  Logs    │      │ Weights  │
                │  stdout  │      │ Volume   │
                └──────────┘      └──────────┘
```

---

## Design Trade-offs

### Simplicity vs. Flexibility

**Choice**: Simplicity

**Trade-off**: Less flexibility in container configuration

**Reasoning**: For LOTF project, flexibility is not needed. Fixed parameters reduce cognitive load and potential errors.

### Generality vs. High-Level Abstractions

**Choice**: Generality

**Trade-off**: No high-level task shortcuts (e.g., `dockrun train residual`)

**Reasoning**: LLM agents can generate detailed commands. High-level abstractions would add complexity without significant benefit.

### Ephemeral vs. Persistent Containers

**Choice**: Ephemeral

**Trade-off**: No container reuse between runs

**Reasoning**: Ephemeral containers ensure reproducibility and isolation. Container reuse introduces state management complexity.

---

## Usage Patterns

### For LLM Agents

```bash
# Generate any detailed command
dockrun --non-interactive \
  uv run python -m lotf [task] \
  [detailed parameters...]

# Examples:
dockrun --non-interactive \
  uv run python -m lotf residual \
  --dataset examples/data.csv \
  --config configs/residual_dynamics.yaml \
  --output checkpoints/model \
  --num-models 5 \
  --num-epochs 200

dockrun --non-interactive \
  uv run python -m lotf hover \
  --config configs/state_hovering.yaml \
  --output checkpoints/policy \
  --num-envs 64
```

### For Users

```bash
# Simple version check
./dockrun --dockrun-version

# Quick command
./dockrun --non-interactive uv run python --version

# Build Dockerfile and image
./dockerfile_utils init
./dockerfile_utils build
```

---

## Extensibility

### Adding New Commands

The design makes it easy to add new commands:

```python
# In dockrun.py, the command is just passed through
# No need to modify dockrun for new lotf commands

# Example: Adding a new lotf command
# Just use it directly:
dockrun --non-interactive uv run python -m lotf new_command [args...]
```

### Modifying Fixed Behavior

If fixed behavior needs to change:

```python
# In dockrun.py, modify build_docker_run_command()
def build_docker_run_command(command_args: list[str]) -> list[str]:
    base_cmd = [
        "docker",
        "run",
        # Modify these lines as needed
        "--gpus=all",  # Change to specific GPU if needed
        "-v",
        f"{Path.cwd()}:/app",  # Change mount point if needed
        "-w",
        "/app",  # Change workdir if needed
        "--rm",  # Remove to keep containers
        "lotf:latest",  # Change image if needed
    ]
    if command_args:
        base_cmd.extend(command_args)
    return base_cmd
```

---

## Anti-Patterns to Avoid

### 1. Don't Add Task-Specific Logic

❌ Wrong:
```python
if "residual" in command:
    # Add residual-specific parameters
```

✅ Correct: Pass all arguments directly to the command

### 2. Don't Create High-Level Abstractions

❌ Wrong:
```bash
dockrun train residual --dataset data.csv
```

✅ Correct:
```bash
dockrun --non-interactive \
  uv run python -m lotf residual --dataset data.csv
```

### 3. Don't Make Fixed Parameters Configurable

❌ Wrong:
```bash
dockrun --gpu 0 --mount /data:/app --workdir /workspace ...
```

✅ Correct: Fixed parameters are implicit

---

## Conclusion

The core design philosophy of `dockrun` and `dockerfile_utils` is:

> **"Do one thing well, with fixed behavior and flexible interface."**

This design prioritizes:
- Simplicity over flexibility
- Generality over high-level abstractions
- Ephemeral over persistent
- Convention over configuration
- LLM agent compatibility over human convenience

The result is a tool that is:
- Easy to understand
- Easy to maintain
- Easy to extend
- Perfect for automation by LLM agents

---

## Related Documentation

- [README.md](../README.md) - Project overview and usage
- [DOCKER_BUILD.md](../DOCKER_BUILD.md) - Docker build process
- [docs/deployment.md](../docs/deployment.md) - Deployment guide
