# Design Decisions

This document records key design decisions made during LOTF development to provide context for future contributors.

---

## Decision 1: python_exec Wrapper - Single Responsibility

**Date:** 2026-02-26
**Status:** ✅ Implemented

### Problem

The original `python_exec` wrapper violated the **Single Responsibility Principle** by attempting to manage multiple concerns:

```bash
# Original implementation (multi-responsibility)
setup_envs() {
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    
    if [ ! -d ".venv" ]; then
        echo "Virtual environment not found. Setting up..."
        uv sync --extra cuda12  # ❌ Should not auto-install
    fi
}

cleanup_envs() {
    unset PYTHONPATH
}

trap cleanup_envs EXIT
setup_envs
uv run python "$@"
```

**Issues:**
1. Auto-ran `uv sync` causing permission errors
2. Violated single responsibility (env setup + package management)
3. Confusing behavior - sometimes installs, sometimes doesn't
4. Permission denied errors when updating `.venv`:
   ```
   ValueError: Destination /home/rec/learning_on_the_fly/checkpoints/policy/state_hovering_params already exists.
   ```

### Solution

Simplified `python_exec` to have a **single responsibility**:

```bash
#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Set PYTHONPATH to project root
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Execute with uv-managed environment
uv run python "$@"
```

**New responsibilities:**
- ✅ Set `PYTHONPATH` to project root
- ✅ Execute `uv run python` for dependency management

**No longer handles:**
- ❌ Checking if `.venv` exists
- ❌ Auto-running `uv sync`
- ❌ Creating virtual environments
- ❌ Cleaning up environment variables

### User Workflow

```bash
# One-time setup (manual, explicit)
git clone https://github.com/WarriorHanamy/learning_on_the_fly.git
cd learning_on_the_fly
uv sync --extra cuda12

# Daily development (python_exec handles environment)
./bin/python_exec -m lotf hover
./bin/python_exec -m lotf track
./bin/python_exec train.py
```

### Benefits

1. **Explicit control**: Users control when to install dependencies
2. **No permission errors**: `python_exec` never modifies `.venv`
3. **Clear responsibility**: Wrapper only sets environment, not package management
4. **Predictable behavior**: Same behavior every time
5. **Single Responsibility Principle**: One clear purpose

---

## Decision 2: Remove Redundant setup.py

**Date:** 2026-02-26
**Status:** ✅ Implemented

### Problem

The project had both modern `pyproject.toml` (PEP 621) and legacy `setup.py`:

**pyproject.toml (modern, complete):**
```toml
[project]
name = "lotf"
version = "0.1.0"
description = "A Python package to learn agile flight using differentiable simulation."
dependencies = [
    "jax>=0.4.30",
    "jaxlib>=0.4.30",
    "flax>=0.8.5",
    # ... full dependency list
]

[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project.scripts]
lotf = "lotf.__main__:main"
```

**setup.py (legacy, redundant):**
```python
from setuptools import setup, find_packages

setup(
    name="lotf",
    version="0.1.0",
    description="A Python package to learn agile flight using differentiable simulation.",
    author="Michael Pan",
    author_email="michael.pan31415@gmail.com",
    packages=find_packages(),
    install_requires=[],  # Empty! Provides no value
)
```

### Analysis

| Aspect | pyproject.toml | setup.py | Verdict |
|---------|----------------|-----------|----------|
| Package metadata | ✅ Complete | ❌ Duplicate | setup.py redundant |
| Version | ✅ "0.1.0" | ❌ "0.1.0" | setup.py redundant |
| Dependencies | ✅ Full list | ❌ Empty | setup.py useless |
| Entry points | ✅ `lotf` CLI | ❌ None | setup.py incomplete |
| Build backend | ✅ Defined | ❌ Duplicate | setup.py redundant |

### Solution

Removed `setup.py` entirely as it provides **zero value**.

**uv sync behavior:**
```
1. Reads pyproject.toml (single source of truth)
2. Creates/updates .venv
3. Installs dependencies from pyproject.toml
4. Completely ignores setup.py  ← This is the key!
```

### Benefits

1. **Single source of truth**: Only `pyproject.toml` defines package metadata
2. **Modern standards**: Follows PEP 621 (2020) packaging
3. **Lightweight**: No separate installation step needed
4. **Source execution**: Direct module execution via `python_exec`
5. **No confusion**: One configuration file instead of two conflicting ones
6. **PEP 621 compliance**: Modern Python packaging standard

### Lightweight Design Workflow

```bash
# Clone and setup (no separate install step)
git clone https://github.com/WarriorHanamy/learning_on_the_fly.git
cd learning_on_the_fly
uv sync --extra cuda12  # Only uses pyproject.toml, no setup.py

# Run directly from source (no pip install needed)
./bin/python_exec -m lotf hover
./bin/python_exec -m lotf track
./bin/python_exec -m lotf residual --dataset data.csv
```

### Verification

All functionality tested and working without `setup.py`:

```bash
# Test 1: Version check
$ ./bin/python_exec -m lotf --version
lotf 0.1.0

# Test 2: CLI help
$ ./bin/python_exec -m lotf hover --help
usage: lotf hover [-h] [--config CONFIG] [--output OUTPUT]

# Test 3: Module import from source
$ ./bin/python_exec -c "import lotf; print(lotf.__file__)"
/home/rec/learning_on_the_fly/lotf/__init__.py
```

---

## Decision 3: Auto-Increment Checkpoint Paths

**Date:** 2026-02-26
**Status:** ✅ Implemented

### Problem

Training scripts would fail when checkpoint directory already exists:

```
ValueError: Destination /home/rec/learning_on_the_fly/checkpoints/policy/state_hovering_params already exists.
```

Users had to manually rename checkpoint directories between runs.

### Solution

Added `get_unique_checkpoint_path()` helper to all training scripts:

```python
def get_unique_checkpoint_path(base_path: Path) -> Path:
    """Generate a unique checkpoint path by appending timestamp if directory exists."""
    path = base_path.resolve()
    if not path.exists():
        return path

    # Generate timestamp suffix
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_path = path.parent / f"{path.name}_{timestamp}"

    print(f"Checkpoint directory exists, using: {new_path}")
    return new_path
```

### Behavior

```bash
# First run
$ ./bin/python_exec -m lotf hover
Policy saved successfully to: checkpoints/policy/state_hovering_params

# Second run (auto-increments)
$ ./bin/python_exec -m lotf hover
Checkpoint directory exists, using: checkpoints/policy/state_hovering_params_20260226_233339
Policy saved successfully to: checkpoints/policy/state_hovering_params_20260226_233339
```

### Files Modified

- `lotf/scripts/train_state_hovering.py`
- `lotf/scripts/train_traj_tracking.py`
- `lotf/scripts/train_residual.py`

---

## Related Commits

- `c22d151`: Simplify python_exec wrapper for single responsibility
- `f83626c`: Remove redundant setup.py for lightweight design
- `40346db`: Add auto-increment for checkpoint paths

---

## Future Considerations

1. **Enable GitHub Issues**: Repository has issues disabled, preventing community feedback
2. **Documentation**: Keep design decisions updated as project evolves
3. **PEP 621**: Monitor for changes in Python packaging standards
