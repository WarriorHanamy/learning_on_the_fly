# Repository Guidelines

## Project Structure & Module Organization
The differentiable-simulation library lives in `lotf/` with task-specific components split by domain: algorithms in `lotf/algos/`, environments in `lotf/envs/`, perception and dynamics models inside `lotf/modules/`, and physics assets under `lotf/simulation/`. Quadrotor-centric objects and sensors live in `lotf/objects/` and `lotf/sensors/`. Use `examples/*/*.ipynb` notebooks for runnable walkthroughs of residual dynamics learning, state hovering, trajectory tracking, and vision hovering. Pretrained weights remain in `checkpoints/` (split into `policy/` and `residual_dynamics/`), while visual assets are stored in `assets/`.

## Build, Test, and Development Commands
```bash
conda env create -f environment.yml   # Provision ROS + CUDA-capable env named "lotf"
conda activate lotf
pip install --use-pep517 -e .         # Editable install for iterative work
pytest                                # Runs Python unit tests (add targets under tests/ or module dirs)
jupyter lab examples/state_hovering   # Launch notebooks for end-to-end workflows
```
Keep ROS-specific builds isolated in dedicated workspaces; this repo expects only the Python package to be editable.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation, `snake_case` for functions/modules, and `CamelCase` for classes (matching current `lotf` APIs). Prefer explicit dataclasses or TypedDicts when passing structured state. Co-locate task-specific configs near their modules (`lotf/envs/**/config.py`) and keep notebook filenames prefixed numerically to preserve execution order. Add short docstrings describing dynamics assumptions, especially when touching differentiable simulation code.

## Testing Guidelines
Pytest, pytest-cov, and stress rerun plugins are bundled in `environment.yml`. Mirror module paths when creating tests (e.g., `tests/envs/test_quadrotor.py` for `lotf/envs/quadrotor.py`). Add dataset-dependent checks behind markers so CI can skip heavyweight ROS/GPU workloads, and include at least one smoke test per notebook workflow by importing the notebook's helper functions into a Python test.

## Commit & Pull Request Guidelines
History shows concise, imperative commits (e.g., `Update README.md`); keep subject lines ≤72 chars and explain non-obvious decisions in the body. PRs should include: summary of behavior changes, references to papers or issues when relevant, reproduction commands, and screenshots or plots (`assets/`) whenever behavior or metrics shift. Re-run critical notebooks and link generated artifacts so reviewers can validate learning curves quickly.
