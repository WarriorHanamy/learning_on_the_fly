#!/usr/bin/env python3
"""Entrypoint for running example tasks.

Usage:
    uv run python -m examples.state_hovering train  # Train base hovering policy from scratch
    uv run python -m examples.state_hovering eval  # Evaluate trained policy
    uv run python -m examples.state_hovering finetune-lora  # Fine-tune policy with LoRA
    uv run python -m examples.state_hovering finetune-full  # Fine-tune all policy parameters
"""

import argparse
import runpy
from pathlib import Path

_MODULE_DIR = Path(__file__).parent

_TASKS = {
    "train": ("train_base_policy", "Train base hovering policy from scratch"),
    "eval": ("eval_policy", "Evaluate trained policy"),
    "finetune-lora": ("finetune_policy_lora", "Fine-tune policy with LoRA"),
    "finetune-full": ("finetune_policy_full", "Fine-tune all policy parameters"),
}


def main():
    parser = argparse.ArgumentParser(description="Example task runner")
    parser.add_argument(
        "task",
        nargs="?",
        choices=list(_TASKS.keys()) + [None],
        help="Task to run. If omitted, lists available tasks.",
    )
    args = parser.parse_args()

    if args.task is None:
        print("Available tasks:")
        for tname, (_, tdesc) in _TASKS.items():
            print(f"  {tname:20s}  {tdesc}")
        return

    stem, _ = _TASKS[args.task]
    script = _MODULE_DIR / (stem + ".py")
    if not script.exists():
        print(f"Script not found: {script}")
        return
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
