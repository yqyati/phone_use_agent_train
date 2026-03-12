"""Prepare InfiniteWeb-Dataset training data for verl GRPO training.

Unlike AndroidWorld (which requires a running Docker container), this script
reads the dataset directory directly — no server needed.

Output: train.parquet + test.parquet in verl's RLHFDataset format.

Usage:
    python prepare_wb_agent_data.py \\
        --dataset-dir /path/to/InfiniteWeb-Dataset \\
        --output-dir ~/data/wb_agent_verl \\
        --train-ratio 0.9
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

SYSTEM_PROMPT = """You are a web browser agent. You will be given a task to complete on a website.
At each step, you will receive the current page screenshot.
Output your next action in the following JSON format inside <action> tags:
<action>{"action_type": "click", "x": 100, "y": 200}</action>

Supported action types:
- click: {"action_type": "click", "x": int, "y": int}
- input_text: {"action_type": "input_text", "text": str}
- scroll: {"action_type": "scroll", "direction": "up|down|left|right"}
- navigate_back: {"action_type": "navigate_back"}
- navigate_home: {"action_type": "navigate_home"}
- keyboard_enter: {"action_type": "keyboard_enter"}
- long_press: {"action_type": "long_press", "x": int, "y": int}
- wait: {"action_type": "wait"}
- status: {"action_type": "status", "goal_status": "success"}

When you believe the task is complete, output:
<action>{"action_type": "status", "goal_status": "success"}</action>

Think step by step before each action."""


def _load_tasks(website_dir: Path) -> list[dict]:
    tasks_file = website_dir / "rewritten_tasks.json"
    if not tasks_file.exists():
        return []
    try:
        data = json.loads(tasks_file.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data.get("tasks", [])
        return data
    except Exception as e:
        print(f"  WARNING: failed to parse {tasks_file}: {e}")
        return []


def _build_row(
    website_id: str,
    task_idx: int,
    goal: str,
    split: str,
    global_idx: int,
) -> dict:
    return {
        "data_source": "wb_agent",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Task: {goal}"},
        ],
        "ability": "web_agent",
        "reward_model": {
            "style": "rule",
            "ground_truth": website_id,  # bookkeeping only
        },
        "extra_info": {
            "split": split,
            "index": global_idx,
            "website_id": website_id,
            "task_idx": task_idx,
            "goal": goal,
            "interaction_kwargs": {
                "name": "wb_agent",
                "website_id": website_id,
                "task_idx": task_idx,
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare InfiniteWeb data for verl GRPO training")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=os.path.expanduser("~/workspace/InfiniteWeb-Dataset"),
        help="Root directory of InfiniteWeb-Dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.expanduser("~/data/wb_agent_verl"),
        help="Output directory for train.parquet and test.parquet",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Fraction of tasks used for training (default: 0.9)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/test split",
    )
    parser.add_argument(
        "--max-websites",
        type=int,
        default=-1,
        help="Limit number of websites (useful for quick testing, -1 = all)",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")

    # Collect all (website_id, task_idx, goal) triples
    rows: list[tuple[str, int, str]] = []
    website_dirs = sorted(d for d in dataset_dir.iterdir() if d.is_dir())

    if args.max_websites > 0:
        website_dirs = website_dirs[:args.max_websites]

    print(f"Scanning {len(website_dirs)} website directories …")
    for website_dir in website_dirs:
        tasks = _load_tasks(website_dir)
        for i, task in enumerate(tasks):
            goal = task.get("instruction") or task.get("name") or f"Complete task {i}"
            rows.append((website_dir.name, i, goal))

    print(f"Total task instances: {len(rows)}")

    # Reproducible shuffle and split
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(rows))
    split_at = int(len(indices) * args.train_ratio)
    train_indices = indices[:split_at]
    test_indices = indices[split_at:]

    def build_df(idx_list: np.ndarray, split: str) -> pd.DataFrame:
        data = []
        for global_idx, i in enumerate(idx_list):
            website_id, task_idx, goal = rows[int(i)]
            data.append(_build_row(website_id, task_idx, goal, split, global_idx))
        return pd.DataFrame(data)

    train_df = build_df(train_indices, "train")
    test_df = build_df(test_indices, "test")

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.parquet")
    test_path = os.path.join(args.output_dir, "test.parquet")
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"Saved {len(train_df):,} train rows  →  {train_path}")
    print(f"Saved {len(test_df):,} test rows   →  {test_path}")
    print("Done.")


if __name__ == "__main__":
    main()
