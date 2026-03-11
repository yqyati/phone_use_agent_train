"""Prepare AndroidWorld training data for verl GRPO training.

This script connects to a running AndroidWorld Docker container, fetches
the task list and goals, then writes train.parquet and test.parquet in the
format expected by verl's RLHFDataset.

Usage:
    # First start a container:
    #   docker run --privileged -p 5001:5000 android_world:latest
    python prepare_android_world_data.py --port 5001 --output_dir ~/data/android_world_verl

The parquet schema matches verl's multi-turn interaction format:
    - data_source
    - prompt          (list of chat messages)
    - ability
    - reward_model    (style=rule, ground_truth=task_type)
    - extra_info      (split, index, interaction_kwargs)
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import requests

SYSTEM_PROMPT = """You are an Android device agent. You will be given a task to complete on an Android device.
At each step, you will receive the current screen as an image.
Output your next action in the following JSON format inside <action> tags:
<action>{"action_type": "click", "x": 100, "y": 200}</action>

Supported action types:
- click: {"action_type": "click", "x": int, "y": int}
- input_text: {"action_type": "input_text", "text": str}
- scroll: {"action_type": "scroll", "direction": "up|down|left|right"}
- long_press: {"action_type": "long_press", "x": int, "y": int}
- navigate_home: {"action_type": "navigate_home"}
- navigate_back: {"action_type": "navigate_back"}
- keyboard_enter: {"action_type": "keyboard_enter"}
- open_app: {"action_type": "open_app", "app_name": str}
- wait: {"action_type": "wait"}
- status: {"action_type": "status", "goal_status": "success"}

When you believe the task is complete, output:
<action>{"action_type": "status", "goal_status": "success"}</action>

Think step by step before each action."""


def wait_for_server(base_url: str, timeout: int = 600) -> None:
    print(f"Waiting for AndroidWorld server at {base_url} ...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            if r.ok:
                print("Server is ready.")
                return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"Server at {base_url} did not become ready within {timeout}s")


def build_row(
    task_type: str,
    task_idx: int,
    goal: str,
    split: str,
    global_idx: int,
) -> dict:
    """Build one parquet row for a (task_type, task_idx) pair."""
    return {
        "data_source": "android_world",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Task: {goal}"},
        ],
        "ability": "android_agent",
        "reward_model": {
            "style": "rule",
            "ground_truth": task_type,  # used only for bookkeeping
        },
        "extra_info": {
            "split": split,
            "index": global_idx,
            "task_type": task_type,
            "task_idx": task_idx,
            "goal": goal,
            "interaction_kwargs": {
                "name": "android_world",
                "task_type": task_type,
                "task_idx": task_idx,
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare AndroidWorld data for verl")
    parser.add_argument("--port", type=int, default=5001,
                        help="Host port of the AndroidWorld Docker container (default: 5001)")
    parser.add_argument("--host", type=str, default="localhost",
                        help="Hostname of the Docker container (default: localhost)")
    parser.add_argument("--n_task_combinations", type=int, default=3,
                        help="Number of parameter instances per task type (default: 3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for task parameter generation (default: 42)")
    parser.add_argument("--task_family", type=str, default="android_world",
                        help="Task family to use (default: android_world)")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Fraction of tasks used for training (default: 0.8)")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.expanduser("~/data/android_world_verl"),
                        help="Output directory for parquet files")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    wait_for_server(base_url)

    # Initialize task suite
    print(f"Initializing suite: n_task_combinations={args.n_task_combinations}, seed={args.seed}")
    r = requests.get(
        f"{base_url}/suite/reinitialize",
        params={
            "n_task_combinations": args.n_task_combinations,
            "seed": args.seed,
            "task_family": args.task_family,
        },
        timeout=120,
    )
    r.raise_for_status()

    # Fetch task list
    r = requests.get(f"{base_url}/suite/task_list", params={"max_index": -1}, timeout=60)
    r.raise_for_status()
    task_types: list[str] = r.json()["task_list"]
    print(f"Found {len(task_types)} task types.")

    # Build rows for all (task_type, task_idx) combinations
    rows = []
    for task_type in task_types:
        # Get number of instances for this task type
        r = requests.get(f"{base_url}/suite/task_length",
                         params={"task_type": task_type}, timeout=30)
        r.raise_for_status()
        n_instances = r.json()["length"]

        for task_idx in range(n_instances):
            # Fetch the natural-language goal
            try:
                r = requests.get(
                    f"{base_url}/task/goal",
                    params={"task_type": task_type, "task_idx": task_idx},
                    timeout=30,
                )
                r.raise_for_status()
                goal = r.json()["goal"]
            except Exception as e:
                print(f"  WARNING: could not get goal for {task_type}[{task_idx}]: {e}")
                goal = f"Complete the task: {task_type}"

            rows.append((task_type, task_idx, goal))

    print(f"Total samples: {len(rows)}")

    # Shuffle and split train / test
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(rows))
    split_at = int(len(indices) * args.train_ratio)
    train_indices = indices[:split_at]
    test_indices = indices[split_at:]

    def build_df(idx_list: np.ndarray, split: str) -> pd.DataFrame:
        data = []
        for global_idx, i in enumerate(idx_list):
            task_type, task_idx, goal = rows[i]
            data.append(build_row(task_type, task_idx, goal, split, global_idx))
        return pd.DataFrame(data)

    train_df = build_df(train_indices, "train")
    test_df = build_df(test_indices, "test")

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.parquet")
    test_path = os.path.join(args.output_dir, "test.parquet")
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"Saved {len(train_df)} train rows  → {train_path}")
    print(f"Saved {len(test_df)} test rows   → {test_path}")
    print("Done.")


if __name__ == "__main__":
    main()
