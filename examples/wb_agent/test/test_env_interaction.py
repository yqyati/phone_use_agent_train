"""End-to-end smoke test: agent ↔ web_env_server one full episode.

Starts web_env_server.py as a subprocess, then drives a complete episode
using WebWorldInteraction directly (no verl trainer needed).

The "agent" here is a simple rule-based stub that just clicks the center
of the screen every step and declares success at max_steps — enough to
verify the full I/O loop works correctly.

Usage:
    cd /root/workspace/verl
    python examples/wb_agent/test/test_env_interaction.py

    # Use a specific website/task:
    python examples/wb_agent/test/test_env_interaction.py \
        --website-id 102_addiction_recovery_s --task-idx 0

    # Keep the server alive after the test (for manual inspection):
    python examples/wb_agent/test/test_env_interaction.py --no-cleanup
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Allow running from any cwd
VERL_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(VERL_ROOT))

import importlib.util, types

# Import web_world_interaction directly to avoid verl.__init__ → omegaconf →
# antlr4 version conflict.
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# Stub out the verl package hierarchy so relative imports resolve
for _pkg in ("verl", "verl.interactions"):
    sys.modules.setdefault(_pkg, types.ModuleType(_pkg))

_load_module("verl.interactions.base", VERL_ROOT / "verl/interactions/base.py")
_wwi = _load_module("verl.interactions.web_world_interaction",
                    VERL_ROOT / "verl/interactions/web_world_interaction.py")
WebWorldInteraction = _wwi.WebWorldInteraction

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASET_DIR = Path(os.path.expanduser("~/workspace/InfiniteWeb-Dataset"))
SERVER_PORT = 16001   # use a high port to avoid conflicts with real workers
MAX_STEPS = 5         # keep the test short
SERVER_STARTUP_WAIT = 5  # seconds to wait for the server to be ready


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_website(dataset_dir: Path, website_id: str | None) -> str:
    if website_id:
        return website_id
    # Pick first directory that has rewritten_tasks.json
    for d in sorted(dataset_dir.iterdir()):
        if d.is_dir() and (d / "rewritten_tasks.json").exists():
            return d.name
    raise RuntimeError(f"No valid website found in {dataset_dir}")


def _wait_for_server(port: int, timeout: float = 10.0) -> bool:
    """Poll /health until the server responds or timeout."""
    import requests
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=2)
            if r.ok:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


# ---------------------------------------------------------------------------
# Stub agent
# ---------------------------------------------------------------------------

def _stub_agent_response(step: int, max_steps: int, obs_text: str) -> str:
    """
    Minimal rule-based agent:
    - Steps 1..max_steps-1: click center of viewport
    - Final step: declare success
    """
    if step >= max_steps - 1:
        action = {"action_type": "status", "goal_status": "success"}
        print(f"  [agent] step={step}  → declare success")
    else:
        action = {"action_type": "click", "x": 640, "y": 400}
        print(f"  [agent] step={step}  → click(640, 400)")
    return f"<action>{json.dumps(action)}</action>"


# ---------------------------------------------------------------------------
# Main test coroutine
# ---------------------------------------------------------------------------

async def run_test(website_id: str, task_idx: int, cleanup: bool):
    print("=" * 60)
    print(f"  wb_agent smoke test")
    print(f"  website : {website_id}")
    print(f"  task_idx: {task_idx}")
    print(f"  port    : {SERVER_PORT}")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. Start web_env_server as a subprocess
    # ------------------------------------------------------------------ #
    server_script = VERL_ROOT / "examples/wb_agent/env_server/web_env_server.py"
    cmd = [
        sys.executable, str(server_script),
        "--port", str(SERVER_PORT),
        "--dataset-dir", str(DATASET_DIR),
    ]
    print(f"\n[1/5] Starting server: {' '.join(cmd)}")
    server_proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        print(f"      Waiting up to {SERVER_STARTUP_WAIT}s for /health …")
        if not _wait_for_server(SERVER_PORT, timeout=SERVER_STARTUP_WAIT):
            # Dump server output for debugging
            server_proc.terminate()
            out, _ = server_proc.communicate(timeout=5)
            print("SERVER OUTPUT:\n", out)
            raise RuntimeError(f"Server on port {SERVER_PORT} did not start in time")
        print("      Server is healthy ✓")

        # ------------------------------------------------------------------ #
        # 2. Build WebWorldInteraction (same as verl does internally)
        # ------------------------------------------------------------------ #
        print("\n[2/5] Initialising WebWorldInteraction …")
        interaction = WebWorldInteraction(config={
            "ports": [SERVER_PORT],
            "max_steps": MAX_STEPS,
            "host": "localhost",
            "success_threshold": 1.0,
        })

        # ------------------------------------------------------------------ #
        # 3. start_interaction
        # ------------------------------------------------------------------ #
        print(f"\n[3/5] start_interaction(website_id={website_id!r}, task_idx={task_idx})")
        instance_id = await interaction.start_interaction(
            website_id=website_id,
            task_idx=task_idx,
        )
        print(f"      instance_id = {instance_id}")

        # ------------------------------------------------------------------ #
        # 4. Episode loop
        # ------------------------------------------------------------------ #
        print(f"\n[4/5] Running episode (max {MAX_STEPS} steps) …\n")
        messages = []
        step = 0
        total_score = 0.0

        while True:
            done, obs_text, score, meta = await interaction.generate_response(
                instance_id=instance_id,
                messages=messages,
            )
            total_score = score
            has_image = "data:image/png;base64," in obs_text
            print(f"  generate_response → done={done}  score={score:.4f}  "
                  f"screenshot={'YES' if has_image else 'NO'}  meta={meta}")

            if done:
                print(f"\n  Episode finished. Final score = {total_score:.4f}")
                break

            # Agent produces a response
            agent_reply = _stub_agent_response(step, MAX_STEPS, obs_text)
            messages.append({"role": "assistant", "content": agent_reply})
            step += 1

        # ------------------------------------------------------------------ #
        # 5. Result
        # ------------------------------------------------------------------ #
        print("\n[5/5] Test result")
        print(f"  Steps taken  : {step}")
        print(f"  Final score  : {total_score:.4f}")
        print(f"  Score > 0    : {'YES' if total_score > 0 else 'NO (expected for stub agent)'}")
        print("\n  PASS — full interaction loop completed without errors ✓")

    finally:
        if cleanup:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()
            print("\n  Server stopped.")
        else:
            print(f"\n  Server left running on port {SERVER_PORT} (--no-cleanup).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global DATASET_DIR
    parser = argparse.ArgumentParser(description="wb_agent end-to-end smoke test")
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--website-id", default=None, help="e.g. 102_addiction_recovery_s")
    parser.add_argument("--task-idx", type=int, default=0)
    parser.add_argument("--no-cleanup", action="store_true", help="Leave server running after test")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.is_dir():
        sys.exit(f"Dataset directory not found: {dataset_dir}")

    DATASET_DIR = dataset_dir
    website_id = _pick_website(DATASET_DIR, args.website_id)

    asyncio.run(run_test(
        website_id=website_id,
        task_idx=args.task_idx,
        cleanup=not args.no_cleanup,
    ))


if __name__ == "__main__":
    main()
