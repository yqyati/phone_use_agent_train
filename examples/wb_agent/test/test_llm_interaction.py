"""End-to-end test: Claude API agent ↔ web_env_server one full episode.

Uses the Anthropic API (claude-sonnet-4-6) as the actual agent.
The agent receives a real screenshot each step and generates real actions.

Usage:
    cd /root/workspace/verl
    python examples/wb_agent/test/test_llm_interaction.py
    python examples/wb_agent/test/test_llm_interaction.py \
        --website-id 102_addiction_recovery_s --task-idx 0 --max-steps 10
"""

import argparse
import asyncio
import importlib.util
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import anthropic

VERL_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(VERL_ROOT))

# ---------------------------------------------------------------------------
# Direct import to avoid omegaconf/antlr4 conflict
# ---------------------------------------------------------------------------

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

for _pkg in ("verl", "verl.interactions"):
    sys.modules.setdefault(_pkg, types.ModuleType(_pkg))

_load_module("verl.interactions.base", VERL_ROOT / "verl/interactions/base.py")
_wwi = _load_module("verl.interactions.web_world_interaction",
                    VERL_ROOT / "verl/interactions/web_world_interaction.py")
WebWorldInteraction = _wwi.WebWorldInteraction
SYSTEM_PROMPT = _wwi.SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASET_DIR = Path(os.path.expanduser("~/workspace/InfiniteWeb-Dataset"))
SERVER_PORT = 16002
MODEL = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_website(dataset_dir: Path, website_id: str | None) -> str:
    if website_id:
        return website_id
    for d in sorted(dataset_dir.iterdir()):
        if d.is_dir() and (d / "rewritten_tasks.json").exists():
            return d.name
    raise RuntimeError(f"No valid website found in {dataset_dir}")


def _wait_for_server(port: int, timeout: float = 10.0) -> bool:
    import requests
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"http://localhost:{port}/health", timeout=2).ok:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _load_task_goal(dataset_dir: Path, website_id: str, task_idx: int) -> str:
    tasks_file = dataset_dir / website_id / "rewritten_tasks.json"
    data = json.loads(tasks_file.read_text())
    tasks = data.get("tasks", data) if isinstance(data, dict) else data
    return tasks[task_idx].get("instruction", f"Complete task {task_idx}")


# ---------------------------------------------------------------------------
# Claude agent
# ---------------------------------------------------------------------------

def _call_claude(client: anthropic.Anthropic, messages: list[dict]) -> str:
    """Send the current conversation to Claude and return its text response."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return response.content[0].text


def _build_user_message(obs_text: str) -> dict:
    """Build an Anthropic message dict from the observation text.

    The observation contains an inline image as a data URL:
        ![page](data:image/png;base64,<b64>)

    We split this into a proper vision message with image + text parts.
    """
    import re
    img_match = re.search(r"!\[.*?\]\((data:image/png;base64,([^)]+))\)", obs_text)
    if img_match:
        b64_data = img_match.group(2)
        # Strip the data-URL prefix if present in group(2)
        if "base64," in b64_data:
            b64_data = b64_data.split("base64,", 1)[1]
        text_only = re.sub(r"!\[.*?\]\(data:image/[^)]+\)\n*", "", obs_text).strip()
        return {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64_data,
                    },
                },
                {"type": "text", "text": text_only or "What is your next action?"},
            ],
        }
    # No image — plain text
    return {"role": "user", "content": obs_text}


# ---------------------------------------------------------------------------
# Main test coroutine
# ---------------------------------------------------------------------------

async def run_test(website_id: str, task_idx: int, max_steps: int, cleanup: bool, save_messages: bool = True):
    goal = _load_task_goal(DATASET_DIR, website_id, task_idx)

    print("=" * 60)
    print(f"  wb_agent  ×  Claude API  smoke test")
    print(f"  model   : {MODEL}")
    print(f"  website : {website_id}")
    print(f"  task    : [{task_idx}] {goal}")
    print(f"  port    : {SERVER_PORT}")
    print(f"  steps   : {max_steps}")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. Start server
    # ------------------------------------------------------------------ #
    server_script = VERL_ROOT / "examples/wb_agent/env_server/web_env_server.py"
    server_proc = subprocess.Popen(
        [sys.executable, str(server_script),
         "--port", str(SERVER_PORT),
         "--dataset-dir", str(DATASET_DIR)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    print(f"\n[1/5] Server starting on port {SERVER_PORT} …")
    if not _wait_for_server(SERVER_PORT, timeout=10):
        server_proc.terminate()
        out, _ = server_proc.communicate(timeout=5)
        print("SERVER OUTPUT:\n", out)
        raise RuntimeError("Server did not start in time")
    print("      Server healthy ✓")

    try:
        # ------------------------------------------------------------------ #
        # 2. Init interaction + Anthropic client
        # ------------------------------------------------------------------ #
        print("\n[2/5] Initialising WebWorldInteraction + Anthropic client …")
        interaction = WebWorldInteraction(config={
            "ports": [SERVER_PORT],
            "max_steps": max_steps,
            "host": "localhost",
            "success_threshold": 2.0,  # never auto-terminate; run full max_steps
        })
        claude = anthropic.Anthropic()

        # ------------------------------------------------------------------ #
        # 3. start_interaction
        # ------------------------------------------------------------------ #
        print(f"\n[3/5] start_interaction …")
        instance_id = await interaction.start_interaction(
            website_id=website_id,
            task_idx=task_idx,
        )
        print(f"      instance_id = {instance_id}")

        # ------------------------------------------------------------------ #
        # 4. Episode loop
        # ------------------------------------------------------------------ #
        print(f"\n[4/5] Episode loop\n")
        conversation: list[dict] = []
        screenshots: list[dict] = []   # {turn, action, screenshot_b64}
        final_score = 0.0
        first_turn = True

        for turn in range(max_steps + 1):
            done, obs_text, score, meta = await interaction.generate_response(
                instance_id=instance_id,
                messages=conversation,
            )
            final_score = score
            has_img = "data:image/png;base64," in obs_text
            print(f"  turn={turn:02d}  done={done}  score={score:.4f}  screenshot={'YES' if has_img else 'NO '}")

            # Extract and save screenshot
            import re as _re
            img_match = _re.search(r"data:image/png;base64,([^)\s\"]+)", obs_text)
            if img_match:
                screenshots.append({
                    "turn": turn,
                    "score": score,
                    "action": None,  # filled in after Claude responds
                    "screenshot_b64": img_match.group(1),
                })

            if done:
                break

            # First turn: inject the task goal into the user message
            user_msg = _build_user_message(obs_text)
            if first_turn:
                goal_prefix = f"Task: {goal}\n\n"
                if isinstance(user_msg["content"], list):
                    user_msg["content"].append({"type": "text", "text": goal_prefix})
                else:
                    user_msg["content"] = goal_prefix + user_msg["content"]
                first_turn = False
            conversation.append(user_msg)

            # Ask Claude for the next action
            agent_reply = _call_claude(claude, conversation)
            print(f"           Claude → {agent_reply[:120].strip()}")
            conversation.append({"role": "assistant", "content": agent_reply})

            # Tag this screenshot with the action Claude chose
            if screenshots:
                action_match = _re.search(r"<action>(.*?)</action>", agent_reply, _re.DOTALL)
                screenshots[-1]["action"] = action_match.group(1).strip() if action_match else agent_reply[:100]

        # ------------------------------------------------------------------ #
        # 5. Summary
        # ------------------------------------------------------------------ #
        print(f"\n[5/5] Result")
        print(f"  Turns       : {turn}")
        print(f"  Final score : {final_score:.4f}")
        print(f"  Success     : {'YES ✓' if final_score >= 1.0 else 'NO (agent did not complete the task)'}")

        if save_messages:
            out_dir = Path(__file__).parent / f"run_{website_id}_task{task_idx}"
            out_dir.mkdir(exist_ok=True)

            # Save messages JSON (images truncated)
            serializable = []
            for msg in conversation:
                if isinstance(msg.get("content"), list):
                    parts = []
                    for part in msg["content"]:
                        if part.get("type") == "image":
                            parts.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "<truncated>"}})
                        else:
                            parts.append(part)
                    serializable.append({"role": msg["role"], "content": parts})
                else:
                    serializable.append(msg)
            (out_dir / "messages.json").write_text(json.dumps(serializable, ensure_ascii=False, indent=2))

            # Save per-step screenshots as HTML viewer
            html_parts = [f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{website_id} task{task_idx}</title>
<style>
  body {{ font-family: monospace; background: #1a1a1a; color: #eee; padding: 20px; }}
  .step {{ border: 1px solid #444; margin: 20px 0; border-radius: 8px; overflow: hidden; }}
  .header {{ background: #333; padding: 10px 16px; font-size: 14px; }}
  .score-0 {{ color: #f66; }} .score-1 {{ color: #6f6; }}
  .canvas-wrap {{ position: relative; display: inline-block; max-width: 100%; }}
  canvas {{ display: block; max-width: 100%; }}
  .action {{ background: #222; padding: 10px 16px; font-size: 13px; color: #9cf; white-space: pre-wrap; }}
</style></head><body>
<h2>Task: {goal}</h2>
<p>Website: {website_id} | task_idx: {task_idx} | Final score: {final_score}</p>
<script>
function drawStep(canvasId, b64, action) {{
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');
  const img = new Image();
  img.onload = function() {{
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    ctx.drawImage(img, 0, 0);
    if (!action) return;
    try {{
      const a = JSON.parse(action);
      if (a.action_type === 'click' || a.action_type === 'long_press') {{
        const x = a.x, y = a.y;
        // crosshair
        ctx.strokeStyle = 'rgba(255,0,0,0.8)';
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(x-20, y); ctx.lineTo(x+20, y); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x, y-20); ctx.lineTo(x, y+20); ctx.stroke();
        // circle
        ctx.beginPath();
        ctx.arc(x, y, 18, 0, 2*Math.PI);
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 3;
        ctx.stroke();
        // label
        ctx.fillStyle = 'rgba(255,0,0,0.85)';
        ctx.font = 'bold 16px monospace';
        ctx.fillText('(' + x + ',' + y + ')', x+22, y-8);
      }}
    }} catch(e) {{}}
  }};
  img.src = 'data:image/png;base64,' + b64;
}}
</script>
"""]
            for s in screenshots:
                score_cls = "score-1" if s["score"] >= 1.0 else "score-0"
                cid = f"c{s['turn']}"
                action_json = s['action'] or ''
                # escape for JS string
                action_js = action_json.replace("\\", "\\\\").replace("'", "\\'")
                html_parts.append(f"""<div class="step">
  <div class="header">Turn {s['turn']} &nbsp;|&nbsp; <span class="{score_cls}">score={s['score']:.4f}</span></div>
  <div class="canvas-wrap"><canvas id="{cid}"></canvas></div>
  <div class="action">→ {action_json or '(episode end)'}</div>
  <script>drawStep('{cid}', '{s["screenshot_b64"]}', '{action_js}');</script>
</div>""")
            html_parts.append("</body></html>")
            html_path = out_dir / "steps.html"
            html_path.write_text("".join(html_parts))

            print(f"\n  Saved to {out_dir}/")
            print(f"  messages.json  — full conversation")
            print(f"  steps.html     — screenshot viewer (open in browser)")
        print()

    finally:
        if cleanup:
            server_proc.terminate()
            server_proc.wait(timeout=5)
            print("  Server stopped.")
        else:
            print(f"  Server left running on port {SERVER_PORT}.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global DATASET_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--website-id", default=None)
    parser.add_argument("--task-idx", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--no-cleanup", action="store_true")
    args = parser.parse_args()

    DATASET_DIR = Path(args.dataset_dir).resolve()
    if not DATASET_DIR.is_dir():
        sys.exit(f"Dataset dir not found: {DATASET_DIR}")

    website_id = _pick_website(DATASET_DIR, args.website_id)
    asyncio.run(run_test(website_id, args.task_idx, args.max_steps, not args.no_cleanup, save_messages=True))


if __name__ == "__main__":
    main()
