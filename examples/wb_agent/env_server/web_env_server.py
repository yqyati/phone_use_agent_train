"""Web environment server for InfiniteWeb-Dataset GUI agent RL training.

Each instance of this server controls one headless Chromium browser via Playwright,
serving one website task at a time. Multiple instances run in parallel (one per port)
to enable large-scale concurrent rollouts during verl GRPO training.

Exposes the same HTTP API surface as the AndroidWorld FastAPI server so that the
interaction layer (web_world_interaction.py) requires minimal adaptation.

Usage:
    pip install playwright fastapi uvicorn
    playwright install chromium
    python web_env_server.py --port 6001 --dataset-dir /path/to/InfiniteWeb-Dataset
"""

import argparse
import asyncio
import base64
import json
import os
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Playwright lazy import to allow module-level usage check
try:
    from playwright.async_api import async_playwright, Browser, BrowserContext, Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

app = FastAPI(title="WebEnvServer")

# ---------------------------------------------------------------------------
# Global environment state (one server = one browser = one active task)
# ---------------------------------------------------------------------------

class _WebEnvState:
    def __init__(self):
        self.playwright = None
        self.browser: Optional["Browser"] = None
        self.context: Optional["BrowserContext"] = None
        self.page: Optional["Page"] = None
        self.website_dir: Optional[Path] = None
        self.current_task: Optional[dict] = None
        self.dataset_dir: Optional[Path] = None
        self.viewport_width: int = 1280
        self.viewport_height: int = 800

_env = _WebEnvState()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def _startup():
    if not PLAYWRIGHT_AVAILABLE:
        raise RuntimeError("playwright not installed. Run: pip install playwright && playwright install chromium")
    _env.playwright = await async_playwright().start()
    _env.browser = await _env.playwright.chromium.launch(
        headless=True,
        args=["--no-sandbox", "--disable-dev-shm-usage"],
    )


@app.on_event("shutdown")
async def _shutdown():
    if _env.context:
        await _env.context.close()
    if _env.browser:
        await _env.browser.close()
    if _env.playwright:
        await _env.playwright.stop()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "browser": _env.browser is not None}


# ---------------------------------------------------------------------------
# Reset  (mirrors /reset on AndroidWorld)
# ---------------------------------------------------------------------------

@app.post("/reset")
async def reset(go_home: bool = True):
    """Close current browser context and open a fresh one."""
    if _env.context:
        await _env.context.close()
    _env.context = await _env.browser.new_context(
        viewport={"width": _env.viewport_width, "height": _env.viewport_height},
        locale="en-US",
    )
    _env.page = await _env.context.new_page()
    _env.current_task = None
    _env.website_dir = None
    return {"status": "reset"}


# ---------------------------------------------------------------------------
# Task management
# ---------------------------------------------------------------------------

def _load_tasks(website_dir: Path) -> list[dict]:
    tasks_file = website_dir / "rewritten_tasks.json"
    if not tasks_file.exists():
        return []
    data = json.loads(tasks_file.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data.get("tasks", [])
    return data  # assume list


@app.post("/task/initialize")
async def initialize_task(website_id: str, task_idx: int):
    """Load a specific website and task, navigate to index.html."""
    if _env.dataset_dir is None:
        raise HTTPException(500, "dataset_dir not configured")

    website_dir = _env.dataset_dir / website_id
    if not website_dir.is_dir():
        raise HTTPException(404, f"Website directory not found: {website_id}")

    tasks = _load_tasks(website_dir)
    if task_idx >= len(tasks):
        raise HTTPException(404, f"task_idx {task_idx} out of range (have {len(tasks)})")

    _env.website_dir = website_dir
    _env.current_task = tasks[task_idx]

    # Fresh browser context for clean localStorage state
    if _env.context:
        await _env.context.close()
    _env.context = await _env.browser.new_context(
        viewport={"width": _env.viewport_width, "height": _env.viewport_height},
        locale="en-US",
    )
    _env.page = await _env.context.new_page()

    # Navigate via file:// — Playwright/Chromium supports localStorage on file:// pages
    index_url = website_dir.resolve().as_uri() + "/index.html"
    try:
        await _env.page.goto(index_url, wait_until="domcontentloaded", timeout=15_000)
    except Exception as e:
        raise HTTPException(500, f"Failed to navigate to {index_url}: {e}")

    return {
        "status": "initialized",
        "website_id": website_id,
        "task_idx": task_idx,
        "goal": _env.current_task["instruction"],
    }


@app.post("/task/tear_down")
async def tear_down_task(website_id: str = "", task_idx: int = 0):
    """Clean up current task (close context, reset state)."""
    if _env.context:
        await _env.context.close()
        _env.context = None
        _env.page = None
    _env.current_task = None
    _env.website_dir = None
    return {"status": "torn_down"}


@app.get("/task/goal")
async def get_task_goal(website_id: str, task_idx: int):
    """Return the natural-language instruction for a task (used during data prep)."""
    if _env.dataset_dir is None:
        raise HTTPException(500, "dataset_dir not configured")
    website_dir = _env.dataset_dir / website_id
    tasks = _load_tasks(website_dir)
    if task_idx >= len(tasks):
        raise HTTPException(404, f"task_idx {task_idx} out of range")
    return {"goal": tasks[task_idx]["instruction"]}


@app.get("/task/score")
async def get_task_score(website_id: str = "", task_idx: int = 0):
    """Compute task completion score (0.0–1.0) by checking localStorage state.

    Scoring logic:
        1. Fetch current localStorage as a JSON string.
        2. Check how many of the task's ``target_ids`` appear anywhere in
           that string (case-insensitive).
        3. Return matches / total_targets as the score.

    This works because InfiniteWeb websites write selected item IDs
    (product IDs, property IDs, booking IDs, etc.) into localStorage when
    the user completes key actions — making target_id presence a reliable
    proxy for task completion.
    """
    if not _env.current_task or not _env.page:
        return {"score": 0.0}

    ground_truth = _env.current_task.get("ground_truth", {})
    target_ids: list[str] = ground_truth.get("target_ids", [])

    if not target_ids:
        return {"score": 0.0}

    try:
        ls_raw: str = await _env.page.evaluate(
            "() => JSON.stringify(Object.assign({}, localStorage))"
        )
        ls_text = (ls_raw or "").lower()
    except Exception:
        return {"score": 0.0}

    matches = sum(1 for tid in target_ids if tid.lower() in ls_text)
    score = matches / len(target_ids)
    return {"score": round(score, 4)}


# ---------------------------------------------------------------------------
# Screenshot
# ---------------------------------------------------------------------------

@app.get("/screenshot")
async def screenshot(wait_to_stabilize: bool = False):
    """Capture current viewport as a base64-encoded PNG data URL."""
    if not _env.page:
        raise HTTPException(400, "No page initialized. Call /task/initialize first.")

    if wait_to_stabilize:
        try:
            await _env.page.wait_for_load_state("networkidle", timeout=3_000)
        except Exception:
            pass

    png_bytes = await _env.page.screenshot(full_page=False, type="png")
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return {"screenshot_b64": f"data:image/png;base64,{b64}"}


# ---------------------------------------------------------------------------
# Action execution
# ---------------------------------------------------------------------------

class _ActionRequest(BaseModel):
    action_type: str
    x: Optional[int] = None
    y: Optional[int] = None
    text: Optional[str] = None
    direction: Optional[str] = None
    # Unused web-side but accepted for API compatibility
    goal_status: Optional[str] = None
    app_name: Optional[str] = None


_SCROLL_DELTA: dict[str, tuple[int, int]] = {
    "up":    (0, -400),
    "down":  (0,  400),
    "left":  (-400, 0),
    "right": ( 400, 0),
}


@app.post("/execute_action")
async def execute_action(action: _ActionRequest):
    """Execute a single agent action on the current page.

    Supported action_types (aligned with Android action space):
        click, input_text, scroll, navigate_back, navigate_home,
        keyboard_enter, long_press, wait, status (no-op on server side).
    """
    if not _env.page:
        raise HTTPException(400, "No page initialized")

    atype = action.action_type
    try:
        if atype == "click":
            await _env.page.mouse.click(action.x, action.y)

        elif atype == "input_text":
            await _env.page.keyboard.type(action.text or "", delay=30)

        elif atype == "scroll":
            dx, dy = _SCROLL_DELTA.get(action.direction or "down", (0, 400))
            await _env.page.mouse.wheel(dx, dy)

        elif atype == "navigate_back":
            await _env.page.go_back(wait_until="domcontentloaded", timeout=10_000)

        elif atype == "navigate_home":
            if _env.website_dir:
                index_url = _env.website_dir.resolve().as_uri() + "/index.html"
                await _env.page.goto(index_url, wait_until="domcontentloaded", timeout=10_000)

        elif atype == "keyboard_enter":
            await _env.page.keyboard.press("Enter")

        elif atype == "long_press":
            await _env.page.mouse.move(action.x, action.y)
            await _env.page.mouse.down()
            await asyncio.sleep(0.8)
            await _env.page.mouse.up()

        elif atype == "wait":
            await asyncio.sleep(1.0)

        elif atype == "status":
            # Handled by interaction layer; server just acknowledges
            pass

        else:
            return {"status": "unknown_action", "action_type": atype}

    except Exception as e:
        # Non-fatal: log and continue so the episode can recover
        return {"status": "error", "error": str(e)}

    # Wait briefly for any triggered navigation / JS to settle
    try:
        await _env.page.wait_for_load_state("domcontentloaded", timeout=2_000)
    except Exception:
        pass

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Web environment server for wb_agent RL training")
    parser.add_argument("--port", type=int, default=6001, help="Port to listen on (default: 6001)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=os.path.expanduser("~/workspace/InfiniteWeb-Dataset"),
        help="Path to InfiniteWeb-Dataset root directory",
    )
    parser.add_argument("--viewport-width", type=int, default=1280)
    parser.add_argument("--viewport-height", type=int, default=800)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _env.dataset_dir = Path(args.dataset_dir).resolve()
    _env.viewport_width = args.viewport_width
    _env.viewport_height = args.viewport_height

    if not _env.dataset_dir.is_dir():
        raise SystemExit(f"Dataset directory not found: {_env.dataset_dir}")

    print(f"Starting WebEnvServer on port {args.port}, dataset: {_env.dataset_dir}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
