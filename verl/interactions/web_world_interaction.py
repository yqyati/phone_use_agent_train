# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""WebWorld interaction for verl multi-turn GRPO training on InfiniteWeb-Dataset.

Mirrors android_world_interaction.py but targets web browser environments
controlled by web_env_server.py (Playwright + FastAPI) instead of Android
Docker containers.

Port pool pattern is identical: each port maps to one running web_env_server
process, ensuring no two rollouts share a browser instance.

Start workers before training:
    for i in $(seq 1 64); do
        python examples/wb_agent/env_server/web_env_server.py \\
            --port $((6000+i)) \\
            --dataset-dir /path/to/InfiniteWeb-Dataset &
    done
"""

import asyncio
import json
import logging
import os
import re
from typing import Any, Optional

import requests

from .base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

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


# ---------------------------------------------------------------------------
# Synchronous HTTP client (runs in a thread via asyncio.to_thread)
# ---------------------------------------------------------------------------

class _WebEnvHttpClient:
    """Minimal synchronous HTTP client for the web_env_server FastAPI server."""

    def __init__(self, port: int, host: str = "localhost"):
        self.base_url = f"http://{host}:{port}"

    def health(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/health", timeout=5)
            return r.ok
        except Exception:
            return False

    def reset(self) -> dict:
        r = requests.post(f"{self.base_url}/reset", timeout=30)
        r.raise_for_status()
        return r.json()

    def initialize_task(self, website_id: str, task_idx: int) -> dict:
        r = requests.post(
            f"{self.base_url}/task/initialize",
            params={"website_id": website_id, "task_idx": task_idx},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def tear_down_task(self, website_id: str, task_idx: int) -> dict:
        r = requests.post(
            f"{self.base_url}/task/tear_down",
            params={"website_id": website_id, "task_idx": task_idx},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def get_task_goal(self, website_id: str, task_idx: int) -> str:
        r = requests.get(
            f"{self.base_url}/task/goal",
            params={"website_id": website_id, "task_idx": task_idx},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["goal"]

    def get_task_score(self, website_id: str, task_idx: int) -> float:
        r = requests.get(
            f"{self.base_url}/task/score",
            params={"website_id": website_id, "task_idx": task_idx},
            timeout=30,
        )
        r.raise_for_status()
        return float(r.json()["score"])

    def get_screenshot(self, wait_to_stabilize: bool = False) -> str:
        """Return screenshot as a base64 PNG data URL."""
        r = requests.get(
            f"{self.base_url}/screenshot",
            params={"wait_to_stabilize": wait_to_stabilize},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["screenshot_b64"]

    def execute_action(self, action_dict: dict) -> dict:
        r = requests.post(f"{self.base_url}/execute_action", json=action_dict, timeout=30)
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_action(text: str) -> dict:
    """Extract the first <action>…</action> block and parse it as JSON.

    Falls back to a wait action if nothing parseable is found.
    """
    match = re.search(r"<action>(.*?)</action>", text, re.DOTALL)
    if match:
        raw = match.group(1).strip()
    else:
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        raw = match.group(0).strip() if match else ""

    if not raw:
        return {"action_type": "wait"}

    try:
        action = json.loads(raw)
        if "action_type" not in action:
            return {"action_type": "wait"}
        return action
    except json.JSONDecodeError:
        logger.warning("Failed to parse action JSON: %s", raw)
        return {"action_type": "wait"}


def _get_last_assistant_content(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, list):
                return " ".join(
                    part.get("text", "") for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            return str(content)
    return ""


# ---------------------------------------------------------------------------
# Main Interaction class
# ---------------------------------------------------------------------------

class WebWorldInteraction(BaseInteraction):
    """verl Interaction that drives a WebWorld browser environment.

    Config keys:
        ports (list[int]): Host ports mapped to individual web_env_server processes.
        max_steps (int): Maximum agent steps per episode (default 25).
        host (str): Hostname for the web_env_server processes (default "localhost").
        success_threshold (float): Minimum score considered a success (default 1.0).
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        ports: list[int] = config.get("ports", [6001])
        self._max_steps: int = config.get("max_steps", 25)
        self._host: str = config.get("host", "localhost")
        self._success_threshold: float = config.get("success_threshold", 1.0)

        # Port pool — one port = one running web_env_server = one browser
        self._port_queue: asyncio.Queue = asyncio.Queue()
        for port in ports:
            self._port_queue.put_nowait(port)

        # instance_id -> state dict
        self._instances: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # BaseInteraction interface
    # ------------------------------------------------------------------

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        website_id: Optional[str] = None,
        task_idx: int = 0,
        **kwargs,
    ) -> str:
        if instance_id is None:
            import uuid
            instance_id = str(uuid.uuid4())

        if website_id is None:
            raise ValueError("website_id must be provided in interaction_kwargs")

        # Acquire a free browser port (blocks until one is available)
        port = await self._port_queue.get()
        client = _WebEnvHttpClient(port=port, host=self._host)

        try:
            await asyncio.to_thread(client.reset)
            await asyncio.to_thread(client.initialize_task, website_id, task_idx)
            initial_screenshot = await asyncio.to_thread(client.get_screenshot, True)
        except Exception as e:
            self._port_queue.put_nowait(port)
            raise RuntimeError(
                f"Failed to initialize web task {website_id}[{task_idx}] on port {port}: {e}"
            ) from e

        self._instances[instance_id] = {
            "port": port,
            "client": client,
            "website_id": website_id,
            "task_idx": task_idx,
            "step": 0,
            "initial_screenshot": initial_screenshot,
            "done": False,
        }
        logger.info(
            "Started interaction %s: website=%s task_idx=%d port=%d",
            instance_id, website_id, task_idx, port,
        )
        return instance_id

    async def generate_response(
        self,
        instance_id: str,
        messages: list[dict[str, Any]],
        **kwargs,
    ) -> tuple[bool, str, float, dict[str, Any]]:
        inst = self._instances.get(instance_id)
        if inst is None:
            return True, "Task already completed.", 0.0, {}

        if inst["done"]:
            return True, "Task already completed.", inst.get("final_score", 0.0), {}

        client: _WebEnvHttpClient = inst["client"]
        website_id: str = inst["website_id"]
        task_idx: int = inst["task_idx"]

        # Step 0: return the initial screenshot so the model sees the page first
        if inst["step"] == 0 and inst.get("initial_screenshot"):
            screenshot_b64 = inst.pop("initial_screenshot")
            obs_text = (
                f"Here is the current page:\n"
                f"![page]({screenshot_b64})\n\n"
                f"What is your next action?"
            )
            inst["step"] += 1
            return False, obs_text, 0.0, {"step": inst["step"]}

        # Parse the agent's last action
        last_text = _get_last_assistant_content(messages)
        action_dict = _parse_action(last_text)

        # Handle explicit task-complete signal
        if (action_dict.get("action_type") == "status" and
                action_dict.get("goal_status") == "success"):
            score = await asyncio.to_thread(client.get_task_score, website_id, task_idx)
            inst["done"] = True
            inst["final_score"] = score
            self._instances.pop(instance_id, None)
            await self._cleanup_instance(inst)
            return True, f"Task declared complete. Score: {score:.2f}", score, {}

        # Execute action in the browser
        try:
            await asyncio.to_thread(client.execute_action, action_dict)
        except Exception as e:
            logger.warning("execute_action failed for %s: %s", instance_id, e)

        inst["step"] += 1

        # Observe new state
        screenshot_b64 = await asyncio.to_thread(client.get_screenshot, True)
        score = await asyncio.to_thread(client.get_task_score, website_id, task_idx)

        done = score >= self._success_threshold or inst["step"] >= self._max_steps

        if done:
            inst["done"] = True
            inst["final_score"] = score

        if score >= self._success_threshold:
            feedback = f"Task completed successfully! Score: {score:.2f}"
        elif done:
            feedback = f"Maximum steps reached. Score: {score:.2f}"
        else:
            feedback = f"Step {inst['step']}/{self._max_steps}. Score so far: {score:.2f}"

        obs_text = (
            f"Here is the current page:\n"
            f"![page]({screenshot_b64})\n\n"
            f"{feedback}\n\n"
            f"What is your next action?"
        )

        if done:
            self._instances.pop(instance_id, None)
            await self._cleanup_instance(inst)

        return done, obs_text, score, {"step": inst["step"], "score": score}

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        inst = self._instances.get(instance_id)
        if inst is None:
            return 0.0
        if inst.get("done"):
            return inst.get("final_score", 0.0)
        client: _WebEnvHttpClient = inst["client"]
        return await asyncio.to_thread(
            client.get_task_score, inst["website_id"], inst["task_idx"]
        )

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        inst = self._instances.pop(instance_id, None)
        if inst is None:
            return
        await self._cleanup_instance(inst)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _cleanup_instance(self, inst: dict) -> None:
        """Tear down the task and return the port to the pool."""
        client: _WebEnvHttpClient = inst["client"]
        port: int = inst["port"]
        try:
            await asyncio.to_thread(
                client.tear_down_task, inst["website_id"], inst["task_idx"]
            )
        except Exception as e:
            logger.warning("Cleanup failed for port %d: %s", port, e)
        finally:
            self._port_queue.put_nowait(port)
            logger.info("Released port %d back to pool", port)
