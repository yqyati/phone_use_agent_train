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

"""AndroidWorld interaction for verl multi-turn GRPO training.

Each interaction instance corresponds to one AndroidWorld task running in a
Docker container. Containers are managed via a port pool so that multiple
rollout workers can share the pool without conflicts.

Docker containers expose the AndroidWorld FastAPI server on port 5000.
The port pool maps host ports (e.g. 5001-5032) to individual containers:

    docker run --privileged -p 5001:5000 android_world:latest
    docker run --privileged -p 5002:5000 android_world:latest
    ...
"""

import asyncio
import base64
import io
import json
import logging
import os
import re
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import requests
from PIL import Image

from .base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

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


# ---------------------------------------------------------------------------
# Synchronous HTTP client (runs in a thread via asyncio.to_thread)
# ---------------------------------------------------------------------------

class _AndroidEnvHttpClient:
    """Minimal synchronous HTTP client for the AndroidWorld FastAPI server."""

    def __init__(self, port: int, host: str = "localhost"):
        self.base_url = f"http://{host}:{port}"

    def health(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/health", timeout=5)
            return r.ok
        except Exception:
            return False

    def reset(self, go_home: bool = True) -> dict:
        r = requests.post(f"{self.base_url}/reset", params={"go_home": go_home}, timeout=30)
        r.raise_for_status()
        return r.json()

    def reinitialize_suite(self, n_task_combinations: int = 3, seed: int = 42,
                           task_family: str = "android_world") -> dict:
        r = requests.get(
            f"{self.base_url}/suite/reinitialize",
            params={"n_task_combinations": n_task_combinations, "seed": seed,
                    "task_family": task_family},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()

    def get_suite_task_list(self, max_index: int = -1) -> list[str]:
        r = requests.get(f"{self.base_url}/suite/task_list",
                         params={"max_index": max_index}, timeout=30)
        r.raise_for_status()
        return r.json()["task_list"]

    def get_suite_task_length(self, task_type: str) -> int:
        r = requests.get(f"{self.base_url}/suite/task_length",
                         params={"task_type": task_type}, timeout=30)
        r.raise_for_status()
        return r.json()["length"]

    def initialize_task(self, task_type: str, task_idx: int) -> dict:
        r = requests.post(
            f"{self.base_url}/task/initialize",
            params={"task_type": task_type, "task_idx": task_idx},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()

    def tear_down_task(self, task_type: str, task_idx: int) -> dict:
        r = requests.post(
            f"{self.base_url}/task/tear_down",
            params={"task_type": task_type, "task_idx": task_idx},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def get_task_goal(self, task_type: str, task_idx: int) -> str:
        r = requests.get(f"{self.base_url}/task/goal",
                         params={"task_type": task_type, "task_idx": task_idx}, timeout=30)
        r.raise_for_status()
        return r.json()["goal"]

    def get_task_score(self, task_type: str, task_idx: int) -> float:
        r = requests.get(f"{self.base_url}/task/score",
                         params={"task_type": task_type, "task_idx": task_idx}, timeout=30)
        r.raise_for_status()
        return float(r.json()["score"])

    def get_screenshot(self, wait_to_stabilize: bool = False) -> np.ndarray:
        r = requests.get(f"{self.base_url}/screenshot",
                         params={"wait_to_stabilize": wait_to_stabilize}, timeout=30)
        r.raise_for_status()
        return np.array(r.json()["pixels"])

    def execute_action(self, action_dict: dict) -> dict:
        r = requests.post(f"{self.base_url}/execute_action", json=action_dict, timeout=30)
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_screenshot(pixels: np.ndarray) -> str:
    """Encode a numpy RGB array as a base64 PNG data URL."""
    if pixels.ndim == 1:
        # Flat pixel array — try to reshape to a square-ish image
        side = int(pixels.shape[0] ** 0.5)
        try:
            pixels = pixels.reshape(side, -1, 3)
        except ValueError:
            # Fall back: treat as a single row
            pixels = pixels.reshape(1, -1, 3)
    img = Image.fromarray(pixels.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _parse_action(text: str) -> dict:
    """Extract the first <action>...</action> block and parse it as JSON.

    Falls back to a wait action if nothing parseable is found.
    """
    match = re.search(r"<action>(.*?)</action>", text, re.DOTALL)
    if match:
        raw = match.group(1).strip()
    else:
        # Try to find a bare JSON object in the text
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
                # Multi-modal content list — extract text parts
                return " ".join(
                    part.get("text", "") for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            return str(content)
    return ""


# ---------------------------------------------------------------------------
# Main Interaction class
# ---------------------------------------------------------------------------

class AndroidWorldInteraction(BaseInteraction):
    """verl Interaction that drives an AndroidWorld Docker container.

    Config keys:
        ports (list[int]): Host ports mapped to individual containers.
            Each port corresponds to one running Docker container with
            ``docker run --privileged -p <port>:5000 android_world:latest``.
        max_steps (int): Maximum number of agent steps per episode (default 20).
        host (str): Hostname for the Docker containers (default "localhost").
        success_threshold (float): Minimum score considered a success (default 1.0).
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        ports: list[int] = config.get("ports", [5001])
        self._max_steps: int = config.get("max_steps", 20)
        self._host: str = config.get("host", "localhost")
        self._success_threshold: float = config.get("success_threshold", 1.0)

        # Port pool — filled lazily on first use to avoid issues at import time
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
        task_type: Optional[str] = None,
        task_idx: int = 0,
        **kwargs,
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())

        if task_type is None:
            raise ValueError("task_type must be provided in interaction_kwargs")

        # Acquire a free container port (blocks until one is available)
        port = await self._port_queue.get()
        client = _AndroidEnvHttpClient(port=port, host=self._host)

        try:
            # Reset environment and initialize the specific task
            await asyncio.to_thread(client.reset, True)
            await asyncio.to_thread(client.initialize_task, task_type, task_idx)

            # Fetch initial screenshot for the first observation
            pixels = await asyncio.to_thread(client.get_screenshot, True)
            initial_screenshot = _encode_screenshot(pixels)
        except Exception as e:
            # Return port to pool on failure
            self._port_queue.put_nowait(port)
            raise RuntimeError(
                f"Failed to initialize AndroidWorld task {task_type}[{task_idx}] "
                f"on port {port}: {e}"
            ) from e

        self._instances[instance_id] = {
            "port": port,
            "client": client,
            "task_type": task_type,
            "task_idx": task_idx,
            "step": 0,
            "initial_screenshot": initial_screenshot,  # injected on first generate_response
            "done": False,
        }
        logger.info("Started interaction %s: task=%s[%d] port=%d",
                    instance_id, task_type, task_idx, port)
        return instance_id

    async def generate_response(
        self,
        instance_id: str,
        messages: list[dict[str, Any]],
        **kwargs,
    ) -> tuple[bool, str, float, dict[str, Any]]:
        inst = self._instances[instance_id]

        if inst["done"]:
            return True, "Task already completed.", inst.get("final_score", 0.0), {}

        client: _AndroidEnvHttpClient = inst["client"]
        task_type: str = inst["task_type"]
        task_idx: int = inst["task_idx"]

        # On the very first call, return the initial screenshot so the model
        # can decide its first action based on actual screen content.
        if inst["step"] == 0 and inst.get("initial_screenshot"):
            screenshot_b64 = inst.pop("initial_screenshot")
            obs_text = (
                f"Here is the current screen:\n"
                f"![screen]({screenshot_b64})\n\n"
                f"What is your next action?"
            )
            inst["step"] += 1
            return False, obs_text, 0.0, {"step": inst["step"]}

        # Parse the agent's last action
        last_text = _get_last_assistant_content(messages)
        action_dict = _parse_action(last_text)

        # Handle explicit task-complete signal from the agent
        if (action_dict.get("action_type") == "status" and
                action_dict.get("goal_status") == "success"):
            score = await asyncio.to_thread(client.get_task_score, task_type, task_idx)
            inst["done"] = True
            inst["final_score"] = score
            self._instances.pop(instance_id, None)
            await self._cleanup_instance(inst)
            return True, f"Task declared complete. Score: {score:.2f}", score, {}

        # Execute action on the device
        try:
            await asyncio.to_thread(client.execute_action, action_dict)
        except Exception as e:
            logger.warning("execute_action failed for %s: %s", instance_id, e)

        inst["step"] += 1

        # Observe new state
        pixels = await asyncio.to_thread(client.get_screenshot, True)
        screenshot_b64 = _encode_screenshot(pixels)
        score = await asyncio.to_thread(client.get_task_score, task_type, task_idx)

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
            f"Here is the current screen:\n"
            f"![screen]({screenshot_b64})\n\n"
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
        client: _AndroidEnvHttpClient = inst["client"]
        return await asyncio.to_thread(
            client.get_task_score, inst["task_type"], inst["task_idx"]
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
        client: _AndroidEnvHttpClient = inst["client"]
        port: int = inst["port"]
        try:
            await asyncio.to_thread(
                client.tear_down_task, inst["task_type"], inst["task_idx"]
            )
            await asyncio.to_thread(client.reset, True)
        except Exception as e:
            logger.warning("Cleanup failed for port %d: %s", port, e)
        finally:
            self._port_queue.put_nowait(port)
            logger.info("Released port %d back to pool", port)
