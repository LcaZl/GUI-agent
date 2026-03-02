import asyncio
import base64
import io
import json
import logging
import os
import socket
from json import JSONDecodeError
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from agentz.constants import OBS_SNIPPET_LEN, STEP_DEFAULT_PAUSE_SEC, STEP_DEFAULT_TIMEOUT_SEC


class OSWorldEnvironment:
    """TCP client for OSWorld server (IPC)."""

    def __init__(self, config):
        """
        Initialize the OSWorld environment client.
        
        Parameters
        ----------
        config : Any
            Environment configuration used for server commands and timeouts.
        
        Returns
        -------
        None
            No return value.
        """
        self.logger = logging.getLogger("OSWorldEnv")
        self.config = config
        self.host = str(os.environ["osworld_host_address"])
        self.port = int(os.environ["osworld_host_port"])
        self.recv_buf = 10_000_000
        default_socket_timeout = float(getattr(self.config, "socket_timeout_sec", 120.0))
        env_socket_timeout = os.environ.get("OSWORLD_SOCKET_TIMEOUT_SEC")
        if env_socket_timeout is None:
            self.socket_timeout_sec = default_socket_timeout
        else:
            try:
                self.socket_timeout_sec = float(env_socket_timeout)
            except Exception:
                self.logger.warning(
                    "Invalid OSWORLD_SOCKET_TIMEOUT_SEC=%r. Using config socket_timeout_sec=%s",
                    env_socket_timeout,
                    default_socket_timeout,
                )
                self.socket_timeout_sec = default_socket_timeout

        # Ready-state fields for async startup.
        self.ready: bool = False
        self._ready_event: asyncio.Event = asyncio.Event()
        self._init_task: Optional[asyncio.Task] = None
        self._init_error: Optional[BaseException] = None

    # --- async startup ---
    def start_init(self) -> None:
        """
        Start async environment initialization (idempotent).
        """
        if self._init_task is not None:
            return
        self._init_task = asyncio.create_task(self._async_init(), name="osworld-init")

    async def _async_init(self) -> None:
        """
        Initialize the remote OSWorld environment.
        
        Returns
        -------
        None
            No return value.
        """
        try:
            # Trigger server-side environment initialization.
            await self._send_async(
                {
                    "cmd": "init_env",
                    "config": self._make_server_config(),
                }
            )
            self.ready = True
        except BaseException as e:
            self._init_error = e
            self.logger.exception("OSWorld init failed")
            raise
        finally:
            self._ready_event.set()

    async def wait_ready(self, timeout: Optional[float] = None) -> None:
        """
        Wait for async initialization completion and raise startup errors.
        """
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)
        except asyncio.TimeoutError as e:
            raise TimeoutError("OSWorldEnvironment not ready within timeout") from e

        if self._init_error is not None:
            # Re-raise the original startup error for debugging.
            raise self._init_error

    # --- async wrapper around blocking _send ---
    async def _send_async(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a blocking request in a worker thread.
        
        Parameters
        ----------
        payload : Dict[str, Any]
            JSON payload sent to the OSWorld server.
        
        Returns
        -------
        Dict[str, Any]
            Decoded JSON response from the server.
        """
        return await asyncio.to_thread(self._send, payload)

    # ----------------------------------------
    # Config to server payload
    # ----------------------------------------
    def _make_server_config(self) -> Dict[str, Any]:
        """
        Build the server configuration payload.
        """
        c = self.config
        return {
            "provider_name": c.provider_name,
            "path_to_vm": str(c.path_to_vm),
            "snapshot_name": c.snapshot_name,
            "headless": c.headless,
            "action_space": c.action_space,
            "require_a11y_tree": c.require_a11y_tree,
            "require_terminal": c.require_terminal,
            "os_type": c.os_type,
            "enable_proxy": c.enable_proxy,
            "client_password": c.client_password,
        }

    # ----------------------------------------
    # TCP
    # ----------------------------------------
    def _send(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a JSON payload to the server and return decoded JSON.
        """
        msg = json.dumps(payload).encode("utf-8")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.settimeout(self.socket_timeout_sec)
            sock.connect((self.host, self.port))
            sock.sendall(msg)

            chunks = []
            while True:
                part = sock.recv(self.recv_buf)
                if not part:
                    break
                chunks.append(part)

            raw = b"".join(chunks).decode("utf-8", errors="replace")
            if not raw.strip():
                # Typical when the server crashes or closes the connection without responding.
                self.logger.error("[osworld] empty response for cmd=%s", payload.get("cmd"))
                raise RuntimeError(f"OSWorld server returned an empty response for cmd={payload.get('cmd')!r}")

            # Some servers may prepend logs; trim to first JSON object/array if needed.
            trimmed = raw.lstrip()
            if trimmed and trimmed[0] not in "{[":
                first_obj = min(
                    [i for i in (trimmed.find("{"), trimmed.find("[")) if i != -1],
                    default=-1,
                )
                if first_obj != -1:
                    trimmed = trimmed[first_obj:]

            try:
                return json.loads(trimmed)
            except JSONDecodeError as e:
                snippet = trimmed[:OBS_SNIPPET_LEN].replace("\n", "\\n")
                self.logger.error(
                    "[osworld] invalid JSON for cmd=%s len=%s snippet=%s",
                    payload.get("cmd"),
                    len(trimmed),
                    snippet,
                )
                raise RuntimeError(
                    f"OSWorld server returned invalid JSON for cmd={payload.get('cmd')!r}: {e}. "
                    f"First {OBS_SNIPPET_LEN} chars: {snippet!r}"
                ) from e
        finally:
            sock.close()

    # ----------------------------------------
    # Screenshot
    # ----------------------------------------
    @staticmethod
    def decode_screenshot(b64_img: Optional[str]) -> Optional[np.ndarray]:
        """
        Decode a base64 screenshot into an RGB numpy image.
        
        Parameters
        ----------
        b64_img : Optional[str]
            Base64-encoded image payload.
        
        Returns
        -------
        Optional[np.ndarray]
            RGB image array, or `None` when input is empty.
        """
        if not b64_img:
            return None
        raw = base64.b64decode(b64_img)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return np.array(img, dtype=np.uint8)

    def _clean_obs(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize raw observation payload from the server.
        
        Parameters
        ----------
        raw : Dict[str, Any]
            Raw observation dictionary returned by OSWorld.
        
        Returns
        -------
        Dict[str, Any]
            Observation dictionary in agent-compatible shape.
        """
        return {
            "screenshot": self.decode_screenshot(raw.get("screenshot")),
            "accessibility_tree": raw.get("accessibility_tree"),
            "terminal": raw.get("terminal"),
            "instruction": raw.get("instruction"),
        }

    # ----------------------------------------
    # Core API
    # ----------------------------------------
    def observe(self) -> Dict[str, Any]:
        """
        Fetch current observation from the environment.
        
        Returns
        -------
        Dict[str, Any]
            Current normalized observation.
        """
        raw = self._send({"cmd": "observe"})
        return self._clean_obs(raw)

    def reset(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reset the environment for a new task.
        
        Parameters
        ----------
        task : Dict[str, Any]
            Task definition passed to server reset.
        
        Returns
        -------
        Dict[str, Any]
            Initial normalized observation after reset.
        """
        raw = self._send({"cmd": "reset", "task": task})
        return self._clean_obs(raw)

    def step(self, action: Any, pause: float = STEP_DEFAULT_PAUSE_SEC) -> Dict[str, Any]:
        """
        Execute one action in the environment.
        
        Parameters
        ----------
        action : Any
            Action payload to execute.
        pause : float
            Pause duration after action, in seconds.
        
        Returns
        -------
        Dict[str, Any]
            Transition payload with observation, reward, done and info.
        """
        raw = self._send({"cmd": "step", "action": action, "pause": pause})
        return {
            "obs": self._clean_obs(raw["obs"]),
            "reward": raw.get("reward"),
            "done": raw.get("done"),
            "info": raw.get("info"),
        }

    # ----------------------------------------
    # Auxiliary endpoints
    # ----------------------------------------
    def python_exec(self, code: str) -> Dict[str, Any]:
        """
        Execute Python code in the VM via inline or script endpoint.
        """
        if "\n" in code:
            return self._send(
                {
                    "cmd": "python_script",
                    "script": code,
                }
            )
        return self._send(
            {
                "cmd": "python_exec",
                "code": code,
            }
        )

    def bash_script(
        self,
        script: str,
        timeout: int = STEP_DEFAULT_TIMEOUT_SEC,
        working_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a bash script in the VM.
        """
        payload: Dict[str, Any] = {
            "cmd": "bash_script",
            "script": script,
            "timeout": int(timeout),
        }
        if working_dir is not None:
            payload["working_dir"] = working_dir
        return self._send(payload)

    def vm_info(self) -> Dict[str, Any]:
        """
        Return basic VM information (platform, screen size, etc.).
        """
        return self._send({"cmd": "vm_info"})

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate current task state via server evaluator.
        """
        return self._send({"cmd": "evaluate"})
