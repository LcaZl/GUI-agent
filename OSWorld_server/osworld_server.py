import base64
import datetime
import inspect
import io
import json
import socket
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_OSWORLD_PATH = PROJECT_ROOT / "OSWorld"
if LOCAL_OSWORLD_PATH.exists():
    # Prefer local checkout over site-packages.
    sys.path.insert(0, str(LOCAL_OSWORLD_PATH))

from desktop_env.desktop_env import DesktopEnv

HOST = "127.0.0.1"
PORT = 6001


def log(msg: str, color: str = "37") -> None:
    """
    Print a timestamped log message using ANSI color codes.

    Parameters
    ----------
    msg : str
        Message text to print.
    color : str
        ANSI color code used for terminal output.

    Returns
    -------
    None
        No return value.
    """
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"\x1b[{color}m[{ts}] {msg}\x1b[0m", flush=True)


def print_dict(
    d: Union[Dict[Any, Any], List[Any], str],
    title: str = "",
    avoid_keys: Optional[List[str]] = None,
    indent: int = 0,
) -> None:
    """
    Print dictionaries and lists recursively in a compact readable format.

    Parameters
    ----------
    d : Union[Dict[Any, Any], List[Any], str]
        Data structure to print.
    title : str
        Optional title shown before the payload.
    avoid_keys : Optional[List[str]]
        Keys excluded from recursive printing.
    indent : int
        Current indentation level.

    Returns
    -------
    None
        No return value.
    """
    if avoid_keys is None:
        avoid_keys = []

    indent_str = "    " * indent

    if title:
        line = f"\n{indent_str}== {title.strip()} =="
        print(line)

    if isinstance(d, dict):
        for key, value in d.items():
            if key in avoid_keys:
                continue
            if isinstance(value, dict):
                print(f"{indent_str}- {key}:")
                print_dict(value, avoid_keys=avoid_keys, indent=indent + 1)
            elif isinstance(value, list):
                print(f"{indent_str}- {key}:")
                for item in value:
                    if isinstance(item, (dict, list)):
                        print_dict(item, avoid_keys=avoid_keys, indent=indent + 1)
                    else:
                        cleaned_item = str(item).replace("\n", " ").replace("\r", " ").strip()
                        print(f"{indent_str}    - {cleaned_item}")
            else:
                cleaned_value = str(value).replace("\n", " ").replace("\r", " ").strip()
                print(f"{indent_str}- {key}: {cleaned_value}")
    elif isinstance(d, list):
        for item in d:
            if isinstance(item, (dict, list)):
                print_dict(item, avoid_keys=avoid_keys, indent=indent)
            else:
                cleaned_item = str(item).replace("\n", " ").replace("\r", " ").strip()
                print(f"{indent_str}- {cleaned_item}")
    else:
        cleaned_data = str(d).replace("\n", " ").replace("\r", " ").strip()
        print(f"{indent_str}{cleaned_data}")


# ------------------------------------------------------------
# Image encoding (numpy -> PNG -> base64)
# ------------------------------------------------------------
def encode_image(img_np: Any) -> Optional[str]:
    """
    Convert an image payload to base64-encoded PNG text.

    Parameters
    ----------
    img_np : Any
        Image payload as ``numpy.ndarray`` or raw bytes.

    Returns
    -------
    Optional[str]
        Base64-encoded PNG payload, or ``None`` if conversion fails.
    """
    if img_np is None:
        return None
    try:
        if isinstance(img_np, np.ndarray):
            if img_np.dtype != np.uint8:
                img_np = img_np.astype(np.uint8)
            if img_np.ndim == 2:
                img_np = np.stack([img_np] * 3, axis=-1)
            if img_np.ndim == 3 and img_np.shape[2] == 4:
                img_np = img_np[:, :, :3]

            pil = Image.fromarray(img_np)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("ascii")

        if isinstance(img_np, (bytes, bytearray)):
            return base64.b64encode(img_np).decode("ascii")

    except Exception as e:
        log(f"encode_image ERROR: {e}", "31")

    return None


def serialize_obs(obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serialize a raw observation to a JSON-friendly payload.

    Parameters
    ----------
    obs : Dict[str, Any]
        Raw observation from DesktopEnv.

    Returns
    -------
    Dict[str, Any]
        Serialized observation with screenshot encoded as base64.
    """
    return {
        "screenshot": encode_image(obs.get("screenshot")),
        "accessibility_tree": obs.get("accessibility_tree"),
        "terminal": obs.get("terminal"),
        "instruction": obs.get("instruction"),
    }


# ------------------------------------------------------------
# Client handler
# ------------------------------------------------------------
def handle_client(conn: socket.socket, addr: Any, env_holder: Dict[str, Any]) -> None:
    """
    Handle one TCP client request and send back a JSON response.

    Parameters
    ----------
    conn : socket.socket
        Accepted client socket.
    addr : Any
        Client address metadata from ``socket.accept``.
    env_holder : Dict[str, Any]
        Shared mutable container that stores the current ``DesktopEnv`` instance.

    Returns
    -------
    None
        No return value.
    """
    _ = addr
    try:
        raw = conn.recv(50_000_000).decode()
        req = json.loads(raw)
        cmd = req.get("cmd")

        log(f"Received command: {cmd}", "36")
        print_dict(req, title="JSON request content:")

        # ------------------------------
        # Initialize DesktopEnv
        # ------------------------------
        if cmd == "init_env":
            cfg = req.get("config", {})
            log("Initializing DesktopEnv with agent config...", "33")

            env_holder["env"] = DesktopEnv(
                provider_name=cfg["provider_name"],
                path_to_vm=cfg["path_to_vm"],
                headless=cfg["headless"],
                action_space=cfg["action_space"],
                require_terminal=cfg["require_terminal"],
                require_a11y_tree=cfg["require_a11y_tree"],
                os_type=cfg["os_type"],
                snapshot_name=cfg["snapshot_name"],
                enable_proxy=cfg["enable_proxy"],
                client_password=cfg["client_password"],
                screen_size=(1920, 1080),
            )

            log("DesktopEnv interface initialized!", "33")
            conn.send(json.dumps({"status": "ok"}).encode())
            return

        # ------------------------------
        # Guard: environment not initialized
        # ------------------------------
        if env_holder["env"] is None:
            conn.send(json.dumps({"error": "DesktopEnv not initialized. Call init_env first."}).encode())
            return

        env = env_holder["env"]

        # ------------------------------
        # Core observation and action commands
        # ------------------------------
        if cmd == "observe":
            resp = serialize_obs(env._get_obs())

        elif cmd == "reset":
            resp = serialize_obs(env.reset(task_config=req.get("task", {})))

        elif cmd == "step":
            obs, reward, done, info = env.step(req.get("action"), pause=req.get("pause", 1))
            resp = {
                "obs": serialize_obs(obs),
                "reward": reward,
                "done": done,
                "info": info,
            }

        # ------------------------------
        # Execute one Python command in the VM
        # ------------------------------
        elif cmd == "python_exec":
            code = req.get("code") or req.get("command")
            if not isinstance(code, str):
                resp = {"error": "python_exec requires string field 'code' (or 'command')."}
            else:
                # Let VM-side exceptions bubble to the main handler.
                result = env.controller.execute_python_command(code)
                resp = {"status": "ok", "result": result}

        # ------------------------------
        # Execute a multi-line Python script in the VM
        # ------------------------------
        elif cmd == "python_script":
            script = req.get("script") or req.get("code")
            if not isinstance(script, str):
                resp = {"error": "python_script requires string field 'script' (or 'code')."}
            else:
                result = env.controller.run_python_script(script)
                # ``run_python_script`` already returns status/output/error fields.
                resp = {"status": "ok", "result": result}

        # ------------------------------
        # Execute shell script, with Python fallback when native call fails
        # ------------------------------
        elif cmd == "bash_script":
            script = req.get("script")
            timeout = req.get("timeout", 30)
            working_dir = req.get("working_dir", None)

            if not isinstance(script, str):
                # Keep response schema consistent with other endpoints.
                resp = {
                    "status": "error",
                    "result": {
                        "status": "error",
                        "returncode": -1,
                        "output": "",
                        "error": "bash_script requires string field 'script'.",
                    },
                }
            else:
                # 1) First attempt: native OSWorld API.
                native = env.controller.run_bash_script(
                    script=script,
                    timeout=int(timeout),
                    working_dir=working_dir,
                )

                if isinstance(native, dict) and native.get("status") == "success":
                    resp = {"status": "ok", "result": native}
                else:
                    # 2) Fallback through ``run_python_script`` + subprocess.
                    log(
                        "run_bash_script() native failed, falling back via run_python_script+subprocess",
                        "33",
                    )

                    py_code = f"""
import subprocess, sys

cmd = {script!r}
proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
sys.stdout.write(proc.stdout)
sys.stderr.write(proc.stderr)
sys.stdout.flush()
sys.stderr.flush()
"""

                    try:
                        py_result = env.controller.run_python_script(py_code)
                        result = {
                            "status": py_result.get("status", "error"),
                            "returncode": py_result.get("return_code", -1),
                            "output": py_result.get("output", ""),
                            "error": py_result.get("error", ""),
                        }
                        resp = {"status": "ok", "result": result}
                    except Exception as e:
                        resp = {
                            "status": "error",
                            "result": {
                                "status": "error",
                                "returncode": -1,
                                "output": "",
                                "error": f"bash_script fallback error: {e!r}",
                            },
                        }

        # ------------------------------
        # Return quick VM metadata
        # ------------------------------
        elif cmd == "vm_info":
            try:
                platform = env.vm_platform
            except Exception:
                platform = None
            try:
                screen_size = env.vm_screen_size
            except Exception:
                screen_size = None

            resp = {
                "status": "ok",
                "platform": platform,
                "screen_size": screen_size,
            }

        elif cmd == "evaluate":
            # Evaluate current task completion via DesktopEnv.
            try:
                metric = env.evaluate()

                # Normalize possible numpy scalar payload.
                if isinstance(metric, (np.generic,)):
                    metric = metric.item()

                # Expose a best-effort boolean success indicator.
                success = None
                if isinstance(metric, (int, float, bool)):
                    mval = float(metric)
                    success = bool(mval >= 1.0) if mval not in (0.0, 1.0) else bool(int(mval) == 1)

                resp = {
                    "status": "ok",
                    "result": {
                        "metric": metric,
                        "success": success,
                    },
                }

            except Exception as e:
                # Keep a predictable schema and include traceback for debugging.
                resp = {
                    "status": "error",
                    "result": {
                        "error": repr(e),
                        "traceback": traceback.format_exc(),
                    },
                }

        # ------------------------------
        # Unknown command
        # ------------------------------
        else:
            resp = {"error": f"Unknown command '{cmd}'"}

        log(f"Command {cmd} processed.", "36")
        print_dict(
            resp,
            title="Content of the answer (screenshot and AT not shown):",
            avoid_keys=["accessibility_tree", "screenshot"],
        )

        screenshot_present = False
        accessibility_tree_present = False
        if isinstance(resp, dict):
            screenshot_present = resp.get("screenshot") is not None
            accessibility_tree_present = resp.get("accessibility_tree") is not None
            obs = resp.get("obs")
            if isinstance(obs, dict):
                screenshot_present = screenshot_present or (obs.get("screenshot") is not None)
                accessibility_tree_present = accessibility_tree_present or (obs.get("accessibility_tree") is not None)

        log(f"Screenshot: {screenshot_present}")
        log(f"Accessibility tree: {accessibility_tree_present}")
        conn.send(json.dumps(resp).encode())

    except Exception as e:
        log("Exception in handler", "31")
        tb = traceback.format_exc()
        print(tb)
        try:
            err = {
                "status": "error",
                "error": repr(e),
                "traceback": tb,
            }
            conn.send(json.dumps(err).encode())
        except Exception:
            pass
    finally:
        conn.close()


# ------------------------------------------------------------
# Main server loop with Ctrl+C support
# ------------------------------------------------------------
def main() -> None:
    """
    Start the OSWorld TCP server loop and serve client commands.

    Returns
    -------
    None
        No return value.
    """
    log("OSWorld Server starting (DesktopEnv not initialized yet)...", "33")
    log(f"DesktopEnv loaded from: {inspect.getfile(DesktopEnv)}", "33")

    env_holder = {"env": None}

    sock = socket.socket()
    sock.bind((HOST, PORT))
    sock.listen(5)

    # Use timeout so Ctrl+C can interrupt the accept loop reliably.
    sock.settimeout(0.5)

    log(f"Server listening on {HOST}:{PORT}", "32")

    try:
        while True:
            try:
                conn, addr = sock.accept()
                threading.Thread(
                    target=handle_client,
                    args=(conn, addr, env_holder),
                    daemon=True,
                ).start()

            except socket.timeout:
                # Loop again to keep keyboard interrupt responsive.
                continue

    except KeyboardInterrupt:
        log("Shutting down server via Ctrl+C ...", "31")

    finally:
        try:
            sock.close()
        except Exception:
            pass

        log("Server terminated cleanly.", "31")
        sys.exit(0)


if __name__ == "__main__":
    main()
