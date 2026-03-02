import argparse
import asyncio
import datetime
import json
import logging
import pathlib
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

from agentz import Agent
from agentz.pydantic_models import ExperimentConfiguration

from dotenv import load_dotenv
load_dotenv()  # loads .env from current working directory by default


def _load_yaml(path: Path) -> Dict[str, Any]:
    """
    Load yaml.
        
        Parameters
        ----------
        path : Path
            Filesystem path.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with computed fields.
        
    """
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_task(path: Path) -> Dict[str, Any]:
    """
    Load task.
        
        Parameters
        ----------
        path : Path
            Filesystem path.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with computed fields.
        
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text) or {}
    if path.suffix.lower() == ".json":
        return json.loads(text)
    raise ValueError(f"Unsupported task file extension: {path.suffix} (use .json/.yaml/.yml)")


def _setup_logging(cfg: Dict[str, Any], main_name: str) -> str:
    """
    Set up logging.
        
        Parameters
        ----------
        cfg : Dict[str, Any]
            Function argument.
        main_name : str
            Function argument.
        
        Returns
        -------
        str
            Resulting string value.
        
    """
    log_dir = pathlib.Path(cfg.get("log_dir", "../data/logs/agent"))
    log_dir.mkdir(exist_ok=True, parents=True)
    log_level = str(cfg.get("log_level", "INFO")).upper()

    logging.getLogger().setLevel(logging.NOTSET)

    log_name = f"{main_name}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
    log_path = log_dir / log_name

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.getLevelName(log_level))
    console.setFormatter(logging.Formatter("%(asctime)s [%(name)s-%(levelname)s]: %(message)s"))
    logging.getLogger().addHandler(console)

    logfile = logging.FileHandler(filename=log_path)
    logfile.setLevel(logging.DEBUG)
    logfile.setFormatter(logging.Formatter("%(asctime)s [%(name)s-%(levelname)s]: %(message)s"))
    logging.getLogger().addHandler(logfile)

    logging.info("Logging configured. log_file=%s", str(log_path))
    return log_name


def _teardown_logging(start_time: datetime.datetime) -> None:
    """
    Tear down logging.
        
        Parameters
        ----------
        start_time : datetime.datetime
            Function argument.
        
        Returns
        -------
        None
            No return value.
        
    """
    logging.info("Close all. Execution time: %s", datetime.datetime.now() - start_time)
    logging.getLogger().handlers.clear()
    logging.shutdown()


def _default_demo_task() -> Dict[str, Any]:
    """
    Return default demo task.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with computed fields.
        
    """
    return {
        "id": "demo-install-spotify",
        "instruction": "I want to install Spotify on my current system. Could you please help me?",
        "config": [
            {
                "type": "execute",
                "parameters": {"command": ["python", "-c", "import pyautogui,time; pyautogui.click(960,540); time.sleep(0.5);"]},
            }
        ],
        "evaluator": {
            "func": "check_include_exclude",
            "result": {"type": "vm_command_line", "command": "which spotify"},
            "expected": {"type": "rule", "rules": {"include": ["spotify"], "exclude": ["not found"]}},
        },
    }


def build_parser() -> argparse.ArgumentParser:
    """
    Build parser.
        
        Returns
        -------
        argparse.ArgumentParser
            Function result.
        
    """
    p = argparse.ArgumentParser(description="Run AgentZ from a YAML configuration file.")
    p.add_argument("--conf", "-c", required=True, metavar="FILE", help="YAML configuration file for the agent.")
    p.add_argument("--task", "-t", default=None, metavar="FILE", help="Task file (.json/.yaml). If omitted, uses a demo task.")
    p.add_argument("--agent-id", default="g01", help="Agent identifier used in logs/state.")
    p.add_argument("--mode", default="bdi", choices=["bdi"], help="Execution mode (extend later if needed).")
    p.add_argument("--max-cycles", type=int, default=0, help="Max cycles for the agent loop (0 = agent default).")
    p.add_argument("--dry-run", action="store_true", help="Load config/task and exit without running the agent.")
    return p


def main() -> None:
    """
    Run main for the current workflow step.
    
    Returns
    -------
    None
        No return value.
    """
    start_time = datetime.datetime.now()
    args = build_parser().parse_args()

    cfg_path = Path(args.conf)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config file: {cfg_path}")

    cfg = _load_yaml(cfg_path)
    log_name = _setup_logging(cfg, main_name="run_agent")
    cfg["log_name"] = log_name

    try:
        settings = ExperimentConfiguration(**cfg)

        task: Dict[str, Any]
        if args.task:
            task_path = Path(args.task)
            if not task_path.exists():
                raise FileNotFoundError(f"Missing task file: {task_path}")
            task = _load_task(task_path)
        else:
            task = _default_demo_task()

        logging.info("Config loaded from: %s", str(cfg_path))
        logging.info("Task id=%s", task.get("id"))

        if args.dry_run:
            logging.info("Dry-run requested. Exiting without executing.")
            return

        agent = asyncio.run(Agent.create(args.agent_id, settings))

        if args.mode == "bdi":
            if args.max_cycles and args.max_cycles > 0:
                agent.run_task_bdi(task=task, max_cycles=args.max_cycles)
            else:
                agent.run_task_bdi(task=task)

    except Exception:
        logging.exception("Fatal error while running agent.")
        raise
    finally:
        _teardown_logging(start_time)


if __name__ == "__main__":
    main()
