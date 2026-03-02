import argparse
import asyncio
import csv
import datetime
import hashlib
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

ROOT_DIR = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT_DIR / "agent"

if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from agentz import Agent
from agentz.pydantic_models import ExperimentConfiguration


def _suppress_warnings() -> None:
    # Suppress all Python warnings for cleaner batch output.
    os.environ["PYTHONWARNINGS"] = "ignore"
    warnings.simplefilter("ignore")
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.handlers.clear()
    warnings_logger.propagate = False
    warnings_logger.setLevel(logging.CRITICAL + 1)


def _repo_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return ROOT_DIR / p


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f) or {}


def _load_tasks(path: Path) -> List[Dict[str, Any]]:
    # Use utf-8-sig to transparently support UTF-8 files with BOM.
    text = path.read_text(encoding="utf-8-sig")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text) or []
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported task file extension: {path.suffix}")

    if isinstance(data, dict):
        return [data]
    if not isinstance(data, list):
        raise ValueError("Task file must contain a list or a single dict task.")

    tasks: List[Dict[str, Any]] = []
    for i, task in enumerate(data):
        if not isinstance(task, dict):
            raise ValueError(f"Task at index {i} is not an object.")
        tasks.append(task)
    return tasks


def _count_csv_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        total_lines = sum(1 for _ in reader)
    return max(0, total_lines - 1)


def _completed_task_ids_from_metrics(
    path: Path,
    *,
    experiment_id: str,
    experiment_name: str,
    experiment_config_hash: str,
) -> set[str]:
    """
    Read completed task ids from the metrics CSV for the current experiment context.

    A row is considered in-scope when either:
    - `experiment_id` matches exactly, or
    - `experiment_name` matches and `experiment_config_hash` matches (or is missing).
    """
    completed: set[str] = set()
    if not path.exists() or path.stat().st_size == 0:
        return completed

    try:
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "task_id" not in reader.fieldnames:
                return completed

            for row in reader:
                task_id = str(row.get("task_id", "")).strip()
                if not task_id:
                    continue

                row_exp_id = str(row.get("experiment_id", "")).strip()
                row_exp_name = str(row.get("experiment_name", "")).strip()
                row_cfg_hash = str(row.get("experiment_config_hash", "")).strip()

                same_experiment = row_exp_id and (row_exp_id == experiment_id)
                same_named_run = (
                    row_exp_name
                    and row_exp_name == experiment_name
                    and ((not row_cfg_hash) or row_cfg_hash == experiment_config_hash)
                )

                if same_experiment or same_named_run:
                    completed.add(task_id)
    except Exception:
        logging.exception("Failed reading metrics CSV for resume-skip logic: %s", path)

    return completed


def _setup_logging(log_dir: Path, log_level: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = log_dir / f"batch_run_{ts}.log"

    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    root_logger.setLevel(logging.NOTSET)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    console.setFormatter(logging.Formatter("%(asctime)s [%(name)s-%(levelname)s]: %(message)s"))
    root_logger.addHandler(console)

    logfile = logging.FileHandler(filename=log_path, encoding="utf-8")
    logfile.setLevel(logging.DEBUG)
    logfile.setFormatter(logging.Formatter("%(asctime)s [%(name)s-%(levelname)s]: %(message)s"))
    root_logger.addHandler(logfile)

    logging.info("Logging configured. log_file=%s", log_path)
    return log_path


def _teardown_logging(start_ts: datetime.datetime) -> None:
    elapsed = datetime.datetime.now() - start_ts
    logging.info("Batch finished. Elapsed=%s", elapsed)
    logging.getLogger().handlers.clear()
    logging.shutdown()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AgentZ on a list of tasks in one terminal execution."
    )
    parser.add_argument(
        "--conf",
        default="agent/config_files/start_agent.yaml",
        help="YAML config file for ExperimentConfiguration.",
    )
    parser.add_argument(
        "--tasks",
        default="data/tasks.json",
        help="Task file (.json/.yaml/.yml) containing a list of tasks.",
    )
    parser.add_argument(
        "--metrics-csv",
        default="data/experiments.csv",
        help="CSV written by append_metrics_csv.",
    )
    parser.add_argument("--agent-id", default="batch-agent", help="Agent identifier.")
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Stable experiment id for this batch run. If omitted, auto-generated.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Human-readable experiment label (e.g., baseline-v1).",
    )
    parser.add_argument("--max-cycles", type=int, default=0, help="0 means use agent default.")
    parser.add_argument("--start-index", type=int, default=0, help="Start index in task list.")
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit.")
    parser.add_argument(
        "--task-ids",
        nargs="*",
        default=None,
        help="Optional list of task ids to run (exact match).",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop batch at first task exception.",
    )
    parser.add_argument(
        "--no-reuse-agent",
        action="store_true",
        help="Create a fresh agent for each task.",
    )
    parser.add_argument(
        "--verbose-ui",
        action="store_true",
        help="Enable UI visualization windows (slower).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Load everything and exit.")
    parser.add_argument(
        "--skip-existing-task-ids",
        action="store_true",
        help="Skip task IDs already present in metrics CSV for the same experiment context.",
    )
    parser.add_argument("--log-dir", default="data/logs/agent", help="Batch log directory.")
    parser.add_argument("--log-level", default="INFO", help="Console log level.")
    return parser.parse_args()


def _write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    start_ts = datetime.datetime.now()
    args = _parse_args()
    _suppress_warnings()

    conf_path = _repo_path(args.conf)
    task_path = _repo_path(args.tasks)
    metrics_path = _repo_path(args.metrics_csv)
    log_dir = _repo_path(args.log_dir)

    if not conf_path.exists():
        raise FileNotFoundError(f"Missing config file: {conf_path}")
    if not task_path.exists():
        raise FileNotFoundError(f"Missing task file: {task_path}")

    _setup_logging(log_dir=log_dir, log_level=args.log_level)

    if load_dotenv is not None:
        load_dotenv(ROOT_DIR / ".env")
    else:
        logging.info("python-dotenv is not installed. Skipping .env loading.")

    try:
        cfg = _load_yaml(conf_path)
        settings = ExperimentConfiguration(**cfg)
        all_tasks = _load_tasks(task_path)

        app_config_json = json.dumps(cfg, ensure_ascii=False, sort_keys=True, default=str)
        app_config_hash = hashlib.sha1(app_config_json.encode("utf-8")).hexdigest()
        experiment_id = args.experiment_id or datetime.datetime.now().strftime("exp_%Y%m%d_%H%M%S")
        experiment_name = args.experiment_name or conf_path.stem
        experiment_started_at = datetime.datetime.now().isoformat()

        indexed_tasks: List[tuple[int, Dict[str, Any]]] = list(enumerate(all_tasks, start=1))
        indexed_tasks = indexed_tasks[args.start_index :]
        if args.task_ids:
            allowed = set(args.task_ids)
            indexed_tasks = [(i, t) for i, t in indexed_tasks if str(t.get("id")) in allowed]
        if args.limit > 0:
            indexed_tasks = indexed_tasks[: args.limit]

        planned_tasks_total = len(indexed_tasks)

        if args.skip_existing_task_ids:
            completed_task_ids = _completed_task_ids_from_metrics(
                metrics_path,
                experiment_id=experiment_id,
                experiment_name=experiment_name,
                experiment_config_hash=app_config_hash,
            )
            if completed_task_ids:
                before = len(indexed_tasks)
                indexed_tasks = [(i, t) for i, t in indexed_tasks if str(t.get("id")) not in completed_task_ids]
                skipped = before - len(indexed_tasks)
                if skipped > 0:
                    logging.info(
                        "Resume-skip enabled: skipped %d already-completed task(s) for experiment context.",
                        skipped,
                    )

        if not indexed_tasks:
            logging.info("No tasks left to execute after filters/resume-skip.")
            return

        reuse_agent = not args.no_reuse_agent

        logging.info("Batch setup complete")
        logging.info("Config: %s", conf_path)
        logging.info("Tasks file: %s", task_path)
        logging.info("Metrics CSV: %s", metrics_path)
        logging.info("Selected tasks: %d", len(indexed_tasks))
        logging.info("Reuse agent: %s", reuse_agent)
        logging.info("Experiment id: %s", experiment_id)
        logging.info("Experiment name: %s", experiment_name)
        logging.info("AppConfig path: %s", conf_path)
        logging.info("AppConfig hash: %s", app_config_hash)

        for queue_idx, (task_index, task) in enumerate(indexed_tasks, start=1):
            logging.info(
                "Task queue [%d/%d] task_index=%d id=%s",
                queue_idx,
                len(indexed_tasks),
                task_index,
                task.get("id"),
            )

        if args.dry_run:
            logging.info("Dry-run requested. Exiting without execution.")
            return

        run_rows: List[Dict[str, Any]] = []
        batch_t0 = time.perf_counter()
        agent: Agent | None = None

        def _create_agent(instance_name: str) -> Agent:
            logging.info("Creating agent instance: %s", instance_name)
            return asyncio.run(Agent.create(instance_name, settings))

        if reuse_agent:
            agent = _create_agent(args.agent_id)

        for queue_idx, (task_index, task) in enumerate(indexed_tasks, start=1):
            task_id = task.get("id")
            instruction = str(task.get("instruction", ""))
            logging.info(
                "=== Task %d/%d | task_index=%d | id=%s ===",
                queue_idx,
                len(indexed_tasks),
                task_index,
                task_id,
            )
            logging.info("Instruction: %s", instruction)

            if not reuse_agent:
                agent_name = f"{args.agent_id}-{queue_idx:03d}"
                agent = _create_agent(agent_name)

            task_t0 = time.perf_counter()
            rows_before = _count_csv_rows(metrics_path)
            episode = None
            task_error = None

            try:
                agent_instance_name = args.agent_id if reuse_agent else f"{args.agent_id}-{queue_idx:03d}"
                run_context = {
                    "experiment_id": experiment_id,
                    "experiment_name": experiment_name,
                    "experiment_started_at": experiment_started_at,
                    "experiment_config_path": str(conf_path),
                    "experiment_config_hash": app_config_hash,
                    "experiment_config_json": app_config_json,
                    "experiment_task_index": task_index,
                    "experiment_tasks_total": planned_tasks_total,
                    "experiment_reuse_agent": reuse_agent,
                    "experiment_agent_id": args.agent_id,
                    "experiment_agent_instance": agent_instance_name,
                    "experiment_max_cycles": args.max_cycles if args.max_cycles > 0 else None,
                }
                run_kwargs = {
                    "task": task,
                    "verbose": bool(args.verbose_ui),
                    "close_memory_on_finish": not reuse_agent,
                    "metrics_path": str(metrics_path),
                    "experiment_context": run_context,
                }
                if args.max_cycles > 0:
                    run_kwargs["max_cycles"] = args.max_cycles
                episode = agent.run_task_bdi(**run_kwargs)
            except Exception as exc:
                task_error = exc
                logging.exception("Task execution raised an exception | task_id=%s", task_id)
            finally:
                duration_sec = time.perf_counter() - task_t0
                rows_after = _count_csv_rows(metrics_path)
                rows_delta = rows_after - rows_before

                status = "ERROR"
                success = False
                episode_id = None
                if episode is not None:
                    status = str(getattr(episode, "status", "UNKNOWN"))
                    success = bool((getattr(episode, "score", None) or {}).get("success", False))
                    episode_id = getattr(episode, "episode_id", None)

                logging.info(
                    "Task result | id=%s | status=%s | success=%s | episode_id=%s | duration_sec=%.2f | metrics_rows_delta=%d",
                    task_id,
                    status,
                    success,
                    episode_id,
                    duration_sec,
                    rows_delta,
                )

                if rows_delta <= 0:
                    logging.info(
                        "Metrics CSV was not incremented for task id=%s. Check error logs and append_metrics_csv.",
                        task_id,
                    )

                run_rows.append(
                    {
                        "batch_timestamp": datetime.datetime.now().isoformat(),
                        "experiment_id": experiment_id,
                        "experiment_name": experiment_name,
                        "experiment_config_hash": app_config_hash,
                        "task_index": task_index,
                        "task_id": task_id,
                        "episode_id": episode_id,
                        "status": status,
                        "success": success,
                        "duration_sec": round(duration_sec, 3),
                        "metrics_rows_delta": rows_delta,
                        "error": "" if task_error is None else f"{type(task_error).__name__}: {task_error}",
                    }
                )

            if task_error is not None:
                if agent is not None:
                    try:
                        agent.stop()
                    except Exception:
                        pass
                    agent = None

                if args.stop_on_error:
                    logging.error("Stopping batch due to --stop-on-error.")
                    break

                if reuse_agent:
                    logging.info("Recreating shared agent after task error.")
                    agent = _create_agent(args.agent_id)

            if (not reuse_agent) and agent is not None:
                try:
                    agent.stop()
                except Exception:
                    pass
                agent = None

        if reuse_agent and agent is not None:
            try:
                agent.stop()
            except Exception:
                pass

        elapsed = time.perf_counter() - batch_t0
        total = len(run_rows)
        total_success = sum(1 for r in run_rows if bool(r["success"]))
        total_error = sum(1 for r in run_rows if str(r["status"]).upper() == "ERROR")
        total_metrics_ok = sum(1 for r in run_rows if int(r["metrics_rows_delta"]) > 0)

        logging.info("=== Batch Summary ===")
        logging.info("Tasks executed: %d", total)
        logging.info("Success count: %d", total_success)
        logging.info("Error count: %d", total_error)
        logging.info("Rows appended in metrics CSV: %d/%d", total_metrics_ok, total)
        logging.info("Total duration: %.2f sec", elapsed)

        summary_dir = ROOT_DIR / "data" / "experiment_runs"
        summary_name = datetime.datetime.now().strftime("batch_summary_%Y%m%d_%H%M%S.csv")
        summary_path = summary_dir / summary_name
        _write_summary_csv(summary_path, run_rows)
        logging.info("Batch summary CSV: %s", summary_path)

    finally:
        _teardown_logging(start_ts)


if __name__ == "__main__":
    main()
