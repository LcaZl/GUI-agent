import argparse
import csv
import datetime as dt
import hashlib
import json
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List


ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_COLUMNS = [
    "timestamp",
    "episode_id",
    "status",
    "success",
    "osworld_status",
    "osworld_metric",
    "os_name",
    "desktop_env",
    "display_server",
    "started_ts_ms",
    "finished_ts_ms",
    "task_id",
    "task_snapshot",
    "task_source",
    "task_proxy",
    "task_fixed_ip",
    "task_possibility_of_env_change",
    "evaluator_func",
    "related_apps",
    "instruction_len_chars",
    "instruction_len_words",
    "chunks_total",
    "chunks_success",
    "chunks_fail",
    "chunk_success_rate",
    "chunk_fail_rate",
    "chunk_steps_total",
    "steps_total",
    "steps_eval_total",
    "steps_success",
    "steps_fail",
    "steps_without_evaluation",
    "step_success_rate",
    "step_eval_coverage",
    "avg_steps_per_chunk",
    "min_steps_per_chunk",
    "max_steps_per_chunk",
    "avg_pause_per_step_sec",
    "total_planned_pause_sec",
    "step_confidence_mean",
    "step_confidence_std",
    "step_confidence_min",
    "step_confidence_max",
    "failing_step_index_mean",
    "failing_step_index_min",
    "failing_step_index_max",
    "first_failure_chunk_index",
    "max_consecutive_chunk_failures",
    "planner_done_judge_fail",
    "planner_fail_judge_success",
    "recoveries_after_fail_total",
    "recoveries_after_fail_opportunities",
    "recovery_after_fail_rate",
    "observations_total",
    "trim_updates_total",
    "llm_requests_total",
    "history_events_total",
    "unique_failure_types",
    "unique_action_types",
    "failure_type_counts",
    "decision_counts",
    "action_type_counts",
    "episode_duration_ms",
    "episode_duration_sec",
    "agent_name",
    "agent_settings_hash",
    "agent_settings_json",
    "experiment_id",
    "experiment_name",
    "experiment_started_at",
    "experiment_config_path",
    "experiment_config_hash",
    "experiment_config_json",
    "experiment_task_index",
    "experiment_tasks_total",
    "experiment_reuse_agent",
    "experiment_agent_id",
    "experiment_agent_instance",
    "experiment_max_cycles",
]


def _repo_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return ROOT_DIR / p


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate dummy experiments CSV for notebook testing."
    )
    parser.add_argument(
        "--output",
        default="data/dummy_experiments.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--schema-from",
        default="data/experiments.csv",
        help="Read header schema from this CSV if available.",
    )
    parser.add_argument(
        "--tasks-file",
        default="data/tasks.json",
        help="Optional tasks file used to reuse real task ids.",
    )
    parser.add_argument("--experiments", type=int, default=3, help="Number of experiments.")
    parser.add_argument(
        "--tasks-per-experiment", type=int, default=10, help="Episodes generated per experiment."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if already present.",
    )
    return parser.parse_args()


def _load_schema(path: Path) -> List[str]:
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                header = []
        if header:
            return header
    return DEFAULT_COLUMNS


def _load_task_ids(path: Path, total_needed: int) -> List[str]:
    fallback = [f"dummy_task_{i + 1:02d}" for i in range(total_needed)]
    if not path.exists():
        return fallback
    try:
        if path.suffix.lower() == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
        else:
            return fallback
        if isinstance(raw, dict):
            raw = [raw]
        if not isinstance(raw, list):
            return fallback
        ids = []
        for i, item in enumerate(raw):
            if isinstance(item, dict):
                ids.append(str(item.get("id", f"task_{i + 1:02d}")))
        if not ids:
            return fallback
        while len(ids) < total_needed:
            ids.extend(ids)
        return ids[:total_needed]
    except Exception:
        return fallback


def _json_str(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _profile(exp_index: int) -> Dict[str, Any]:
    presets = [
        {
            "name": "baseline",
            "success_p": 0.45,
            "duration_min": 70,
            "duration_max": 180,
            "max_cycles": 40,
            "temperature": 0.5,
            "num_rollouts": 2,
        },
        {
            "name": "memory_tuned",
            "success_p": 0.65,
            "duration_min": 60,
            "duration_max": 150,
            "max_cycles": 45,
            "temperature": 0.3,
            "num_rollouts": 3,
        },
        {
            "name": "high_budget",
            "success_p": 0.78,
            "duration_min": 50,
            "duration_max": 135,
            "max_cycles": 55,
            "temperature": 0.2,
            "num_rollouts": 4,
        },
    ]
    if exp_index < len(presets):
        return presets[exp_index]
    base = presets[-1].copy()
    base["name"] = f"variant_{exp_index + 1}"
    base["success_p"] = min(0.9, 0.6 + 0.03 * exp_index)
    base["max_cycles"] = 45 + exp_index
    return base


def _make_agent_config(profile: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "planner_settings": {
            "max_plan_steps": profile["max_cycles"],
            "num_rollouts": profile["num_rollouts"],
        },
        "gpt_client_settings": {"model": "gpt-4o", "temperature": profile["temperature"]},
        "memory_settings": {"initialize_memory": True, "memory_name": "mem00"},
        "plan_executor_settings": {"default_pause_sec": 1.0},
    }


def _sample_action_counts(rng: random.Random, steps_total: int) -> Dict[str, int]:
    action_types = ["CLICK", "TYPE", "WAIT", "PYTHON", "SCROLL"]
    counts = {k: 0 for k in action_types}
    for _ in range(max(1, steps_total)):
        counts[rng.choice(action_types)] += 1
    return {k: v for k, v in counts.items() if v > 0}


def _sample_failure_counts(rng: random.Random, failed: bool) -> Dict[str, int]:
    if not failed:
        return {}
    options = [
        "ACTION_INEFFECTIVE",
        "WRONG_TARGET",
        "UI_NOT_READY",
        "ENV_LIMITATION",
        "UNCLEAR",
    ]
    picked = rng.choice(options)
    return {picked: 1}


def _safe_rate(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def _build_row(
    fieldnames: List[str],
    rng: random.Random,
    *,
    exp_index: int,
    task_index: int,
    tasks_total: int,
    task_id: str,
    exp_id: str,
    exp_name: str,
    exp_start: dt.datetime,
    agent_cfg: Dict[str, Any],
    max_cycles: int,
) -> Dict[str, Any]:
    success = rng.random() < _profile(exp_index)["success_p"]
    status = "SUCCESS" if success else "FAIL"

    chunks_total = rng.randint(2, 6)
    chunks_success = chunks_total if success else rng.randint(0, chunks_total - 1)
    chunks_fail = chunks_total - chunks_success

    steps_total = rng.randint(chunks_total + 1, chunks_total * 4)
    steps_eval_total = max(1, steps_total - rng.randint(0, 2))
    steps_success = min(steps_eval_total, rng.randint(0, steps_eval_total) if not success else rng.randint(max(1, steps_eval_total - 2), steps_eval_total))
    steps_fail = max(0, steps_eval_total - steps_success)
    steps_without_evaluation = max(0, steps_total - steps_eval_total)

    avg_pause = round(rng.uniform(0.8, 1.8), 3)
    total_pause = round(avg_pause * steps_total, 3)
    step_conf_mean = round(rng.uniform(0.75, 0.98) if success else rng.uniform(0.35, 0.8), 3)
    step_conf_std = round(rng.uniform(0.03, 0.18), 3)
    step_conf_min = max(0.0, round(step_conf_mean - step_conf_std * 1.5, 3))
    step_conf_max = min(1.0, round(step_conf_mean + step_conf_std * 1.5, 3))

    failing_step = 0 if success else rng.randint(0, max(0, steps_eval_total - 1))
    first_failure_chunk = 0 if success else rng.randint(0, max(0, chunks_total - 1))
    max_consecutive_chunk_failures = 0 if success else rng.randint(1, min(3, chunks_fail))

    recoveries_opportunities = rng.randint(0, 2) if success else rng.randint(1, 3)
    recoveries_total = (
        recoveries_opportunities if success else rng.randint(0, recoveries_opportunities)
    )
    recovery_rate = _safe_rate(recoveries_total, recoveries_opportunities)

    observations_total = chunks_total + rng.randint(1, 4)
    trim_updates_total = rng.randint(0, 4)
    llm_requests_total = chunks_total * 2 + rng.randint(0, 4)
    history_events_total = observations_total + llm_requests_total + rng.randint(0, 4)

    failure_type_counts = _sample_failure_counts(rng, failed=not success)
    action_type_counts = _sample_action_counts(rng, steps_total=steps_total)
    decision_counts = {"CONTINUE": chunks_total}
    decision_counts["DONE" if success else "FAIL"] = 1

    unique_failure_types = len(failure_type_counts)
    unique_action_types = len(action_type_counts)

    duration_sec = round(rng.uniform(_profile(exp_index)["duration_min"], _profile(exp_index)["duration_max"]), 3)
    duration_ms = int(duration_sec * 1000)

    started = exp_start + dt.timedelta(minutes=(task_index - 1) * 3)
    finished = started + dt.timedelta(seconds=duration_sec)
    started_ts_ms = int(started.timestamp() * 1000)
    finished_ts_ms = int(finished.timestamp() * 1000)

    related_apps = rng.sample(
        ["os", "firefox", "terminal", "files", "settings"], k=rng.randint(1, 2)
    )
    instruction_words = rng.randint(10, 35)
    instruction_chars = instruction_words * rng.randint(5, 8)

    agent_cfg_json = _json_str(agent_cfg)
    agent_cfg_hash = hashlib.sha1(agent_cfg_json.encode("utf-8")).hexdigest()

    values: Dict[str, Any] = {
        "timestamp": finished.isoformat(),
        "episode_id": uuid.uuid4().hex,
        "status": status,
        "success": success,
        "osworld_status": "ok",
        "osworld_metric": 1.0 if success else 0.0,
        "os_name": "Ubuntu 22.04.5 LTS",
        "desktop_env": "GNOME",
        "display_server": "x11",
        "started_ts_ms": started_ts_ms,
        "finished_ts_ms": finished_ts_ms,
        "task_id": task_id,
        "task_snapshot": "os",
        "task_source": "dummy://synthetic",
        "task_proxy": False,
        "task_fixed_ip": False,
        "task_possibility_of_env_change": "low",
        "evaluator_func": "exact_match",
        "related_apps": _json_str(related_apps),
        "instruction_len_chars": instruction_chars,
        "instruction_len_words": instruction_words,
        "chunks_total": chunks_total,
        "chunks_success": chunks_success,
        "chunks_fail": chunks_fail,
        "chunk_success_rate": _safe_rate(chunks_success, chunks_total),
        "chunk_fail_rate": _safe_rate(chunks_fail, chunks_total),
        "chunk_steps_total": steps_total,
        "steps_total": steps_total,
        "steps_eval_total": steps_eval_total,
        "steps_success": steps_success,
        "steps_fail": steps_fail,
        "steps_without_evaluation": steps_without_evaluation,
        "step_success_rate": _safe_rate(steps_success, steps_eval_total),
        "step_eval_coverage": _safe_rate(steps_eval_total, steps_total),
        "avg_steps_per_chunk": _safe_rate(steps_total, chunks_total),
        "min_steps_per_chunk": 1,
        "max_steps_per_chunk": max(2, int(round(_safe_rate(steps_total, chunks_total) + 1))),
        "avg_pause_per_step_sec": avg_pause,
        "total_planned_pause_sec": total_pause,
        "step_confidence_mean": step_conf_mean,
        "step_confidence_std": step_conf_std,
        "step_confidence_min": step_conf_min,
        "step_confidence_max": step_conf_max,
        "failing_step_index_mean": failing_step,
        "failing_step_index_min": failing_step,
        "failing_step_index_max": failing_step,
        "first_failure_chunk_index": first_failure_chunk,
        "max_consecutive_chunk_failures": max_consecutive_chunk_failures,
        "planner_done_judge_fail": 0,
        "planner_fail_judge_success": 0,
        "recoveries_after_fail_total": recoveries_total,
        "recoveries_after_fail_opportunities": recoveries_opportunities,
        "recovery_after_fail_rate": recovery_rate,
        "observations_total": observations_total,
        "trim_updates_total": trim_updates_total,
        "llm_requests_total": llm_requests_total,
        "history_events_total": history_events_total,
        "unique_failure_types": unique_failure_types,
        "unique_action_types": unique_action_types,
        "failure_type_counts": _json_str(failure_type_counts),
        "decision_counts": _json_str(decision_counts),
        "action_type_counts": _json_str(action_type_counts),
        "episode_duration_ms": duration_ms,
        "episode_duration_sec": duration_sec,
        "agent_name": "batch-agent",
        "agent_settings_hash": agent_cfg_hash,
        "agent_settings_json": agent_cfg_json,
        "experiment_id": exp_id,
        "experiment_name": exp_name,
        "experiment_started_at": exp_start.isoformat(),
        "experiment_config_path": f"agent/config_files/{exp_name}.yaml",
        "experiment_config_hash": agent_cfg_hash,
        "experiment_config_json": agent_cfg_json,
        "experiment_task_index": task_index,
        "experiment_tasks_total": tasks_total,
        "experiment_reuse_agent": True,
        "experiment_agent_id": "batch-agent",
        "experiment_agent_instance": f"batch-agent-{exp_index + 1}",
        "experiment_max_cycles": max_cycles,
    }

    row = {col: "" for col in fieldnames}
    for key, value in values.items():
        if key in row:
            row[key] = value
    return row


def main() -> None:
    args = _parse_args()
    output_path = _repo_path(args.output)
    schema_path = _repo_path(args.schema_from)
    tasks_path = _repo_path(args.tasks_file)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )

    fieldnames = _load_schema(schema_path)
    rng = random.Random(args.seed)

    total_tasks_needed = max(1, args.tasks_per_experiment)
    task_ids = _load_task_ids(tasks_path, total_tasks_needed)
    now = dt.datetime.now().replace(microsecond=0)

    rows: List[Dict[str, Any]] = []
    for exp_index in range(args.experiments):
        profile = _profile(exp_index)
        exp_name = profile["name"]
        exp_id = f"dummy_exp_{now.strftime('%Y%m%d_%H%M%S')}_{exp_index + 1:02d}"
        exp_start = now + dt.timedelta(hours=exp_index)
        agent_cfg = _make_agent_config(profile)
        for task_index in range(1, args.tasks_per_experiment + 1):
            task_id = task_ids[task_index - 1]
            row = _build_row(
                fieldnames,
                rng,
                exp_index=exp_index,
                task_index=task_index,
                tasks_total=args.tasks_per_experiment,
                task_id=task_id,
                exp_id=exp_id,
                exp_name=exp_name,
                exp_start=exp_start,
                agent_cfg=agent_cfg,
                max_cycles=profile["max_cycles"],
            )
            rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Dummy CSV created: {output_path}")
    print(f"Rows written: {len(rows)}")
    print(f"Experiments: {args.experiments}")
    print(f"Tasks per experiment: {args.tasks_per_experiment}")


if __name__ == "__main__":
    main()
