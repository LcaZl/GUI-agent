from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Optional
import csv
import json
from pathlib import Path
from datetime import datetime
from statistics import pstdev


def _enum_to_str(value: Any) -> Optional[str]:
    """
    Process enum to str.
        
        Parameters
        ----------
        value : Any
            Input value.
        
        Returns
        -------
        Optional[str]
            Function result.
        
    """
    if value is None:
        return None
    if hasattr(value, "value"):
        try:
            return str(value.value)
        except Exception:
            pass
    return str(value)


def _normalize_action_type(action_type: Any) -> str:
    """
    Normalize action type.
        
        Parameters
        ----------
        action_type : Any
            Function argument.
        
        Returns
        -------
        str
            Resulting string value.
        
    """
    if action_type is None:
        return "(none)"
    if hasattr(action_type, "value"):
        try:
            return str(action_type.value).strip().upper() or "(none)"
        except Exception:
            pass
    s = str(action_type).strip().upper()
    return s or "(none)"


def _mean(values: list[float]) -> Optional[float]:
    """
    Process mean.
        
        Parameters
        ----------
        values : list[float]
            Function argument.
        
        Returns
        -------
        Optional[float]
            Function result.
        
    """
    if not values:
        return None
    return sum(values) / float(len(values))


def _std(values: list[float]) -> Optional[float]:
    """
    Process std.
        
        Parameters
        ----------
        values : list[float]
            Function argument.
        
        Returns
        -------
        Optional[float]
            Function result.
        
    """
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return float(pstdev(values))


def compute_episode_metrics(history_manager, episode: Optional[Any] = None) -> Dict[str, Any]:
    """
    Compute a compact, episode-level metrics bundle from HistoryManager.
    
    This function is intentionally dependency-light and side-effect free.
    """
    chunks = list(getattr(history_manager, "chunks_history", []) or [])
    steps = list(getattr(history_manager, "steps_history", []) or [])
    observations = list(getattr(history_manager, "observations_history", []) or [])
    trim_info = list(getattr(history_manager, "trim_info", []) or [])
    llm_requests = list(getattr(history_manager, "llm_requests", []) or [])
    full_history = list(getattr(history_manager, "full_history", []) or [])

    chunks_total = len(chunks)
    steps_total = len(steps)
    observations_total = len(observations)

    chunk_success = 0
    chunk_fail = 0
    failure_type_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    action_type_counts: Counter[str] = Counter()

    planner_done_judge_fail = 0
    planner_fail_judge_success = 0

    chunk_step_lengths: list[int] = []
    failing_step_indexes: list[int] = []
    confidence_values: list[float] = []

    first_failure_chunk_index: Optional[int] = None
    max_consecutive_chunk_failures = 0
    _current_fail_streak = 0

    for ch in chunks:
        chunk_step_lengths.append(len(getattr(ch, "steps", []) or []))

        if bool(getattr(ch, "overall_success", False)):
            chunk_success += 1
            _current_fail_streak = 0
        else:
            chunk_fail += 1
            _current_fail_streak += 1
            if first_failure_chunk_index is None:
                first_failure_chunk_index = len(chunk_step_lengths) - 1

        if _current_fail_streak > max_consecutive_chunk_failures:
            max_consecutive_chunk_failures = _current_fail_streak

        ft = _enum_to_str(getattr(ch, "failure_type", None))
        if ft:
            failure_type_counts[ft] += 1

        decision = _enum_to_str(getattr(ch, "decision", None))
        if decision:
            decision_counts[decision] += 1

        # Planner/Judge alignment checks
        if decision == "DONE" and bool(getattr(ch, "overall_success", False)) is False:
            planner_done_judge_fail += 1
        if decision == "FAIL" and bool(getattr(ch, "overall_success", False)) is True:
            planner_fail_judge_success += 1

        failing_step_index = getattr(ch, "failing_step_index", None)
        if isinstance(failing_step_index, int) and failing_step_index >= 0:
            failing_step_indexes.append(failing_step_index)

    # Step-level aggregation
    step_success = 0
    step_fail = 0
    step_total_eval = 0
    for ch in chunks:
        for ev in getattr(ch, "steps_eval", []) or []:
            if ev is None:
                continue
            step_total_eval += 1
            if bool(getattr(ev, "success", False)):
                step_success += 1
            else:
                step_fail += 1

            confidence = getattr(ev, "confidence", None)
            if isinstance(confidence, (int, float)):
                confidence_values.append(float(confidence))

    total_planned_pause_sec = 0.0
    for step in steps:
        action_type = _normalize_action_type(getattr(step, "action_type", None))
        action_type_counts[action_type] += 1

        pause = getattr(step, "pause", None)
        if isinstance(pause, (int, float)) and pause > 0:
            total_planned_pause_sec += float(pause)

    # Recovery effectiveness: P(success | previous chunk failed)
    recover_success = 0
    recover_total = 0
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        curr = chunks[i]
        if bool(getattr(prev, "overall_success", False)) is False:
            recover_total += 1
            if bool(getattr(curr, "overall_success", False)) is True:
                recover_success += 1

    avg_steps_per_chunk: Optional[float] = None
    min_steps_per_chunk: Optional[int] = None
    max_steps_per_chunk: Optional[int] = None
    chunk_steps_total = sum(chunk_step_lengths)
    if chunks_total > 0:
        avg_steps_per_chunk = chunk_steps_total / float(chunks_total)
        min_steps_per_chunk = min(chunk_step_lengths)
        max_steps_per_chunk = max(chunk_step_lengths)

    episode_duration_ms: Optional[int] = None
    if episode is not None:
        started_ts_ms = getattr(episode, "started_ts_ms", None)
        finished_ts_ms = getattr(episode, "finished_ts_ms", None)
        if isinstance(started_ts_ms, int) and isinstance(finished_ts_ms, int):
            delta = finished_ts_ms - started_ts_ms
            if delta >= 0:
                episode_duration_ms = delta

    metrics: Dict[str, Any] = {
        "chunks_total": chunks_total,
        "chunks_success": chunk_success,
        "chunks_fail": chunk_fail,
        "chunk_success_rate": (chunk_success / chunks_total) if chunks_total else None,
        "chunk_fail_rate": (chunk_fail / chunks_total) if chunks_total else None,
        "chunk_steps_total": chunk_steps_total,
        "steps_total": steps_total,
        "steps_eval_total": step_total_eval,
        "steps_success": step_success,
        "steps_fail": step_fail,
        "steps_without_evaluation": max(0, steps_total - step_total_eval),
        "step_success_rate": (step_success / step_total_eval) if step_total_eval else None,
        "step_eval_coverage": (step_total_eval / steps_total) if steps_total else None,
        "avg_steps_per_chunk": avg_steps_per_chunk,
        "min_steps_per_chunk": min_steps_per_chunk,
        "max_steps_per_chunk": max_steps_per_chunk,
        "avg_pause_per_step_sec": (total_planned_pause_sec / steps_total) if steps_total else None,
        "total_planned_pause_sec": total_planned_pause_sec,
        "step_confidence_mean": _mean(confidence_values),
        "step_confidence_std": _std(confidence_values),
        "step_confidence_min": min(confidence_values) if confidence_values else None,
        "step_confidence_max": max(confidence_values) if confidence_values else None,
        "failing_step_index_mean": _mean([float(v) for v in failing_step_indexes]),
        "failing_step_index_min": min(failing_step_indexes) if failing_step_indexes else None,
        "failing_step_index_max": max(failing_step_indexes) if failing_step_indexes else None,
        "first_failure_chunk_index": first_failure_chunk_index,
        "max_consecutive_chunk_failures": max_consecutive_chunk_failures,
        "planner_done_judge_fail": planner_done_judge_fail,
        "planner_fail_judge_success": planner_fail_judge_success,
        "recoveries_after_fail_total": recover_success,
        "recoveries_after_fail_opportunities": recover_total,
        "recovery_after_fail_rate": (recover_success / recover_total) if recover_total else None,
        "observations_total": observations_total,
        "trim_updates_total": len(trim_info),
        "llm_requests_total": len(llm_requests),
        "history_events_total": len(full_history),
        "unique_failure_types": len(failure_type_counts),
        "unique_action_types": len(action_type_counts),
        "failure_type_counts": dict(failure_type_counts),
        "decision_counts": dict(decision_counts),
        "action_type_counts": dict(action_type_counts),
        "episode_duration_ms": episode_duration_ms,
        "episode_duration_sec": (episode_duration_ms / 1000.0) if episode_duration_ms is not None else None,
    }

    return metrics


def append_metrics_csv(
    path: str = "data/experiments.csv",
    episode: Optional[Any] = None,
    run_context: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Append metrics to a CSV file.
    
    If schema evolves (new columns), existing rows are rewritten once with the
    merged header so the CSV remains consistent.
    """
    if episode is None:
        return

    score = getattr(episode, "score", None) or {}
    metrics = score.get("stats")
    if metrics is None:
        return

    task = getattr(episode, "task", None) or {}
    task_id = task.get("id")
    task_instruction = str(task.get("instruction") or getattr(episode, "instruction", "") or "")

    row: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
    }

    row.update(
        {
            "episode_id": getattr(episode, "episode_id", None),
            "status": getattr(episode, "status", None),
            "success": score.get("success", None),
            "osworld_status": score.get("status", None),
            "osworld_metric": score.get("metric", None),
            "os_name": getattr(episode, "os_name", None),
            "desktop_env": getattr(episode, "desktop_env", None),
            "display_server": getattr(episode, "display_server", None),
            "started_ts_ms": getattr(episode, "started_ts_ms", None),
            "finished_ts_ms": getattr(episode, "finished_ts_ms", None),
            "task_id": task_id,
            "task_snapshot": task.get("snapshot"),
            "task_source": task.get("source"),
            "task_proxy": task.get("proxy"),
            "task_fixed_ip": task.get("fixed_ip"),
            "task_possibility_of_env_change": task.get("possibility_of_env_change"),
            "evaluator_func": (task.get("evaluator") or {}).get("func"),
            "related_apps": task.get("related_apps"),
            "instruction_len_chars": len(task_instruction),
            "instruction_len_words": len([w for w in task_instruction.strip().split() if w]),
        }
    )

    def _serialize_cell(value: Any) -> Any:
        """
        Process serialize cell.
        
        Parameters
        ----------
        value : Any
            Input value.
        
        Returns
        -------
        Any
            Function result.
        
        """
        if isinstance(value, (dict, list, tuple, set)):
            return json.dumps(value, ensure_ascii=False)
        return value

    # Flatten metrics; serialize nested structures
    for k, v in metrics.items():
        row[k] = _serialize_cell(v)

    row["osworld_metric"] = _serialize_cell(row.get("osworld_metric"))
    row["related_apps"] = _serialize_cell(row.get("related_apps"))

    context = run_context or {}
    for key, value in context.items():
        k = str(key).strip()
        if not k:
            continue
        col = k if k not in row else f"ctx_{k}"
        row[col] = _serialize_cell(value)

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_fieldnames: list[str] = []
    existing_rows: list[Dict[str, Any]] = []
    file_exists = out_path.exists()
    if file_exists and out_path.stat().st_size > 0:
        with out_path.open("r", newline="", encoding="utf-8") as f_in:
            reader = csv.DictReader(f_in)
            existing_fieldnames = list(reader.fieldnames or [])
            existing_rows = list(reader)

    if existing_fieldnames:
        merged_fieldnames = list(existing_fieldnames)
        for key in row.keys():
            if key not in merged_fieldnames:
                merged_fieldnames.append(key)
    else:
        merged_fieldnames = list(row.keys())

    if file_exists and existing_fieldnames and merged_fieldnames != existing_fieldnames:
        with out_path.open("w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=merged_fieldnames)
            writer.writeheader()
            for old_row in existing_rows:
                normalized = {k: old_row.get(k, "") for k in merged_fieldnames}
                writer.writerow(normalized)

    write_header = (not file_exists) or (out_path.stat().st_size == 0)
    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=merged_fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in merged_fieldnames})
