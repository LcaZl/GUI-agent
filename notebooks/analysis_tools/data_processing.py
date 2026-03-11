import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_RUN_SPECS, EXP_ORDER, PLOT_CONDITION_ORDER
from .plot_helpers import ordered_conditions


def resolve_existing_path(path_like):
    p = Path(path_like)
    candidates = [p, Path.cwd() / p, Path("notebooks") / p]
    for cand in candidates:
        if cand.exists():
            return cand
    return p
def load_experiments_csv(force_csv_path=None, candidate_dirs=None):
    if candidate_dirs is None:
        candidate_dirs = [Path("../thesis_experiment"), Path("thesis_experiment")]

    candidates = []
    for directory in candidate_dirs:
        if directory.exists():
            candidates.extend(directory.glob("experiments_*.csv"))

    candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No experiments CSV found in thesis_experiment/")

    def inspect_candidate(path: Path):
        try:
            tmp = pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            return {
                "path": str(path),
                "rows": 0,
                "mtime": path.stat().st_mtime,
                "exp_codes": [],
                "has_known_experiments": False,
                "tasks_per_exp_min": 0,
            }

        exp_codes = []
        if "experiment_name" in tmp.columns:
            for name in tmp["experiment_name"].dropna().astype(str):
                match = re.search(r"(exp0[1-4])", name)
                if match:
                    exp_codes.append(match.group(1))

        exp_set = sorted(set(exp_codes))
        tasks_per_exp_min = 0
        if "experiment_name" in tmp.columns and "task_id" in tmp.columns and len(tmp):
            grp = tmp.assign(
                exp_code=tmp["experiment_name"].astype(str).str.extract(r"(exp0[1-4])", expand=False)
            )
            grp = grp[grp["exp_code"].isin(EXP_ORDER)]
            if len(grp):
                tasks_per_exp = grp.groupby("exp_code", observed=False)["task_id"].nunique()
                tasks_per_exp_min = int(tasks_per_exp.min()) if len(tasks_per_exp) else 0

        return {
            "path": str(path),
            "rows": int(len(tmp)),
            "mtime": path.stat().st_mtime,
            "exp_codes": exp_set,
            "has_known_experiments": bool(exp_set),
            "tasks_per_exp_min": tasks_per_exp_min,
        }

    meta = pd.DataFrame([inspect_candidate(p) for p in candidates])
    meta["mtime"] = pd.to_datetime(meta["mtime"], unit="s")
    meta = meta.sort_values("mtime", ascending=False)

    if force_csv_path:
        csv_path = resolve_existing_path(force_csv_path)
    else:
        complete = meta[meta["has_known_experiments"]].copy()
        if len(complete):
            complete = complete.sort_values(["mtime", "rows"], ascending=[False, False])
            csv_path = Path(complete.iloc[0]["path"])
        else:
            csv_path = Path(meta.iloc[0]["path"])

    return csv_path, pd.read_csv(csv_path, encoding="utf-8-sig"), meta


def extract_cfg_value(cfg_obj, path, default=np.nan):
    cur = cfg_obj
    for part in path:
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def to_bool(value):
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def canonical_exp(name):
    match = re.search(r"(exp0[1-4])", str(name or ""))
    return match.group(1) if match else "other"


def to_num(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def parse_counter_cell(value):
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return {}
    text = str(value).strip()
    if not text:
        return {}
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def infer_memory_mode(row):
    enable_tms_trim = row.get("cfg_enable_tms_trim", np.nan)
    initialize_memory = row.get("cfg_initialize_memory", np.nan)

    if not pd.isna(enable_tms_trim) and not bool(enable_tms_trim):
        return "no_stm"
    if not pd.isna(initialize_memory) and not bool(initialize_memory):
        return "warm_ltm"
    if not pd.isna(initialize_memory) and bool(initialize_memory):
        return "baseline"
    return "unknown"


def infer_vision_mode(row):
    value = row.get("cfg_use_vision", np.nan)
    if not pd.isna(value):
        return "on" if bool(value) else "off"
    return "unknown"


def infer_stm_mode(row):
    value = row.get("cfg_enable_tms_trim", np.nan)
    if not pd.isna(value):
        return "on" if bool(value) else "off"
    return "unknown"


def build_condition_label(row):
    return f"{row.get('memory_label', 'Unknown')} | {row.get('vision_label', 'Vision ?')}"


def preprocess_experiments(df_raw, run_id=None, run_label=None):
    df = df_raw.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df["success_bool"] = df["success"].map(to_bool) if "success" in df.columns else False
    df["status_norm"] = (
        df["status"].astype(str).str.upper().str.strip() if "status" in df.columns else "UNKNOWN"
    )
    df["exp_name_raw"] = df["experiment_name"].fillna("unknown") if "experiment_name" in df.columns else "unknown"

    if run_id is not None:
        df["run_id"] = run_id
    if run_label is not None:
        df["run_label"] = run_label

    df["exp_code"] = df["exp_name_raw"].map(canonical_exp)
    if df["exp_code"].isin(EXP_ORDER).any():
        df = df[df["exp_code"].isin(EXP_ORDER)].copy()

    df["exp_uid"] = df["exp_code"]
    df["cfg_obj"] = [{} for _ in range(len(df))]
    if "experiment_config_json" in df.columns:
        parsed = []
        for raw_value in df["experiment_config_json"]:
            try:
                parsed.append(json.loads(raw_value) if isinstance(raw_value, str) and raw_value.strip() else {})
            except Exception:
                parsed.append({})
        df["cfg_obj"] = parsed

    df["cfg_initialize_memory"] = df["cfg_obj"].map(
        lambda obj: extract_cfg_value(obj, ["memory_settings", "initialize_memory"])
    )
    df["cfg_enable_tms_trim"] = df["cfg_obj"].map(
        lambda obj: extract_cfg_value(obj, ["memory_settings", "enable_tms_trim"])
    )
    df["cfg_memory_name"] = df["cfg_obj"].map(
        lambda obj: extract_cfg_value(obj, ["memory_settings", "memory_name"])
    )
    df["cfg_use_vision"] = df["cfg_obj"].map(
        lambda obj: extract_cfg_value(obj, ["perception_settings", "use_vision"])
    )

    numeric_cols = [
        "chunks_total",
        "chunks_success",
        "chunks_fail",
        "chunk_success_rate",
        "steps_total",
        "steps_success",
        "steps_fail",
        "step_success_rate",
        "total_planned_pause_sec",
        "avg_pause_per_step_sec",
        "episode_duration_sec",
        "recovery_after_fail_rate",
        "recoveries_after_fail_total",
        "recoveries_after_fail_opportunities",
        "first_failure_chunk_index",
        "max_consecutive_chunk_failures",
        "planner_done_judge_fail",
        "planner_fail_judge_success",
        "llm_requests_total",
        "osworld_metric",
    ]
    for column in numeric_cols:
        if column in df.columns:
            df[column] = df[column].map(to_num)

    for column in ["failure_type_counts", "action_type_counts", "decision_counts"]:
        if column in df.columns:
            df[column] = df[column].map(parse_counter_cell)

    df["memory_mode"] = df.apply(infer_memory_mode, axis=1)
    df["stm_mode"] = df.apply(infer_stm_mode, axis=1)
    df["vision_mode"] = df.apply(infer_vision_mode, axis=1)
    df["memory_label"] = df["memory_mode"].map(
        {"baseline": "Baseline", "warm_ltm": "Warm LTM", "no_stm": "No STM"}
    ).fillna("Unknown")
    df["stm_label"] = df["stm_mode"].map({"on": "STM ON", "off": "STM OFF"}).fillna("STM ?")
    df["vision_label"] = df["vision_mode"].map({"on": "Vision ON", "off": "Vision OFF"}).fillna("Vision ?")
    df["plot_condition"] = df.apply(build_condition_label, axis=1)

    df["exp_sort"] = df["exp_code"].apply(lambda value: EXP_ORDER.index(value) if value in EXP_ORDER else 999)
    df["plot_sort"] = df["plot_condition"].map(
        lambda value: PLOT_CONDITION_ORDER.index(value) if value in PLOT_CONDITION_ORDER else 999
    )
    return df.sort_values(["plot_sort", "timestamp"], kind="stable").reset_index(drop=True)


def load_experiment_runs(run_specs=None):
    if run_specs is None:
        run_specs = DEFAULT_RUN_SPECS

    run_datasets = []
    run_inventory = []
    for spec in run_specs:
        csv_path, df_raw, candidate_meta = load_experiments_csv(force_csv_path=spec["csv_path"])
        df = preprocess_experiments(df_raw, run_id=spec["run_id"], run_label=spec["run_label"])
        run_datasets.append(
            {
                "run_id": spec["run_id"],
                "run_label": spec["run_label"],
                "csv_path": str(csv_path),
                "candidate_meta": candidate_meta,
                "df": df,
            }
        )
        run_inventory.append(
            {
                "run_id": spec["run_id"],
                "run_label": spec["run_label"],
                "csv_path": str(csv_path),
                "rows": len(df),
                "conditions": ", ".join(ordered_conditions(df["plot_condition"].dropna().astype(str).unique())),
            }
        )
    return run_datasets, pd.DataFrame(run_inventory)


def combine_runs(run_datasets):
    runs = [(item["run_id"], item["df"].copy()) for item in run_datasets]
    df_all = pd.concat([df.assign(run_id=run_id) for run_id, df in runs], ignore_index=True)
    if "exp_code" in df_all.columns:
        df_all["exp_code"] = pd.Categorical(df_all["exp_code"], categories=EXP_ORDER, ordered=True)
    if "plot_condition" in df_all.columns:
        df_all["plot_condition"] = pd.Categorical(
            df_all["plot_condition"],
            categories=PLOT_CONDITION_ORDER,
            ordered=True,
        )
    return df_all


def recompute_task_means(df_all, task_col="task_id", exp_col="plot_condition"):
    dframe = df_all.dropna(subset=[task_col, exp_col]).copy()
    dframe["chunks_total"] = pd.to_numeric(dframe["chunks_total"], errors="coerce")
    dframe["steps_total"] = pd.to_numeric(dframe["steps_total"], errors="coerce")
    return (
        dframe.groupby([task_col, exp_col], observed=False)
        .agg(chunks_mean=("chunks_total", "mean"), steps_mean=("steps_total", "mean"))
        .reset_index()
    )


def build_analysis_context(run_specs=None):
    run_datasets, run_inventory = load_experiment_runs(run_specs=run_specs)
    df_all = combine_runs(run_datasets)
    task_means = recompute_task_means(df_all, task_col="task_id", exp_col="plot_condition")
    chunks_piv = task_means.pivot(index="task_id", columns="plot_condition", values="chunks_mean").round(2)
    steps_piv = task_means.pivot(index="task_id", columns="plot_condition", values="steps_mean").round(2)
    ordered_cols = [col for col in PLOT_CONDITION_ORDER if col in chunks_piv.columns]
    chunks_piv = chunks_piv.reindex(columns=ordered_cols)
    steps_piv = steps_piv.reindex(columns=ordered_cols)
    return {
        "run_datasets": run_datasets,
        "run_inventory": run_inventory,
        "df_all": df_all,
        "chunks_piv": chunks_piv,
        "steps_piv": steps_piv,
    }
