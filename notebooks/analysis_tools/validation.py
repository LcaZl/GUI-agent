import pandas as pd

from .config import PLOT_CONDITION_ORDER
from .dashboards import build_chunk_dashboard_tables


def validate_run_completeness(df_all, cond_col="plot_condition", task_col="task_id", run_col="run_id"):
    return {
        "rows_total": int(len(df_all)),
        "conditions": df_all.groupby(cond_col, observed=False).size().to_dict(),
        "tasks_per_condition": df_all.groupby(cond_col, observed=False)[task_col].nunique().to_dict(),
        "runs_per_condition": df_all.groupby(cond_col, observed=False)[run_col].nunique().to_dict(),
        "duplicate_run_task_condition": int(df_all.duplicated([run_col, task_col, cond_col]).sum()),
        "ordered_conditions": [cond for cond in PLOT_CONDITION_ORDER if cond in df_all[cond_col].astype(str).unique()],
    }


def validate_chunk_quality_consistency(df_all, cond_col="plot_condition", task_col="task_id"):
    weighted = build_chunk_dashboard_tables(df_all, cond_col=cond_col, task_col=task_col, rate_mode="weighted")
    episode_mean = build_chunk_dashboard_tables(df_all, cond_col=cond_col, task_col=task_col, rate_mode="episode_mean")

    comparison = pd.DataFrame(
        {
            "condition": weighted["conditions"],
            "weighted_chunk_rate_mean": weighted["rc"]["mean"].reindex(weighted["conditions"]).to_numpy(),
            "episode_mean_chunk_rate_mean": episode_mean["rc"]["mean"].reindex(weighted["conditions"]).to_numpy(),
            "mean_chunks_success_per_run": weighted["cnt"]["chunks_success"].reindex(weighted["conditions"]).to_numpy(),
            "mean_chunks_fail_per_run": weighted["cnt"]["chunks_fail"].reindex(weighted["conditions"]).to_numpy(),
        }
    )
    comparison["difference_weighted_minus_episode_mean"] = (
        comparison["weighted_chunk_rate_mean"] - comparison["episode_mean_chunk_rate_mean"]
    )
    return comparison


def summarize_analysis_context(context):
    df_all = context["df_all"]
    return {
        "run_inventory": context["run_inventory"].copy(),
        "completeness": validate_run_completeness(df_all),
        "chunk_quality_comparison": validate_chunk_quality_consistency(df_all),
    }
