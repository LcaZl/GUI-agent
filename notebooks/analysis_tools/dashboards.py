import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from .config import PLOT_CONDITION_ORDER
from .plot_helpers import (
    make_compact_condition_labels,
    make_two_line_labels,
    ordered_conditions,
    save_figure,
    sum_counters,
    truncate_cmap,
)
from .style import (
    ANNOTATION_FONTSIZE,
    AXIS_LABEL_FONTSIZE,
    HEATMAP_CELL_FONTSIZE,
    LEGEND_FONTSIZE,
    LEGEND_TITLE_FONTSIZE,
    PANEL_TITLE_FONTSIZE,
    SUPTITLE_FONTSIZE,
    XTICK_FONTSIZE,
    YTICK_FONTSIZE,
)


def _style_ax(ax, xtick_fontsize, ytick_fontsize, xtick_rotation, xtick_pad, grid_alpha):
    ax.grid(True, axis="y", alpha=grid_alpha)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=xtick_rotation, pad=xtick_pad, labelsize=xtick_fontsize)
    ax.tick_params(axis="y", labelsize=ytick_fontsize)
    for label in ax.get_xticklabels():
        label.set_ha("center")


def _annotate_binary_stack(ax, x, lower_vals, upper_vals, fontsize):
    for idx, (lower, upper) in enumerate(zip(lower_vals, upper_vals)):
        total = lower + upper
        if total <= 0:
            continue
        lower_pct = 100.0 * lower / total
        upper_pct = 100.0 * upper / total
        if lower > 0:
            ax.text(x[idx], lower / 2.0, f"{lower_pct:.0f}%", ha="center", va="center", fontsize=fontsize, color="black")
        if upper > 0:
            ax.text(x[idx], lower + upper / 2.0, f"{upper_pct:.0f}%", ha="center", va="center", fontsize=fontsize, color="white")


def _annotate_bar_tops(ax, x, vals, fontsize, fmt="{:.0f}%", dy=0.02, scale=1.0):
    ymax = ax.get_ylim()[1] if ax.get_ylim()[1] else 1.0
    for idx, value in enumerate(vals):
        if np.isnan(value):
            continue
        ax.text(x[idx], value + ymax * dy, fmt.format(value * scale), ha="center", va="bottom", fontsize=fontsize)


def _annotate_bar_bases(ax, x, vals, fontsize, fmt="{:.0f}%", base_frac=0.04, scale=1.0):
    ymax = ax.get_ylim()[1] if ax.get_ylim()[1] else 1.0
    y = ymax * base_frac
    for idx, value in enumerate(vals):
        if np.isnan(value) or value <= 0:
            continue
        ax.text(x[idx], y, fmt.format(value * scale), ha="center", va="bottom", fontsize=fontsize)


def _annotate_stacked_shares(ax, x, matrix, fontsize, threshold=0.08):
    bottom = np.zeros(len(x), dtype=float)
    for vals in matrix:
        vals = np.asarray(vals, dtype=float)
        for idx, value in enumerate(vals):
            if value <= 0 or value < threshold:
                continue
            color = "white" if value >= 0.22 else "black"
            ax.text(x[idx], bottom[idx] + value / 2.0, f"{100.0 * value:.0f}%", ha="center", va="center", fontsize=fontsize, color=color)
        bottom += vals


def plot_task_condition_success_heatmap_light_continuous(
    df,
    cond_col="plot_condition",
    outpath_png=None,
    annotate=True,
    figsize=(14.5, 7.0),
    task_id_chars=22,
    cell_fontsize=HEATMAP_CELL_FONTSIZE,
    ytick_fontsize=YTICK_FONTSIZE,
    xtick_fontsize=XTICK_FONTSIZE,
    axis_label_fontsize=AXIS_LABEL_FONTSIZE,
    xlabels_two_lines=True,
    title="",
    title_fontsize=PANEL_TITLE_FONTSIZE,
    annotation_as_percent=True,
    colorbar_label="Success rate",
    colorbar_label_fontsize=AXIS_LABEL_FONTSIZE,
    colorbar_tick_fontsize=12,
    left_margin=0.08,
    right_margin=0.95,
    bottom_margin=0.22,
    top_margin=0.92,
    include_mean_row=True,
    mean_row_label="Mean",
):
    grouped = (
        df.groupby(["run_id", "task_id", cond_col], observed=False)["success_bool"]
        .mean()
        .reset_index(name="run_rate")
    )
    mean_tbl = (
        grouped.groupby(["task_id", cond_col], observed=False)["run_rate"]
        .mean()
        .reset_index(name="mean_rate")
    )
    pivot = mean_tbl.pivot(index="task_id", columns=cond_col, values="mean_rate").fillna(0.0)
    pivot = pivot.reindex(columns=[c for c in PLOT_CONDITION_ORDER if c in pivot.columns])
    pivot = pivot.loc[sorted(pivot.index.astype(str))]
    if include_mean_row and len(pivot):
        mean_row = pd.DataFrame([pivot.mean(axis=0)], index=[mean_row_label])
        pivot = pd.concat([pivot, mean_row], axis=0)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        pivot.to_numpy(),
        cmap=truncate_cmap(plt.get_cmap("Blues"), 0.10, 0.70),
        norm=plt.Normalize(vmin=0, vmax=1),
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xlabel("")
    ax.set_ylabel("Task", fontsize=axis_label_fontsize, labelpad=12)
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xticks(np.arange(pivot.shape[1]))
    xlabels = make_two_line_labels(pivot.columns) if xlabels_two_lines else [str(c) for c in pivot.columns]
    ax.set_xticklabels(xlabels, rotation=0, ha="center", fontsize=xtick_fontsize)
    ax.tick_params(axis="x", pad=10, length=0)

    ylabels = []
    for task in pivot.index:
        task_str = str(task)
        if include_mean_row and task_str == mean_row_label:
            ylabels.append(mean_row_label)
        else:
            ylabels.append(task_str[:task_id_chars] + ("…" if len(task_str) > task_id_chars else ""))
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(ylabels, fontsize=ytick_fontsize)
    ax.tick_params(axis="y", pad=8)

    if annotate:
        zvals = pivot.to_numpy()
        for i in range(zvals.shape[0]):
            for j in range(zvals.shape[1]):
                label = f"{100.0 * zvals[i, j]:.0f}%" if annotation_as_percent else f"{zvals[i, j]:.2f}"
                ax.text(j, i, label, ha="center", va="center", fontsize=cell_fontsize)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(colorbar_label, fontsize=colorbar_label_fontsize, labelpad=12)
    cbar.ax.tick_params(labelsize=colorbar_tick_fontsize)

    fig.subplots_adjust(left=left_margin, right=right_margin, top=top_margin, bottom=bottom_margin)
    save_figure(fig, outpath_png=outpath_png, dpi=350)
    return fig, ax, pivot


def build_chunk_dashboard_tables(df, cond_col="plot_condition", task_col="task_id", rate_mode="weighted"):
    conds = ordered_conditions(df[cond_col].dropna().astype(str).unique().tolist())

    run_counts = (
        df.groupby(["run_id", cond_col], observed=False)[["chunks_success", "chunks_fail", "chunks_total"]]
        .sum(min_count=1)
        .reset_index()
    )
    run_counts["weighted_chunk_rate"] = run_counts["chunks_success"] / run_counts["chunks_total"].replace(0, np.nan)

    run_episode_counts = (
        df.groupby(["run_id", cond_col], observed=False)[["chunks_success", "chunks_fail", "chunks_total"]]
        .mean(numeric_only=True)
        .reset_index()
    )

    run_episode_rates = (
        df.groupby(["run_id", cond_col], observed=False)["chunk_success_rate"]
        .mean()
        .reset_index(name="episode_mean_chunk_rate")
    )
    run_rates = run_counts.merge(run_episode_rates, on=["run_id", cond_col], how="left")

    rate_column = "weighted_chunk_rate" if rate_mode == "weighted" else "episode_mean_chunk_rate"
    rc = run_rates.groupby(cond_col, observed=False)[rate_column].agg(["mean", "std"]).reindex(conds)

    task_counts = (
        df.groupby([task_col, cond_col], observed=False)[["chunks_success", "chunks_total"]]
        .sum(min_count=1)
        .reset_index()
    )
    task_counts["task_weighted_chunk_rate"] = task_counts["chunks_success"] / task_counts["chunks_total"].replace(0, np.nan)
    task_episode_rates = (
        df.groupby([task_col, cond_col], observed=False)["chunk_success_rate"]
        .mean()
        .reset_index(name="task_episode_chunk_rate")
    )
    task_rates = task_counts.merge(task_episode_rates, on=[task_col, cond_col], how="left")
    task_rate_column = "task_weighted_chunk_rate" if rate_mode == "weighted" else "task_episode_chunk_rate"

    cnt = (
        run_episode_counts.groupby(cond_col, observed=False)[["chunks_success", "chunks_fail"]]
        .mean()
        .reindex(conds)
    )
    g_run_dyn = (
        df.groupby(["run_id", cond_col], observed=False)[["first_failure_chunk_index", "max_consecutive_chunk_failures"]]
        .mean()
        .reset_index()
    )
    dyn_mean = g_run_dyn.groupby(cond_col, observed=False)[["first_failure_chunk_index", "max_consecutive_chunk_failures"]].mean().reindex(conds)
    dyn_std = g_run_dyn.groupby(cond_col, observed=False)[["first_failure_chunk_index", "max_consecutive_chunk_failures"]].std().reindex(conds).fillna(0)

    return {
        "conditions": conds,
        "run_rates": run_rates,
        "task_rates": task_rates,
        "rate_column": rate_column,
        "task_rate_column": task_rate_column,
        "rc": rc,
        "cnt": cnt,
        "dyn_mean": dyn_mean,
        "dyn_std": dyn_std,
    }


def plot_chunk_dashboard_v3(
    df,
    outpath_png=None,
    title="Chunk-level diagnostics across conditions",
    figsize=(20, 15),
    width_ratios=(1.08, 1.18),
    height_ratios=(1.08, 1.08),
    hspace=0.42,
    wspace=0.24,
    bar_width=0.58,
    bar_edgecolor="black",
    bar_linewidth=0.6,
    err_capsize=5,
    scatter_size=48,
    scatter_alpha=0.95,
    scatter_edgecolor="white",
    scatter_linewidth=0.6,
    jitter_frac=0.40,
    main_blue="#56B4E9",
    contrast_orange="#D55E00",
    blue_light="#8ECAE6",
    blue_dark="#1F77B4",
    suptitle_fontsize=SUPTITLE_FONTSIZE,
    panel_title_fontsize=PANEL_TITLE_FONTSIZE,
    axis_label_fontsize=AXIS_LABEL_FONTSIZE,
    xtick_fontsize=XTICK_FONTSIZE,
    ytick_fontsize=YTICK_FONTSIZE,
    legend_fontsize=LEGEND_FONTSIZE,
    suptitle_y=0.98,
    title_pad=0,
    title_y=1.18,
    legend_outside=True,
    legend_y=1.08,
    legend_ncol=2,
    legend_columnspacing=1.2,
    legend_handlelength=1.6,
    legend_handletextpad=0.5,
    legend_labelspacing=0.6,
    xlabels_two_lines=True,
    compact_condition_labels=False,
    xtick_rotation=0,
    xtick_pad=8,
    grid_alpha=0.25,
    ypad_top=0.06,
    ylim_a_max=1.15,
    save_dpi=350,
    cond_col="plot_condition",
    task_col="task_id",
    chunk_rate_mode="weighted",
    binary_label_fontsize=ANNOTATION_FONTSIZE,
    rate_label_fontsize=ANNOTATION_FONTSIZE,
    show_quality_error_bars=True,
    show_quality_bar_labels=True,
    quality_bar_label_position="top",
    quality_bar_label_base_frac=0.04,
    left_margin=0.07,
    right_margin=0.98,
    bottom_margin=0.08,
    top_margin=0.93,
):
    tables = build_chunk_dashboard_tables(df, cond_col=cond_col, task_col=task_col, rate_mode=chunk_rate_mode)
    conds = tables["conditions"]
    x = np.arange(len(conds))
    if compact_condition_labels:
        xlabels = make_compact_condition_labels(conds)
    else:
        xlabels = make_two_line_labels(conds) if xlabels_two_lines else [str(c) for c in conds]

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=height_ratios, width_ratios=width_ratios, wspace=wspace, hspace=hspace)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    quality_yerr = tables["rc"]["std"].fillna(0).to_numpy() if show_quality_error_bars else None
    quality_err_kw = {"capsize": err_capsize} if show_quality_error_bars else {}
    axA.bar(
        x,
        tables["rc"]["mean"].fillna(0).to_numpy(),
        yerr=quality_yerr,
        width=bar_width,
        color=main_blue,
        edgecolor=bar_edgecolor,
        linewidth=bar_linewidth,
        **quality_err_kw,
    )

    rng = np.random.default_rng(42)
    max_point = 0.0
    for idx, cond in enumerate(conds):
        vals = tables["task_rates"].loc[
            tables["task_rates"][cond_col].astype(str) == cond,
            tables["task_rate_column"],
        ].dropna().to_numpy()
        if len(vals):
            max_point = max(max_point, float(np.max(vals)))
            jitter = (rng.random(len(vals)) - 0.5) * (bar_width * jitter_frac)
            axA.scatter(
                np.full(len(vals), idx) + jitter,
                vals,
                s=scatter_size,
                alpha=scatter_alpha,
                color=contrast_orange,
                edgecolors=scatter_edgecolor,
                linewidths=scatter_linewidth,
                zorder=3,
            )

    top = max(1.0 + ypad_top, max_point + 0.03)
    axA.set_ylim(0, min(ylim_a_max, top))
    axA.set_xticks(x)
    axA.set_xticklabels(xlabels)
    axA.set_ylabel("Chunk success rate $r_c$", fontsize=axis_label_fontsize)
    axA.set_title("(a) Chunk quality ($r_c$)", fontsize=panel_title_fontsize, pad=title_pad, y=title_y)
    _style_ax(axA, xtick_fontsize, ytick_fontsize, xtick_rotation, xtick_pad, grid_alpha)
    if show_quality_bar_labels:
        values = tables["rc"]["mean"].fillna(0).to_numpy()
        if quality_bar_label_position == "base":
            _annotate_bar_bases(
                axA,
                x,
                values,
                rate_label_fontsize,
                scale=100.0,
                base_frac=quality_bar_label_base_frac,
            )
        else:
            _annotate_bar_tops(axA, x, values, rate_label_fontsize, scale=100.0)

    succ = tables["cnt"]["chunks_success"].fillna(0).to_numpy()
    fail = tables["cnt"]["chunks_fail"].fillna(0).to_numpy()
    h1 = axB.bar(x, succ, width=bar_width, label="Chunks OK", color=blue_light, edgecolor=bar_edgecolor, linewidth=bar_linewidth)
    h2 = axB.bar(x, fail, bottom=succ, width=bar_width, label="Chunks FAIL", color=blue_dark, edgecolor=bar_edgecolor, linewidth=bar_linewidth)
    axB.set_xticks(x)
    axB.set_xticklabels(xlabels)
    axB.set_ylabel("Mean chunks per episode", fontsize=axis_label_fontsize)
    axB.set_title("(b) Chunk outcomes", fontsize=panel_title_fontsize, pad=title_pad, y=title_y)
    _style_ax(axB, xtick_fontsize, ytick_fontsize, xtick_rotation, xtick_pad, grid_alpha)
    _annotate_binary_stack(axB, x, succ, fail, binary_label_fontsize)
    if legend_outside:
        axB.legend(
            [h1[0], h2[0]],
            ["Chunks OK", "Chunks FAIL"],
            loc="center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=legend_ncol,
            frameon=False,
            fontsize=legend_fontsize,
            columnspacing=legend_columnspacing,
            handlelength=legend_handlelength,
            handletextpad=legend_handletextpad,
            borderaxespad=0.0,
            labelspacing=legend_labelspacing,
        )

    axC.bar(
        x,
        tables["dyn_mean"]["first_failure_chunk_index"].fillna(0).to_numpy(),
        yerr=tables["dyn_std"]["first_failure_chunk_index"].fillna(0).to_numpy(),
        capsize=err_capsize,
        width=bar_width,
        color=main_blue,
        edgecolor=bar_edgecolor,
        linewidth=bar_linewidth,
    )
    axC.set_xticks(x)
    axC.set_xticklabels(xlabels)
    axC.set_ylabel("Mean index", fontsize=axis_label_fontsize)
    axC.set_title("(c) First failure position", fontsize=panel_title_fontsize, pad=title_pad, y=title_y)
    _style_ax(axC, xtick_fontsize, ytick_fontsize, xtick_rotation, xtick_pad, grid_alpha)

    axD.bar(
        x,
        tables["dyn_mean"]["max_consecutive_chunk_failures"].fillna(0).to_numpy(),
        yerr=tables["dyn_std"]["max_consecutive_chunk_failures"].fillna(0).to_numpy(),
        capsize=err_capsize,
        width=bar_width,
        color=main_blue,
        edgecolor=bar_edgecolor,
        linewidth=bar_linewidth,
    )
    axD.set_xticks(x)
    axD.set_xticklabels(xlabels)
    axD.set_ylabel("Mean max streak", fontsize=axis_label_fontsize)
    axD.set_title("(d) Failure persistence", fontsize=panel_title_fontsize, pad=title_pad, y=title_y)
    _style_ax(axD, xtick_fontsize, ytick_fontsize, xtick_rotation, xtick_pad, grid_alpha)

    if title is not None:
        fig.suptitle(title, y=suptitle_y, fontsize=suptitle_fontsize)
    fig.subplots_adjust(left=left_margin, right=right_margin, bottom=bottom_margin, top=top_margin, wspace=wspace, hspace=hspace)
    save_figure(fig, outpath_png=outpath_png, dpi=save_dpi)
    return fig, (axA, axB, axC, axD), tables


def build_steps_summary_table(df, cond_col="plot_condition"):
    base_cols = [
        "steps_total",
        "steps_success",
        "steps_fail",
        "step_success_rate",
        "avg_pause_per_step_sec",
        "llm_requests_total",
    ]
    for column in base_cols:
        if column not in df.columns:
            raise ValueError(f"Missing column: {column}")

    g_base = (
        df.groupby(["run_id", cond_col], observed=False)[base_cols]
        .mean()
        .reset_index()
    )

    rows = []
    has_actions = "action_type_counts" in df.columns
    for (run_id, cond), sub in df.groupby(["run_id", cond_col], observed=False, dropna=False):
        if not has_actions:
            rows.append({"run_id": run_id, cond_col: cond, "WAIT_share": np.nan, "PYTHON_share": np.nan})
            continue
        cnt = sum_counters(sub["action_type_counts"])
        tot = sum(cnt.values()) if cnt else 0.0
        if tot <= 0:
            rows.append({"run_id": run_id, cond_col: cond, "WAIT_share": np.nan, "PYTHON_share": np.nan})
            continue
        rows.append(
            {
                "run_id": run_id,
                cond_col: cond,
                "WAIT_share": cnt.get("WAIT", 0.0) / tot,
                "PYTHON_share": cnt.get("PYTHON", 0.0) / tot,
            }
        )
    g_act = pd.DataFrame(rows)
    g = g_base.merge(g_act, on=["run_id", cond_col], how="left")
    agg = g.groupby(cond_col, observed=False, dropna=False).mean(numeric_only=True).reset_index()
    agg[cond_col] = agg[cond_col].astype(str)
    ordered = ordered_conditions(agg[cond_col].tolist())
    agg = agg.set_index(cond_col).reindex(ordered).reset_index()

    def fmt_pct(value, decimals=1):
        return "-" if pd.isna(value) else f"{100 * value:.{decimals}f}%"

    return pd.DataFrame(
        {
            "Condition": agg[cond_col].astype(str),
            "Step success rate $r_s$": agg["step_success_rate"].map(lambda value: fmt_pct(value, 1)),
            "Steps evaluated": agg["steps_total"].round(1),
            "Steps OK": agg["steps_success"].round(1),
            "Steps FAIL": agg["steps_fail"].round(1),
            "WAIT share": agg["WAIT_share"].map(lambda value: fmt_pct(value, 1)),
            "PYTHON share": agg["PYTHON_share"].map(lambda value: fmt_pct(value, 1)),
            "Avg pause/step (s)": agg["avg_pause_per_step_sec"].round(2),
            "LLM calls": agg["llm_requests_total"].round(1),
        }
    )


def plot_steps_dashboard_v8(
    df,
    cond_col="plot_condition",
    outpath_png=None,
    save_dpi=350,
    figsize=(20, 15),
    bar_width=0.58,
    hspace=0.46,
    wspace=0.24,
    axis_label_fontsize=AXIS_LABEL_FONTSIZE,
    xtick_fontsize=XTICK_FONTSIZE,
    ytick_fontsize=YTICK_FONTSIZE,
    panel_title_fontsize=PANEL_TITLE_FONTSIZE,
    title_pad=0,
    title_y=1.18,
    xlabels_two_lines=True,
    compact_condition_labels=False,
    y_headroom_frac=0.10,
    ylabel_C="Mean action share",
    ylabel_D="Mean failure share",
    legend_outside=True,
    legend_fontsize=LEGEND_FONTSIZE,
    legend_ncol=2,
    legend_y=1.08,
    emphasize_D_edge=True,
    binary_label_fontsize=ANNOTATION_FONTSIZE,
    rate_label_fontsize=ANNOTATION_FONTSIZE,
    action_share_label_fontsize=ANNOTATION_FONTSIZE,
    show_quality_error_bars=True,
    annotate_action_shares=True,
    left_margin=0.06,
    right_margin=0.80,
    bottom_margin=0.07,
    top_margin=0.95,
):
    req = ["run_id", cond_col, "step_success_rate", "steps_success", "steps_fail", "steps_total", "success_bool"]
    for column in req:
        if column not in df.columns:
            raise ValueError(f"Missing column: {column}")

    conds = ordered_conditions(df[cond_col].dropna().astype(str).unique().tolist())
    x = np.arange(len(conds))
    if compact_condition_labels:
        xlabels = make_compact_condition_labels(conds)
    else:
        xlabels = make_two_line_labels(conds) if xlabels_two_lines else [str(c) for c in conds]

    g_run = (
        df.groupby(["run_id", cond_col], observed=False, dropna=False)[
            ["step_success_rate", "steps_success", "steps_fail", "steps_total"]
        ]
        .mean(numeric_only=True)
        .reset_index()
    )
    rs = g_run.groupby(cond_col, observed=False)["step_success_rate"].agg(["mean", "std"]).reindex(conds)
    sf = g_run.groupby(cond_col, observed=False)[["steps_success", "steps_fail"]].mean().reindex(conds)

    pivot_actions, action_mean_counts = None, {}
    if "action_type_counts" in df.columns:
        long_rows = []
        count_rows = []
        for (run_id, cond), sub in df.groupby(["run_id", cond_col], observed=False, dropna=False):
            cnt = sum_counters(sub["action_type_counts"])
            tot = sum(cnt.values()) if cnt else 0.0
            if tot <= 0:
                continue
            for key, value in cnt.items():
                long_rows.append({"run_id": run_id, cond_col: cond, "action_type": str(key), "prop": float(value) / tot})
                count_rows.append({"run_id": run_id, cond_col: cond, "action_type": str(key), "count": float(value)})
        long_df = pd.DataFrame(long_rows)
        if not long_df.empty:
            comp = long_df.groupby([cond_col, "action_type"], observed=False)["prop"].mean().reset_index()
            pivot_actions = comp.pivot(index=cond_col, columns="action_type", values="prop").fillna(0.0).reindex(conds)
        count_df = pd.DataFrame(count_rows)
        if not count_df.empty:
            action_mean_counts = count_df.groupby("action_type", observed=False)["count"].mean().to_dict()

    pivot_fail, fail_mean_counts = None, {}
    if "failure_type_counts" in df.columns:
        long_rows = []
        count_rows = []
        for (run_id, cond), sub in df.groupby(["run_id", cond_col], observed=False, dropna=False):
            cnt = sum_counters(sub["failure_type_counts"])
            tot = sum(cnt.values()) if cnt else 0.0
            if tot <= 0:
                continue
            for key, value in cnt.items():
                long_rows.append({"run_id": run_id, cond_col: cond, "failure_type": str(key), "prop": float(value) / tot})
                count_rows.append({"run_id": run_id, cond_col: cond, "failure_type": str(key), "count": float(value)})
        ft = pd.DataFrame(long_rows)
        if not ft.empty:
            ft_agg = ft.groupby([cond_col, "failure_type"], observed=False)["prop"].mean().reset_index()
            pivot_fail = ft_agg.pivot(index=cond_col, columns="failure_type", values="prop").fillna(0.0).reindex(conds)
        count_df = pd.DataFrame(count_rows)
        if not count_df.empty:
            fail_mean_counts = count_df.groupby("failure_type", observed=False)["count"].mean().to_dict()

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, wspace=wspace, hspace=hspace)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    quality_yerr = rs["std"].fillna(0).to_numpy() if show_quality_error_bars else None
    quality_err_kw = {"capsize": 5} if show_quality_error_bars else {}
    axA.bar(
        x,
        rs["mean"].to_numpy(),
        yerr=quality_yerr,
        width=bar_width,
        color="#56B4E9",
        edgecolor="black",
        linewidth=0.6,
        **quality_err_kw,
    )
    axA.set_ylim(0, 1.05 + y_headroom_frac)
    axA.set_xticks(x)
    axA.set_xticklabels(xlabels)
    axA.set_ylabel("Step success rate $r_s$", fontsize=axis_label_fontsize)
    axA.set_title("(a) Step quality", fontsize=panel_title_fontsize, pad=title_pad, y=title_y)
    _style_ax(axA, xtick_fontsize, ytick_fontsize, 0, 8, 0.25)
    _annotate_bar_tops(axA, x, rs["mean"].to_numpy(), rate_label_fontsize, scale=100.0)

    ok = sf["steps_success"].fillna(0).to_numpy()
    fail = sf["steps_fail"].fillna(0).to_numpy()
    hb1 = axB.bar(x, ok, width=bar_width, color="#8ECAE6", edgecolor="black", linewidth=0.6)
    hb2 = axB.bar(x, fail, bottom=ok, width=bar_width, color="#1F77B4", edgecolor="black", linewidth=0.6)
    axB.set_xticks(x)
    axB.set_xticklabels(xlabels)
    axB.set_ylabel("Mean steps per episode", fontsize=axis_label_fontsize)
    axB.set_title("(b) Step outcomes", fontsize=panel_title_fontsize, pad=title_pad, y=title_y)
    _style_ax(axB, xtick_fontsize, ytick_fontsize, 0, 8, 0.25)
    axB.set_ylim(0, float(np.max(ok + fail)) * (1.0 + y_headroom_frac) if len(ok) else 1.0)
    _annotate_binary_stack(axB, x, ok, fail, binary_label_fontsize)
    if legend_outside:
        axB.legend(
            [hb1[0], hb2[0]],
            # [f"Steps OK ({float(sf['steps_success'].mean()):.1f})", f"Steps FAIL ({float(sf['steps_fail'].mean()):.1f})"],
            [f"Steps OK", f"Steps FAIL"],
            loc="center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=legend_ncol,
            frameon=False,
            fontsize=legend_fontsize,
        )

    palette = ["#8ECAE6", "#56B4E9", "#219EBC", "#1F77B4", "#2A9DF4", "#0077B6", "#48CAE4", "#00B4D8"]
    if pivot_actions is None:
        axC.axis("off")
    else:
        bottom = np.zeros(len(pivot_actions))
        handles = []
        labels = []
        for idx, key in enumerate(pivot_actions.columns):
            vals = pivot_actions[key].to_numpy()
            handle = axC.bar(x, vals, bottom=bottom, width=bar_width, color=palette[idx % len(palette)], edgecolor="black", linewidth=0.6)
            bottom += vals
            handles.append(handle[0])
            labels.append(f"{key} ({action_mean_counts.get(str(key), 0.0):.1f})")
        axC.set_xticks(x)
        axC.set_xticklabels(xlabels)
        axC.set_ylim(0, 1.0 + y_headroom_frac)
        axC.set_ylabel(ylabel_C, fontsize=axis_label_fontsize)
        axC.set_title("(c) Step action types", fontsize=panel_title_fontsize, pad=title_pad, y=title_y)
        _style_ax(axC, xtick_fontsize, ytick_fontsize, 0, 8, 0.25)
        if annotate_action_shares:
            _annotate_stacked_shares(axC, x, [pivot_actions[key].to_numpy() for key in pivot_actions.columns], action_share_label_fontsize)
        if legend_outside:
            axC.legend(handles, labels, loc="center", bbox_to_anchor=(0.5, legend_y), ncol=legend_ncol, frameon=False, fontsize=legend_fontsize)

    if pivot_fail is None:
        axD.axis("off")
    else:
        bottom = np.zeros(len(pivot_fail))
        handles = []
        labels = []
        for idx, key in enumerate(pivot_fail.columns):
            vals = pivot_fail[key].to_numpy()
            handle = axD.bar(
                x,
                vals,
                bottom=bottom,
                width=bar_width,
                color=palette[idx % len(palette)],
                edgecolor="black" if emphasize_D_edge else None,
                linewidth=0.7 if emphasize_D_edge else 0.0,
                alpha=0.95,
            )
            bottom += vals
            handles.append(handle[0])
            labels.append(f"{key} ({fail_mean_counts.get(str(key), 0.0):.1f})")
        axD.set_xticks(x)
        axD.set_xticklabels(xlabels)
        axD.set_ylim(0, 1.0 + y_headroom_frac)
        axD.set_ylabel(ylabel_D, fontsize=axis_label_fontsize)
        axD.set_title("(d) Failure type composition", fontsize=panel_title_fontsize, pad=title_pad, y=title_y)
        _style_ax(axD, xtick_fontsize, ytick_fontsize, 0, 8, 0.25)
        if legend_outside:
            axD.legend(handles, labels, loc="center", bbox_to_anchor=(0.5, legend_y), ncol=legend_ncol, frameon=False, fontsize=legend_fontsize)

    fig.subplots_adjust(left=left_margin, right=right_margin, bottom=bottom_margin, top=top_margin, wspace=wspace, hspace=hspace)
    save_figure(fig, outpath_png=outpath_png, dpi=save_dpi)
    return fig, (axA, axB, axC, axD), pivot_fail


def plot_task_difficulty_v3(
    df,
    task_col="task_id",
    figsize=None,
    fig_width=16,
    row_height=0.55,
    min_fig_height=6.0,
    tight_layout_rect=(0, 0, 1, 0.98),
    task_label_max_chars=36,
    sort_by=("success_rate", "duration_mean"),
    sort_ascending=(True, False),
    main_blue="#56B4E9",
    contrast_orange="#D55E00",
    bar_edgecolor="black",
    suptitle=None,
    suptitle_fontsize=SUPTITLE_FONTSIZE,
    panel_title="Task success profile",
    panel_title_fontsize=PANEL_TITLE_FONTSIZE,
    axis_label_fontsize=AXIS_LABEL_FONTSIZE,
    xtick_fontsize=XTICK_FONTSIZE,
    ytick_fontsize=YTICK_FONTSIZE,
    annotation_fontsize=ANNOTATION_FONTSIZE,
    bar_height=0.72,
    bar_linewidth=0.6,
    bar_alpha=1.0,
    grid_alpha=0.25,
    xlim=(0.0, 1.04),
    show_reference_line=True,
    reference_line_value=None,
    reference_line_linestyle="--",
    reference_line_linewidth=1.5,
    reference_line_alpha=0.9,
    show_bar_values=True,
    bar_value_fmt="{:.0%}",
    bar_value_dx=0.015,
    outpath_png=None,
    save_dpi=350,
):
    needed = [task_col, "success_bool"]
    for column in needed:
        if column not in df.columns:
            raise ValueError(f"Missing column: {column}")

    cols = [task_col, "success_bool"]
    for extra in ["episode_duration_sec", "llm_requests_total", "chunk_success_rate", "step_success_rate"]:
        if extra in df.columns:
            cols.append(extra)

    agg_dict = {"success_rate": ("success_bool", "mean")}
    agg_dict["duration_mean"] = ("episode_duration_sec", "mean") if "episode_duration_sec" in cols else ("success_bool", "mean")
    agg_dict["llm_mean"] = ("llm_requests_total", "mean") if "llm_requests_total" in cols else ("success_bool", "mean")
    agg_dict["chunk_rate_mean"] = ("chunk_success_rate", "mean") if "chunk_success_rate" in cols else ("success_bool", "mean")
    agg_dict["step_rate_mean"] = ("step_success_rate", "mean") if "step_success_rate" in cols else ("success_bool", "mean")

    prof = (
        df[cols]
        .groupby(task_col, as_index=False)
        .agg(**agg_dict)
        .sort_values(list(sort_by), ascending=list(sort_ascending))
        .reset_index(drop=True)
    )
    prof["task_label"] = prof[task_col].astype(str).map(
        lambda text: text if len(text) <= task_label_max_chars else text[: task_label_max_chars - 1] + "…"
    )

    fig_height = max(min_fig_height, row_height * len(prof))
    if figsize is None:
        figsize = (fig_width, fig_height)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=suptitle_fontsize, y=0.995)

    y = np.arange(len(prof))
    bars = ax.barh(y, prof["success_rate"].to_numpy(), height=bar_height, color=main_blue, edgecolor=bar_edgecolor, linewidth=bar_linewidth, alpha=bar_alpha)
    ax.set_yticks(y)
    ax.set_yticklabels(prof["task_label"])
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("Success rate across runs & conditions", fontsize=axis_label_fontsize)
    ax.set_ylabel("Task", fontsize=axis_label_fontsize)
    ax.set_title(panel_title, fontsize=panel_title_fontsize, pad=16)
    ax.grid(True, axis="x", alpha=grid_alpha)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=xtick_fontsize)
    ax.tick_params(axis="y", labelsize=ytick_fontsize)

    if show_reference_line:
        ref_value = float(prof["success_rate"].mean()) if reference_line_value is None else float(reference_line_value)
        ax.axvline(ref_value, color=contrast_orange, linestyle=reference_line_linestyle, linewidth=reference_line_linewidth, alpha=reference_line_alpha)

    if show_bar_values:
        for rect, value in zip(bars, prof["success_rate"].to_numpy()):
            ax.text(min(value + bar_value_dx, xlim[1] - 0.01), rect.get_y() + rect.get_height() / 2, bar_value_fmt.format(value), va="center", ha="left", fontsize=annotation_fontsize)

    fig.tight_layout(rect=tight_layout_rect)
    save_figure(fig, outpath_png=outpath_png, dpi=save_dpi)
    return fig, ax, prof


def plot_episode_cost_dashboard_v3(
    df,
    cond_col,
    outpath_png=None,
    title=None,
    figsize=(20, 8),
    wspace=0.28,
    hspace=0.20,
    point_size=55,
    point_alpha=0.35,
    point_edgecolor="black",
    point_linewidth=0.5,
    mean_marker="D",
    mean_marker_size=220,
    mean_alpha=1.0,
    mean_edgecolor="black",
    mean_linewidth=0.9,
    err_capsize=5,
    err_linewidth=1.2,
    jitter=0.10,
    random_seed=42,
    main_blue="#56B4E9",
    contrast_orange="#D55E00",
    blue_light="#8ECAE6",
    blue_dark="#1F77B4",
    fallback_cmap_name="tab10",
    suptitle_fontsize=SUPTITLE_FONTSIZE,
    panel_title_fontsize=PANEL_TITLE_FONTSIZE,
    axis_label_fontsize=AXIS_LABEL_FONTSIZE,
    xtick_fontsize=XTICK_FONTSIZE,
    ytick_fontsize=YTICK_FONTSIZE,
    legend_fontsize=LEGEND_FONTSIZE,
    legend_title_fontsize=LEGEND_TITLE_FONTSIZE,
    suptitle_y=0.98,
    title_pad=0,
    title_y=1.08,
    show_legend=True,
    legend_title="Experiment",
    legend_loc="center left",
    legend_bbox_to_anchor=(1.02, 0.5),
    legend_frameon=False,
    legend_ncol=1,
    legend_columnspacing=1.2,
    legend_handlelength=1.6,
    legend_handletextpad=0.5,
    legend_labelspacing=0.6,
    legend_borderaxespad=0.0,
    title_left="(A) Average LLM requests per episode",
    title_right="(B) Average episode duration",
    xlabel="Experiment",
    ylabel_left="Average LLM requests per episode",
    ylabel_right="Average duration per episode (sec)",
    xlabels_two_lines=True,
    compact_condition_labels=False,
    xtick_rotation=0,
    xtick_pad=8,
    grid_alpha=0.25,
    save_dpi=350,
    left_margin=0.07,
    right_margin=0.86,
    bottom_margin=0.12,
    top_margin=0.93,
):
    needed = ["run_id", cond_col, "llm_requests_total", "episode_duration_sec"]
    for column in needed:
        if column not in df.columns:
            raise ValueError(f"Missing column: {column}")

    plot_df = df.copy()
    plot_df[cond_col] = plot_df[cond_col].astype(str)
    agg_dict = {
        "avg_llm_requests": ("llm_requests_total", "mean"),
        "avg_duration_sec": ("episode_duration_sec", "mean"),
        "n_episodes": ("episode_id", "nunique") if "episode_id" in df.columns else ("run_id", "size"),
    }
    run_tbl = plot_df.groupby(["run_id", cond_col], as_index=False).agg(**agg_dict)

    summary_tbl = (
        run_tbl.groupby(cond_col, as_index=False)
        .agg(
            mean_llm_requests=("avg_llm_requests", "mean"),
            std_llm_requests=("avg_llm_requests", "std"),
            mean_duration_sec=("avg_duration_sec", "mean"),
            std_duration_sec=("avg_duration_sec", "std"),
            n_runs=("run_id", "nunique"),
            total_episodes=("n_episodes", "sum"),
        )
    )
    summary_tbl[cond_col] = summary_tbl[cond_col].astype(str)
    ordered = ordered_conditions(summary_tbl[cond_col].tolist())
    summary_tbl = summary_tbl.set_index(cond_col).reindex(ordered).reset_index()
    summary_tbl["std_llm_requests"] = summary_tbl["std_llm_requests"].fillna(0)
    summary_tbl["std_duration_sec"] = summary_tbl["std_duration_sec"].fillna(0)

    conds = summary_tbl[cond_col].dropna().astype(str).tolist()
    x = np.arange(len(conds))
    cond_to_x = {cond: idx for idx, cond in enumerate(conds)}
    if compact_condition_labels:
        xlabels = make_compact_condition_labels(conds)
    else:
        xlabels = make_two_line_labels(conds) if xlabels_two_lines else [str(c) for c in conds]

    preferred_colors = [main_blue, blue_dark, blue_light, contrast_orange]
    if len(conds) <= len(preferred_colors):
        palette = {cond: preferred_colors[idx] for idx, cond in enumerate(conds)}
    else:
        cmap = plt.get_cmap(fallback_cmap_name)
        palette = {cond: cmap(idx % cmap.N) for idx, cond in enumerate(conds)}

    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"wspace": wspace, "hspace": hspace})
    axA, axB = axes
    if title is not None:
        fig.suptitle(title, y=suptitle_y, fontsize=suptitle_fontsize)

    rng = np.random.default_rng(random_seed)
    for cond in conds:
        sub = run_tbl.loc[run_tbl[cond_col].astype(str) == cond].copy()
        xj = cond_to_x[cond] + rng.uniform(-jitter, jitter, size=len(sub))
        axA.scatter(xj, sub["avg_llm_requests"], s=point_size, alpha=point_alpha, color=palette[cond], edgecolors=point_edgecolor, linewidths=point_linewidth, zorder=2)
        axB.scatter(xj, sub["avg_duration_sec"], s=point_size, alpha=point_alpha, color=palette[cond], edgecolors=point_edgecolor, linewidths=point_linewidth, zorder=2)

    for _, row in summary_tbl.iterrows():
        cond = str(row[cond_col])
        xc = cond_to_x[cond]
        axA.errorbar(xc, row["mean_llm_requests"], yerr=row["std_llm_requests"], fmt="none", ecolor="black", elinewidth=err_linewidth, capsize=err_capsize, zorder=3)
        axA.scatter(xc, row["mean_llm_requests"], s=mean_marker_size, marker=mean_marker, alpha=mean_alpha, color=palette[cond], edgecolors=mean_edgecolor, linewidths=mean_linewidth, zorder=4)
        axB.errorbar(xc, row["mean_duration_sec"], yerr=row["std_duration_sec"], fmt="none", ecolor="black", elinewidth=err_linewidth, capsize=err_capsize, zorder=3)
        axB.scatter(xc, row["mean_duration_sec"], s=mean_marker_size, marker=mean_marker, alpha=mean_alpha, color=palette[cond], edgecolors=mean_edgecolor, linewidths=mean_linewidth, zorder=4)

    for ax, ylabel, panel_title in ((axA, ylabel_left, title_left), (axB, ylabel_right, title_right)):
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        ax.set_xlabel(xlabel, fontsize=axis_label_fontsize)
        ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)
        ax.set_title(panel_title, fontsize=panel_title_fontsize, pad=title_pad, y=title_y)
        _style_ax(ax, xtick_fontsize, ytick_fontsize, xtick_rotation, xtick_pad, grid_alpha)

    if show_legend:
        legend_handles = [
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor=palette[cond], markeredgecolor=point_edgecolor, markeredgewidth=point_linewidth, markersize=np.sqrt(point_size), alpha=0.95, label=str(cond))
            for cond in conds
        ]
        axB.legend(
            handles=legend_handles,
            title=legend_title,
            loc=legend_loc,
            bbox_to_anchor=legend_bbox_to_anchor,
            frameon=legend_frameon,
            fontsize=legend_fontsize,
            title_fontsize=legend_title_fontsize,
            ncol=legend_ncol,
            columnspacing=legend_columnspacing,
            handlelength=legend_handlelength,
            handletextpad=legend_handletextpad,
            borderaxespad=legend_borderaxespad,
            labelspacing=legend_labelspacing,
        )

    fig.subplots_adjust(left=left_margin, right=right_margin, bottom=bottom_margin, top=top_margin, wspace=wspace)
    save_figure(fig, outpath_png=outpath_png, dpi=save_dpi)
    return fig, axes, run_tbl, summary_tbl


def plot_error_dashboard_v3(
    df,
    cond_col="exp_label",
    run_col="run_id",
    outpath_png=None,
    title=None,
    figsize=(20, 6.8),
    width_ratios=(1.08, 1.18),
    wspace=0.24,
    bar_width=0.58,
    bar_edgecolor="black",
    bar_linewidth=0.6,
    err_capsize=5,
    err_linewidth=1.2,
    main_blue="#56B4E9",
    blue_light="#8ECAE6",
    blue_dark="#1F77B4",
    suptitle_fontsize=SUPTITLE_FONTSIZE,
    panel_title_fontsize=PANEL_TITLE_FONTSIZE,
    axis_label_fontsize=AXIS_LABEL_FONTSIZE,
    xtick_fontsize=XTICK_FONTSIZE,
    ytick_fontsize=YTICK_FONTSIZE,
    legend_fontsize=LEGEND_FONTSIZE,
    suptitle_y=0.98,
    title_pad=0,
    title_y=1.18,
    legend_outside=True,
    legend_y=1.08,
    legend_loc="center",
    legend_ncol=2,
    legend_columnspacing=1.2,
    legend_handlelength=1.6,
    legend_handletextpad=0.5,
    legend_labelspacing=0.6,
    title_A="(A) Recovery opportunities",
    title_B="(B) Recovery rate",
    ylabel_A="Mean opportunities per episode",
    ylabel_B="Recovery rate $r_{rec}$",
    xlabels_two_lines=True,
    compact_condition_labels=False,
    xtick_rotation=0,
    xtick_pad=8,
    grid_alpha=0.25,
    ylim_rate=(0.0, 1.0),
    save_dpi=350,
    binary_label_fontsize=ANNOTATION_FONTSIZE,
    left_margin=0.07,
    right_margin=0.98,
    bottom_margin=0.12,
    top_margin=0.93,
):
    recovery_metrics = [
        "recoveries_after_fail_total",
        "recoveries_after_fail_opportunities",
        "recovery_after_fail_rate",
    ]
    available_recovery = [col for col in recovery_metrics if col in df.columns]
    if not available_recovery:
        raise ValueError("None of the expected recovery columns were found in df.")

    conds = ordered_conditions(df[cond_col].dropna().astype(str).unique().tolist())
    x = np.arange(len(conds))

    if run_col is not None and run_col in df.columns:
        rec_run_tbl = (
            df.groupby([run_col, cond_col], observed=False, as_index=False)[available_recovery]
            .mean(numeric_only=True)
        )
        rec_summary_tbl = rec_run_tbl.groupby(cond_col, observed=False)[available_recovery].agg(["mean", "std"])
    else:
        rec_summary_tbl = df.groupby(cond_col, observed=False)[available_recovery].agg(["mean", "std"])

    rec_summary_tbl.columns = [
        cond_col if col[0] == cond_col else f"{col[0]}_{col[1]}"
        for col in rec_summary_tbl.columns.to_flat_index()
    ]
    rec_summary_tbl = rec_summary_tbl.reset_index(drop=True)
    rec_summary_tbl[cond_col] = pd.Series(conds)
    rec_summary_tbl = rec_summary_tbl.set_index(cond_col).reindex(conds).reset_index()

    for col in ["recoveries_after_fail_total_std", "recoveries_after_fail_opportunities_std", "recovery_after_fail_rate_std"]:
        if col in rec_summary_tbl.columns:
            rec_summary_tbl[col] = rec_summary_tbl[col].fillna(0)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=width_ratios, wspace=wspace)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    if compact_condition_labels:
        xlabels = make_compact_condition_labels(conds)
    else:
        xlabels = make_two_line_labels(conds) if xlabels_two_lines else [str(c) for c in conds]

    if {"recoveries_after_fail_total_mean", "recoveries_after_fail_opportunities_mean"}.issubset(rec_summary_tbl.columns):
        recovered = rec_summary_tbl["recoveries_after_fail_total_mean"].fillna(0).to_numpy()
        opportunities = rec_summary_tbl["recoveries_after_fail_opportunities_mean"].fillna(0).to_numpy()
        not_recovered = np.clip(opportunities - recovered, a_min=0, a_max=None)
        h1 = axA.bar(x, recovered, width=bar_width, label="Recovered", color=blue_light, edgecolor=bar_edgecolor, linewidth=bar_linewidth)
        h2 = axA.bar(x, not_recovered, bottom=recovered, width=bar_width, label="Not recovered", color=blue_dark, edgecolor=bar_edgecolor, linewidth=bar_linewidth)
        axA.set_xticks(x)
        axA.set_xticklabels(xlabels)
        axA.set_ylabel(ylabel_A, fontsize=axis_label_fontsize)
        axA.set_title(title_A, fontsize=panel_title_fontsize, pad=title_pad, y=title_y)
        _style_ax(axA, xtick_fontsize, ytick_fontsize, xtick_rotation, xtick_pad, grid_alpha)
        _annotate_binary_stack(axA, x, recovered, not_recovered, binary_label_fontsize)
        if legend_outside:
            axA.legend(
                [h1[0], h2[0]],
                ["Recovered", "Not recovered"],
                loc=legend_loc,
                bbox_to_anchor=(0.5, legend_y),
                ncol=legend_ncol,
                frameon=False,
                fontsize=legend_fontsize,
                columnspacing=legend_columnspacing,
                handlelength=legend_handlelength,
                handletextpad=legend_handletextpad,
                labelspacing=legend_labelspacing,
            )
    else:
        axA.axis("off")

    if "recovery_after_fail_rate_mean" in rec_summary_tbl.columns:
        axB.bar(
            x,
            rec_summary_tbl["recovery_after_fail_rate_mean"].fillna(0).to_numpy(),
            yerr=rec_summary_tbl.get("recovery_after_fail_rate_std", pd.Series(np.zeros(len(conds)))).fillna(0).to_numpy(),
            capsize=err_capsize,
            width=bar_width,
            color=main_blue,
            edgecolor=bar_edgecolor,
            linewidth=bar_linewidth,
            error_kw={"elinewidth": err_linewidth},
        )
        axB.set_xticks(x)
        axB.set_xticklabels(xlabels)
        axB.set_ylim(*ylim_rate)
        axB.set_ylabel(ylabel_B, fontsize=axis_label_fontsize)
        axB.set_title(title_B, fontsize=panel_title_fontsize, pad=title_pad, y=title_y)
        _style_ax(axB, xtick_fontsize, ytick_fontsize, xtick_rotation, xtick_pad, grid_alpha)
    else:
        axB.axis("off")

    if title is not None:
        fig.suptitle(title, y=suptitle_y, fontsize=suptitle_fontsize)
    fig.subplots_adjust(left=left_margin, right=right_margin, bottom=bottom_margin, top=top_margin, wspace=wspace)
    save_figure(fig, outpath_png=outpath_png, dpi=save_dpi)
    return fig, (axA, axB), rec_summary_tbl
