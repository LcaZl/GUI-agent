import numpy as np
import pandas as pd
import matplotlib.colors as mcolors

from .config import PLOT_CONDITION_ORDER


def truncate_cmap(cmap, minval=0.15, maxval=0.80, n=256):
    new_colors = cmap(np.linspace(minval, maxval, n))
    return mcolors.LinearSegmentedColormap.from_list("trunc", new_colors)


def make_two_line_labels(labels):
    out = []
    for label in labels:
        text = str(label)
        if "|" in text:
            left, right = [part.strip() for part in text.split("|", 1)]
            out.append(left + "\n" + right)
        else:
            out.append(text.replace(" ", "\n", 1))
    return out


def make_compact_condition_labels(labels):
    out = []
    for label in labels:
        text = str(label)
        if "|" not in text:
            out.append(text)
            continue
        left, right = [part.strip() for part in text.split("|", 1)]
        if right.startswith("Vision "):
            suffix = right.replace("Vision ", "", 1).strip()
            out.append(left + "\nVision\n" + suffix)
        else:
            out.append(left + "\n" + right.replace(" ", "\n"))
    return out


def ordered_conditions(values, order=None):
    vals = [str(v) for v in values if pd.notna(v)]
    uniq = list(dict.fromkeys(vals))
    ref = PLOT_CONDITION_ORDER if order is None else order
    ordered = [c for c in ref if c in uniq]
    remainder = [c for c in uniq if c not in ordered]
    return ordered + remainder


def save_figure(fig, outpath_png=None, dpi=350):
    if outpath_png:
        fig.savefig(outpath_png, dpi=dpi, bbox_inches="tight")


def sum_counters(series):
    total = {}
    for value in series.dropna():
        if isinstance(value, dict):
            for key, count in value.items():
                total[str(key)] = total.get(str(key), 0.0) + float(count)
    return total
