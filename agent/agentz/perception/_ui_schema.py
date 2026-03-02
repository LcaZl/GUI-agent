from typing import Any, Dict, List, Optional

import pandas as pd
from agentz.constants import (
    AREA_DENOM_FLOOR,
    AREA_NORM_WEIGHT,
    DEPTH_DEFAULT,
    DEPTH_NORM_PENALTY,
    ENABLED_WEIGHT_SCHEMA,
    INTERACTIVE_WEIGHT_SCHEMA,
    SCORE_DEFAULT,
    SRC_PREF_WEIGHT,
    VISIBLE_BONUS,
)

STANDARD_UI_COLUMNS: List[str] = [  # Canonical column order for UI dataframe.
    "source",
    "type",
    "role",
    "content",
    "value",
    "x1",
    "y1",
    "x2",
    "y2",
    "cx",
    "cy",
    "score",
    "vision_score",
    "fusion_score",
    "fusion_matched",
    "a11y_id",
    "a11y_role",
    "a11y_showing",
    "a11y_visible",
    "a11y_enabled",
    "a11y_focused",
    "a11y_selected",
    "a11y_checked",
    "a11y_expanded",
    "a11y_states_raw",
    "a11y_actions_raw",
    "a11y_node_id",
    "a11y_parent_id",
    "a11y_depth",
    "a11y_child_index",
    "a11y_is_interactive",
    # in RAW_A11Y_COLUMNS
    "app_name",
    "window_name",
    "window_active",

]

_SCORE_COLUMNS: tuple[str, ...] = ("score", "vision_score", "fusion_score")
_VALID_SOURCES: set[str] = {"vision", "a11y", "fusion"}
_PLACEHOLDER_LABELS: set[str] = {"<unlabeled>", "unlabeled"}
_UNLABELED_ACTIONABLE_ROLES: set[str] = {"slider", "scroll-bar", "spin-button"}
_DEDUP_PROTECTED_ROLES: set[str] = {"slider", "scroll-bar", "spin-button", "level-bar"}


def _isna(v: Any) -> bool:
    """
    Process isna.
        
        Parameters
        ----------
        v : Any
            Function argument.
        
        Returns
        -------
        bool
            True when the condition is satisfied, otherwise False.
        
    """
    try:
        return pd.isna(v)
    except Exception:
        return v is None


def _coerce_float_or_none(v: Any) -> Optional[float]:
    """
    Coerce float or none.
        
        Parameters
        ----------
        v : Any
            Function argument.
        
        Returns
        -------
        Optional[float]
            Function result.
        
    """
    if _isna(v):
        return None
    try:
        return float(v)
    except Exception:
        return None


def _coerce_bool_or_none(v: Any) -> Optional[bool]:
    """
    Coerce bool or none.
        
        Parameters
        ----------
        v : Any
            Function argument.
        
        Returns
        -------
        Optional[bool]
            Function result.
        
    """
    if _isna(v):
        return None
    if v is None:
        return None
    return bool(v)


def _normalize_source(v: Any) -> Optional[str]:
    """
    Normalize source.
        
        Parameters
        ----------
        v : Any
            Function argument.
        
        Returns
        -------
        Optional[str]
            Function result.
        
    """
    if _isna(v):
        return None
    s = str(v).strip().lower()
    if not s:
        return None
    return s if s in _VALID_SOURCES else None


def ensure_click_center(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure click center.
        
        Parameters
        ----------
        df : pd.DataFrame
            Function argument.
        
        Returns
        -------
        pd.DataFrame
            Function result.
        
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    for col in ("cx", "cy", "x1", "y1", "x2", "y2"):
        if col not in out.columns:
            out[col] = None

    def compute_center(row: pd.Series) -> tuple[Optional[float], Optional[float]]:
        """
        Compute center.
        
        Parameters
        ----------
        row : pd.Series
            Function argument.
        
        Returns
        -------
        tuple[Optional[float], Optional[float]]
            Tuple with computed values.
        
        """
        x1, y1, x2, y2 = row.get("x1"), row.get("y1"), row.get("x2"), row.get("y2")
        if any(_isna(v) for v in (x1, y1, x2, y2)):
            return None, None
        try:
            return (float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0
        except Exception:
            return None, None

    for idx, row in out.iterrows():
        if _isna(row.get("cx")) or _isna(row.get("cy")):
            cx, cy = compute_center(row)
            if cx is not None and cy is not None:
                out.at[idx, "cx"] = cx
                out.at[idx, "cy"] = cy

    return out


def standardize_ui_df(df: pd.DataFrame, defaults: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Standardize ui df.
        
        Parameters
        ----------
        df : pd.DataFrame
            Function argument.
        defaults : Optional[Dict[str, Any]]
            Function argument.
        
        Returns
        -------
        pd.DataFrame
            Function result.
        
    """
    if df is None:
        return pd.DataFrame(columns=STANDARD_UI_COLUMNS)

    defaults = defaults or {}
    out = df.copy()

    for col in STANDARD_UI_COLUMNS:
        if col not in out.columns:
            out[col] = defaults.get(col, None)

    out = ensure_click_center(out)
    for col in _SCORE_COLUMNS:
        if col in out.columns:
            coerced = out[col].apply(_coerce_float_or_none).tolist()
            out[col] = pd.Series(
                [None if _isna(v) else float(v) for v in coerced],
                index=out.index,
                dtype=object,
            )
    if "fusion_matched" in out.columns:
        out["fusion_matched"] = out["fusion_matched"].apply(_coerce_bool_or_none)
    if "source" in out.columns:
        out["source"] = out["source"].apply(_normalize_source)
    return out[STANDARD_UI_COLUMNS].copy()

def dedupe_ui_df_by_label(
    df: pd.DataFrame,
    *,
    label_col: str = "content",
    prefer_source: str = "a11y",
    spatial_iou_threshold: float = 0.75,
    max_per_label: Optional[int] = None,
) -> pd.DataFrame:
    """
    Spatially-aware label deduplication.
    Keeps multiple rows with the same label when they refer to distinct
    regions of the screen; removes only near-overlapping duplicates.
    
    Ranking preference (descending):
      1) source == prefer_source
      2) a11y_is_interactive True
      3) a11y_enabled True
      4) a11y_visible True
      5) bbox area (bigger wins as a generic proxy for "main" target)
      6) score (higher wins)
      7) smaller a11y_depth (more surface-level)
    
    Deduplication rule per label:
      - sort by rank
      - keep a row unless it overlaps (IoU >= spatial_iou_threshold)
        with an already-kept row of the same label
      - optionally cap survivors per label with max_per_label
    """
    if df is None or df.empty or label_col not in df.columns:
        return df

    out = df.copy()

    def _norm(s: Any) -> str:
        """
        Process norm.
        
        Parameters
        ----------
        s : Any
            Function argument.
        
        Returns
        -------
        str
            Resulting string value.
        
        """
        try:
            if pd.isna(s):
                return ""
        except Exception:
            pass
        txt = str(s).strip().casefold()
        return "" if txt in _PLACEHOLDER_LABELS else txt

    def _safe_box(r: pd.Series) -> Optional[tuple[float, float, float, float]]:
        """
        Safely handle box.
        
        Parameters
        ----------
        r : pd.Series
            Function argument.
        
        Returns
        -------
        Optional[tuple[float, float, float, float]]
            Tuple with computed values.
        
        """
        try:
            bx1 = float(r.get("x1"))
            by1 = float(r.get("y1"))
            bx2 = float(r.get("x2"))
            by2 = float(r.get("y2"))
            if any(pd.isna(v) for v in (bx1, by1, bx2, by2)):
                return None
            if bx2 <= bx1 or by2 <= by1:
                return None
            return (bx1, by1, bx2, by2)
        except Exception:
            return None

    def _safe_center(r: pd.Series) -> Optional[tuple[float, float]]:
        """
        Safely handle center.
        
        Parameters
        ----------
        r : pd.Series
            Function argument.
        
        Returns
        -------
        Optional[tuple[float, float]]
            Tuple with computed values.
        
        """
        try:
            cx = float(r.get("cx"))
            cy = float(r.get("cy"))
            if any(pd.isna(v) for v in (cx, cy)):
                return None
            return (cx, cy)
        except Exception:
            return None

    def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        """
        Process bbox iou.
        
        Parameters
        ----------
        a : tuple[float, float, float, float]
            Function argument.
        b : tuple[float, float, float, float]
            Function argument.
        
        Returns
        -------
        float
            Floating-point result value.
        
        """
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter
        return 0.0 if denom <= 0.0 else inter / denom

    def _center_dist(a: tuple[float, float], b: tuple[float, float]) -> float:
        """
        Process center dist.
        
        Parameters
        ----------
        a : tuple[float, float]
            Function argument.
        b : tuple[float, float]
            Function argument.
        
        Returns
        -------
        float
            Floating-point result value.
        
        """
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return (dx * dx + dy * dy) ** 0.5

    role_src = "a11y_role" if "a11y_role" in out.columns else ("role" if "role" in out.columns else None)
    out["_label_norm"] = out[label_col].apply(_norm)
    if role_src is None:
        out["_role_norm"] = ""
    else:
        out["_role_norm"] = out[role_src].apply(_norm)

    # Drop unlabeled rows by default, but keep known actionable controls
    # that are often unlabeled in accessibility trees (e.g., volume sliders).
    if "a11y_actions_raw" in out.columns:
        actions_raw = (
            out["a11y_actions_raw"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.casefold()
        )
    else:
        actions_raw = pd.Series("", index=out.index, dtype="object")
    keep_unlabeled = out["_role_norm"].isin(_UNLABELED_ACTIONABLE_ROLES) | actions_raw.str.len().gt(0)
    out = out[(out["_label_norm"] != "") | keep_unlabeled].copy()
    if out.empty:
        return out.drop(columns=["_label_norm", "_role_norm"], errors="ignore")

    src_pref = (out.get("source", "") == prefer_source).astype(int)
    role_present = out["_role_norm"].astype(str).str.len().gt(0).astype(int)

    def _bool_col(name: str) -> pd.Series:
        """
        Process bool col.
        
        Parameters
        ----------
        name : str
            Function argument.
        
        Returns
        -------
        pd.Series
            Function result.
        
        """
        s = out.get(name, pd.Series(pd.NA, index=out.index, dtype="boolean"))
        s = s.astype("boolean")
        return s.fillna(False).astype(int)

    def _num_col(name: str, default: float = 0.0) -> pd.Series:
        """
        Process num col.
        
        Parameters
        ----------
        name : str
            Function argument.
        default : Optional[float]
            Function argument.
        
        Returns
        -------
        pd.Series
            Function result.
        
        """
        s = out.get(name)
        if isinstance(s, pd.Series):
            return pd.to_numeric(s, errors="coerce")
        return pd.Series(default, index=out.index, dtype="float64")

    interactive = _bool_col("a11y_is_interactive")
    enabled = _bool_col("a11y_enabled")
    visible = _bool_col("a11y_visible")

    x1 = _num_col("x1")
    y1 = _num_col("y1")
    x2 = _num_col("x2")
    y2 = _num_col("y2")
    area = ((x2 - x1).clip(lower=0) * (y2 - y1).clip(lower=0)).fillna(0.0)

    score = _num_col("score", default=SCORE_DEFAULT).fillna(SCORE_DEFAULT)
    depth = _num_col("a11y_depth", default=DEPTH_DEFAULT).fillna(DEPTH_DEFAULT)

    area_norm = area / (area.max() if area.max() > 0 else AREA_DENOM_FLOOR)
    depth_norm = depth / (depth.max() if depth.max() > 0 else AREA_DENOM_FLOOR)

    out["_rank"] = (
        src_pref * SRC_PREF_WEIGHT
        + role_present * 50_000
        + interactive * INTERACTIVE_WEIGHT_SCHEMA
        + enabled * ENABLED_WEIGHT_SCHEMA
        + visible * VISIBLE_BONUS
        + area_norm * AREA_NORM_WEIGHT
        + score
        - depth_norm * DEPTH_NORM_PENALTY
    )

    iou_th = float(spatial_iou_threshold)
    if iou_th < 0.0:
        iou_th = 0.0
    if iou_th > 1.0:
        iou_th = 1.0

    # If a label has both rows with/without role, drop no-role rows when spatially equivalent.
    roleful = out[out["_role_norm"] != ""]
    no_role = out[out["_role_norm"] == ""]
    if not roleful.empty and not no_role.empty:
        drop_ids: List[int] = []
        for idx, row in no_role.iterrows():
            same_label = roleful[roleful["_label_norm"] == row["_label_norm"]]
            if same_label.empty:
                continue
            box = _safe_box(row)
            center = _safe_center(row)
            should_drop = False
            for _, r2 in same_label.iterrows():
                box2 = _safe_box(r2)
                center2 = _safe_center(r2)
                if box is not None and box2 is not None and _bbox_iou(box, box2) >= iou_th:
                    should_drop = True
                    break
                if center is not None and center2 is not None and _center_dist(center, center2) <= 20.0:
                    should_drop = True
                    break
            if should_drop:
                drop_ids.append(idx)
        if drop_ids:
            out = out.drop(index=drop_ids)

    # Dedup only when both normalized label and role match.
    kept_groups: List[pd.DataFrame] = []
    for _, group in out.groupby(["_label_norm", "_role_norm"], sort=False):
        g = group.sort_values(by="_rank", ascending=False).copy()
        selected_rows = []
        selected_boxes: List[tuple[float, float, float, float]] = []
        selected_centers: List[tuple[float, float]] = []

        for _, row in g.iterrows():
            box = _safe_box(row)
            center = _safe_center(row)

            is_dup = False
            if box is not None:
                for kbox in selected_boxes:
                    if _bbox_iou(box, kbox) >= iou_th:
                        is_dup = True
                        break
            elif center is not None:
                for kc in selected_centers:
                    if _center_dist(center, kc) <= 8.0:
                        is_dup = True
                        break

            if not is_dup:
                selected_rows.append(row)
                if box is not None:
                    selected_boxes.append(box)
                if center is not None:
                    selected_centers.append(center)

        if selected_rows:
            kept_groups.append(pd.DataFrame(selected_rows))

    if kept_groups:
        out = pd.concat(kept_groups, ignore_index=True)
    else:
        out = out.head(0)

    # Optional cap per label after dedup (keeps highest-rank rows),
    # but never cap critical a11y-backed controls.
    if max_per_label is not None and int(max_per_label) > 0 and not out.empty:
        is_protected_role = out["_role_norm"].isin(_DEDUP_PROTECTED_ROLES)
        has_a11y_role = out.get("a11y_role", pd.Series(index=out.index, dtype="object")).notna()
        protected_mask = is_protected_role & has_a11y_role

        protected = out[protected_mask].copy()
        regular = out[~protected_mask].copy()

        if not regular.empty:
            regular = regular.sort_values(by="_rank", ascending=False)
            regular = regular.groupby("_label_norm", sort=False).head(int(max_per_label)).copy()

        if not protected.empty and not regular.empty:
            out = pd.concat([regular, protected], ignore_index=True)
        elif not protected.empty:
            out = protected
        else:
            out = regular

    return out.drop(columns=["_label_norm", "_role_norm", "_rank"], errors="ignore")
