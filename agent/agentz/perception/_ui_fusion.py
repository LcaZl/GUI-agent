# _ui_fusion.py
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from agentz.constants import (
    DEFAULT_MATCH_COLUMN,
    DEDUP_IOU_SAME_LABEL,
    ENABLED_WEIGHT_FUSION,
    FUSION_WEIGHT,
    INTERACTIVE_WEIGHT_FUSION,
    MAX_CENTER_DIST_PX,
    MIN_FUSION_SCORE,
    MIN_IOU_CANDIDATE,
    MIN_POS_ONLY_SCORE,
    MIN_TXT_ONLY_SCORE,
    POS_WEIGHT,
    SHOWING_WEIGHT,
    TXT_WEIGHT,
    VISION_A11Y_SUPPRESS_CENTER_DIST_PX,
    VISION_A11Y_SUPPRESS_IOU,
    VISION_A11Y_SUPPRESS_MIN_OVERLAP,
    VISION_A11Y_SUPPRESS_MIN_TARGET_COVER,
    VISION_WEIGHT,
)

_ACTIONABLE_A11Y_ROLES = {
    "push-button",
    "toggle-button",
    "switch",
    "check-box",
    "radio-button",
    "menu-item",
    "combo-box",
    "entry",
    "link",
    "spin-button",
    "slider",
    "page-tab",
    "icon",
}


def _normalize_text(value: Any) -> str:
    """
    Normalize text.
        
        Parameters
        ----------
        value : Any
            Input value.
        
        Returns
        -------
        str
            Resulting string value.
        
    """
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip().casefold()


def _similarity(a: str, b: str) -> float:
    """
    Process similarity.
        
        Parameters
        ----------
        a : str
            Function argument.
        b : str
            Function argument.
        
        Returns
        -------
        float
            Floating-point result value.
        
    """
    if not a or not b:
        return 0.0
    return SequenceMatcher(a=a, b=b).ratio()


def _bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    """
    Process bbox iou.
        
        Parameters
        ----------
        a : Tuple[float, float, float, float]
            Function argument.
        b : Tuple[float, float, float, float]
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


def _bbox_intersection_area(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    """
    Process bbox intersection area.
        
        Parameters
        ----------
        a : Tuple[float, float, float, float]
            Function argument.
        b : Tuple[float, float, float, float]
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
    return iw * ih


def _box_area(b: Tuple[float, float, float, float]) -> float:
    """
    Process box area.
        
        Parameters
        ----------
        b : Tuple[float, float, float, float]
            Function argument.
        
        Returns
        -------
        float
            Floating-point result value.
        
    """
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _is_actionable_a11y_row(row: Dict[str, Any]) -> bool:
    """
    Return whether is actionable a11y row.
    
    Parameters
    ----------
    row : Dict[str, Any]
        Function argument.
    
    Returns
    -------
    bool
        True when the condition is satisfied, otherwise False.
    """
    inter = row.get("a11y_is_interactive")
    if inter is True:
        return True
    role = _normalize_text(row.get("a11y_role"))
    if role in _ACTIONABLE_A11Y_ROLES:
        return True
    actions_raw = _normalize_text(row.get("a11y_actions_raw"))
    return bool(actions_raw)


def _center_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """
    Process center distance.
        
        Parameters
        ----------
        a : Tuple[float, float]
            Function argument.
        b : Tuple[float, float]
            Function argument.
        
        Returns
        -------
        float
            Floating-point result value.
        
    """
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


def _is_vision_duplicate_of_a11y_target(
    *,
    vision_box: Optional[Tuple[float, float, float, float]],
    vision_center: Optional[Tuple[float, float]],
    a11y_box: Optional[Tuple[float, float, float, float]],
    a11y_center: Optional[Tuple[float, float]],
) -> bool:
    """
    Suppress vision-only rows that are effectively the same target as an a11y-backed row.
    We use three robust cues:
    - high IoU
    - strong containment (intersection / min area)
    - near-identical centers
    """
    if vision_box is not None and a11y_box is not None:
        iou = _bbox_iou(vision_box, a11y_box)
        inter = _bbox_intersection_area(vision_box, a11y_box)
        v_area = _box_area(vision_box)
        a_area = _box_area(a11y_box)
        v_cover = (inter / v_area) if v_area > 0.0 else 0.0
        a_cover = (inter / a_area) if a_area > 0.0 else 0.0

        if iou >= VISION_A11Y_SUPPRESS_IOU:
            return True
        # Containment-like suppression only if overlap is high from the vision perspective
        # and also non-trivial on the a11y target (avoid killing text inside huge containers).
        if v_cover >= VISION_A11Y_SUPPRESS_MIN_OVERLAP and a_cover >= VISION_A11Y_SUPPRESS_MIN_TARGET_COVER:
            return True

        # Center proximity alone is too aggressive in dense UIs.
        # Use it only when there is at least some geometric overlap.
        if vision_center is not None and a11y_center is not None:
            if _center_distance(vision_center, a11y_center) <= VISION_A11Y_SUPPRESS_CENTER_DIST_PX and iou >= 0.20:
                return True

    # Fallback for rows without usable boxes.
    if vision_box is None and a11y_box is None and vision_center is not None and a11y_center is not None:
        if _center_distance(vision_center, a11y_center) <= VISION_A11Y_SUPPRESS_CENTER_DIST_PX:
            return True

    return False


def _safe_box(r: pd.Series) -> Optional[Tuple[float, float, float, float]]:
    """
    Safely handle box.
        
        Parameters
        ----------
        r : pd.Series
            Function argument.
        
        Returns
        -------
        Optional[Tuple[float, float, float, float]]
            Tuple with computed values.
        
    """
    try:
        x1, y1, x2, y2 = float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)
    except Exception:
        return None


def _safe_center(r: pd.Series) -> Optional[Tuple[float, float]]:
    """
    Safely handle center.
        
        Parameters
        ----------
        r : pd.Series
            Function argument.
        
        Returns
        -------
        Optional[Tuple[float, float]]
            Tuple with computed values.
        
    """
    try:
        return (float(r["cx"]), float(r["cy"]))
    except Exception:
        b = _safe_box(r)
        if b is None:
            return None
        x1, y1, x2, y2 = b
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """
    Safely handle float.
        
        Parameters
        ----------
        value : Any
            Input value.
        default : Optional[float]
            Function argument.
        
        Returns
        -------
        Optional[float]
            Function result.
        
    """
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return default


def _source_priority(source: Any) -> float:
    """
    Process source priority.
        
        Parameters
        ----------
        source : Any
            Function argument.
        
        Returns
        -------
        float
            Floating-point result value.
        
    """
    src = _normalize_text(source)
    if src == "fusion":
        return 3.0
    if src == "a11y":
        return 2.0
    if src == "vision":
        return 1.0
    return 0.0


def _pos_score(v_box, a_box, v_center, a_center) -> float:
    """
    Process pos score.
        
        Parameters
        ----------
        v_box : Any
            Function argument.
        a_box : Any
            Function argument.
        v_center : Any
            Function argument.
        a_center : Any
            Function argument.
        
        Returns
        -------
        float
            Floating-point result value.
        
    """
    if v_box is not None and a_box is not None:
        iou = _bbox_iou(v_box, a_box)
        if iou > 0.0:
            return iou
    if v_center is None or a_center is None:
        return 0.0
    d = _center_distance(v_center, a_center)
    if d >= MAX_CENTER_DIST_PX:
        return 0.0
    return max(0.0, 1.0 - (d / MAX_CENTER_DIST_PX))


def _rank_row(r: pd.Series) -> float:
    """
    Process rank row.
        
        Parameters
        ----------
        r : pd.Series
            Function argument.
        
        Returns
        -------
        float
            Floating-point result value.
        
    """
    inter = 1.0 if bool(r.get("a11y_is_interactive")) else 0.0
    enabled = 1.0 if r.get("a11y_enabled") is True else 0.0
    showing = 1.0 if r.get("a11y_showing") is True else 0.0
    fscore = _safe_float(r.get("fusion_score"), default=0.0) or 0.0
    vscore = _safe_float(r.get("vision_score"), default=0.0) or 0.0
    src = _source_priority(r.get("source"))
    return (
        src * 100.0
        + inter * INTERACTIVE_WEIGHT_FUSION
        + enabled * ENABLED_WEIGHT_FUSION
        + showing * SHOWING_WEIGHT
        + fscore * FUSION_WEIGHT
        + vscore * VISION_WEIGHT
    )


def dedup_spatial_same_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate spatial same label.
        
        Parameters
        ----------
        df : pd.DataFrame
            Function argument.
        
        Returns
        -------
        pd.DataFrame
            Function result.
        
    """
    if df is None or df.empty or "content" not in df.columns:
        return df

    tmp = df.copy()
    tmp["_label_norm"] = tmp["content"].apply(_normalize_text)

    kept = []
    for label, group in tmp.groupby("_label_norm", sort=False):
        if not label:
            kept.append(group.drop(columns=["_label_norm"], errors="ignore"))
            continue

        group = group.copy()
        group["_box"] = [_safe_box(r) for _, r in group.iterrows()]
        group["_rank"] = group.apply(_rank_row, axis=1)
        group = group.sort_values(by="_rank", ascending=False)

        selected_rows = []
        selected_boxes = []

        for _, r in group.iterrows():
            b = r.get("_box")
            if b is None:
                selected_rows.append(r)
                continue
            is_dup = False
            for kb in selected_boxes:
                if kb is None:
                    continue
                if _bbox_iou(b, kb) >= DEDUP_IOU_SAME_LABEL:
                    is_dup = True
                    break
            if not is_dup:
                selected_rows.append(r)
                selected_boxes.append(b)

        kept.append(pd.DataFrame(selected_rows).drop(columns=["_label_norm", "_box", "_rank"], errors="ignore"))

    return pd.concat(kept, ignore_index=True) if kept else df.head(0)


class UIFusion:
    def __init__(self, match_column: str = DEFAULT_MATCH_COLUMN) -> None:
        """
        Initialize class dependencies and runtime state.
        
        Parameters
        ----------
        match_column : Optional[str]
            Function argument.
        
        Returns
        -------
        None
            No return value.
        """
        self.match_column = str(match_column)

    def fuse(self, vision_df: pd.DataFrame, a11y_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fuse accessibility and vision UI candidates into one table.
        
        Parameters
        ----------
        vision_df : pd.DataFrame
            Function argument.
        a11y_df : pd.DataFrame
            Function argument.
        
        Returns
        -------
        pd.DataFrame
            Function result.
        
        """
        vision = vision_df.copy() if vision_df is not None and not vision_df.empty else pd.DataFrame()
        a11y = a11y_df.copy() if a11y_df is not None and not a11y_df.empty else pd.DataFrame()

        if not vision.empty:
            if "content" not in vision.columns:
                vision["content"] = None
            if "source" not in vision.columns:
                vision["source"] = "vision"
            vision["_content_norm"] = vision["content"].apply(_normalize_text)
            vision["_box"] = [_safe_box(r) for _, r in vision.iterrows()]
            vision["_center"] = [_safe_center(r) for _, r in vision.iterrows()]

        if a11y.empty:
            if vision.empty:
                return pd.DataFrame()
            out = vision.copy()
            out["source"] = "vision"
            vscore = pd.to_numeric(out.get("vision_score", out.get("score")), errors="coerce")
            out["vision_score"] = vscore.where(vscore.notna(), None).astype(object)
            out["score"] = out["vision_score"]
            out["fusion_score"] = None
            out["fusion_matched"] = False
            # Ensure a11y columns exist (including new app/window columns)
            for col in [
                "a11y_id","a11y_role","a11y_showing","a11y_visible","a11y_enabled","a11y_focused","a11y_selected",
                "a11y_checked","a11y_expanded","a11y_states_raw","a11y_actions_raw","a11y_node_id","a11y_parent_id",
                "a11y_depth","a11y_child_index","a11y_is_interactive",
                "app_name","window_name","window_active",
            ]:
                if col not in out.columns:
                    out[col] = None
            return out

        if self.match_column not in a11y.columns:
            a11y[self.match_column] = a11y.get("name", "").apply(_normalize_text)

        out_rows: List[Dict[str, Any]] = []
        matched_vision_idx: set[int] = set()

        for _, a in a11y.iterrows():
            abox = _safe_box(a)
            ac = _safe_center(a)
            a_name_norm = a.get(self.match_column, "")
            a_name_norm = a_name_norm if isinstance(a_name_norm, str) else _normalize_text(a_name_norm)

            best_v = None
            best_final = -1.0
            best_vidx = None

            if not vision.empty:
                for vidx, v in vision.iterrows():
                    v_box = v.get("_box")
                    v_center = v.get("_center")

                    vtxt = v.get("_content_norm", "")
                    txt = _similarity(vtxt, a_name_norm) if vtxt and a_name_norm else 0.0

                    # Standard path: geometry-aware fusion.
                    if (abox is not None) or (ac is not None):
                        pos = _pos_score(v_box, abox, v_center, ac)
                        if pos <= 0.0:
                            continue

                        if pos < MIN_IOU_CANDIDATE:
                            if v_center is None or ac is None:
                                continue
                            if _center_distance(v_center, ac) > MAX_CENTER_DIST_PX:
                                continue

                        if vtxt and a_name_norm:
                            final = POS_WEIGHT * pos + TXT_WEIGHT * txt
                            ok = final >= MIN_FUSION_SCORE
                        else:
                            final = pos
                            ok = final >= MIN_POS_ONLY_SCORE
                    else:
                        # Fallback path for a11y controls with missing geometry
                        # (e.g., slider/level-bar nodes exported without bounds).
                        if (v_box is None) and (v_center is None):
                            continue
                        if not (vtxt and a_name_norm):
                            continue
                        final = txt
                        ok = final >= MIN_TXT_ONLY_SCORE

                    if ok and final > best_final:
                        best_final = final
                        best_v = v
                        best_vidx = int(vidx)

            matched = best_v is not None

            # --- a11y-master row + propagate app/window attribution ---
            matched_vscore = (
                _safe_float(best_v.get("vision_score")) if matched else None
            )
            if matched and matched_vscore is None:
                matched_vscore = _safe_float(best_v.get("score"))
            row_source = "fusion" if matched else "a11y"
            row_vscore = matched_vscore if matched else 1.0
            row_fusion_score = _safe_float(best_final) if matched else None
            row_score = row_fusion_score if matched else 1.0

            # Keep native a11y geometry when present; otherwise inherit from matched vision row.
            row_x1 = a.get("x1")
            row_y1 = a.get("y1")
            row_x2 = a.get("x2")
            row_y2 = a.get("y2")
            row_cx = a.get("cx")
            row_cy = a.get("cy")
            if matched:
                if _safe_float(row_x1) is None:
                    row_x1 = best_v.get("x1")
                if _safe_float(row_y1) is None:
                    row_y1 = best_v.get("y1")
                if _safe_float(row_x2) is None:
                    row_x2 = best_v.get("x2")
                if _safe_float(row_y2) is None:
                    row_y2 = best_v.get("y2")
                if _safe_float(row_cx) is None:
                    row_cx = best_v.get("cx")
                if _safe_float(row_cy) is None:
                    row_cy = best_v.get("cy")

            row = {
                "source": row_source,
                "type": "node",
                "role": a.get("role"),
                "content": a.get("name"),
                "value": a.get("value"),
                "x1": row_x1,
                "y1": row_y1,
                "x2": row_x2,
                "y2": row_y2,
                "cx": row_cx,
                "cy": row_cy,
                "score": row_score,
                "vision_score": row_vscore,
                "fusion_score": row_fusion_score,
                "fusion_matched": bool(matched),
                "a11y_id": a.get("a11y_id"),
                "a11y_role": a.get("role"),
                "a11y_showing": a.get("showing"),
                "a11y_visible": a.get("visible"),
                "a11y_enabled": a.get("enabled"),
                "a11y_focused": a.get("focused"),
                "a11y_selected": a.get("selected"),
                "a11y_checked": a.get("checked"),
                "a11y_expanded": a.get("expanded"),
                "a11y_states_raw": a.get("states_raw"),
                "a11y_actions_raw": a.get("actions_raw"),
                "a11y_node_id": a.get("node_id"),
                "a11y_parent_id": a.get("parent_id"),
                "a11y_depth": a.get("depth"),
                "a11y_child_index": a.get("child_index"),
                "a11y_is_interactive": a.get("is_interactive"),
                # NEW
                "app_name": a.get("app_name"),
                "window_name": a.get("window_name"),
                "window_active": a.get("window_active"),
            }
            out_rows.append(row)

            if matched and best_vidx is not None:
                matched_vision_idx.add(best_vidx)

        # Append vision-only rows (unmatched) + keep app/window fields as None
        if not vision.empty:
            # Prefer a11y-backed rows when vision-only rows are clearly overlapping the same target.
            a11y_targets: List[Tuple[Optional[Tuple[float, float, float, float]], Optional[Tuple[float, float]]]] = []
            for row in out_rows:
                if _normalize_text(row.get("source")) == "vision":
                    continue
                if not _is_actionable_a11y_row(row):
                    continue
                a11y_targets.append((_safe_box(row), _safe_center(row)))

            for vidx, v in vision.iterrows():
                if int(vidx) in matched_vision_idx:
                    continue

                v_box = v.get("_box")
                v_center = v.get("_center")
                duplicate_of_a11y = False
                for a_box, a_center in a11y_targets:
                    if _is_vision_duplicate_of_a11y_target(
                        vision_box=v_box,
                        vision_center=v_center,
                        a11y_box=a_box,
                        a11y_center=a_center,
                    ):
                        duplicate_of_a11y = True
                        break
                if duplicate_of_a11y:
                    continue

                out_rows.append(
                    {
                        "source": "vision",
                        "type": v.get("type"),
                        "role": v.get("role"),
                        "content": v.get("content"),
                        "value": v.get("value", None),
                        "x1": v.get("x1"),
                        "y1": v.get("y1"),
                        "x2": v.get("x2"),
                        "y2": v.get("y2"),
                        "cx": v.get("cx"),
                        "cy": v.get("cy"),
                        "score": _safe_float(v.get("vision_score"), default=_safe_float(v.get("score"))),
                        "vision_score": _safe_float(v.get("vision_score"), default=_safe_float(v.get("score"))),
                        "fusion_score": None,
                        "fusion_matched": False,
                        "a11y_id": None,
                        "a11y_role": None,
                        "a11y_showing": None,
                        "a11y_visible": None,
                        "a11y_enabled": None,
                        "a11y_focused": None,
                        "a11y_selected": None,
                        "a11y_checked": None,
                        "a11y_expanded": None,
                        "a11y_states_raw": None,
                        "a11y_actions_raw": None,
                        "a11y_node_id": None,
                        "a11y_parent_id": None,
                        "a11y_depth": None,
                        "a11y_child_index": None,
                        "a11y_is_interactive": None,
                        # NEW
                        "app_name": None,
                        "window_name": None,
                        "window_active": None,
                    }
                )

        fused = pd.DataFrame(out_rows)
        return fused
