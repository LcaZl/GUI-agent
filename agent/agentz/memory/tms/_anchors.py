from __future__ import annotations

from typing import Any, Dict, List, Tuple

from agentz.pydantic_models._tms_models import SpatialAnchor
from agentz.constants import ANCHOR_LABEL_MAX_CHARS, ANCHOR_ROLE_MAX_CHARS


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
    s = str(s or "").strip().lower()
    s = " ".join(s.split())
    return s


_INTERACTIVE_ROLES = {
    "push-button",
    "toggle-button",
    "switch",
    "check-box",
    "radio-button",
    "entry",
    "combo-box",
    "menu-item",
    "page-tab",
    "slider",
    "spin-button",
    "link",
}

_LOW_SIGNAL_ROLES = {
    "label",
    "text",
    "list-item",
    "grouping",
    "scroll-pane",
    "panel",
    "info-bar",
    "section",
    "table-cell",
}

_ROLE_PRIORITY = {
    "entry": 7,
    "combo-box": 6,
    "toggle-button": 6,
    "switch": 6,
    "check-box": 6,
    "radio-button": 5,
    "page-tab": 5,
    "menu-item": 5,
    "slider": 5,
    "push-button": 3,
}


def _as_bool(v: Any) -> bool:
    """
    Return bool.
        
        Parameters
        ----------
        v : Any
            Function argument.
        
        Returns
        -------
        bool
            True when the condition is satisfied, otherwise False.
        
    """
    return v is True


def _as_float(v: Any, default: float = 0.0) -> float:
    """
    Return float.
        
        Parameters
        ----------
        v : Any
            Function argument.
        default : Optional[float]
            Function argument.
        
        Returns
        -------
        float
            Floating-point result value.
        
    """
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _is_stateful(el: Any) -> bool:
    """
    Return whether is stateful.
    
    Parameters
    ----------
    el : Any
        Function argument.
    
    Returns
    -------
    bool
        True when the condition is satisfied, otherwise False.
    """
    return bool(
        _as_bool(getattr(el, "focused", None))
        or _as_bool(getattr(el, "selected", None))
        or _as_bool(getattr(el, "checked", None))
        or _as_bool(getattr(el, "expanded", None))
    )


def _source_priority(src: str) -> int:
    """
    Process source priority.
        
        Parameters
        ----------
        src : str
            Function argument.
        
        Returns
        -------
        int
            Integer result value.
        
    """
    if src == "a11y":
        return 3
    if src == "fusion":
        return 2
    if src == "vision":
        return 1
    return 0


def _role_cap(role: str) -> int:
    # Prevent button grids from saturating anchors.
    """
    Process role cap.
        
        Parameters
        ----------
        role : str
            Function argument.
        
        Returns
        -------
        int
            Integer result value.
        
    """
    if role == "push-button":
        return 10
    if role in {"label", "text"}:
        return 4
    return 12


def _ctx_role_cap(role: str) -> int:
    """
    Process ctx role cap.
        
        Parameters
        ----------
        role : str
            Function argument.
        
        Returns
        -------
        int
            Integer result value.
        
    """
    if role == "push-button":
        return 6
    return 8


def build_spatial_anchors(
    ui_elements: Dict[str, Any],
    *,
    grid: int,
    max_anchors: int,
) -> List[SpatialAnchor]:
    """
    Build high-signal spatial anchors from ui_elements with optional app/window context.
    The selection favors active/actionable/stable controls and reduces noisy duplicates.
    """
    anchors: List[SpatialAnchor] = []
    if not isinstance(ui_elements, dict):
        return anchors
    if max_anchors <= 0:
        return anchors

    candidates: List[Tuple[Tuple[float, ...], str, str, int, int, str]] = []

    for _, el in ui_elements.items():
        label = getattr(el, "label", None) or getattr(el, "value", None) or ""
        role = getattr(el, "a11y_role", None) or getattr(el, "kind", None) or getattr(el, "role", None) or ""

        app = getattr(el, "app_name", None) or ""
        win = getattr(el, "window_name", None) or ""
        context = _norm(win) or _norm(app)

        label = _norm(label)
        role = _norm(role)

        if context and label:
            label = f"{context}:{label}"

        cc = getattr(el, "center_coords", None)
        x = getattr(cc, "x", None)
        y = getattr(cc, "y", None)
        if not label or not role or x is None or y is None:
            continue
        if label == "<unlabeled>":
            continue

        qx = int(x) // grid
        qy = int(y) // grid

        actionable = _as_bool(getattr(el, "actionable", None))
        stateful = _is_stateful(el)
        win_active = _as_bool(getattr(el, "window_active", None))
        src = _norm(getattr(el, "source", None) or "")
        conf = max(
            _as_float(getattr(el, "fusion_score", None), default=0.0),
            _as_float(getattr(el, "score", None), default=0.0),
            _as_float(getattr(el, "vision_score", None), default=0.0),
        )
        role_prio = _ROLE_PRIORITY.get(role, 1 if role in _INTERACTIVE_ROLES else 0)

        # Remove very noisy anchors:
        # - vision-only passive text/icon
        # - passive structural/container rows
        if src == "vision" and not actionable and not stateful:
            continue
        if role in _LOW_SIGNAL_ROLES and not actionable and not stateful:
            continue

        rank = (
            1.0 if win_active else 0.0,
            1.0 if stateful else 0.0,
            1.0 if actionable else 0.0,
            float(role_prio),
            float(_source_priority(src)),
            conf,
        )
        candidates.append((rank, label, role, qx, qy, context))

    # Best first.
    candidates.sort(reverse=True, key=lambda x: x[0])

    seen_exact = set()
    role_count: Dict[str, int] = {}
    ctx_role_count: Dict[Tuple[str, str], int] = {}

    for _, label, role, qx, qy, context in candidates:
        exact_key = (label, role, qx, qy)
        if exact_key in seen_exact:
            continue

        # Near-duplicate suppression: same label/role almost same quantized position.
        is_near_dup = False
        for k in seen_exact:
            if k[0] == label and k[1] == role and abs(k[2] - qx) <= 1 and abs(k[3] - qy) <= 1:
                is_near_dup = True
                break
        if is_near_dup:
            continue

        if role_count.get(role, 0) >= _role_cap(role):
            continue
        ctx_key = (context or "", role)
        if context and ctx_role_count.get(ctx_key, 0) >= _ctx_role_cap(role):
            continue

        label_txt = str(label)
        role_txt = str(role)
        if ANCHOR_LABEL_MAX_CHARS is not None:
            label_txt = label_txt[:ANCHOR_LABEL_MAX_CHARS]
        if ANCHOR_ROLE_MAX_CHARS is not None:
            role_txt = role_txt[:ANCHOR_ROLE_MAX_CHARS]
        anchors.append(SpatialAnchor(label=label_txt, role=role_txt, qx=qx, qy=qy))
        seen_exact.add(exact_key)
        role_count[role] = role_count.get(role, 0) + 1
        if context:
            ctx_role_count[ctx_key] = ctx_role_count.get(ctx_key, 0) + 1

        if len(anchors) >= max_anchors:
            break

    return anchors
