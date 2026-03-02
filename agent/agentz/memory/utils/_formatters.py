from __future__ import annotations

import logging
import re
from typing import Any, List, Optional

from agentz.pydantic_models import ExecutedChunk, UIElement
from agentz.constants import (
    APP_NAME_MAX_CHARS,
    DEFAULT_MAX_ANCHORS,
    DEFAULT_MAX_VALUE_CHARS,
    MIN_LABEL_LEN,
    PROJECT_MAX_VALUE_CHARS,
    TERMINAL_PROMPT_MAX_CHARS,
    TERMINAL_PROMPT_MAX_LINES,
    TRIM_MAX_VALUE_CHARS,
    UI_FULL_MAX_ITEMS,
    UI_MAX_ITEMS,
    UI_TRIM_MAX_ITEMS,
    UI_TRIM_QUANTIZE_PX,
    WINDOW_NAME_MAX_CHARS,
)
from agentz.pydantic_models._tms_models import RetrievedNodeForPrompt, SpatialAnchor, TMSNode

_RE_MOSTLY_NONALNUM = re.compile(r"^[\W_]+$", re.UNICODE)  # Detect non-informative labels.
_RE_MOSTLY_NUM = re.compile(r"^[0-9\W_]+$", re.UNICODE)  # Detect mostly numeric labels.
_RE_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")  # Strip ANSI escapes.
_LOG = logging.getLogger("UIFormatter")



def _clean_one_line(s: str) -> str:
    """
    Process clean one line.
        
        Parameters
        ----------
        s : str
            Function argument.
        
        Returns
        -------
        str
            Resulting string value.
        
    """
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def terminal_text_for_prompt(
    terminal_content: Any,
    *,
    max_chars: int = TERMINAL_PROMPT_MAX_CHARS,
    max_lines: int = TERMINAL_PROMPT_MAX_LINES,
) -> str:
    """
    Normalize terminal transcript for prompt usage.
    Keeps the most recent lines, strips ANSI escapes, and truncates safely.
    """
    if terminal_content is None:
        return "(terminal not available)"

    s = str(terminal_content or "").strip()
    if not s or s.lower() in {"none", "null", "nan"}:
        return "(terminal not available)"

    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _RE_ANSI_ESCAPE.sub("", s)

    lines = [ln.rstrip() for ln in s.split("\n")]
    if max_lines and len(lines) > max_lines:
        lines = lines[-max_lines:]
    s = "\n".join(lines).strip()

    if not s:
        return "(terminal not available)"

    if max_chars and len(s) > max_chars:
        s = s[-max_chars:]
        s = "... (truncated, showing most recent output)\n" + s

    return s


def format_anchor_lines(
    anchors: List[SpatialAnchor],
    *,
    max_items: Optional[int] = None,
    empty_text: str = "",
) -> str:
    """
    Format anchor lines.
        
        Parameters
        ----------
        anchors : List[SpatialAnchor]
            Function argument.
        max_items : Optional[int]
            Function argument.
        empty_text : Optional[str]
            Function argument.
        
        Returns
        -------
        str
            Resulting string value.
        
    """
    if not anchors:
        return empty_text
    if max_items is not None and max_items > 0 and len(anchors) > max_items:
        anchors = anchors[:max_items]
    return "\n".join([f"- {a.as_key()}" for a in anchors])


def project_tms_node_for_prompt(
    node: TMSNode,
    *,
    max_value_chars: Optional[int] = DEFAULT_MAX_VALUE_CHARS,
    max_anchors: Optional[int] = DEFAULT_MAX_ANCHORS,
) -> RetrievedNodeForPrompt:
    """
    Process project tms node for prompt.
        
        Parameters
        ----------
        node : TMSNode
            Function argument.
        max_value_chars : Optional[int]
            Function argument.
        max_anchors : Optional[int]
            Function argument.
        
        Returns
        -------
        RetrievedNodeForPrompt
            Function result.
        
    """
    value = node.value or ""
    if max_value_chars is not None:
        value = value[:max_value_chars]
    anchors = node.anchor_keys()
    if max_anchors is not None:
        anchors = anchors[:max_anchors]
    return RetrievedNodeForPrompt(
        node_id=node.node_id,
        title=node.title,
        status=node.status,
        value=value,
        last_outcome=(node.last_outcome or None),
        last_guidance=(node.last_guidance or None),
        last_success=node.last_success,
        anchors=anchors,
    )


def project_tms_nodes_for_prompt(
    nodes: List[TMSNode],
    *,
    max_nodes: int,
    max_value_chars: Optional[int] = PROJECT_MAX_VALUE_CHARS,
    max_anchors: Optional[int] = DEFAULT_MAX_ANCHORS,
) -> List[RetrievedNodeForPrompt]:
    """
    Process project tms nodes for prompt.
        
        Parameters
        ----------
        nodes : List[TMSNode]
            Function argument.
        max_nodes : int
            Function argument.
        max_value_chars : Optional[int]
            Function argument.
        max_anchors : Optional[int]
            Function argument.
        
        Returns
        -------
        List[RetrievedNodeForPrompt]
            List with computed values.
        
    """
    out: List[RetrievedNodeForPrompt] = []
    for n in (nodes or [])[:max_nodes]:
        out.append(project_tms_node_for_prompt(n, max_value_chars=max_value_chars, max_anchors=max_anchors))
    return out


def format_trim_nodes(
    nodes: List[RetrievedNodeForPrompt],
    *,
    max_value_chars: Optional[int] = TRIM_MAX_VALUE_CHARS,
) -> str:
    """
    Format trim nodes.
        
        Parameters
        ----------
        nodes : List[RetrievedNodeForPrompt]
            Function argument.
        max_value_chars : Optional[int]
            Function argument.
        
        Returns
        -------
        str
            Resulting string value.
        
    """
    if not nodes:
        return ""
    lines: List[str] = []
    for n in nodes:
        val = (n.value or "")
        if max_value_chars is not None:
            val = val[:max_value_chars]
        lines.append(
            f"- node_id={n.node_id} | title={n.title} | status={n.status} | value={val} "
            f"| last_success={n.last_success} | anchors={n.anchors}"
        )
    return "\n".join(lines)


def format_planner_nodes(nodes: List[RetrievedNodeForPrompt]) -> str:
    """
    Format planner nodes.
        
        Parameters
        ----------
        nodes : List[RetrievedNodeForPrompt]
            Function argument.
        
        Returns
        -------
        str
            Resulting string value.
        
    """
    if not nodes:
        return ""
    lines: List[str] = []
    for n in nodes:
        lines.append(f"\n[Node {n.node_id}] {n.title} | status={n.status}")
        lines.append(f"Value: {n.value}")
        if n.last_success is not None:
            lines.append(f"Last success: {n.last_success}")
        if n.last_outcome:
            lines.append(f"Last outcome: {n.last_outcome}")
        if n.last_guidance:
            lines.append(f"Last guidance: {n.last_guidance}")
        if n.anchors:
            lines.append(f"Anchors: {n.anchors}")
    return "\n".join(lines)


def _looks_uninformative_text(s: str) -> bool:
    """
    Process looks uninformative text.
        
        Parameters
        ----------
        s : str
            Function argument.
        
        Returns
        -------
        bool
            True when the condition is satisfied, otherwise False.
        
    """
    s = (s or "").strip()
    if not s:
        return True
    if len(s) <= 2:
        return True
    if _RE_MOSTLY_NONALNUM.match(s) or _RE_MOSTLY_NUM.match(s):
        return True
    return False


def _short_app_window(el) -> str:
    """
    Process short app window.
        
        Parameters
        ----------
        el : Any
            Function argument.
        
        Returns
        -------
        str
            Resulting string value.
        
    """
    app = _clean_one_line(getattr(el, "app_name", "") or "")
    win = _clean_one_line(getattr(el, "window_name", "") or "")
    wact = getattr(el, "window_active", None)

    if not app and not win and wact is None:
        return ""

    if APP_NAME_MAX_CHARS is not None and len(app) > APP_NAME_MAX_CHARS:
        app = app[:APP_NAME_MAX_CHARS] + "..."
    if WINDOW_NAME_MAX_CHARS is not None and len(win) > WINDOW_NAME_MAX_CHARS:
        win = win[:WINDOW_NAME_MAX_CHARS] + "..."

    parts = []
    if app:
        parts.append(f"app={app}")
    if win:
        parts.append(f"win={win}")
    if wact is True:
        parts.append("active")
    elif wact is False:
        parts.append("inactive")

    return " | " + ",".join(parts)


def _element_label(el) -> str:
    """
    Process element label.
        
        Parameters
        ----------
        el : Any
            Function argument.
        
        Returns
        -------
        str
            Resulting string value.
        
    """
    lab = getattr(el, "label", None) or ""
    val = getattr(el, "value", None) or ""
    cont = getattr(el, "content", None) or ""
    s = lab or val or cont or ""
    return _clean_one_line(s)


def _element_role(el) -> str:
    """
    Process element role.
        
        Parameters
        ----------
        el : Any
            Function argument.
        
        Returns
        -------
        str
            Resulting string value.
        
    """
    return getattr(el, "a11y_role", None) or getattr(el, "kind", None) or "unknown"


def ui_elements_string(ui_elements: dict[str, UIElement]) -> str:
    """
    High-signal UI digest for Planner and Judge.
    """
    lines: list[str] = []
    if not ui_elements:
        return ""
    critical_roles = {
        "switch",
        "toggle-button",
        "slider",
        "check-box",
        "radio-button",
        "push-button",
    }
    interactive_roles = {
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
        "scroll-bar",
        "page-tab",
        "icon",
    }

    def score_of(el: UIElement) -> float:
        """
        Process score of.
        
        Parameters
        ----------
        el : UIElement
            Function argument.
        
        Returns
        -------
        float
            Floating-point result value.
        
        """
        fs = getattr(el, "fusion_score", None)
        if isinstance(fs, (int, float)):
            return float(fs)
        sc = getattr(el, "score", None)
        return float(sc) if isinstance(sc, (int, float)) else 0.0

    def is_stateful(el: UIElement) -> bool:
        """
        Return whether is stateful.
        
        Parameters
        ----------
        el : UIElement
            Function argument.
        
        Returns
        -------
        bool
            True when the condition is satisfied, otherwise False.
        """
        return bool(
            getattr(el, "focused", False)
            or getattr(el, "selected", False)
            or getattr(el, "checked", False)
            or getattr(el, "expanded", False)
        )

    def is_interactive(el: UIElement) -> bool:
        """
        Return whether is interactive.
        
        Parameters
        ----------
        el : UIElement
            Function argument.
        
        Returns
        -------
        bool
            True when the condition is satisfied, otherwise False.
        """
        if getattr(el, "actionable", None) is True:
            return True
        acts = getattr(el, "actions", None) or []
        if acts:
            return True
        role = _element_role(el).strip().lower()
        if role in interactive_roles:
            return True
        return False

    def keep(el: UIElement) -> bool:
        """
        Keep only lines that match the selected predicate.
        
        Parameters
        ----------
        el : UIElement
            Function argument.
        
        Returns
        -------
        bool
            True when the condition is satisfied, otherwise False.
        
        """
        label = _element_label(el)
        if is_interactive(el) or is_stateful(el):
            return True
        if _looks_uninformative_text(label):
            return False
        if MIN_LABEL_LEN is None:
            return True
        return len(label) >= MIN_LABEL_LEN

    items = [(el_id, el) for el_id, el in ui_elements.items() if el is not None and keep(el)]
    raw_total = len(ui_elements)
    kept_total = len(items)

    def key(item) -> tuple:
        """
        Process key.
        
        Parameters
        ----------
        item : Any
            Function argument.
        
        Returns
        -------
        tuple
            Tuple with computed values.
        
        """
        el_id, el = item
        window_active = 1 if getattr(el, "window_active", False) else 0
        actionable = 1 if getattr(el, "actionable", None) is True else 0
        focused = 1 if getattr(el, "focused", False) else 0
        stateful = 1 if is_stateful(el) else 0
        enabled = 1 if getattr(el, "enabled", None) is not False else 0
        fused = 1 if getattr(el, "fusion_matched", False) else 0
        sc = score_of(el)
        cx = int(getattr(el.center_coords, "x", 0) or 0)
        cy = int(getattr(el.center_coords, "y", 0) or 0)
        return (-window_active, -actionable, -focused, -stateful, -enabled, -fused, -sc, cy, cx, str(el_id))

    items.sort(key=key)

    max_items = UI_MAX_ITEMS
    critical_items: list[tuple[str, UIElement]] = []
    other_items: list[tuple[str, UIElement]] = []
    for item in items:
        role = _element_role(item[1]).strip().lower()
        if role in critical_roles:
            critical_items.append(item)
        else:
            other_items.append(item)

    critical_quota = min(max_items, max(8, max_items // 3))
    selected: list[tuple[str, UIElement]] = []
    selected.extend(critical_items[:critical_quota])
    remaining = max_items - len(selected)
    if remaining > 0:
        selected.extend(other_items[:remaining])
    if len(selected) < max_items:
        selected_ids = {el_id for el_id, _ in selected}
        for el_id, el in critical_items[critical_quota:]:
            if el_id in selected_ids:
                continue
            selected.append((el_id, el))
            selected_ids.add(el_id)
            if len(selected) >= max_items:
                break

    selected = selected[:max_items]
    selected_id_set = {el_id for el_id, _ in selected}
    dropped_items = [(el_id, el) for el_id, el in items if el_id not in selected_id_set]

    critical_total = len(critical_items)
    critical_selected = sum(1 for _, el in selected if _element_role(el).strip().lower() in critical_roles)
    active_selected = sum(1 for _, el in selected if getattr(el, "window_active", False) is True)
    actionable_selected = sum(1 for _, el in selected if getattr(el, "actionable", None) is True)
    _LOG.info(
        "UI prompt selection | raw=%d | kept=%d | selected=%d | max=%d | dropped=%d | critical=%d/%d | active=%d | actionable=%d",
        raw_total,
        kept_total,
        len(selected),
        max_items,
        max(0, kept_total - len(selected)),
        critical_selected,
        critical_total,
        active_selected,
        actionable_selected,
    )
    if selected:
        selected_preview = "; ".join(
            f"{el_id}:{_element_role(el)}:{_element_label(el)[:24]}"
            for el_id, el in selected[:10]
        )
        _LOG.info("UI prompt selected_top=%s", selected_preview)
    if dropped_items:
        dropped_preview = "; ".join(
            f"{el_id}:{_element_role(el)}:{_element_label(el)[:24]}"
            for el_id, el in dropped_items[:10]
        )
        _LOG.info("UI prompt dropped_top=%s", dropped_preview)

    for el_id, el in selected:
        flags = []

        if getattr(el, "visible", None) is True:
            flags.append("visible")
        if getattr(el, "enabled", None) is False:
            flags.append("disabled")
        if getattr(el, "focused", False):
            flags.append("focused")
        if getattr(el, "selected", False):
            flags.append("selected")
        if getattr(el, "checked", False):
            flags.append("checked")
        if getattr(el, "expanded", False):
            flags.append("expanded")
        if getattr(el, "actionable", None) is False:
            flags.append("not-actionable")
        if getattr(el, "fusion_matched", False):
            flags.append("fused")

        flags_str = ",".join(flags) if flags else "none"

        actions = ",".join(getattr(el, "actions", None) or [])

        role = _element_role(el)
        label = _element_label(el)

        cx = int(getattr(el.center_coords, "x", 0) or 0)
        cy = int(getattr(el.center_coords, "y", 0) or 0)

        # Planner prompt: keep app/window context only when informative.
        app = _clean_one_line(getattr(el, "app_name", "") or "")
        win = _clean_one_line(getattr(el, "window_name", "") or "")
        wact = getattr(el, "window_active", None)
        if app.strip().lower() == "gnome-shell" and not win:
            app = ""
        ctx_parts = []
        if app:
            if APP_NAME_MAX_CHARS is not None and len(app) > APP_NAME_MAX_CHARS:
                app = app[:APP_NAME_MAX_CHARS] + "..."
            ctx_parts.append(f"app={app}")
        if win:
            if WINDOW_NAME_MAX_CHARS is not None and len(win) > WINDOW_NAME_MAX_CHARS:
                win = win[:WINDOW_NAME_MAX_CHARS] + "..."
            ctx_parts.append(f"win={win}")
        if wact is True:
            ctx_parts.append("active")
        elif wact is False and win:
            ctx_parts.append("inactive")
        ctx_str = ",".join(ctx_parts) if ctx_parts else ""

        sc = score_of(el)
        src = str(getattr(el, "source", "") or "").strip().lower()

        parts = [
            f'role={role}',
            f'label="{label}"',
            f"flags={flags_str}",
            f"center=({cx},{cy})",
        ]
        if actions:
            parts.append(f"actions={actions}")
        if sc > 0:
            parts.append(f"score={sc:.2f}")
        if src and src != "fusion":
            parts.append(f"src={src}")
        if ctx_str:
            parts.append(f"ctx={ctx_str}")

        lines.append(f"- [{' | '.join(parts)}]")

    return "\n".join(lines)


def ui_elements_string_full(ui_elements: dict[str, UIElement], max_items: int = UI_FULL_MAX_ITEMS) -> str:
    """
    Full UI digest for Judge.
    Includes as many elements as possible with minimal filtering.
    """
    lines: list[str] = []
    if not ui_elements:
        return ""

    items = list(ui_elements.items())

    def key(item) -> tuple:
        """
        Process key.
        
        Parameters
        ----------
        item : Any
            Function argument.
        
        Returns
        -------
        tuple
            Tuple with computed values.
        
        """
        el_id, el = item
        cx = int(getattr(getattr(el, "center_coords", None), "x", 0) or 0)
        cy = int(getattr(getattr(el, "center_coords", None), "y", 0) or 0)
        return (cy, cx, str(el_id))

    items.sort(key=key)

    total = len(items)
    if max_items is not None and max_items > 0 and total > max_items:
        items = items[:max_items]

    for el_id, el in items:
        role = _element_role(el)
        label = _element_label(el)
        value = _clean_one_line(getattr(el, "value", "") or "")

        flags = []
        if getattr(el, "visible", None) is True:
            flags.append("visible")
        if getattr(el, "enabled", None) is False:
            flags.append("disabled")
        if getattr(el, "focused", False):
            flags.append("focused")
        if getattr(el, "selected", False):
            flags.append("selected")
        if getattr(el, "checked", False):
            flags.append("checked")
        if getattr(el, "expanded", False):
            flags.append("expanded")
        if getattr(el, "actionable", None) is False:
            flags.append("not-actionable")
        if getattr(el, "fusion_matched", False):
            flags.append("fused")

        flags_str = ",".join(flags) if flags else "none"

        actions = ",".join(getattr(el, "actions", None) or []) or "none"

        cx = int(getattr(getattr(el, "center_coords", None), "x", 0) or 0)
        cy = int(getattr(getattr(el, "center_coords", None), "y", 0) or 0)

        aw = _short_app_window(el)

        if value and value != label:
            value_part = f' | value="{value}"'
        else:
            value_part = ""

        lines.append(
            f'- [id={el_id} | role={role} | label="{label}"{value_part} | '
            f'flags={flags_str} | actions={actions} | center=({cx},{cy}){aw}]'
        )

    if max_items is not None and max_items > 0 and total > max_items:
        lines.append(f"... ({total - max_items} more elements omitted)")

    return "\n".join(lines)


def ui_elements_string_for_trim(ui_elements: dict[str, UIElement]) -> str:
    """
    Low-noise UI digest for TRIM.
    Focused on semantic/spatial context, not execution.
    """
    lines: list[str] = []
    if not ui_elements:
        return ""

    def keep(el: UIElement) -> bool:
        """
        Keep only rows that satisfy the selection rule.
        
        Parameters
        ----------
        el : UIElement
            Function argument.
        
        Returns
        -------
        bool
            True when the condition is satisfied, otherwise False.
        
        """
        label = _element_label(el)
        if getattr(el, "actionable", None) is True:
            return True
        if (
            getattr(el, "focused", False)
            or getattr(el, "selected", False)
            or getattr(el, "checked", False)
            or getattr(el, "expanded", False)
        ):
            return True
        if _looks_uninformative_text(label):
            return False
        if MIN_LABEL_LEN is None:
            return True
        return len(label) >= MIN_LABEL_LEN

    items = [(el_id, el) for el_id, el in ui_elements.items() if el is not None and keep(el)]

    items = items[:UI_TRIM_MAX_ITEMS]

    for _, el in items:
        role = _element_role(el)
        label = _element_label(el)

        cx = int(getattr(el.center_coords, "x", 0) or 0)
        cy = int(getattr(el.center_coords, "y", 0) or 0)

        qx = int(round(cx / UI_TRIM_QUANTIZE_PX) * UI_TRIM_QUANTIZE_PX)
        qy = int(round(cy / UI_TRIM_QUANTIZE_PX) * UI_TRIM_QUANTIZE_PX)

        aw = _short_app_window(el)

        lines.append(f'- [role={role} | label="{label}" | at=({qx},{qy}){aw}]')

    return "\n".join(lines)


def chunk_digest_for_tms(last_chunk: Optional[ExecutedChunk]) -> str:
    """
    Produce a compact, TRIM-oriented textual digest of the last executed chunk.
    """
    if last_chunk is None:
        return "- No executed chunk available."

    c = last_chunk
    lines: list[str] = []

    lines.append(f"MACRO GOAL: {c.macro_goal}")
    lines.append(f"OVERALL SUCCESS: {c.overall_success}")

    if c.failing_step_index is not None:
        lines.append(f"FAILING STEP INDEX: {c.failing_step_index}")

    if hasattr(c, "failure_type") and c.failure_type is not None:
        lines.append(f"FAILURE TYPE: {c.failure_type}")

    if hasattr(c, "post_chunk_state"):
        lines.append(f"POST-CHUNK STATE: {c.post_chunk_state}")

    if not c.overall_success and c.failing_step_index is not None:
        try:
            idx = int(c.failing_step_index)
        except Exception:
            idx = None

        if idx is not None:
            step = next((s for s in c.steps if int(getattr(s, "index", -1)) == idx), None)
            eval_by_index = {
                int(ev.index): ev
                for ev in (c.steps_eval or [])
                if ev is not None and getattr(ev, "index", None) is not None
            }
            step_eval = eval_by_index.get(idx)

            if step is not None and step_eval is not None:
                lines.append(
                    f"FAILED STEP SUMMARY: step={idx}, "
                    f"description='{step.description}', "
                    f"reason='{step_eval.failure_reason}'"
                )

    return "\n".join(lines)
