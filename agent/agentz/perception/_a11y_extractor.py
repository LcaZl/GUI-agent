import logging
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from agentz.pydantic_models import AccessibilityExtractorSettings
from ._ui_schema import standardize_ui_df
from agentz.constants import A11Y_TEXT_ROLE_MIN_LEN, NAME_MIN_LEN, NODE_ID_PREFIX

RAW_A11Y_COLUMNS: List[str] = [  # Columns expected from raw a11y dump.
    "source",
    "a11y_id",
    "node_id",
    "parent_id",
    "depth",
    "child_index",
    "tag",
    "role",
    "name",
    "value",
    "name_norm",
    "role_norm",
    "showing",
    "visible",
    "enabled",
    "focused",
    "selected",
    "checked",
    "expanded",
    "states_raw",
    "states",
    "actions_raw",
    "actions",
    "is_interactive",
    "x1",
    "y1",
    "x2",
    "y2",
    "cx",
    "cy",
    # app/window attribution
    "app_name",
    "window_name",
    "window_active",
]

# Keep contract stable (Linux/AT-SPI-ish)
# Note: do not exclude "scroll-bar" globally; these controls can be task-critical (e.g., volume/brightness sliders).
_EXCLUDE_TAGS = {"panel", "window", "filler", "frame", "separator"}  # Tags to skip.

_RE_TWO_NUMS = re.compile(r"(-?\d+(?:\.\d+)?)\D+(-?\d+(?:\.\d+)?)")  # Extract two numbers (int/float bounds).
_RE_MOSTLY_NONALNUM = re.compile(r"^[\W_]+$", re.UNICODE)  # Filter non-informative text.
_RE_MOSTLY_NUM = re.compile(r"^[0-9\W_]+$", re.UNICODE)  # Filter mostly numeric text.
_PLACEHOLDER_LABELS = {"<unlabeled>", "unlabeled"}  # Synthetic placeholders to treat as empty labels.
_NO_COORD_ALLOWED_TAGS = {"slider", "scroll-bar", "spin-button", "level-bar"}  # Controls sometimes exported without geometry.


def _strip_xml_namespace(s: str) -> str:
    """
    Strip xml namespace.
        
        Parameters
        ----------
        s : str
            Function argument.
        
        Returns
        -------
        str
            Resulting string value.
        
    """
    return s.split("}", 1)[-1] if isinstance(s, str) else str(s)


def _normalize_text(v: Any) -> str:
    """
    Normalize text.
        
        Parameters
        ----------
        v : Any
            Function argument.
        
        Returns
        -------
        str
            Resulting string value.
        
    """
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v).strip()


def _normalize_for_match(v: Any) -> str:
    """
    Normalize for match.
        
        Parameters
        ----------
        v : Any
            Function argument.
        
        Returns
        -------
        str
            Resulting string value.
        
    """
    return _normalize_text(v).casefold()


def _normalize_label_text(v: Any) -> str:
    """
    Normalize label text.
        
        Parameters
        ----------
        v : Any
            Function argument.
        
        Returns
        -------
        str
            Resulting string value.
        
    """
    txt = _normalize_text(v)
    if txt.casefold() in _PLACEHOLDER_LABELS:
        return ""
    return txt


def _parse_coord_tuple(value: Any) -> Tuple[float, float]:
    """
    Parse coord tuple.
        
        Parameters
        ----------
        value : Any
            Input value.
        
        Returns
        -------
        Tuple[float, float]
            Tuple with computed values.
        
    """
    if value is None:
        return -1.0, -1.0
    s = str(value)
    m = _RE_TWO_NUMS.search(s)
    if not m:
        return -1.0, -1.0
    try:
        return float(m.group(1)), float(m.group(2))
    except Exception:
        return -1.0, -1.0


def _parse_size_tuple(value: Any) -> Tuple[float, float]:
    """
    Parse size tuple.
        
        Parameters
        ----------
        value : Any
            Input value.
        
        Returns
        -------
        Tuple[float, float]
            Tuple with computed values.
        
    """
    if value is None:
        return -1.0, -1.0
    s = str(value)
    m = _RE_TWO_NUMS.search(s)
    if not m:
        return -1.0, -1.0
    try:
        return float(m.group(1)), float(m.group(2))
    except Exception:
        return -1.0, -1.0


def _iter_attrib_localnames(node: ET.Element) -> List[Tuple[str, str]]:
    """
    Iterate over attrib localnames.
        
        Parameters
        ----------
        node : ET.Element
            Function argument.
        
        Returns
        -------
        List[Tuple[str, str]]
            List with computed values.
        
    """
    out: List[Tuple[str, str]] = []
    for k, v in (node.attrib or {}).items():
        local = _strip_xml_namespace(k).split(":", 1)[-1].strip().lower()
        out.append((local, str(v)))
    return out


def _has_attr_localname(node: ET.Element, key_local: str) -> bool:
    """
    Return whether has attr localname.
    
    Parameters
    ----------
    node : ET.Element
        Function argument.
    key_local : str
        Function argument.
    
    Returns
    -------
    bool
        True when the condition is satisfied, otherwise False.
    """
    key_local = key_local.strip().lower()
    for local, _ in _iter_attrib_localnames(node):
        if local == key_local:
            return True
    return False


def _get_bool_attr_by_localname(node: Optional[ET.Element], key_local: str, default: str = "false") -> str:
    """
    Return get bool attr by localname.
    
    Parameters
    ----------
    node : Optional[ET.Element]
        Function argument.
    key_local : str
        Function argument.
    default : Optional[str]
        Function argument.
    
    Returns
    -------
    str
        Resulting string value.
    """
    if node is None:
        return default
    key_local = key_local.strip().lower()
    for local, v in _iter_attrib_localnames(node):
        if local == key_local:
            return str(v).strip().lower()
    return default


def _get_attr_value_by_localname(node: Optional[ET.Element], key_local: str, default: Optional[str] = None) -> Optional[str]:
    """
    Return get attr value by localname.
    
    Parameters
    ----------
    node : Optional[ET.Element]
        Function argument.
    key_local : str
        Function argument.
    default : Optional[str]
        Function argument.
    
    Returns
    -------
    Optional[str]
        Function result.
    """
    if node is None:
        return default
    key_local = key_local.strip().lower()
    for local, v in _iter_attrib_localnames(node):
        if local == key_local:
            return v
    return default


def _extract_actions_from_attrs(node: ET.Element) -> List[str]:
    """
    Extract actions from attrs.
        
        Parameters
        ----------
        node : ET.Element
            Function argument.
        
        Returns
        -------
        List[str]
            List with computed values.
        
    """
    actions: List[str] = []
    for local, _ in _iter_attrib_localnames(node):
        if local.endswith("_desc") or local.endswith("_kb"):
            base = local.rsplit("_", 1)[0].strip()
            if base:
                actions.append(base)
    return sorted(set(actions))


def _extract_states_from_attrs(node: ET.Element) -> List[str]:
    """
    Extract states from attrs.
        
        Parameters
        ----------
        node : ET.Element
            Function argument.
        
        Returns
        -------
        List[str]
            List with computed values.
        
    """
    states: List[str] = []
    for local, v in _iter_attrib_localnames(node):
        if str(v).strip().lower() == "true":
            states.append(local)
    return states


def _extract_state_truth_map(node: ET.Element) -> Dict[str, bool]:
    """
    Extract state truth map.
        
        Parameters
        ----------
        node : ET.Element
            Function argument.
        
        Returns
        -------
        Dict[str, bool]
            Dictionary with computed fields.
        
    """
    out: Dict[str, bool] = {}
    for local, v in _iter_attrib_localnames(node):
        val = str(v).strip().lower()
        if val == "true":
            out[local] = True
        elif val == "false":
            out[local] = False
    return out


def _looks_uninformative_label(txt: str) -> bool:
    """
    Conservative noise filter for a11y labels:
    - empty
    - very short (<=2)
    - mostly numbers / punctuation
    """
    s = (txt or "").strip()
    if not s:
        return True
    if s.casefold() in _PLACEHOLDER_LABELS:
        return True
    if len(s) <= 2:
        return True
    if _RE_MOSTLY_NONALNUM.match(s) or _RE_MOSTLY_NUM.match(s):
        return True
    return False


class AccessibilityTreeExtractor:
    """
    Extract a structured DataFrame from a raw accessibility tree (XML).

    Noise-reduction changes:
    - When at least one active window exists for a non-shell app, keep only nodes belonging to active windows.
    - Drop non-informative label/text nodes unless they are interactive or stateful.
    """

    def __init__(self, settings: AccessibilityExtractorSettings) -> None:
        """
        Initialize `AccessibilityTreeExtractor` dependencies and runtime state.
        
        Parameters
        ----------
        settings : AccessibilityExtractorSettings
            Runtime settings for this component.
        
        Returns
        -------
        None
            No return value.
        """
        self.settings = settings
        self.logger = logging.getLogger("AccessibilityTreeExtractor")

    def extract(self, accessibility_tree_xml: Any, screen_size: Optional[Tuple[int, int]] = None) -> pd.DataFrame:
        """
        Extract UI elements from the accessibility tree.
        
        Parameters
        ----------
        accessibility_tree_xml : Any
            Function argument.
        screen_size : Optional[Tuple[int, int]]
            Function argument.
        
        Returns
        -------
        pd.DataFrame
            Function result.
        
        """
        if not isinstance(accessibility_tree_xml, str):
            return pd.DataFrame(columns=RAW_A11Y_COLUMNS)
        xml = accessibility_tree_xml.strip()
        if not xml.startswith("<"):
            return pd.DataFrame(columns=RAW_A11Y_COLUMNS)

        try:
            root = ET.fromstring(xml)
        except ET.ParseError as e:
            self.logger.warning("Failed to parse accessibility tree: %s", e)
            return pd.DataFrame(columns=RAW_A11Y_COLUMNS)

        include_invisible = bool(getattr(self.settings, "include_invisible", False))
        keep_only_active_window_nodes = bool(getattr(self.settings, "keep_only_active_window_nodes", True))
        assume_normalized_bounds = bool(getattr(self.settings, "assume_normalized_bounds", True))
        clamp_bounds_to_screen = bool(getattr(self.settings, "clamp_bounds_to_screen", True))
        max_bounds_area_ratio = float(getattr(self.settings, "max_bounds_area_ratio", 0.95))

        screen_w: Optional[int] = None
        screen_h: Optional[int] = None
        try:
            if screen_size is not None:
                sw, sh = int(screen_size[0]), int(screen_size[1])
                if sw > 0 and sh > 0:
                    screen_w, screen_h = sw, sh
        except Exception:
            screen_w, screen_h = None, None

        # Build parent pointers (ElementTree nodes do not store parent references)
        parent_map: Dict[int, ET.Element] = {}
        for p in root.iter():
            for c in list(p):
                parent_map[id(c)] = p

        def _ancestor_by_tag(node: ET.Element, tag_name: str) -> Optional[ET.Element]:
            """
            Process ancestor by tag.
                        
                        Parameters
                        ----------
                        node : ET.Element
                            Function argument.
                        tag_name : str
                            Function argument.
                        
                        Returns
                        -------
                        Optional[ET.Element]
                            Function result.
                        
            """
            t = tag_name.strip().lower()
            cur = node
            while id(cur) in parent_map:
                cur = parent_map[id(cur)]
                if _strip_xml_namespace(cur.tag).strip().lower() == t:
                    return cur
            return None

        def _get_app_name(node: ET.Element) -> Optional[str]:
            """
            Return get app name.
            
            Parameters
            ----------
            node : ET.Element
                Function argument.
            
            Returns
            -------
            Optional[str]
                Function result.
            """
            app_el = _ancestor_by_tag(node, "application")
            if app_el is None:
                return None
            name = _normalize_text(app_el.attrib.get("name"))
            return name or None

        def _get_frame_el(node: ET.Element) -> Optional[ET.Element]:
            """
            Return get frame el.
            
            Parameters
            ----------
            node : ET.Element
                Function argument.
            
            Returns
            -------
            Optional[ET.Element]
                Function result.
            """
            return _ancestor_by_tag(node, "frame")

        # 1) Active app pruning: controlled by active-window policy, not by visibility policy.
        if keep_only_active_window_nodes:
            to_keep = self._find_active_applications_to_keep(root)
            self._prune_applications(root, to_keep)

        # Precompute: for each app, does it have an active frame?
        # (We will use this to keep only active-window nodes for that app)
        app_has_active_frame: Dict[str, bool] = {}
        for n in root.iter():
            if _strip_xml_namespace(n.tag).strip().lower() != "frame":
                continue
            active = _get_bool_attr_by_localname(n, "active", "false") == "true"
            if not active:
                continue
            parent_app = _ancestor_by_tag(n, "application")
            if parent_app is None:
                continue
            appn = _normalize_text(parent_app.attrib.get("name"))
            if appn:
                app_has_active_frame[appn] = True

        # 2) Filter nodes by showing/visible and by screencoord presence
        preserved_nodes: List[ET.Element] = []
        for node in root.iter():
            tag = _strip_xml_namespace(node.tag).strip()
            if tag in _EXCLUDE_TAGS:
                continue

            showing_flag = _get_bool_attr_by_localname(node, "showing", "false") == "true"
            visible_attr_present = _has_attr_localname(node, "visible")
            visible_flag = (_get_bool_attr_by_localname(node, "visible", "false") == "true") if visible_attr_present else None

            # include_invisible=True: keep also invisible rows (if they have coordinates).
            # include_invisible=False: keep only visually present rows.
            if not include_invisible:
                if visible_attr_present:
                    if visible_flag is not True:
                        continue
                elif showing_flag is not True:
                    continue

            x, y = _parse_coord_tuple(_get_attr_value_by_localname(node, "screencoord", "(-1, -1)"))
            if x < 0 or y < 0:
                if tag.strip().lower() not in _NO_COORD_ALLOWED_TAGS:
                    continue

            preserved_nodes.append(node)

        if not preserved_nodes:
            return pd.DataFrame(columns=RAW_A11Y_COLUMNS)

        # 3) Stable node_id mapping based on preserved order
        node_id_by_elem: Dict[int, str] = {id(n): f"{NODE_ID_PREFIX}_{i}" for i, n in enumerate(preserved_nodes)}

        # 4) Compute parent_id/depth/child_index for preserved nodes
        parent_id_map: Dict[int, Optional[str]] = {}
        depth_map: Dict[int, int] = {}
        child_index_map: Dict[int, int] = {}

        stack: List[Tuple[ET.Element, Optional[ET.Element], int]] = [(root, None, 0)]
        while stack:
            n, parent, depth = stack.pop()
            children = list(n)
            for i in reversed(range(len(children))):
                stack.append((children[i], n, depth + 1))

            if id(n) not in node_id_by_elem:
                continue

            parent_id_map[id(n)] = node_id_by_elem.get(id(parent)) if parent is not None else None
            depth_map[id(n)] = depth
            if parent is None:
                child_index_map[id(n)] = 0
            else:
                try:
                    child_index_map[id(n)] = list(parent).index(n)
                except Exception:
                    child_index_map[id(n)] = 0

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
        }
        interactive_states = {"focusable", "editable", "selectable", "checked", "expanded"}
        selectable_roles = {
            "menu-item",
            "list-item",
            "tree-item",
            "page-tab",
            "table-cell",
            "option",
            "radio-button",
        }
        checkable_roles = {
            "check-box",
            "radio-button",
            "toggle-button",
            "switch",
            "menu-item",
        }
        expandable_roles = {
            "tree-item",
            "menu-item",
            "combo-box",
            "page-tab",
            "disclosure-triangle",
            "toggle-button",
        }

        def _tri_state_flag(
            state_map: Dict[str, bool],
            *,
            positive_keys: set[str],
            applicable: bool,
            negative_true_keys: Optional[set[str]] = None,
        ) -> Optional[bool]:
            """
            Process tri state flag.
                        
                        Parameters
                        ----------
                        state_map : Dict[str, bool]
                            Function argument.
                        positive_keys : set[str]
                            Function argument.
                        applicable : bool
                            Function argument.
                        negative_true_keys : Optional[set[str]]
                            Function argument.
                        
                        Returns
                        -------
                        Optional[bool]
                            Function result.
                        
            """
            negative_true_keys = negative_true_keys or set()
            for key in positive_keys:
                if state_map.get(key) is True:
                    return True
            for key in negative_true_keys:
                if state_map.get(key) is True:
                    return False
            relevant = positive_keys.union(negative_true_keys)
            if any(key in state_map for key in relevant):
                return False
            if applicable:
                return False
            return None

        def _keep_row(
            *,
            app_name: Optional[str],
            role_norm: str,
            name: str,
            actions: List[str],
            st: set,
            is_interactive: bool,
            focused: Optional[bool],
            selected: Optional[bool],
            checked: Optional[bool],
            expanded: Optional[bool],
            window_active: Optional[bool],
        ) -> bool:
            """
            Main noise filter.
            """
            # 1) Active-window gating for non-shell apps when an active window exists
            if keep_only_active_window_nodes and app_name and app_name != "gnome-shell":
                if app_has_active_frame.get(app_name) is True:
                    # keep only nodes under the active frame (except state-carrying controls)
                    if window_active is not True and not (focused or selected or checked or expanded):
                        return False

            # 2) Keep any interactive element
            if is_interactive or role_norm in interactive_roles or actions:
                return True

            # 3) Keep anything with meaningful state
            if focused or selected or checked or expanded or bool(st.intersection(interactive_states)):
                return True

            # 4) For non-interactive text/labels: keep only if informative
            rn = (role_norm or "").strip()
            nm = (name or "").strip()

            # drop empty / tiny / numeric-ish labels
            if _looks_uninformative_label(nm):
                return False

            # If it looks like a generic label/text, require a bit more substance
            if rn in {"label", "text"}:
                # keep if it's a “real” message (No Results, Application Found, …)
                if len(nm) >= A11Y_TEXT_ROLE_MIN_LEN:
                    return True
                return False

            # otherwise: conservative keep if label is long enough
            return len(nm) >= NAME_MIN_LEN

        # 5) Emit rows (with noise filter)
        rows: List[Dict[str, Any]] = []

        for node in preserved_nodes:
            node_id = node_id_by_elem[id(node)]
            parent_id = parent_id_map.get(id(node))
            depth = depth_map.get(id(node), 0)
            child_index = child_index_map.get(id(node), 0)

            tag = _strip_xml_namespace(node.tag).strip()
            role = tag  # tag->role mapping

            name = _normalize_label_text(_get_attr_value_by_localname(node, "name", node.attrib.get("name"))) or _normalize_label_text(node.text) or ""
            value = _normalize_label_text(_get_attr_value_by_localname(node, "value", node.attrib.get("value"))) or ""

            state_truth = _extract_state_truth_map(node)
            states = sorted(k for k, v in state_truth.items() if v is True)
            actions = _extract_actions_from_attrs(node)
            st = set(states)

            showing_flag = _get_bool_attr_by_localname(node, "showing", "false") == "true"
            visible_attr_present = _has_attr_localname(node, "visible")
            visible_flag = (_get_bool_attr_by_localname(node, "visible", "false") == "true") if visible_attr_present else None

            showing = bool(showing_flag)
            visible = visible_flag  # may be None

            role_norm = _normalize_for_match(role)
            name_norm = _normalize_for_match(name)

            is_interactive = bool(actions) or bool(st.intersection(interactive_states)) or (role_norm in interactive_roles)

            enabled_applicable = bool(is_interactive) or bool(
                {"enabled", "sensitive", "disabled", "insensitive"}.intersection(state_truth.keys())
            )
            focused_applicable = bool(is_interactive) or bool(
                {"focused", "active", "focusable"}.intersection(state_truth.keys())
            )
            selected_applicable = (role_norm in selectable_roles) or bool(
                {"selected", "unselected"}.intersection(state_truth.keys())
            )
            checked_applicable = (role_norm in checkable_roles) or bool(
                {"checked", "unchecked"}.intersection(state_truth.keys())
            )
            expanded_applicable = (role_norm in expandable_roles) or bool(
                {"expanded", "collapsed"}.intersection(state_truth.keys())
            )

            enabled = _tri_state_flag(
                state_truth,
                positive_keys={"enabled", "sensitive"},
                negative_true_keys={"disabled", "insensitive"},
                applicable=enabled_applicable,
            )
            focused = _tri_state_flag(
                state_truth,
                positive_keys={"focused", "active"},
                applicable=focused_applicable,
            )
            selected = _tri_state_flag(
                state_truth,
                positive_keys={"selected"},
                negative_true_keys={"unselected"},
                applicable=selected_applicable,
            )
            checked = _tri_state_flag(
                state_truth,
                positive_keys={"checked"},
                negative_true_keys={"unchecked"},
                applicable=checked_applicable,
            )
            expanded = _tri_state_flag(
                state_truth,
                positive_keys={"expanded"},
                negative_true_keys={"collapsed"},
                applicable=expanded_applicable,
            )

            # Bounds
            x, y = _parse_coord_tuple(_get_attr_value_by_localname(node, "screencoord", "(-1, -1)"))
            w, h = _parse_size_tuple(_get_attr_value_by_localname(node, "size", "(-1, -1)"))

            x1 = float(x) if x >= 0 else None
            y1 = float(y) if y >= 0 else None
            x2 = float(x + w) if (x >= 0 and w >= 0) else None
            y2 = float(y + h) if (y >= 0 and h >= 0) else None
            cx = (x1 + x2) / 2.0 if x1 is not None and x2 is not None else None
            cy = (y1 + y2) / 2.0 if y1 is not None and y2 is not None else None

            # Optional bounds normalization/clamping when screen size is known.
            if screen_w is not None and screen_h is not None and None not in (x1, y1, x2, y2):
                if assume_normalized_bounds:
                    if all(0.0 <= float(v) <= 1.0 for v in (x1, y1, x2, y2)):
                        x1 = float(x1) * float(screen_w)
                        x2 = float(x2) * float(screen_w)
                        y1 = float(y1) * float(screen_h)
                        y2 = float(y2) * float(screen_h)

                if clamp_bounds_to_screen:
                    max_x = float(screen_w - 1)
                    max_y = float(screen_h - 1)
                    x1 = min(max(float(x1), 0.0), max_x)
                    y1 = min(max(float(y1), 0.0), max_y)
                    x2 = min(max(float(x2), 0.0), max_x)
                    y2 = min(max(float(y2), 0.0), max_y)

                if x2 <= x1 or y2 <= y1:
                    continue

                screen_area = float(screen_w * screen_h)
                if screen_area > 0:
                    area = float(x2 - x1) * float(y2 - y1)
                    if area > max_bounds_area_ratio * screen_area:
                        continue

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

            # App/window attribution via ancestors
            app_el = _ancestor_by_tag(node, "application")
            frame_el = _get_frame_el(node)

            app_name = _normalize_text(app_el.attrib.get("name")) if app_el is not None else None
            app_name = app_name or None

            window_name = _normalize_text(frame_el.attrib.get("name")) if frame_el is not None else None
            window_name = window_name or None

            window_active = (_get_bool_attr_by_localname(frame_el, "active", "false") == "true") if frame_el is not None else None

            # ---- noise filter gate ----
            if not _keep_row(
                app_name=app_name,
                role_norm=role_norm,
                name=name,
                actions=actions,
                st=st,
                is_interactive=bool(is_interactive),
                focused=focused,
                selected=selected,
                checked=checked,
                expanded=expanded,
                window_active=window_active,
            ):
                continue

            row: Dict[str, Any] = {
                "source": "a11y",
                "a11y_id": _normalize_text(node.attrib.get("id")) or node_id,
                "node_id": node_id,
                "parent_id": parent_id,
                "depth": depth,
                "child_index": child_index,
                "tag": tag,
                "role": role,
                "name": name,
                "value": value,
                "name_norm": name_norm,
                "role_norm": role_norm,
                "showing": showing,
                "visible": visible,
                "enabled": enabled,
                "focused": focused,
                "selected": selected,
                "checked": checked,
                "expanded": expanded,
                "states_raw": " ".join(sorted(states)),
                "states": sorted(states),
                "actions_raw": " ".join(actions),
                "actions": actions,
                "is_interactive": bool(is_interactive),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cx": cx,
                "cy": cy,
                "app_name": app_name,
                "window_name": window_name,
                "window_active": window_active,
            }
            rows.append(row)

        df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=RAW_A11Y_COLUMNS)
        for c in RAW_A11Y_COLUMNS:
            if c not in df.columns:
                df[c] = None
        return df[RAW_A11Y_COLUMNS]

    def to_ui_df(self, a11y_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert to ui df.
        
        Parameters
        ----------
        a11y_df : pd.DataFrame
            Function argument.
        
        Returns
        -------
        pd.DataFrame
            Function result.
        """
        if a11y_df is None or a11y_df.empty:
            return standardize_ui_df(pd.DataFrame(), defaults={"source": "a11y", "type": "node"})

        ui = pd.DataFrame(
            {
                "source": "a11y",
                "type": "node",
                "role": a11y_df.get("role"),
                "content": a11y_df.get("name"),
                "value": a11y_df.get("value"),
                "x1": a11y_df.get("x1"),
                "y1": a11y_df.get("y1"),
                "x2": a11y_df.get("x2"),
                "y2": a11y_df.get("y2"),
                "cx": a11y_df.get("cx"),
                "cy": a11y_df.get("cy"),
                "score": 1.0,
                "vision_score": 1.0,
                "fusion_score": None,
                "fusion_matched": False,
                "a11y_id": a11y_df.get("a11y_id"),
                "a11y_role": a11y_df.get("role"),
                "a11y_showing": a11y_df.get("showing"),
                "a11y_visible": a11y_df.get("visible"),
                "a11y_enabled": a11y_df.get("enabled"),
                "a11y_focused": a11y_df.get("focused"),
                "a11y_selected": a11y_df.get("selected"),
                "a11y_checked": a11y_df.get("checked"),
                "a11y_expanded": a11y_df.get("expanded"),
                "a11y_states_raw": a11y_df.get("states_raw"),
                "a11y_actions_raw": a11y_df.get("actions_raw"),
                "a11y_node_id": a11y_df.get("node_id"),
                "a11y_parent_id": a11y_df.get("parent_id"),
                "a11y_depth": a11y_df.get("depth"),
                "a11y_child_index": a11y_df.get("child_index"),
                "a11y_is_interactive": a11y_df.get("is_interactive"),
                "app_name": a11y_df.get("app_name"),
                "window_name": a11y_df.get("window_name"),
                "window_active": a11y_df.get("window_active"),
            }
        )
        return standardize_ui_df(ui, defaults={"source": "a11y", "type": "node"})

    def _find_active_applications_to_keep(self, root: ET.Element) -> List[str]:
        """
        Find active applications to keep.
        
        Parameters
        ----------
        root : ET.Element
            Function argument.
        
        Returns
        -------
        List[str]
            List with computed values.
        
        """
        to_keep = ["gnome-shell"]
        apps_with_active_tag: List[str] = []

        for application in list(root):
            app_tag = _strip_xml_namespace(application.tag).strip().lower()
            if app_tag != "application":
                continue

            app_name = _normalize_text(application.attrib.get("name"))
            for frame in list(application):
                if _get_bool_attr_by_localname(frame, "active", "false") == "true":
                    apps_with_active_tag.append(app_name)

        if apps_with_active_tag:
            to_keep.append(apps_with_active_tag[-1])
        return to_keep

    def _prune_applications(self, root: ET.Element, to_keep: List[str]) -> None:
        """
        Prune applications.
        
        Parameters
        ----------
        root : ET.Element
            Function argument.
        to_keep : List[str]
            Function argument.
        
        Returns
        -------
        None
            No return value.
        
        """
        keep_set = set(to_keep)
        for application in list(root):
            app_tag = _strip_xml_namespace(application.tag).strip().lower()
            if app_tag != "application":
                continue
            app_name = _normalize_text(application.attrib.get("name"))
            if app_name not in keep_set:
                root.remove(application)
