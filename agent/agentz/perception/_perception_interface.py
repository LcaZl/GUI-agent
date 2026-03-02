# _perception_interface.py
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
from typing import Any, Dict, Optional, List

import pandas as pd

from ._ui_fusion import UIFusion
from ._a11y_extractor import AccessibilityTreeExtractor
from ._ui_schema import standardize_ui_df, dedupe_ui_df_by_label
from ._omniparser import OmniParserLocal

from agentz.pydantic_models import PerceptionSettings, UIElement, BBCoords, CenterCoords, Observation
from agentz.constants import (
    DEFAULT_FUSED_PREFIX,
    DEFAULT_VISION_PREFIX,
    ROUND_COORDS_NDIGITS,
    STATE_TOKENS_MAX_N,
    UI_DEDUP_MAX_PER_LABEL,
    VISION_SCORE_MIN_ICON_TEXT,
)

_PLACEHOLDER_LABELS = {"<unlabeled>", "unlabeled"}
_UNLABELED_CONTROL_ROLES = {"slider", "scroll-bar", "spin-button"}


class PerceptionInterface:
    def __init__(
        self,
        settings: PerceptionSettings,
        parallel: bool = True,
        debug_visualizations: bool = True,
        enable_label_dedup: bool = True,
        label_dedup_prefer_source: str = "fusion",
    ) -> None:
        """
        Initialize class dependencies and runtime state.
        
        Parameters
        ----------
        settings : PerceptionSettings
            Component settings.
        parallel : Optional[bool]
            Function argument.
        debug_visualizations : Optional[bool]
            Function argument.
        enable_label_dedup : Optional[bool]
            Function argument.
        label_dedup_prefer_source : Optional[str]
            Function argument.
        
        Returns
        -------
        None
            No return value.
        """
        self.use_vision = bool(getattr(settings, "use_vision", True))
        self.vision = OmniParserLocal(settings=settings.omniparser_settings) if self.use_vision else None
        self.a11y = AccessibilityTreeExtractor(settings=settings.a11y_extractor_settings)
        self.fuser = UIFusion()
        self.parallel = bool(parallel)
        self._obs_counter = 0
        self.logger = logging.getLogger("Perception")

        self.debug_visualizations = bool(debug_visualizations)
        self.debug_dir = Path(settings.debug_directory)
        self.vision_prefix = str(DEFAULT_VISION_PREFIX)
        self.fused_prefix = str(DEFAULT_FUSED_PREFIX)
        self.enable_label_dedup = bool(enable_label_dedup)
        self.label_dedup_prefer_source = str(label_dedup_prefer_source)
        if not self.use_vision:
            self.logger.info("Perception configured in a11y-only mode (vision disabled).")

    def process(self, perception: Dict[str, Any]) -> Observation:
        """
        Process one environment observation through the perception pipeline.
        
        Parameters
        ----------
        perception : Dict[str, Any]
            Perception component or payload.
        
        Returns
        -------
        Observation
            Function result.
        
        """
        if perception.get("obs", None) is not None:
            observation = perception.get("obs")
            reward = perception.get("reward")
            done = perception.get("done")
            info = perception.get("info")
        else:
            observation = perception
            reward = None
            done = None
            info = None

        screenshot = observation.get("screenshot")
        accessibility_tree = observation.get("accessibility_tree")
        terminal = observation.get("terminal")

        screen_size = None
        try:
            if screenshot is not None and hasattr(screenshot, "shape"):
                h, w = screenshot.shape[:2]
                screen_size = (int(w), int(h))
        except Exception:
            screen_size = None

        if self.parallel and self.use_vision:
            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_vision = pool.submit(self.vision.parse, screenshot)
                fut_a11y = pool.submit(self.a11y.extract, accessibility_tree, screen_size=screen_size)
                vision_df = fut_vision.result()
                a11y_raw = fut_a11y.result()
        else:
            if self.use_vision:
                vision_df = self.vision.parse(screenshot)
            else:
                vision_df = pd.DataFrame()
            a11y_raw = self.a11y.extract(accessibility_tree, screen_size=screen_size)

        fused_df = self.fuser.fuse(vision_df, a11y_raw)

        screen_info = standardize_ui_df(vision_df, defaults={"source": "vision"})
        ally_info = self.a11y.to_ui_df(a11y_raw)
        fused_data = standardize_ui_df(fused_df)
        fused_before_dedup = len(fused_data)
        critical_roles = {"slider", "scroll-bar", "spin-button", "level-bar"}
        if not fused_data.empty:
            role_series_before = (
                fused_data.get("a11y_role", pd.Series(index=fused_data.index, dtype="object"))
                .fillna("")
                .astype(str)
                .str.strip()
                .str.casefold()
            )
            critical_before = int(role_series_before.isin(critical_roles).sum())
        else:
            critical_before = 0

        if self.enable_label_dedup:
            fused_data = dedupe_ui_df_by_label(
                fused_data,
                label_col="content",
                prefer_source=self.label_dedup_prefer_source,
                spatial_iou_threshold=0.75,
                max_per_label=UI_DEDUP_MAX_PER_LABEL,
            )
        fused_after_dedup = len(fused_data)
        if not fused_data.empty:
            role_series_after = (
                fused_data.get("a11y_role", pd.Series(index=fused_data.index, dtype="object"))
                .fillna("")
                .astype(str)
                .str.strip()
                .str.casefold()
            )
            critical_after = int(role_series_after.isin(critical_roles).sum())
        else:
            critical_after = 0
        self.logger.info(
            "Perception fusion filter | fused_before=%d | fused_after=%d | dedup_removed=%d | critical_controls_before=%d | critical_controls_after=%d | dedup_enabled=%s",
            fused_before_dedup,
            fused_after_dedup,
            max(0, fused_before_dedup - fused_after_dedup),
            critical_before,
            critical_after,
            str(self.enable_label_dedup).lower(),
        )

        img_path: Optional[str] = None
        if self.debug_visualizations and self.use_vision and self.vision is not None:
            try:
                self.debug_dir.mkdir(exist_ok=True, parents=True)
                img_path = self.vision.save_visualization(
                    screenshot, fused_data, str(self.debug_dir), prefix=self.fused_prefix
                )
            except Exception:
                img_path = None

        ui_elements: Dict[str, UIElement] = {}

        def _f(v: Any) -> Optional[float]:
            """
            Process f.
                        
                        Parameters
                        ----------
                        v : Any
                            Function argument.
                        
                        Returns
                        -------
                        Optional[float]
                            Function result.
                        
            """
            try:
                if pd.isna(v):
                    return None
            except Exception:
                pass
            try:
                return float(v)
            except Exception:
                return None

        def _b(v: Any) -> Optional[bool]:
            """
            Process b.
                        
                        Parameters
                        ----------
                        v : Any
                            Function argument.
                        
                        Returns
                        -------
                        Optional[bool]
                            Function result.
                        
            """
            try:
                if pd.isna(v):
                    return None
            except Exception:
                pass
            if v is None:
                return None
            return bool(v)

        def _s(v: Any) -> Optional[str]:
            """
            Process s.
                        
                        Parameters
                        ----------
                        v : Any
                            Function argument.
                        
                        Returns
                        -------
                        Optional[str]
                            Function result.
                        
            """
            try:
                if pd.isna(v):
                    return None
            except Exception:
                pass
            if v is None:
                return None
            txt = str(v).strip()
            if txt.casefold() in _PLACEHOLDER_LABELS:
                return None
            return txt if txt else None

        # helper: parse raw strings to short lists (low token)
        def _split_tokens(s: Any, max_n: int) -> Optional[List[str]]:
            """
            Split tokens.
                        
                        Parameters
                        ----------
                        s : Any
                            Function argument.
                        max_n : int
                            Function argument.
                        
                        Returns
                        -------
                        Optional[List[str]]
                            List with computed values.
                        
            """
            if s is None:
                return None
            try:
                if pd.isna(s):
                    return None
            except Exception:
                pass
            txt = str(s).strip()
            if not txt:
                return None
            toks = [t for t in txt.replace(",", " ").split() if t]
            return toks[:max_n] if toks else None

        for idx, row in fused_data.iterrows():
            a11y_is_interactive = _b(row.get("a11y_is_interactive"))

            label_txt = _s(row.get("content"))
            if label_txt is None:
                label_txt = _s(row.get("value"))
            if label_txt is None:
                role_txt = _s(row.get("a11y_role")) or _s(row.get("role"))
                if role_txt is not None and role_txt.casefold() in _UNLABELED_CONTROL_ROLES:
                    role_pretty = role_txt.replace("-", " ").strip().title()
                    value_txt = _s(row.get("value"))
                    label_txt = f"{role_pretty} ({value_txt})" if value_txt else role_pretty
            if label_txt is None:
                continue
            focused = _b(row.get("a11y_focused"))
            selected = _b(row.get("a11y_selected"))
            checked = _b(row.get("a11y_checked"))
            expanded = _b(row.get("a11y_expanded"))
            vscore = _f(row.get("vision_score"))
            
            x1, y1, x2, y2 = _f(row.get("x1")), _f(row.get("y1")), _f(row.get("x2")), _f(row.get("y2"))
            cx, cy = _f(row.get("cx")), _f(row.get("cy"))

            if cx is None or cy is None:
                if None not in (x1, y1, x2, y2):
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
            if cx is None or cy is None:
                # Avoid emitting elements with synthetic (0,0) centers when geometry is incomplete.
                continue

            bb_coords = BBCoords(
                x_1=round(x1, ROUND_COORDS_NDIGITS) if x1 is not None else 0.0,
                y_1=round(y1, ROUND_COORDS_NDIGITS) if y1 is not None else 0.0,
                x_2=round(x2, ROUND_COORDS_NDIGITS) if x2 is not None else 0.0,
                y_2=round(y2, ROUND_COORDS_NDIGITS) if y2 is not None else 0.0,
            )
            center_coords = CenterCoords(
                x=round(cx, ROUND_COORDS_NDIGITS) if cx is not None else 0.0,
                y=round(cy, ROUND_COORDS_NDIGITS) if cy is not None else 0.0,
            )

            a11y_visible = _b(row.get("a11y_visible"))
            a11y_enabled = _b(row.get("a11y_enabled"))

            # actionable: prefer a11y
            actionable = a11y_is_interactive
            if actionable is None:
                if (
                    str(row.get("type")) in {"icon", "text"}
                    and vscore is not None
                ):
                    actionable = bool(vscore >= VISION_SCORE_MIN_ICON_TEXT)
                else:
                    actionable = None

            # NEW: app/window attribution from fused_data
            app_name = _s(row.get("app_name"))
            window_name = _s(row.get("window_name"))
            window_active = _b(row.get("window_active"))

            ui_elements[f"ui_{idx}"] = UIElement(
                id=f"ui_{idx}",
                kind=row.get("type"),
                label=label_txt or "",
                value=_s(row.get("value")),
                source=row.get("source"),
                a11y_role=None if pd.isna(row.get("a11y_role")) else row.get("a11y_role"),
                visible=a11y_visible,
                enabled=a11y_enabled,
                actionable=actionable,
                focused=focused,
                selected=selected,
                checked=checked,
                expanded=expanded,
                actions=_split_tokens(row.get("a11y_actions_raw"), max_n=8),
                states=_split_tokens(row.get("a11y_states_raw"), max_n=STATE_TOKENS_MAX_N),
                a11y_id=_s(row.get("a11y_id")),
                a11y_node_id=_s(row.get("a11y_node_id")),
                a11y_parent_id=_s(row.get("a11y_parent_id")),
                a11y_depth=None if pd.isna(row.get("a11y_depth")) else int(row.get("a11y_depth")),
                a11y_child_index=None if pd.isna(row.get("a11y_child_index")) else int(row.get("a11y_child_index")),
                score=_f(row.get("score")),
                vision_score=_f(row.get("vision_score")),
                fusion_score=_f(row.get("fusion_score")),
                fusion_matched=_b(row.get("fusion_matched")),
                # NEW
                app_name=app_name,
                window_name=window_name,
                window_active=window_active,
                bb_coords=bb_coords.model_dump(),
                center_coords=center_coords.model_dump(),
            )

        term_str = ""
        if terminal is not None:
            term_str = str(terminal)
            if term_str.strip().lower() in {"none", "null"}:
                term_str = ""

        out_obs = Observation(
            observation_id=self._obs_counter,
            screen_info=screen_info,
            ally_info=ally_info,
            a11y_raw=a11y_raw,
            fused_data=fused_data,
            ui_elements=ui_elements,
            terminal_content=term_str,
            screenshot=screenshot,
            reward=reward,
            info=info,
            done=done,
        )

        try:
            vision_count = len(vision_df) if vision_df is not None else 0
            a11y_count = len(a11y_raw) if a11y_raw is not None else 0
            fused_count = len(fused_data) if fused_data is not None else 0
        except Exception:
            vision_count = 0
            a11y_count = 0
            fused_count = 0

        term_str = "" if terminal is None else str(terminal)
        term_len = len(term_str.strip())
        terminal_ui = False
        app_counts: Dict[str, int] = {}
        win_counts: Dict[str, int] = {}

        def _is_terminal_window_visible(el: UIElement) -> bool:
            """
            Return whether is terminal window visible.
            
            Parameters
            ----------
            el : UIElement
                Function argument.
            
            Returns
            -------
            bool
                True when the condition is satisfied, otherwise False.
            """
            app = str(getattr(el, "app_name", "") or "").lower()
            win = str(getattr(el, "window_name", "") or "").lower()
            role = str(getattr(el, "a11y_role", "") or getattr(el, "kind", "") or "").lower()
            is_active = bool(getattr(el, "window_active", False))
            is_focused = bool(getattr(el, "focused", False))

            # Avoid false positives from app-grid icons named "Terminal":
            # require terminal context from app/window plus active/focused evidence.
            terminal_hints = ("gnome-terminal", "terminal", "ptyxis", "xterm", "console")
            in_terminal_context = any(h in app for h in terminal_hints) or any(h in win for h in terminal_hints)

            if in_terminal_context and (is_active or is_focused):
                return True

            # Fallback: terminal-specific role with active/focused evidence.
            if "terminal" in role and (is_active or is_focused):
                return True

            return False

        try:
            for _, el in (out_obs.ui_elements or {}).items():
                app = str(getattr(el, "app_name", "") or "").lower()
                win = str(getattr(el, "window_name", "") or "").lower()
                if app:
                    app_counts[app] = app_counts.get(app, 0) + 1
                if win:
                    win_counts[win] = win_counts.get(win, 0) + 1
                if _is_terminal_window_visible(el):
                    terminal_ui = True
                    break
        except Exception:
            terminal_ui = False
        self.logger.info(
            "Perception obs_id=%s ui=%d vision=%d a11y=%d fused=%d terminal_chars=%d terminal_ui=%s",
            self._obs_counter,
            len(out_obs.ui_elements),
            vision_count,
            a11y_count,
            fused_count,
            term_len,
            terminal_ui,
        )
        if app_counts:
            top_apps = sorted(app_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            self.logger.info("Perception apps_top3=%s", top_apps)
        if win_counts:
            top_wins = sorted(win_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            self.logger.info("Perception windows_top3=%s", top_wins)

        self._obs_counter += 1
        return out_obs
