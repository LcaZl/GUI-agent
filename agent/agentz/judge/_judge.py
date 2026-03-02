import json
import logging
from typing import Any, Dict, List, Optional, Literal, Tuple

from agentz.memory.core import HistoryManager
from pydantic import BaseModel, Field, ConfigDict

from agentz.pydantic_models import ChunkEvaluation, ActionChunk, Observation, ExperimentConfiguration, GPTClientRequest
from agentz.tools import Tools

from ._prompts import JUDGE_LAST_CHUNK
from agentz.constants import JUDGE_TEMPERATURE, UI_DELTA_MAX_LABELS

class Judge:
    
    def __init__(self, settings: ExperimentConfiguration, tools: Tools):
        """
        Initialize the judge component.
        
        Parameters
        ----------
        settings : ExperimentConfiguration
            Full runtime configuration for the current agent instance.
        tools : Tools
            Tool container exposing shared clients and utilities.
        
        Returns
        -------
        None
            No return value.
        """
        self.settings = settings
        self.tools = tools
        self.logger = logging.getLogger("Judge")

    @staticmethod
    def _ui_delta(
        before_ui: Dict[str, Any],
        after_ui: Dict[str, Any],
        max_labels: int = UI_DELTA_MAX_LABELS,
    ) -> Dict[str, Any]:
        """
        Compute a compact UI label delta between two observations.
        
        Parameters
        ----------
        before_ui : Dict[str, Any]
            UI elements before step execution.
        after_ui : Dict[str, Any]
            UI elements after step execution.
        max_labels : int
            Maximum number of labels returned for each side of the diff.
        
        Returns
        -------
        Dict[str, Any]
            Pair `(added_labels, removed_labels)` represented as two lists.
        """
        def labelset(ui: Dict[str, Any]) -> List[str]:
            """
            Extract non-empty labels from a UI mapping.
            
            Parameters
            ----------
            ui : Dict[str, Any]
                Mapping of element ids to element payloads.
            
            Returns
            -------
            List[str]
                Flat list of labels found in the mapping.
            """
            labels = []
            for _, el in (ui or {}).items():
                try:
                    d = el.model_dump() if hasattr(el, "model_dump") else dict(el)
                    lab = (d.get("label") or "").strip()
                    if lab:
                        labels.append(lab)
                except Exception:
                    continue
            return labels

        b = labelset(before_ui)
        a = labelset(after_ui)

        bset, aset = set(b), set(a)
        added = sorted(list(aset - bset))[:max_labels]
        removed = sorted(list(bset - aset))[:max_labels]

        return added, removed

    def evaluate_outcome(self, history_manager : HistoryManager) -> ChunkEvaluation:

        """
        Evaluate the active chunk using before/after observations.
        
        Parameters
        ----------
        history_manager : HistoryManager
            History manager containing run context and observations.
        
        Returns
        -------
        ChunkEvaluation
            Structured outcome with per-step judgments and recovery guidance.
        """
        before_obs = history_manager.active_chunk_first_observation
        after_obs = history_manager.last_observation

        chunk = history_manager.active_chunk
        steps = chunk.steps

        lines : list[str] = []

        lines.append(f"Goal: {chunk.macro_goal}\n")


        # Build a compact per-step view for the judge.
        for i, step in enumerate(steps):
            lines.append(f"- STEP {i} - {step.index}")
            lines.append(f"  - Details:")    
            lines.append(f"     description: {step.description}")    
            lines.append(f"     expected_outcome: {step.expected_outcome}")    
            lines.append(f"     action_type: {step.action_type}")    
            lines.append(f"     command: {step.command}")    
            lines.append(f"     pause: {step.pause}")  

        added_labels, removed_labels = self._ui_delta(before_obs.ui_elements, after_obs.ui_elements)

        ui_delta_lines : list[str] = []
        ui_delta_lines.append(f"Added labels: {added_labels}")
        ui_delta_lines.append(f"Removed labels: {removed_labels}")

        try:
            tb = str(getattr(before_obs, "terminal_content", "") or "").strip()
            ta = str(getattr(after_obs, "terminal_content", "") or "").strip()
            self.logger.info(
                "Judge evidence | ui_before=%d | ui_after=%d | terminal_before_chars=%d | terminal_after_chars=%d",
                len(before_obs.ui_elements or {}),
                len(after_obs.ui_elements or {}),
                len(tb),
                len(ta),
            )
        except Exception:
            pass

        terminal_before = history_manager.terminal_text_for_prompt(before_obs.terminal_content)
        terminal_after = history_manager.terminal_text_for_prompt(after_obs.terminal_content)

        prompt = JUDGE_LAST_CHUNK.format(
            chunk = "\n".join(lines),
            ui_delta = "\n".join(ui_delta_lines),
            ui_elements_before = history_manager.ui_elements_string_full(before_obs.ui_elements),
            ui_elements_after = history_manager.ui_elements_string_full(after_obs.ui_elements),
            terminal_before = terminal_before,
            terminal_after = terminal_after,
        )


        images: List[Any] = [before_obs.screenshot, after_obs.screenshot]

        request = GPTClientRequest(
            prompt=prompt,
            tool_schema=ChunkEvaluation,
            overrides={"temperature": JUDGE_TEMPERATURE},
        )

        self.logger.info(f"Evaluating chunk. steps={len(steps)} images={len(images)}")

        evaluation = self.tools.gpt_client.chat_with_tool_and_images(
            request,
            images=images,
            image_detail="low",
        )
        
        history_manager.update(
            {"entity": "Judge", "request": request},
            tags=["llm_prompt"]
        )

        return evaluation
