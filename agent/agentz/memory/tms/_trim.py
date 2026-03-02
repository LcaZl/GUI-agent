# _trim_v2.py
from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Optional

from agentz.memory.core import HistoryManager
from agentz.memory.tms._prompts import TRIM_PROMPT
from pydantic import BaseModel, Field, ConfigDict
from agentz.pydantic_models import GPTClientRequest

from agentz.pydantic_models._tms_models import (
    SpatialAnchor,
    TMSNode,
    RetrievedNodeForPrompt,
    TRIMToolOutput,
)
from agentz.constants import (
    TRIM_DEFAULT_GRID,
    TRIM_MAX_ANCHORS_IN_PROMPT,
    TRIM_MAX_NODES_IN_PROMPT,
    TRIM_NODES_MAX_VALUE_CHARS,
    TRIM_PROJECT_MAX_ANCHORS,
    TRIM_PROJECT_MAX_VALUE_CHARS,
)
from ._anchors import build_spatial_anchors
from ..utils._formatters import format_anchor_lines, format_trim_nodes, project_tms_nodes_for_prompt

# -----------------------------
# TRIM (LLM-driven)
# -----------------------------

class TRIMLLM:
    """
    Paper TRIM module:
      1) Input decomposition into subtasks
      2) Intent classification (NEW/UPDATE/CHECK/ROLLBACK)
      3) Mapping to existing node(s) when applicable
    """

    def __init__(
        self,
        gpt_client: Any,
        grid: int = TRIM_DEFAULT_GRID,
        max_nodes_in_prompt: int = TRIM_MAX_NODES_IN_PROMPT,
        max_anchors_in_prompt: int = TRIM_MAX_ANCHORS_IN_PROMPT,
        overrides: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize class dependencies and runtime state.
        
        Parameters
        ----------
        gpt_client : Any
            GPT client manager.
        grid : Optional[int]
            Function argument.
        max_nodes_in_prompt : Optional[int]
            Function argument.
        max_anchors_in_prompt : Optional[int]
            Function argument.
        overrides : Optional[Dict[str, Any]]
            Function argument.
        
        Returns
        -------
        Any
            Function result.
        """
        self.gpt_client = gpt_client
        self.grid = grid
        self.max_nodes_in_prompt = max_nodes_in_prompt
        self.max_anchors_in_prompt = max_anchors_in_prompt
        self.overrides = overrides or {}
        self.logger = logging.getLogger("TRIM")

    # -------------
    # Anchors helper
    # -------------

    def observation_to_anchors(self, observation: Any) -> List[SpatialAnchor]:
        """
        Extract spatial anchors from your Observation(ui_elements).
        We keep it defensive: if fields missing, returns empty list.
        """
        ui = getattr(observation, "ui_elements", None)
        return build_spatial_anchors(ui, grid=self.grid, max_anchors=self.max_anchors_in_prompt)

    # --------------------
    # Node projection helper
    # --------------------

    def project_nodes_for_prompt(self, nodes: List[TMSNode]) -> List[RetrievedNodeForPrompt]:
        """
        Provide a compact view of nodes to the LLM:
        id, title, status, value, last_outcome, last_success, few anchors.
        """
        return project_tms_nodes_for_prompt(
            nodes,
            max_nodes=self.max_nodes_in_prompt,
            max_value_chars=TRIM_PROJECT_MAX_VALUE_CHARS,
            max_anchors=TRIM_PROJECT_MAX_ANCHORS,
        )

    # --------------------
    # Main TRIM call
    # --------------------

    def run(
        self,
        task_instruction: str,
        tms_nodes: List[TMSNode],
        history_manager : HistoryManager,
        current_observation: Optional[Any] = None,
        cid: Optional[str] = None,
        chunk_digest : str = None,
    ) -> TRIMToolOutput:
        """
        Execute run.
        
        Parameters
        ----------
        task_instruction : str
            Function argument.
        tms_nodes : List[TMSNode]
            Function argument.
        history_manager : HistoryManager
            History manager with observations and executed chunks.
        current_observation : Optional[Any]
            Function argument.
        cid : Optional[str]
            Function argument.
        chunk_digest : Optional[str]
            Function argument.
        
        Returns
        -------
        TRIMToolOutput
            Function result.
        
        """
        anchors = self.observation_to_anchors(current_observation) if current_observation is not None else []
        candidates = self.project_nodes_for_prompt(tms_nodes)

        anchors_txt = format_anchor_lines(anchors, empty_text="")
        nodes_txt = format_trim_nodes(candidates, max_value_chars=TRIM_NODES_MAX_VALUE_CHARS)
        chunk_digest = chunk_digest or ""

        self.logger.info(
            "TRIM input | anchors=%d | candidate_nodes=%d | chunk_digest_len=%d",
            len(anchors),
            len(candidates),
            len(chunk_digest or ""),
        )
        last_chunk = history_manager.get_last_chunk()
        failure_type = getattr(last_chunk, "failure_type", None)
        failing_step = "(none)"
        if last_chunk is not None and getattr(last_chunk, "failing_step_index", None) is not None:
            try:
                idx = int(last_chunk.failing_step_index)
                if 0 <= idx < len(last_chunk.steps):
                    step = last_chunk.steps[idx]
                    failing_step = f"{step.description} | expected={step.expected_outcome}"
            except Exception:
                failing_step = "(unknown)"

        prompt = TRIM_PROMPT.format(
            task_instruction=task_instruction,
            anchors=anchors_txt,
            nodes=nodes_txt,
            chunks_digest=chunk_digest,
            failure_type=str(failure_type),
            failing_step=failing_step,
        )

        request = GPTClientRequest(prompt=prompt, tool_schema=TRIMToolOutput, overrides=self.overrides or None, cid=cid)
        response = self.gpt_client.chat_with_tool(request)

        history_manager.update(
            {"entity": "TRIM", "request": request},
            tags=["llm_prompt"]
        )

        # Expected: response already parsed into TRIMToolOutput by your infra
        # If your infra returns dict, wrap it: TRIMToolOutput.model_validate(response)
        if isinstance(response, TRIMToolOutput):
            op_counts = Counter([d.op for d in response.decisions])
            self.logger.info(
                "TRIM output | decisions=%d | ops=%s",
                len(response.decisions),
                dict(op_counts),
            )
            return response

    
        out = TRIMToolOutput.model_validate(response)
        op_counts = Counter([d.op for d in out.decisions])
        self.logger.info(
            "TRIM output | decisions=%d | ops=%s",
            len(out.decisions),
            dict(op_counts),
        )
        return out
