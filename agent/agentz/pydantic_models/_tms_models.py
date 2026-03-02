# _tms_models_v2.py
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ConfigDict


# -----------------------------
# Paper concepts: intents & ops
# -----------------------------

class TRIMIntent(str, Enum):
    """
    Paper: Intent classification.
    - NEW: new subtask (create new node)
    - UPDATE: modify / replace an existing node (same subtask goal but changed plan/value)
    - CHECK: verification-only / query against memory (no structural change)
    - ROLLBACK: revert a node to a previous revision (requires history)
    """
    NEW = "NEW"
    UPDATE = "UPDATE"
    CHECK = "CHECK"
    ROLLBACK = "ROLLBACK"
    INACTIVATE = "INACTIVATE"


class TMSOp(str, Enum):
    """
    Paper: graph update operations.
    """
    ADD = "ADD"
    REPLACE = "REPLACE"
    INACTIVATE = "INACTIVATE"
    ROLLBACK = "ROLLBACK"
    NOOP = "NOOP"


class NodeStatus(str, Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"


# -----------------------------
# Node revision history (paper)
# -----------------------------

class NodeRevision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rev_id: int = Field(..., ge=0, description="Monotonic revision id for the node.")
    created_step: int = Field(..., ge=0, description="Global step index when revision created.")
    value: str = Field(..., description="Node 'value' at this revision (summary/state/solution).")
    summary: Optional[str] = Field(default=None, description="Optional short rationale for this revision.")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata.")


# -----------------------------
# Spatial anchors (paper: spatial memory)
# -----------------------------

class SpatialAnchor(BaseModel):
    """
    Compact spatial cue extracted from an Observation: (label, role, quantized position).
    Paper uses spatial memory as robust cue for UI / page recognition.
    """
    model_config = ConfigDict(extra="forbid")

    label: str
    role: str
    qx: int
    qy: int

    def as_key(self) -> str:
        """
        Return key.
        
        Returns
        -------
        str
            Resulting string value.
        
        """
        return f"{self.label}|{self.role}|{self.qx}|{self.qy}"


# -----------------------------
# Graph structures
# -----------------------------

class TMSEdge(BaseModel):
    """
    Dependency edge (DAG): parent -> child
    Paper: TMS is a graph of subtasks with dependencies.
    """
    model_config = ConfigDict(extra="forbid")

    parent_id: str
    child_id: str
    label: str = Field(default="depends_on")


class TMSNode(BaseModel):
    """
    Node in TMS.
    Paper: each node stores task/subtask representation and revisions.
    """
    model_config = ConfigDict(extra="forbid")

    node_id: str
    title: str = Field(..., description="Short canonical name of subtask (macro-goal).")
    status: NodeStatus = Field(default=NodeStatus.ACTIVE)
    created_step: int = Field(..., ge=0)

    # "value" is the current best summary / working memory for this subtask.
    value: str = Field(default="", description="Current node value/summary/state.")
    revisions: List[NodeRevision] = Field(default_factory=list)

    # Spatial anchors: union of anchors seen when working on this node
    anchors: List[SpatialAnchor] = Field(default_factory=list)

    # Lightweight outcomes/traces (keep minimal; detailed traces remain in your HistoryManager)
    last_outcome: Optional[str] = Field(default=None, description="Last short outcome string.")
    last_guidance: Optional[str] = Field(default=None, description="Last planner guidance.")
    last_success: Optional[bool] = Field(default=None)
    last_updated_step: Optional[int] = Field(default=None, ge=0)

    def anchor_keys(self) -> List[str]:
        """
        Run anchor keys for the current workflow step.
        
        Returns
        -------
        List[str]
            List with computed output entries.
        """
        return [a.as_key() for a in self.anchors]


# -----------------------------
# Retrieval payload (for prompts)
# -----------------------------

class RetrievedNodeForPrompt(BaseModel):
    """
    Minimal node projection to inject into planner prompt.
    """
    model_config = ConfigDict(extra="forbid")

    node_id: str
    title: str
    status: NodeStatus
    value: str
    last_outcome: Optional[str] = None
    last_guidance: Optional[str] = None
    last_success: Optional[bool] = None

    # Important: include a few anchors as cues (paper: spatial grounding)
    anchors: List[str] = Field(default_factory=list)


class RetrievedSubgraph(BaseModel):
    """
    Paper: context retrieval returns a subgraph G' subset G.
    """
    model_config = ConfigDict(extra="forbid")

    nodes: List[RetrievedNodeForPrompt] = Field(default_factory=list)
    edges: List[TMSEdge] = Field(default_factory=list)


# -----------------------------
# TRIM tool schema (LLM output)
# -----------------------------

class TRIMSubtaskDecision(BaseModel):
    """
    Paper: TRIM = (Input Decomposition) + (Intent Classification) + (Mapping).
    This is what we ask the LLM to output as a tool call.
    """
    model_config = ConfigDict(extra="forbid")

    subtask: str = Field(..., description="A decomposed subtask (macro-goal).")

    intent: TRIMIntent = Field(..., description="NEW/UPDATE/CHECK/ROLLBACK.")
    op: TMSOp = Field(..., description="ADD/REPLACE/INACTIVATE/ROLLBACK/NOOP.")

    # Mapping: if UPDATE/REPLACE/ROLLBACK/INACTIVATE, identify the target node
    target_node_id: Optional[str] = Field(default=None)

    # If creating a new node, optional canonical title (else derived from subtask)
    proposed_title: Optional[str] = Field(default=None)

    # Optional new value proposal (paper: node value as working memory)
    proposed_value: Optional[str] = Field(default=None)

    # Dependencies (DAG edges) - ids may refer to existing nodes or new placeholders
    depends_on: List[str] = Field(default_factory=list, description="Parent node_ids this subtask depends on.")

    # If rollback, which revision (optional; if None use latest-1)
    rollback_to_rev: Optional[int] = Field(default=None)

    rationale: Optional[str] = Field(default=None, description="Short rationale for the decision.")


class TRIMToolOutput(BaseModel):
    """
    The tool output expected from LLM.
    """
    model_config = ConfigDict(extra="forbid")

    decisions: List[TRIMSubtaskDecision] = Field(default_factory=list)
    global_notes: Optional[str] = Field(default=None)