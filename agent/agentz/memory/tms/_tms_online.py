# _tms_online_v2.py
from __future__ import annotations

import math
import logging
import uuid
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from agentz.pydantic_models import ExecutedChunk
from agentz.constants import (
    TMS_ANCHOR_CAP,
    TMS_ANCHOR_MAX_ITEMS,
    TMS_DIVERSITY_TITLE_SIM,
    TMS_DIVERSITY_VALUE_SIM,
    TMS_EST_SIZE_OVERHEAD,
    TMS_FIND_NODE_SIM_THRESHOLD,
    TMS_GRID,
    TMS_INACTIVE_KEEP_SCORE,
    TMS_INACTIVE_PENALTY,
    TMS_INACTIVATE_MIN_FAIL_STREAK,
    TMS_JACCARD_MIN_LEN,
    TMS_MAX_ANCHORS_PER_OBS,
    TMS_MAX_NODES_IN_PROMPT,
    TMS_NODE_ID_LEN,
    TMS_PROJECT_MAX_ANCHORS,
    TMS_PROJECT_MAX_VALUE_CHARS,
    TMS_RECENCY_DECAY,
    TMS_SCORE_ANCHOR_W,
    TMS_SCORE_LEXICAL_W,
    TMS_SCORE_RECENCY_W,
    TMS_TOKEN_BUDGET_CHARS,
)
from ._anchors import build_spatial_anchors
from ..utils._formatters import format_anchor_lines, format_planner_nodes, project_tms_node_for_prompt
from ..utils._similarity import jaccard_words
from agentz.pydantic_models._tms_models import (
    NodeRevision,
    NodeStatus,
    SpatialAnchor,
    TMSEdge,
    TMSNode,
    TRIMToolOutput,
    TRIMSubtaskDecision,
    TMSOp,
    RetrievedNodeForPrompt,
    RetrievedSubgraph,
)

class OnlineTMS:
    """
    Paper: Task Memory Structure (TMS) + Update + Retrieve.

    This module is designed to be called from your Agent loop:
      - TRIM produces TRIMToolOutput (decomposition + intents + ops)
      - OnlineTMS.apply_trim_output(...) mutates the graph (ADD/REPLACE/INACTIVATE/ROLLBACK)
      - OnlineTMS.retrieve_subgraph(...) produces the context subgraph G' to feed Planner prompts
    """

    def __init__(
        self,
        grid: int = TMS_GRID,
        max_anchors_per_obs: int = TMS_MAX_ANCHORS_PER_OBS,
        max_nodes_in_prompt: int = TMS_MAX_NODES_IN_PROMPT,
        token_budget_chars: int = TMS_TOKEN_BUDGET_CHARS,  # approximate budget (chars) for context; tune later
    ):
        """
        Initialize class dependencies and runtime state.
        
        Parameters
        ----------
        grid : Optional[int]
            Function argument.
        max_anchors_per_obs : Optional[int]
            Function argument.
        max_nodes_in_prompt : Optional[int]
            Function argument.
        token_budget_chars : Optional[int]
            Function argument.
        
        Returns
        -------
        Any
            Function result.
        """
        self.grid = grid
        self.max_anchors_per_obs = max_anchors_per_obs
        self.max_nodes_in_prompt = max_nodes_in_prompt
        self.token_budget_chars = token_budget_chars
        self.logger = logging.getLogger("OnlineTMS")

        self._step: int = 0
        self._nodes: Dict[str, TMSNode] = {}
        self._edges: List[TMSEdge] = []
        self._fail_streak_by_node: Dict[str, int] = {}

    # -----------------------------
    # Basic helpers
    # -----------------------------

    def reset(self) -> None:
        """
        Run reset for the current workflow step.
        
        Returns
        -------
        None
            No return value.
        """
        self._step = 0
        self._nodes = {}
        self._edges = []
        self._fail_streak_by_node = {}

    @property
    def step(self) -> int:
        """
        Process step.
        
        Returns
        -------
        int
            Integer result value.
        
        """
        return self._step

    def bump_step(self) -> None:
        """
        Run bump step for the current workflow step.
        
        Returns
        -------
        None
            No return value.
        """
        self._step += 1

    def nodes(self) -> List[TMSNode]:
        """
        Run nodes for the current workflow step.
        
        Returns
        -------
        List[TMSNode]
            List with computed output entries.
        """
        return list(self._nodes.values())

    def edges(self) -> List[TMSEdge]:
        """
        Run edges for the current workflow step.
        
        Returns
        -------
        List[TMSEdge]
            List with computed output entries.
        """
        return list(self._edges)

    def get_node(self, node_id: str) -> Optional[TMSNode]:
        """
        Return get node.
        
        Parameters
        ----------
        node_id : str
            Identifier value.
        
        Returns
        -------
        Optional[TMSNode]
            Function result.
        """
        return self._nodes.get(node_id)

    def _new_node_id(self) -> str:
        """
        Process new node id.
        
        Returns
        -------
        str
            Resulting string value.
        
        """
        return uuid.uuid4().hex[:TMS_NODE_ID_LEN]

    # -----------------------------
    # Anchors extraction (spatial)
    # -----------------------------

    def observation_to_anchors(self, observation: Any) -> List[SpatialAnchor]:
        """
        Process observation to anchors.
        
        Parameters
        ----------
        observation : Any
            Observation payload.
        
        Returns
        -------
        List[SpatialAnchor]
            List with computed values.
        
        """
        ui = getattr(observation, "ui_elements", None)
        return build_spatial_anchors(ui, grid=self.grid, max_anchors=self.max_anchors_per_obs)

    # -----------------------------
    # Paper: Update (ADD/REPLACE/INACTIVATE/ROLLBACK)
    # -----------------------------

    def apply_trim_output(
        self,
        trim_out: TRIMToolOutput,
        last_chunk:  ExecutedChunk,
        current_observation: Optional[Any] = None,
    ) -> None:
        """
        Apply TRIM decisions to the TMS graph.
        
        Paper: Update phase supports:
          - ADD: create node
          - REPLACE: update node value (append revision)
          - INACTIVATE: deactivate obsolete node
          - ROLLBACK: restore an older revision
        """
        anchors = self.observation_to_anchors(current_observation) if current_observation is not None else []
        op_counts = Counter([d.op for d in (trim_out.decisions or [])])
        self.logger.info(
            "TMS apply | decisions=%d | ops=%s | anchors=%d | last_success=%s | failure_type=%s",
            len(trim_out.decisions),
            dict(op_counts),
            len(anchors),
            getattr(last_chunk, "overall_success", None),
            getattr(last_chunk, "failure_type", None),
        )
        self.logger.info(
            "TMS apply | pre_state | nodes=%d | edges=%d",
            len(self._nodes),
            len(self._edges),
        )
        for i, dec in enumerate(trim_out.decisions or []):
            self.logger.info(
                "TRIM decision | idx=%d | op=%s | intent=%s | target=%s | deps=%d | rationale=%s",
                i,
                getattr(dec, "op", None),
                getattr(dec, "intent", None),
                getattr(dec, "target_node_id", None),
                len(getattr(dec, "depends_on", []) or []),
                (getattr(dec, "rationale", None) or "").strip()[:200],
            )

        # First pass: ensure nodes exist for ADD decisions
        created_map: Dict[str, str] = {}  # optional mapping for placeholder titles, if any
        add_sequence: List[str] = []
        add_with_deps: Set[str] = set()
        for dec in trim_out.decisions:
            if dec.op == TMSOp.ADD:
                existing_id = self._find_similar_node(dec.proposed_title or dec.subtask)
                if existing_id:
                    # Dedup: reuse existing node instead of creating a duplicate
                    created_map[dec.subtask] = existing_id
                    add_sequence.append(existing_id)
                    continue
                node_id = self._new_node_id()
                title = (dec.proposed_title or dec.subtask).strip()
                add_value = (last_chunk.post_chunk_state or "").strip()
                value_source = "post_chunk_state"
                if not add_value:
                    add_value = (dec.proposed_value or "").strip()
                    value_source = "proposed_value"
                node = TMSNode(
                    node_id=node_id,
                    title=title,
                    status=NodeStatus.ACTIVE,
                    created_step=self._step,
                    value=add_value,
                    revisions=[],
                    anchors=list(anchors),
                    last_guidance=last_chunk.planner_guidance,
                    last_updated_step=self._step,
                    last_success = last_chunk.overall_success,
                    last_outcome = last_chunk.post_chunk_state
                )
                # Initialize revision history with initial value if any
                if node.value:
                    node.revisions.append(
                        NodeRevision(
                            rev_id=0,
                            created_step=self._step,
                            value=node.value,
                            summary="init",
                        )
                    )

                self._nodes[node_id] = node
                created_map[dec.subtask] = node_id
                add_sequence.append(node_id)
                self.logger.info(
                    "TMS ADD | node_id=%s | subtask=%s | value_source=%s",
                    node_id,
                    dec.subtask,
                    value_source,
                )

        # Second pass: apply REPLACE/INACTIVATE/ROLLBACK and edges
        edges_before = len(self._edges)
        for dec in trim_out.decisions:
            target_id = dec.target_node_id

            # If LLM referenced a subtask string that was just created (common),
            # allow mapping via created_map.
            if not target_id and dec.op in {TMSOp.REPLACE, TMSOp.INACTIVATE, TMSOp.ROLLBACK}:
                target_id = created_map.get(dec.subtask)

            if dec.op == TMSOp.REPLACE:
                if not target_id or target_id not in self._nodes:
                    # If mapping failed, fallback: create a node (best-effort).
                    node_id = self._new_node_id()
                    title = (dec.proposed_title or dec.subtask).strip()
                    fallback_value = (last_chunk.post_chunk_state or "").strip()
                    value_source = "post_chunk_state"
                    if not fallback_value:
                        fallback_value = (dec.proposed_value or "").strip()
                        value_source = "proposed_value"
                    self._nodes[node_id] = TMSNode(
                        node_id=node_id,
                        title=title,
                        status=NodeStatus.ACTIVE,
                        created_step=self._step,
                        value=fallback_value,
                        revisions=[],
                        anchors=list(anchors),
                        last_guidance=last_chunk.planner_guidance,
                        last_updated_step=self._step,
                        last_success = last_chunk.overall_success,
                        last_outcome = last_chunk.post_chunk_state
                    )
                    if fallback_value:
                        self._nodes[node_id].revisions.append(
                            NodeRevision(
                                rev_id=0,
                                created_step=self._step,
                                value=fallback_value,
                                summary="init_fallback",
                            )
                        )
                    target_id = node_id
                    self.logger.info(
                        "TMS REPLACE fallback | created node_id=%s | subtask=%s | value_source=%s",
                        node_id,
                        dec.subtask,
                        value_source,
                    )

                node = self._nodes[target_id]
                node.status = NodeStatus.ACTIVE
                node.last_updated_step = self._step
                node.last_success = last_chunk.overall_success
                node.last_outcome = last_chunk.post_chunk_state
                node.last_guidance = last_chunk.planner_guidance

                # Merge anchors (paper: accumulate spatial memory)
                self._merge_anchors(node, anchors)

                # Determine authoritative value (Judge > TRIM)
                new_val = (last_chunk.post_chunk_state or "").strip()
                value_source = "post_chunk_state"
                if not new_val:
                    new_val = (dec.proposed_value or "").strip()
                    value_source = "proposed_value"

                if new_val and node.value != new_val:
                    self.logger.info(
                        "TMS REPLACE | node_id=%s | value_source=%s | updated=True",
                        target_id,
                        value_source,
                    )
                    node.value = new_val
                    rev_id = (node.revisions[-1].rev_id + 1) if node.revisions else 0
                    node.revisions.append(
                        NodeRevision(
                            rev_id=rev_id,
                            created_step=self._step,
                            value=new_val,
                            summary="judge_update",
                        )
                    )
                else:
                    self.logger.info(
                        "TMS REPLACE | node_id=%s | value_source=%s | updated=False",
                        target_id,
                        value_source,
                    )

            elif dec.op == TMSOp.INACTIVATE:
                # Guardrail: only allow inactivation for structural failures
                if last_chunk.failure_type in {"WRONG_TARGET", "ACTION_INEFFECTIVE"}:
                    if target_id and target_id in self._nodes:
                        # Only inactivate after repeated failures on the same node
                        streak = self._fail_streak_by_node.get(target_id, 0)
                        if not last_chunk.overall_success:
                            streak += 1
                        if streak >= TMS_INACTIVATE_MIN_FAIL_STREAK:
                            node = self._nodes[target_id]
                            node.status = NodeStatus.INACTIVE
                            node.last_updated_step = self._step
                        else:
                            self.logger.info(
                                "TMS INACTIVATE skipped | node_id=%s | fail_streak=%d",
                                target_id,
                                streak,
                            )
                else:
                    self.logger.info(
                        "TMS INACTIVATE skipped | failure_type=%s",
                        last_chunk.failure_type,
                    )

            elif dec.op == TMSOp.ROLLBACK:
                if target_id and target_id in self._nodes:
                    node = self._nodes[target_id]
                    self._rollback_node(node, dec.rollback_to_rev)

            elif dec.op == TMSOp.NOOP:
                pass

            elif dec.op == TMSOp.ADD:
                pass  # already created above

            # Dependencies: add edges parent -> child
            # If this decision created a node and we want edges, infer child_id.
            child_id = None
            if dec.op == TMSOp.ADD:
                child_id = created_map.get(dec.subtask)
            elif target_id and target_id in self._nodes:
                child_id = target_id

            if child_id and dec.depends_on:
                add_with_deps.add(child_id)
                for parent in dec.depends_on:
                    # parent might be a subtask string that was created; try map
                    parent_id = parent
                    if parent_id not in self._nodes:
                        parent_id = created_map.get(parent, parent_id)
                        if parent_id not in self._nodes:
                            parent_id = self._find_similar_node(parent)

                    if parent_id in self._nodes and parent_id != child_id:
                        self._add_edge(parent_id, child_id)

        # Update failure streaks for touched nodes
        touched_nodes = set()
        for dec in trim_out.decisions:
            if dec.op == TMSOp.ADD:
                nid = created_map.get(dec.subtask)
            else:
                nid = dec.target_node_id or created_map.get(dec.subtask)
            if nid:
                touched_nodes.add(nid)
        for nid in touched_nodes:
            if last_chunk.overall_success:
                self._fail_streak_by_node[nid] = 0
            else:
                self._fail_streak_by_node[nid] = self._fail_streak_by_node.get(nid, 0) + 1

        # Auto-link sequential ADD nodes when no explicit dependencies provided
        if len(add_sequence) >= 2:
            for i in range(1, len(add_sequence)):
                child_id = add_sequence[i]
                if child_id in add_with_deps:
                    continue
                parent_id = add_sequence[i - 1]
                self._add_edge(parent_id, child_id)
        edges_after = len(self._edges)
        if edges_after == edges_before and len(trim_out.decisions or []) > 1:
            empty_deps = sum(1 for d in (trim_out.decisions or []) if not (d.depends_on or []))
            self.logger.info(
                "TMS apply | no edges added | decisions=%d | empty_dep_decisions=%d",
                len(trim_out.decisions or []),
                empty_deps,
            )
            # Heuristic: if there is exactly one INACTIVATE target and one ADD, link old -> new
            inact_targets = [
                d.target_node_id
                for d in (trim_out.decisions or [])
                if d.op == TMSOp.INACTIVATE and d.target_node_id
            ]
            add_nodes = [created_map.get(d.subtask) for d in (trim_out.decisions or []) if d.op == TMSOp.ADD]
            inact_targets = [t for t in inact_targets if t in self._nodes]
            add_nodes = [n for n in add_nodes if n in self._nodes]
            if len(inact_targets) == 1 and len(add_nodes) == 1:
                self._add_edge(inact_targets[0], add_nodes[0])
                self.logger.info(
                    "TMS apply | edge_added_fallback | parent=%s | child=%s",
                    inact_targets[0],
                    add_nodes[0],
                )

        self.bump_step()
        self.logger.info(
            "TMS state | step=%d | nodes=%d | edges=%d",
            self._step,
            len(self._nodes),
            len(self._edges),
        )

    def _merge_anchors(self, node: TMSNode, anchors: List[SpatialAnchor]) -> None:
        """
        Process merge anchors.
        
        Parameters
        ----------
        node : TMSNode
            Function argument.
        anchors : List[SpatialAnchor]
            Function argument.
        
        Returns
        -------
        None
            No return value.
        
        """
        if not anchors:
            return
        existing = set(node.anchor_keys())
        for a in anchors:
            k = a.as_key()
            if k not in existing:
                node.anchors.append(a)
                existing.add(k)

        # Cap anchors to avoid unbounded growth
        if len(node.anchors) > TMS_ANCHOR_CAP:
            node.anchors = node.anchors[-TMS_ANCHOR_CAP:]

    def _rollback_node(self, node: TMSNode, rollback_to_rev: Optional[int]) -> None:
        """
        Process rollback node.
        
        Parameters
        ----------
        node : TMSNode
            Function argument.
        rollback_to_rev : Optional[int]
            Function argument.
        
        Returns
        -------
        None
            No return value.
        
        """
        if not node.revisions:
            return

        # Default: latest-1 if possible
        if rollback_to_rev is None:
            if len(node.revisions) >= 2:
                rollback_to_rev = node.revisions[-2].rev_id
            else:
                rollback_to_rev = node.revisions[-1].rev_id

        # Find revision
        rev = next((r for r in node.revisions if r.rev_id == rollback_to_rev), None)
        if not rev:
            return

        node.value = rev.value
        node.status = NodeStatus.ACTIVE
        node.last_updated_step = self._step

        # Append a "rollback" revision marker (keeps history monotonic)
        new_rev_id = node.revisions[-1].rev_id + 1
        node.revisions.append(
            NodeRevision(
                rev_id=new_rev_id,
                created_step=self._step,
                value=node.value,
                summary=f"rollback_to={rollback_to_rev}",
            )
        )

    def _add_edge(self, parent_id: str, child_id: str) -> None:
        # Guard against cycles: do not add edge if it would create a cycle
        """
        Process add edge.
        
        Parameters
        ----------
        parent_id : str
            Identifier value.
        child_id : str
            Identifier value.
        
        Returns
        -------
        None
            No return value.
        
        """
        if self._creates_cycle(parent_id, child_id):
            self.logger.info(
                "TMS EDGE skip (cycle) | parent=%s | child=%s",
                parent_id,
                child_id,
            )
            return
        # Deduplicate
        for e in self._edges:
            if e.parent_id == parent_id and e.child_id == child_id:
                return
        self._edges.append(TMSEdge(parent_id=parent_id, child_id=child_id, label="depends_on"))

    def _creates_cycle(self, parent_id: str, child_id: str) -> bool:
        """
        Check if adding parent_id -> child_id would introduce a cycle.
        """
        if parent_id == child_id:
            return True
        # If parent is reachable from child, adding edge creates a cycle
        visited: Set[str] = set()
        stack: List[str] = [child_id]
        while stack:
            nid = stack.pop()
            if nid == parent_id:
                return True
            if nid in visited:
                continue
            visited.add(nid)
            for e in self._edges:
                if e.parent_id == nid:
                    stack.append(e.child_id)
        return False

    def _find_similar_node(self, title: Optional[str]) -> Optional[str]:
        """
        Dedup helper: find existing node by title similarity.
        """
        t = self._norm(title or "")
        if not t:
            return None
        best_id = None
        best_score = 0.0
        for nid, node in self._nodes.items():
            score = jaccard_words(t, self._norm(node.title), min_len=TMS_JACCARD_MIN_LEN)
            if score > best_score:
                best_score = score
                best_id = nid
        if best_score >= TMS_FIND_NODE_SIM_THRESHOLD:
            return best_id
        return None

    def _diversify_nodes(
        self,
        nodes: List[RetrievedNodeForPrompt],
        *,
        scored_map: Dict[str, float],
        max_sim_title: float = TMS_DIVERSITY_TITLE_SIM,
        max_sim_value: float = TMS_DIVERSITY_VALUE_SIM,
    ) -> List[RetrievedNodeForPrompt]:
        """
        Diversity cap: avoid near-duplicates while keeping distinct approaches.
        Nodes are assumed pre-sorted by relevance.
        """
        kept: List[RetrievedNodeForPrompt] = []
        for n in nodes:
            is_dup = False
            for k in kept:
                title_sim = jaccard_words(self._norm(n.title), self._norm(k.title), min_len=TMS_JACCARD_MIN_LEN)
                value_sim = jaccard_words(self._norm(n.value or ""), self._norm(k.value or ""), min_len=TMS_JACCARD_MIN_LEN)
                if title_sim >= max_sim_title and value_sim >= max_sim_value:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(n)
        return kept

    # -----------------------------
    # Paper: Retrieve (subgraph selection with cost)
    # -----------------------------

    def retrieve_subgraph(
        self,
        task_instruction: str,
        current_observation: Optional[Any] = None,
        k_seed: int = 2,
        expand_hops: int = 1,
    ) -> RetrievedSubgraph:
        """
        Paper: retrieve a subgraph G' subset G minimizing cost under budget.
        Here we implement a pragmatic approximation:
          1) score nodes by relevance(query, anchors) + recency
          2) take k_seed seeds
          3) expand by dependencies (ancestors/children) up to expand_hops
          4) prune by approximate token budget (chars) + node cap
        """
        anchors = self.observation_to_anchors(current_observation) if current_observation is not None else []
        anchor_keys = set(a.as_key() for a in anchors)

        # Score nodes
        scored: List[Tuple[float, str]] = []
        for node_id, node in self._nodes.items():
            score = self._relevance_score(task_instruction, node, anchor_keys)
            if score > 0:
                scored.append((score, node_id))

        scored.sort(reverse=True, key=lambda x: x[0])
        scored_map = dict(scored)
        seed_ids = [nid for _, nid in scored[:k_seed]]
        top_scored = []
        for s, nid in scored[:3]:
            n = self._nodes.get(nid)
            top_scored.append(f"{nid}:{s:.3f}:{getattr(n, 'status', None)}")
        if top_scored:
            self.logger.info("TMS score top3 | %s", ", ".join(top_scored))

        # Expand to subgraph
        selected: Set[str] = set()
        frontier: Set[str] = set(seed_ids)

        for _ in range(expand_hops + 1):
            selected |= frontier
            next_frontier: Set[str] = set()
            for nid in frontier:
                next_frontier |= self._neighbors(nid)
            frontier = next_frontier - selected

        # Prune: remove inactive unless strongly relevant
        selected = self._prune_inactive(selected, scored_map=scored_map)

        # Build edges in induced subgraph
        edges = [e for e in self._edges if e.parent_id in selected and e.child_id in selected]

        # Project nodes, prune by budget
        nodes_proj = [self._project_node(self._nodes[nid]) for nid in selected if nid in self._nodes]
        nodes_proj.sort(
            key=lambda n: (
                n.status != NodeStatus.ACTIVE,
                -float(scored_map.get(n.node_id, 0.0)),
                n.title.lower(),
            )
        )

        nodes_proj = self._diversify_nodes(nodes_proj, scored_map=scored_map)
        nodes_proj = self._prune_by_budget(nodes_proj, edges)

        # After pruning nodes, prune edges again
        kept_ids = set(n.node_id for n in nodes_proj)
        edges = [e for e in edges if e.parent_id in kept_ids and e.child_id in kept_ids]

        self.logger.info(
            "TMS retrieve | total_nodes=%d | scored=%d | selected=%d | kept=%d | edges=%d",
            len(self._nodes),
            len(scored),
            len(selected),
            len(nodes_proj),
            len(edges),
        )

        return RetrievedSubgraph(nodes=nodes_proj, edges=edges)

    def _neighbors(self, node_id: str) -> Set[str]:
        """
        Process neighbors.
        
        Parameters
        ----------
        node_id : str
            Identifier value.
        
        Returns
        -------
        Set[str]
            Function result.
        
        """
        nbrs: Set[str] = set()
        for e in self._edges:
            if e.parent_id == node_id:
                nbrs.add(e.child_id)
            elif e.child_id == node_id:
                nbrs.add(e.parent_id)
        return nbrs

    def _prune_inactive(self, selected: Set[str], scored_map: Dict[str, float]) -> Set[str]:
        """
        Prefer ACTIVE nodes; keep INACTIVE only if highly relevant.
        """
        out: Set[str] = set()
        for nid in selected:
            n = self._nodes.get(nid)
            if not n:
                continue
            if n.status == NodeStatus.ACTIVE:
                out.add(nid)
            else:
                # Keep inactive only if score is high
                if scored_map.get(nid, 0.0) >= TMS_INACTIVE_KEEP_SCORE:
                    out.add(nid)
        return out

    def _project_node(self, node: TMSNode) -> RetrievedNodeForPrompt:
        """
        Process project node.
        
        Parameters
        ----------
        node : TMSNode
            Function argument.
        
        Returns
        -------
        RetrievedNodeForPrompt
            Function result.
        
        """
        return project_tms_node_for_prompt(
            node,
            max_value_chars=TMS_PROJECT_MAX_VALUE_CHARS,
            max_anchors=TMS_PROJECT_MAX_ANCHORS,
        )

    def _prune_by_budget(self, nodes: List[RetrievedNodeForPrompt], edges: List[TMSEdge]) -> List[RetrievedNodeForPrompt]:
        """
        Approx token budget -> use chars.
        Also cap number of nodes.
        """
        if not nodes:
            return nodes

        # Always cap hard
        nodes = nodes[: self.max_nodes_in_prompt]

        def est_size(n: RetrievedNodeForPrompt) -> int:
            """
            Process est size.
                        
                        Parameters
                        ----------
                        n : RetrievedNodeForPrompt
                            Function argument.
                        
                        Returns
                        -------
                        int
                            Integer result value.
                        
            """
            return (
                len(n.node_id)
                + len(n.title)
                + len(n.value or "")
                + len(n.last_outcome or "")
                + len(n.last_guidance or "")
                + sum(len(a) for a in (n.anchors or []))
                + TMS_EST_SIZE_OVERHEAD
            )

        total = 0
        kept: List[RetrievedNodeForPrompt] = []
        for n in nodes:
            s = est_size(n)
            if total + s > self.token_budget_chars:
                continue
            kept.append(n)
            total += s

        return kept

    def _relevance_score(self, query: str, node: TMSNode, obs_anchor_keys: Set[str]) -> float:
        """
        Simple relevance proxy:
          - lexical overlap between query and node.title/value
          - anchor overlap between current obs and node.anchors
          - recency bonus
          - inactive penalty
        """
        q = self._norm(query)
        title = self._norm(node.title)
        val = self._norm(node.value or "")

        lexical = jaccard_words(q, title + " " + val, min_len=TMS_JACCARD_MIN_LEN)

        node_anchors = set(node.anchor_keys())
        anchor_overlap = 0.0
        if obs_anchor_keys and node_anchors:
            inter = len(obs_anchor_keys.intersection(node_anchors))
            denom = max(1, len(obs_anchor_keys))
            anchor_overlap = inter / denom

        # recency: more recent -> higher score
        age = (self._step - (node.last_updated_step or node.created_step))
        recency = math.exp(-TMS_RECENCY_DECAY * max(0, age))  # decays with age

        score = (
            TMS_SCORE_LEXICAL_W * lexical
            + TMS_SCORE_ANCHOR_W * anchor_overlap
            + TMS_SCORE_RECENCY_W * recency
        )

        if node.status == NodeStatus.INACTIVE:
            score *= TMS_INACTIVE_PENALTY

        return score

    @staticmethod
    def _norm(s: str) -> str:
        """
        Process norm.
        
        Parameters
        ----------
        s : str
            Function argument.
        
        Returns
        -------
        str
            Resulting string value.
        
        """
        return "".join(ch.lower() if ch.isalnum() else " " for ch in (s or "")).strip()

    # -----------------------------
    # Build planner context (what you inject into prompts)
    # -----------------------------

    def build_planner_context(
        self,
        task_instruction: str,
        current_observation: Optional[Any] = None,
    ) -> str:
        """
        This is the "memory context" string you can pass to your Planner prompt.
        It corresponds to paper: Retrieve -> return G' to condition response generation.
        """
        subg = self.retrieve_subgraph(task_instruction=task_instruction, current_observation=current_observation)

        anchors = self.observation_to_anchors(current_observation) if current_observation is not None else []
        anchors_txt = format_anchor_lines(
            anchors,
            max_items=TMS_ANCHOR_MAX_ITEMS,
            empty_text="- (no anchors)",
        )

        nodes_txt = format_planner_nodes(subg.nodes)

        edges_txt = ""
        if subg.edges:
            edges_txt = "\n".join([f"- {e.parent_id} -> {e.child_id} ({e.label})" for e in subg.edges])
        else:
            edges_txt = "- (no edges)"

        return f"""This section summarizes relevant past progress and failures.
- ACTIVE nodes represent viable macro-goals.
- INACTIVE nodes represent paths that should NOT be retried.
- Node values describe the last known confirmed state.
- Anchors help match the current UI to known states.
- Use this context to avoid repeating mistakes and to choose the next macro-goal.

Current observation anchors:
{anchors_txt}

Relevant task memory nodes:
{nodes_txt if nodes_txt.strip() else "- (no nodes)"}

Dependencies between nodes:
{edges_txt}""".strip()
