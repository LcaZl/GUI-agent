from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import json
import logging
import re

from agentz.memory import EpisodicMemory, OnlineTMS
from agentz.memory.utils._signatures import (
    build_stable_signature_from_string,
    build_ui_signature_from_elements,
    compute_task_key,
    signature_tokens,
)
from agentz.memory.utils._similarity import jaccard_set
from agentz.memory.core import HistoryManager

from agentz.actuators import PlanExecutor
from agentz.tools.openai_api import GPTClientManager
from agentz.ACI import OSWorldEnvironment

from agentz.pydantic_models import (
    GPTClientRequest,
    Observation,
    PlannerSettings,
    ActionChunk,
    ExecutedChunk,
    StepEvaluation,
)

from agentz.perception import PerceptionInterface
from ._prompts import PROMPT_PLANNING_NF, PROMPT_PLANNING_WF
from agentz.constants import (
    CHUNK_POOL_CAP_MIN,
    CHUNK_POOL_CAP_PER_CASE,
    DEFAULT_PRIORITY_FALLBACK,
    EPISODIC_PER_QUERY_LIMIT,
    MEMORY_ADAPTIVE_BALANCED_BEST_MIN,
    MEMORY_ADAPTIVE_BALANCED_CLOSE_DELTA,
    MEMORY_ADAPTIVE_BALANCED_FAIL_CAP,
    MEMORY_ADAPTIVE_STRICT_BEST_MIN,
    MEMORY_ADAPTIVE_STRICT_CLOSE_DELTA,
    MEMORY_ADAPTIVE_STRICT_FAIL_CAP,
    MEMORY_ADAPTIVE_WEAK_FAIL_CAP,
    MEMORY_PATTERN_FAIL_HIGH_SIM_OVERRIDE,
    MEMORY_PATTERN_FAIL_MIN_SEEN,
    MEMORY_PATTERN_SUCCESS_HIGH_SIM_OVERRIDE,
    MEMORY_PATTERN_SUCCESS_MIN_SEEN,
    MEMORY_UI_REF_FLOOR,
    PLANNER_RETRIEVAL_CHUNK_LIMIT,
    PLANNER_RETRIEVAL_EPISODE_LIMIT,
    PLANNER_RETRIEVAL_PATTERN_LIMIT,
    PLANNER_RETRIEVAL_STEP_LIMIT,
    PLANNER_RETRIEVAL_UI_LIMIT,
    PLANNER_LAST_CHUNK_EVAL_MAX_CHARS,
    PATTERN_POOL_CAP_MIN,
    PATTERN_POOL_CAP_PER_CASE,
)


class Planner:
    """
    Planner runner (glue code).

    Episodic memory digest supports:
      - persistent patterns (get_pattern_bundle)
      - retained episode chunks (get_chunk_bundle)

    Anti-noise policy:
      - if we cannot hydrate at least 1 CASE (pattern or chunk), return "".

    Progress-aware retrieval policy (NEW):
      - If last_chunk was SUCCESS:
          * include 1 "nearby" SUCCESS pattern (highest post-UI similarity)
          * include up to 2 "pitfalls" FAIL patterns (highest pre-UI similarity)
          * avoid spamming old unrelated SUCCESS patterns
      - If last_chunk was FAIL:
          * prioritize FAIL patterns similar to current UI (pre/post similarity)
          * fallback to nearest patterns if none fail
    """

    def __init__(
        self,
        settings: PlannerSettings,
        gpt_client: GPTClientManager,
        env: OSWorldEnvironment,
        perception: PerceptionInterface,
        executor: PlanExecutor,
        mem_root: Optional[str] = None,
    ) -> None:
        """
        Initialize planner dependencies and runtime state.

        Parameters
        ----------
        settings : PlannerSettings
            Planner configuration parameters.
        gpt_client : GPTClientManager
            LLM client used to generate the next action chunk.
        env : OSWorldEnvironment
            Environment adapter used by downstream components.
        perception : PerceptionInterface
            Perception interface used to process environment observations.
        executor : PlanExecutor
            Executor used to run the planned steps.
        mem_root : Optional[str]
            Optional memory root path used only for logging context.

        Returns
        -------
        None
            No return value.
        """
        self.settings = settings
        self.gpt_client = gpt_client
        self.env = env
        self.perception = perception
        self.executor = executor

        self.logger = logging.getLogger("S3Planner")
        self.logger.info("Memory enabled (external session factory). mem_root=%s", mem_root)

    # -------------------------
    # Public API
    # -------------------------

    def propose_next_steps(
        self,
        task: Dict[str, Any],
        history_manager: HistoryManager,
        memory: EpisodicMemory,
        system_info: Optional[Dict[str, Any]] = None,
        tms: OnlineTMS = None,
    ) -> Tuple[ActionChunk, GPTClientRequest]:

        """
        Generate the next action chunk from task, state, and memory context.

        Parameters
        ----------
        task : Dict[str, Any]
            Current task payload, including the high-level instruction.
        history_manager : HistoryManager
            Run history with observations, steps, and chunk outcomes.
        memory : EpisodicMemory
            Episodic memory store used to retrieve relevant past cases.
        system_info : Optional[Dict[str, Any]]
            System metadata from the active VM (OS, DE, display server).
        tms : Optional[OnlineTMS]
            Optional TMS graph used for online retrieval and constraints.

        Returns
        -------
        Tuple[ActionChunk, GPTClientRequest]
            Planned action chunk and the exact prompt request sent to the LLM.
        """
        last_chunk: Optional[ExecutedChunk] = history_manager.get_last_chunk()
        last_observation: Observation = history_manager.get_last_observation()
        ui_elements: str = history_manager.ui_elements_string(last_observation.ui_elements)
        terminal_transcript: str = history_manager.terminal_text_for_prompt(last_observation.terminal_content)
        if last_chunk is not None:
           last_outcome =  "success" if bool(last_chunk.overall_success) else "fail"
        else:
            last_outcome = "No Previous Chunk"

        self.logger.info(
            "Planner call | task_id=%s | last_chunk=%s | ui_elements=%d | tms_nodes=%s",
            task.get("id"),
            last_outcome,
            len(last_observation.ui_elements or {}),
            len(tms.nodes()) if tms is not None else "n/a",
        )
        ui_digest_lines = 0
        if ui_elements:
            ui_digest_lines = ui_elements.count("\n") + 1
        self.logger.info(
            "Planner UI digest | raw_ui=%d | digest_lines=%d | digest_chars=%d",
            len(last_observation.ui_elements or {}),
            ui_digest_lines,
            len(ui_elements or ""),
        )
        term_len = len((terminal_transcript or "").strip())
        terminal_ui = False

        def _is_terminal_window_visible(el: Any) -> bool:
            """
            Return whether is terminal window visible.
            
            Parameters
            ----------
            el : Any
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

            # Avoid false positives from launcher/grid icons named "Terminal".
            terminal_hints = ("gnome-terminal", "terminal", "ptyxis", "xterm", "console")
            in_terminal_context = any(h in app for h in terminal_hints) or any(h in win for h in terminal_hints)

            if in_terminal_context and (is_active or is_focused):
                return True

            if "terminal" in role and (is_active or is_focused):
                return True

            return False

        try:
            for _, el in (last_observation.ui_elements or {}).items():
                if _is_terminal_window_visible(el):
                    terminal_ui = True
                    break
        except Exception:
            terminal_ui = False
        self.logger.info(
            "Planner inputs | terminal_chars=%d | terminal_ui=%s | last_failure=%s",
            term_len,
            terminal_ui,
            getattr(last_chunk, "failure_type", None),
        )
        if term_len > 0 and not terminal_ui:
            self.logger.info(
                "Planner inputs | terminal_transcript_present_but_no_terminal_ui | terminal_chars=%d",
                term_len,
            )
        terminal_visibility = "visible" if terminal_ui else "not_visible"
        if terminal_visibility == "not_visible":
            terminal_transcript = "(terminal not visible in current UI)"
        last_failing_step = "(none)"
        if last_chunk is not None and last_chunk.failing_step_index is not None:
            try:
                idx = int(last_chunk.failing_step_index)
                if 0 <= idx < len(last_chunk.steps):
                    step = last_chunk.steps[idx]
                    cmd_txt = "null" if step.command is None else str(step.command).strip()
                    if len(cmd_txt) > 220:
                        cmd_txt = cmd_txt[:220] + "...(truncated)"
                    last_failing_step = (
                        f"{step.description} | expected={step.expected_outcome} "
                        f"| action_type={step.action_type} | command={cmd_txt}"
                    )
            except Exception:
                pass
        retry_meta = self._compute_prompt_retry_metadata(history_manager=history_manager, last_chunk=last_chunk)
        self.logger.info(
            "Planner retry-meta | same_intent_retry=%d | wait_streak=%d | no_ui_change_streak=%d | last_chunk_ui_changed=%s",
            int(retry_meta["same_intent_retry_count"]),
            int(retry_meta["consecutive_wait_steps"]),
            int(retry_meta["no_ui_change_streak"]),
            str(retry_meta["last_chunk_ui_changed"]),
        )

        episodic_memory_digest: str = self._build_episodic_memory_digest(
            task=task,
            last_observation=last_observation,
            last_chunk=last_chunk,
            memory=memory,
            system_info=system_info,
            per_query_limit=EPISODIC_PER_QUERY_LIMIT,
            max_cases=3,
        )

        # gating in prompt
        episodic_memory_block: str = episodic_memory_digest.strip()
        if episodic_memory_block == "":
            episodic_memory_block = "EPISODIC MEMORY: none."

        #return episodic_memory_block
        if last_chunk is None:
            prompt = PROMPT_PLANNING_NF.format(
                instruction=task["instruction"],
                evaluator=json.dumps(task["evaluator"], indent=2),
                system_info=system_info["summary"] if system_info is not None else "",
                episodic_memory=episodic_memory_block,
                terminal_visibility=terminal_visibility,
                terminal_transcript=terminal_transcript,
                ui_elements=ui_elements,
            )

        else:
            tms_context: str = ""
            retrieval_query = task.get("instruction", "")
            judge_guidance = ""
            if last_chunk is not None and getattr(last_chunk, "planner_guidance", None):
                judge_guidance = str(last_chunk.planner_guidance).strip()
            last_chunk_evaluation_raw = history_manager.last_chunk_digest_for_tms()
            last_chunk_evaluation = self._clip_prompt_text(
                last_chunk_evaluation_raw,
                max_chars=PLANNER_LAST_CHUNK_EVAL_MAX_CHARS,
            )
            if len(last_chunk_evaluation) < len(last_chunk_evaluation_raw):
                self.logger.info(
                    "Planner prompt clip | field=last_chunk_evaluation | raw_chars=%d | clipped_chars=%d",
                    len(last_chunk_evaluation_raw),
                    len(last_chunk_evaluation),
                )

            retrieval_query = self._build_tms_retrieval_query(
                task_instruction=task.get("instruction", ""),
                last_chunk=last_chunk,
                last_failing_step=last_failing_step,
                judge_guidance=judge_guidance,
            )
            if tms is not None:
                tms_context = tms.build_planner_context(
                    task_instruction=retrieval_query,
                    current_observation=last_observation,
                )
            self.logger.info(
                "Planner TMS query | chars=%d | text=%s",
                len(retrieval_query or ""),
                (retrieval_query or "")[:400],
            )

            prompt = PROMPT_PLANNING_WF.format(
                instruction=task["instruction"],
                evaluator=json.dumps(task["evaluator"], indent=2),
                system_info=system_info.get("summary", "") if system_info is not None else "",
                tms_context=tms_context,
                episodic_memory=episodic_memory_block,
                terminal_visibility=terminal_visibility,
                terminal_transcript=terminal_transcript,
                ui_elements=ui_elements,
                last_chunk_evaluation=last_chunk_evaluation,
                last_failing_step=last_failing_step,
                judge_guidance=judge_guidance or "(none)",
                same_intent_retry_count=retry_meta["same_intent_retry_count"],
                consecutive_wait_steps=retry_meta["consecutive_wait_steps"],
                no_ui_change_streak=retry_meta["no_ui_change_streak"],
                last_chunk_ui_changed=retry_meta["last_chunk_ui_changed"]
            )

        #return prompt
        self.logger.info(
            "Planner prompt | chars=%d | episodic_memory_chars=%d | tms_chars=%d",
            len(prompt),
            len(episodic_memory_block),
            len(tms_context) if last_chunk is not None else 0,
        )

        request = GPTClientRequest(
            prompt=prompt,
            tool_schema=ActionChunk,
            overrides={"temperature": 0.0},
        )
        #return request
        response: ActionChunk = self.gpt_client.chat_with_tool_and_images(
            request,
            images=[history_manager.last_observation.screenshot],
            image_detail="low",
        )
        history_manager.update(
            {"entity": "Planner", "request": request},
            tags=["llm_prompt"]
        )

        return response

    @staticmethod
    def _clip_prompt_text(text: Any, max_chars: int) -> str:
        """
        Truncate text to a maximum length with an explicit suffix.

        Parameters
        ----------
        text : Any
            Input value converted to string before clipping.
        max_chars : int
            Maximum number of characters allowed in the output.

        Returns
        -------
        str
            Original string when already within budget, otherwise a clipped
            string ending with ``... (truncated for prompt budget)``.
        """
        s = "" if text is None else str(text)
        if max_chars <= 0 or len(s) <= max_chars:
            return s
        suffix = "\n... (truncated for prompt budget)"
        keep = max(0, max_chars - len(suffix))
        return s[:keep] + suffix

    @staticmethod
    def _memory_reliability_tier(*, ui_ref: float, seen_count: Optional[int]) -> str:
        """
        Classify memory-case reliability for prompt consumption.

        Parameters
        ----------
        ui_ref : float
            UI similarity reference score used for transferability
            (typically ``sim_post`` for successes, ``sim_pre`` for failures).
        seen_count : Optional[int]
            Number of times the pattern was observed in memory.

        Returns
        -------
        str
            Reliability bucket: ``HIGH``, ``MEDIUM``, or ``LOW``.
        """
        try:
            sim = float(ui_ref)
        except Exception:
            sim = 0.0
        try:
            seen = int(seen_count) if seen_count is not None else 1
        except Exception:
            seen = 1

        if sim >= 0.72 and seen >= 2:
            return "HIGH"
        if sim >= 0.50:
            return "MEDIUM"
        return "LOW"

    # -------------------------
    # Episodic memory digest
    # -------------------------

    def _build_episodic_memory_digest(
        self,
        task: Dict[str, Any],
        last_observation: Observation,
        last_chunk: Optional[ExecutedChunk],
        memory: EpisodicMemory,
        system_info: Optional[Dict[str, Any]] = None,
        per_query_limit: int = EPISODIC_PER_QUERY_LIMIT,
        max_cases: int = 3,
    ) -> str:
        """
        Build the episodic-memory prompt section for the planner.

        Parameters
        ----------
        task : Dict[str, Any]
            Current task payload with the user instruction.
        last_observation : Observation
            Latest observation used to compute current UI similarity.
        last_chunk : Optional[ExecutedChunk]
            Last executed chunk used to bias retrieval toward success/failure.
        memory : EpisodicMemory
            Episodic store queried for relevant prior cases.
        system_info : Optional[Dict[str, Any]]
            Environment metadata used to compute a task key when available.
        per_query_limit : int
            Max number of memory hits retrieved per query.
        max_cases : int
            Maximum number of hydrated cases included in the digest.

        Returns
        -------
        str
            Formatted digest string for prompt injection, or an empty string
            when no reliable case can be assembled.
        """
        instruction: str = str(task["instruction"])

        # optional task_key filter (best effort)
        task_key: Optional[str] = self._maybe_compute_task_key(instruction=instruction, system_info=system_info)
        self.logger.info("Planner memory | task_key=%s", task_key or "(none)")

        # current UI signature (for similarity scoring)
        current_ui_sig = build_ui_signature_from_elements(last_observation.ui_elements, stable=True, include_context=True)
        cur_tokens = signature_tokens(current_ui_sig)

        # ---------- 1) Current situation summary ----------
        situation_lines: List[str] = []
        situation_lines.append("CURRENT RUN CONTEXT:")
        situation_lines.append(f"- user_task: {instruction}")

        failure_query: Optional[str] = None
        state_query: Optional[str] = None
        guidance_query: Optional[str] = None

        if last_chunk is None:
            situation_lines.append("- last_chunk: NONE (first planning call)")
        else:
            situation_lines.append(f"- last_macro_goal: {last_chunk.macro_goal}")
            situation_lines.append(f"- last_post_state: {last_chunk.post_chunk_state}")
            situation_lines.append(f"- last_overall_success: {last_chunk.overall_success}")

            if last_chunk.overall_success is False:
                situation_lines.append(f"- last_failure_type: {last_chunk.failure_type}")
                situation_lines.append(f"- last_failing_step_index: {last_chunk.failing_step_index}")

                fr: Optional[str] = None
                fx: Optional[str] = None

                if last_chunk.failing_step_index is not None:
                    idx: int = int(last_chunk.failing_step_index)
                    ev_by_index: Dict[int, StepEvaluation] = {int(ev.index): ev for ev in last_chunk.steps_eval}
                    if idx in ev_by_index:
                        ev = ev_by_index[idx]
                        fr = ev.failure_reason
                        fx = ev.fix_suggestion

                if fr is not None:
                    situation_lines.append(f"- last_failure_reason: {fr}")
                if fx is not None:
                    situation_lines.append(f"- last_fix_suggestion: {fx}")

                failure_type_str: str = ""
                if last_chunk.failure_type is not None and hasattr(last_chunk.failure_type, "value"):
                    failure_type_str = last_chunk.failure_type.value
                elif last_chunk.failure_type is not None:
                    failure_type_str = str(last_chunk.failure_type)

                failure_query = " ".join(
                    s
                    for s in [
                        instruction,
                        failure_type_str,
                        fr or "",
                        fx or "",
                        last_chunk.post_chunk_state,
                    ]
                    if s.strip() != ""
                )
            else:
                state_query = " ".join(
                    s for s in [instruction, last_chunk.macro_goal, last_chunk.post_chunk_state] if s.strip() != ""
                )
                guidance_query = " ".join(
                    s for s in [instruction, last_chunk.planner_guidance] if s.strip() != ""
                )

        # ---------- 2) Query list ----------
        queries: List[Tuple[str, str]] = [("task", instruction)]
        if state_query is not None:
            queries.append(("state", state_query))
        if guidance_query is not None:
            queries.append(("guidance", guidance_query))
        if failure_query is not None:
            queries.append(("failure", failure_query))

        # dedup queries
        seen_q: Set[str] = set()
        uniq_queries: List[Tuple[str, str]] = []
        for tag, q in queries:
            qq = (q or "").strip()
            if qq == "" or qq in seen_q:
                continue
            seen_q.add(qq)
            uniq_queries.append((tag, qq))

        # priority: failure, state, guidance, task
        priority = {"failure": 0, "state": 1, "guidance": 2, "task": 3}
        uniq_queries.sort(key=lambda x: priority.get(x[0], DEFAULT_PRIORITY_FALLBACK))

        # ---------- 3) Retrieval ----------
        raw_hits: List[Dict[str, Any]] = []
        strict_kind_limits = {
            "pattern": int(PLANNER_RETRIEVAL_PATTERN_LIMIT),
            "chunk": int(PLANNER_RETRIEVAL_CHUNK_LIMIT),
            "step": int(PLANNER_RETRIEVAL_STEP_LIMIT),
            "episode": int(PLANNER_RETRIEVAL_EPISODE_LIMIT),
            "ui": int(PLANNER_RETRIEVAL_UI_LIMIT),
        }
        for tag, q in uniq_queries:
            # new retriever supports task_key + kind_limits; keep balanced defaults
            hits = memory.retriever.search(
                q,
                limit=per_query_limit,
                require_fts=False,
                task_key=task_key,
                kind_limits=strict_kind_limits,
            )
            for h in hits:
                h2 = dict(h)
                h2["_qtag"] = tag
                raw_hits.append(h2)

        if not raw_hits:
            self.logger.info("Planner memory | no retrieval hits")
            return ""

        # ---------- 3b) Dedup hits ----------
        def hit_key(h: Dict[str, Any]) -> Tuple[Any, ...]:
            """
            Trigger key.
                        
                        Parameters
                        ----------
                        h : Dict[str, Any]
                            Function argument.
                        
                        Returns
                        -------
                        Tuple[Any, ...]
                            Tuple with computed values.
                        
            """
            return (
                h.get("kind"),
                h.get("pattern_id"),
                h.get("episode_id"),
                h.get("chunk_id"),
                h.get("step_id"),
                h.get("observation_id"),
                h.get("text"),
            )

        dedup_hits: List[Dict[str, Any]] = []
        seen_h: Set[Tuple[Any, ...]] = set()
        dup_count = 0
        for h in raw_hits:
            k = hit_key(h)
            if k in seen_h:
                dup_count += 1
                continue
            seen_h.add(k)
            dedup_hits.append(h)
        self.logger.info(
            "Planner memory | hits=%d | deduped=%d | unique=%d",
            len(raw_hits),
            dup_count,
            len(dedup_hits),
        )
        kind_counts: Dict[str, int] = {}
        for h in dedup_hits:
            k = str(h.get("kind", "")).lower()
            kind_counts[k] = kind_counts.get(k, 0) + 1
        if kind_counts:
            self.logger.info("Planner memory | kinds=%s", kind_counts)

        # ---------- 4) Candidate pools ----------
        # Important: collect MORE than max_cases so we can choose 1 nearby success + pitfalls failures.
        pattern_pool_cap = max(PATTERN_POOL_CAP_MIN, max_cases * PATTERN_POOL_CAP_PER_CASE)
        chunk_pool_cap = max(CHUNK_POOL_CAP_MIN, max_cases * CHUNK_POOL_CAP_PER_CASE)

        pattern_ids: List[int] = []
        seen_p: Set[int] = set()
        chunk_ids: List[int] = []
        seen_c: Set[int] = set()

        invalid_pattern = 0
        invalid_chunk = 0
        for h in dedup_hits:
            kind = str(h.get("kind", "")).lower()
            if kind == "pattern":
                pid_val = h.get("pattern_id", None)
                if pid_val is None:
                    invalid_pattern += 1
                    continue
                try:
                    pid = int(pid_val)
                except Exception:
                    invalid_pattern += 1
                    continue
                if pid in seen_p:
                    continue
                seen_p.add(pid)
                pattern_ids.append(pid)
                if len(pattern_ids) >= pattern_pool_cap:
                    break

        if len(pattern_ids) < pattern_pool_cap:
            for h in dedup_hits:
                kind = str(h.get("kind", "")).lower()
                if kind not in ("chunk", "step"):
                    continue
                cid_val = h.get("chunk_id", None)
                if cid_val is None:
                    invalid_chunk += 1
                    continue
                try:
                    cid = int(cid_val)
                except Exception:
                    invalid_chunk += 1
                    continue
                if cid in seen_c:
                    continue
                seen_c.add(cid)
                chunk_ids.append(cid)
                if len(chunk_ids) >= chunk_pool_cap:
                    break
        if invalid_pattern or invalid_chunk:
            self.logger.info(
                "Planner memory | invalid ids | pattern=%d | chunk=%d",
                invalid_pattern,
                invalid_chunk,
            )

        if not pattern_ids and not chunk_ids:
            self.logger.info("Planner memory | no pattern/chunk candidates after dedup")
            return ""
        self.logger.info(
            "Planner memory | candidate_ids | patterns=%d | chunks=%d",
            len(pattern_ids),
            len(chunk_ids),
        )

        # ---------- 5) Hydrate patterns (for rerank) ----------
        pattern_cases: List[Dict[str, Any]] = []
        for pid in pattern_ids:
            try:
                pbundle = memory.retriever.get_pattern_bundle(pattern_id=pid)
            except Exception:
                continue

            p = pbundle.get("pattern", {}) or {}
            steps = pbundle.get("steps", []) or []

            overall_success = bool(p.get("overall_success", False))
            pre_sig = build_stable_signature_from_string((p.get("pre_ui_signature") or "").strip())
            post_sig = build_stable_signature_from_string((p.get("post_ui_signature") or "").strip())

            sim_pre = jaccard_set(cur_tokens, signature_tokens(pre_sig))
            sim_post = jaccard_set(cur_tokens, signature_tokens(post_sig))

            # optional recency tie-breaker
            last_seen = p.get("last_seen_ts_ms", None)
            try:
                last_seen_val = int(last_seen) if last_seen is not None else 0
            except Exception:
                last_seen_val = 0

            pattern_cases.append(
                {
                    "pbundle": pbundle,
                    "pattern": p,
                    "steps": steps,
                    "overall_success": overall_success,
                    "sim_pre": float(sim_pre),
                    "sim_post": float(sim_post),
                    "last_seen_ts_ms": last_seen_val,
                }
            )

        if pattern_cases:
            top_cases = sorted(
                pattern_cases,
                key=lambda c: (float(c["sim_post"]), float(c["sim_pre"])),
                reverse=True,
            )[:3]
            top_str = ", ".join(
                [
                    f"{(c.get('pattern') or {}).get('pattern_id', '?')}:pre={c['sim_pre']:.2f}:post={c['sim_post']:.2f}:{'S' if c['overall_success'] else 'F'}"
                    for c in top_cases
                ]
            )
            self.logger.info("Planner memory | top patterns: %s", top_str)

        # ---------- 6) Adaptive gating + progress-aware pattern selection ----------
        picked_patterns: List[Dict[str, Any]] = []

        last_success = bool(last_chunk.overall_success) if last_chunk is not None else False
        last_failed = (last_chunk is not None and bool(last_chunk.overall_success) is False)
        adaptive_mode = "none"
        adaptive_fail_cap = MEMORY_ADAPTIVE_BALANCED_FAIL_CAP

        def _sort_success_key(c: Dict[str, Any]) -> Tuple[float, int]:
            # prefer post match, then recency
            """
            Sort success key.
            
            Parameters
            ----------
            c : Dict[str, Any]
                Function argument.
            
            Returns
            -------
            Tuple[float, int]
                Tuple with computed values.
            
            """
            return (float(c["sim_post"]), int(c["last_seen_ts_ms"]))

        def _sort_fail_key(c: Dict[str, Any]) -> Tuple[float, float, int]:
            # prefer pre match, then post match, then recency
            """
            Sort fail key.
            
            Parameters
            ----------
            c : Dict[str, Any]
                Function argument.
            
            Returns
            -------
            Tuple[float, float, int]
                Tuple with computed values.
            
            """
            return (float(c["sim_pre"]), float(c["sim_post"]), int(c["last_seen_ts_ms"]))

        def _sort_ui_ref_key(c: Dict[str, Any]) -> Tuple[float, int, int]:
            """
            Sort by adaptive UI relevance, then consistency, then recency.
            """
            return (float(c.get("ui_ref", 0.0)), int(c.get("seen_count", 1)), int(c.get("last_seen_ts_ms", 0)))

        filtered_pattern_cases: List[Dict[str, Any]] = pattern_cases
        if pattern_cases:
            for c in pattern_cases:
                p = c.get("pattern", {}) or {}
                try:
                    seen_count = int(p.get("seen_count", 1) or 1)
                except Exception:
                    seen_count = 1
                c["seen_count"] = seen_count
                c["ui_ref"] = float(c["sim_post"]) if bool(c.get("overall_success", False)) else float(c["sim_pre"])

            def _passes_seen_gate(c: Dict[str, Any]) -> bool:
                """
                Gate patterns with weak historical support unless UI match is very high.
                """
                is_success = bool(c.get("overall_success", False))
                ui_ref = float(c.get("ui_ref", 0.0))
                seen = int(c.get("seen_count", 1) or 1)
                if is_success:
                    return (seen >= int(MEMORY_PATTERN_SUCCESS_MIN_SEEN)) or (
                        ui_ref >= float(MEMORY_PATTERN_SUCCESS_HIGH_SIM_OVERRIDE)
                    )
                p = c.get("pattern", {}) or {}
                has_failure_detail = bool((p.get("failure_reason") or "").strip()) or bool(
                    (p.get("fix_suggestion") or "").strip()
                )
                if ui_ref >= float(MEMORY_PATTERN_FAIL_HIGH_SIM_OVERRIDE):
                    return True
                return (seen >= int(MEMORY_PATTERN_FAIL_MIN_SEEN)) and has_failure_detail

            best_ui_ref = max(float(c.get("ui_ref", 0.0)) for c in pattern_cases)
            if best_ui_ref >= MEMORY_ADAPTIVE_STRICT_BEST_MIN:
                adaptive_mode = "strict"
                adaptive_close_delta = MEMORY_ADAPTIVE_STRICT_CLOSE_DELTA
                adaptive_fail_cap = MEMORY_ADAPTIVE_STRICT_FAIL_CAP
            elif best_ui_ref >= MEMORY_ADAPTIVE_BALANCED_BEST_MIN:
                adaptive_mode = "balanced"
                adaptive_close_delta = MEMORY_ADAPTIVE_BALANCED_CLOSE_DELTA
                adaptive_fail_cap = MEMORY_ADAPTIVE_BALANCED_FAIL_CAP
            else:
                adaptive_mode = "weak"
                adaptive_close_delta = None
                adaptive_fail_cap = MEMORY_ADAPTIVE_WEAK_FAIL_CAP

            base_candidates = [c for c in pattern_cases if float(c.get("ui_ref", 0.0)) >= MEMORY_UI_REF_FLOOR]
            base_candidates = [c for c in base_candidates if _passes_seen_gate(c)]
            if not base_candidates:
                self.logger.info(
                    "Planner memory | adaptive_gate no_base_candidates_after_seen_gate | floor=%.3f | best_ui_ref=%.3f",
                    float(MEMORY_UI_REF_FLOOR),
                    float(best_ui_ref),
                )
                filtered_pattern_cases = []

            if adaptive_mode in ("strict", "balanced"):
                cutoff = max(MEMORY_UI_REF_FLOOR, float(best_ui_ref) - float(adaptive_close_delta))
                filtered_pattern_cases = [
                    c for c in base_candidates if float(c.get("ui_ref", 0.0)) >= float(cutoff)
                ]
            else:
                success_pool = sorted(
                    [c for c in base_candidates if bool(c.get("overall_success", False)) is True],
                    key=_sort_success_key,
                    reverse=True,
                )
                fail_pool = sorted(
                    [c for c in base_candidates if bool(c.get("overall_success", False)) is False],
                    key=_sort_fail_key,
                    reverse=True,
                )
                filtered_pattern_cases = []
                if success_pool:
                    filtered_pattern_cases.append(success_pool[0])
                if fail_pool and adaptive_fail_cap > 0:
                    filtered_pattern_cases.append(fail_pool[0])

            kept_success = sum(1 for c in filtered_pattern_cases if bool(c.get("overall_success", False)) is True)
            kept_fail = sum(1 for c in filtered_pattern_cases if bool(c.get("overall_success", False)) is False)
            self.logger.info(
                "Planner memory | adaptive_gate mode=%s | best_ui_ref=%.3f | raw=%d | kept=%d | kept_success=%d | kept_fail=%d | fail_cap=%d",
                adaptive_mode,
                float(best_ui_ref),
                len(pattern_cases),
                len(filtered_pattern_cases),
                kept_success,
                kept_fail,
                int(adaptive_fail_cap),
            )

            success_cases = [c for c in filtered_pattern_cases if c["overall_success"] is True]
            fail_cases = [c for c in filtered_pattern_cases if c["overall_success"] is False]

            if last_success:
                success_cases.sort(key=_sort_success_key, reverse=True)
                if success_cases:
                    picked_patterns.append(success_cases[0])

                fail_cases.sort(key=_sort_fail_key, reverse=True)
                pit_cap = min(int(adaptive_fail_cap), max(0, max_cases - len(picked_patterns)))
                for c in fail_cases:
                    if len(picked_patterns) >= max_cases:
                        break
                    if pit_cap <= 0:
                        break
                    picked_patterns.append(c)
                    pit_cap -= 1

                if not picked_patterns:
                    filtered_pattern_cases.sort(key=_sort_success_key, reverse=True)
                    picked_patterns = filtered_pattern_cases[:max_cases]

                if len(picked_patterns) < max_cases:
                    already = {int((c["pattern"] or {}).get("pattern_id", -1)) for c in picked_patterns}
                    for c in success_cases[1:]:
                        pid_val = int((c["pattern"] or {}).get("pattern_id", -1))
                        if pid_val in already:
                            continue
                        picked_patterns.append(c)
                        if len(picked_patterns) >= max_cases:
                            break

            elif last_failed:
                fail_cases.sort(key=_sort_fail_key, reverse=True)
                fail_take = min(max_cases, int(adaptive_fail_cap))
                picked_patterns = fail_cases[:fail_take]

                if len(picked_patterns) < max_cases:
                    success_cases.sort(key=_sort_success_key, reverse=True)
                    already = {int((c["pattern"] or {}).get("pattern_id", -1)) for c in picked_patterns}
                    for c in success_cases:
                        pid_val = int((c["pattern"] or {}).get("pattern_id", -1))
                        if pid_val in already:
                            continue
                        picked_patterns.append(c)
                        if len(picked_patterns) >= max_cases:
                            break

                if not picked_patterns:
                    filtered_pattern_cases.sort(key=_sort_success_key, reverse=True)
                    picked_patterns = filtered_pattern_cases[:max_cases]
            else:
                filtered_pattern_cases.sort(key=_sort_success_key, reverse=True)
                picked_patterns = filtered_pattern_cases[:max_cases]

        # ---------- 7) If patterns are not enough, hydrate chunks as fallback ----------
        picked_chunk_bundles: List[Dict[str, Any]] = []
        remaining_slots = max_cases - len(picked_patterns)
        # Keep chunks as strict fallback only when no pattern survives gating.
        if remaining_slots > 0 and chunk_ids and not picked_patterns:
            for cid in chunk_ids:
                if len(picked_chunk_bundles) >= remaining_slots:
                    break
                try:
                    bundle = memory.retriever.get_chunk_bundle(chunk_id=cid)
                except Exception:
                    continue
                picked_chunk_bundles.append({"chunk_id": cid, "bundle": bundle})

        if not picked_patterns and not picked_chunk_bundles:
            self.logger.info("Planner memory | no usable cases after hydration")
            return ""

        if picked_patterns:
            picked_ids = [str((c.get("pattern") or {}).get("pattern_id", "?")) for c in picked_patterns]
            self.logger.info("Planner memory | picked patterns=%s", ",".join(picked_ids))

        self.logger.info(
            "Planner memory | hits=%d | patterns=%d | chunk_candidates=%d | picked_patterns=%d | picked_chunks=%d",
            len(dedup_hits),
            len(pattern_cases),
            len(chunk_ids),
            len(picked_patterns),
            len(picked_chunk_bundles),
        )

        # ---------- 8) Build digest ----------
        lines: List[str] = []
        lines.extend(situation_lines)

        lines.append("")
        lines.append("RETRIEVAL QUERIES USED:")
        for tag, q in uniq_queries:
            lines.append(f"- [{tag}] {q}")

        lines.append("")
        lines.append("RETRIEVED PAST CASES (HINTS):")
        lines.append("- Prioritize HIGH reliability cases.")
        lines.append("- Use MEDIUM reliability only if aligned with CURRENT UI.")
        lines.append("- Treat LOW reliability as weak hints only.")

        recovery_patterns: List[str] = []
        case_counter = 0

        useful_patterns: List[Dict[str, Any]] = []
        avoid_patterns: List[Dict[str, Any]] = []
        low_reliability_patterns: List[Dict[str, Any]] = []
        for pc in picked_patterns:
            p = pc.get("pattern", {}) or {}
            overall_success = bool(pc.get("overall_success", False))
            ui_ref = float(pc.get("sim_post", 0.0)) if overall_success else float(pc.get("sim_pre", 0.0))
            tier = self._memory_reliability_tier(ui_ref=ui_ref, seen_count=p.get("seen_count"))
            pc["reliability_tier"] = tier
            pc["reliability_ui_ref"] = ui_ref

            if tier == "LOW":
                low_reliability_patterns.append(pc)
            elif overall_success:
                useful_patterns.append(pc)
            else:
                avoid_patterns.append(pc)

        useful_chunks: List[Dict[str, Any]] = []
        avoid_chunks: List[Dict[str, Any]] = []
        for x in picked_chunk_bundles:
            bundle = x.get("bundle", {}) or {}
            c = bundle.get("chunk", {}) or {}
            if bool(c.get("overall_success", False)):
                useful_chunks.append(x)
            else:
                avoid_chunks.append(x)

        def _emit_pattern_case(pc: Dict[str, Any]) -> None:
            """
            Append one pattern case to digest lines.
            """
            nonlocal case_counter
            p = pc.get("pattern", {}) or {}
            steps = pc.get("steps", []) or []
            overall_success = bool(pc.get("overall_success", False))
            tier = str(pc.get("reliability_tier", "LOW"))
            ui_ref = float(pc.get("reliability_ui_ref", 0.0))

            case_counter += 1
            pid = p.get("pattern_id", None)
            lines.append(f"- CASE {case_counter}: kind=PATTERN id={pid}")
            if p.get("source_episode_id") is not None:
                lines.append(f"  source: ep={p.get('source_episode_id')} chunk_index={p.get('source_chunk_index')}")
            lines.append(
                f"  reliability: {tier} | ui_ref={ui_ref:.3f}"
                + (f" | seen_count={p.get('seen_count')}" if p.get("seen_count") is not None else "")
            )
            if tier == "LOW":
                lines.append("  reliability_note: weak transferability; follow only if CURRENT UI strongly confirms.")

            goal = str(p.get("macro_goal", "") or "")
            decision = str(p.get("decision", "") or "")
            failure_type = p.get("failure_type", None)
            failing_step_index = p.get("failing_step_index", None)
            post_state = str(p.get("post_chunk_state", "") or "")
            guidance = str(p.get("planner_guidance", "") or "")

            lines.append(f"  goal: {goal}")
            lines.append(
                f"  outcome: {'SUCCESS' if overall_success else 'FAIL'}"
                + (f" | decision={decision}" if decision else "")
                + (f" | failure_type={failure_type}" if failure_type is not None and str(failure_type) != "" else "")
                + (f" | failing_step={failing_step_index}" if failing_step_index is not None else "")
            )

            if post_state.strip():
                lines.append(f"  post_state: {post_state}")
            if guidance.strip():
                lines.append(f"  guidance: {guidance}")

            pre_sig = (p.get("pre_ui_signature") or "").strip()
            post_sig = (p.get("post_ui_signature") or "").strip()
            if pre_sig or post_sig:
                lines.append(f"  ui_match: pre={pc['sim_pre']:.3f} post={pc['sim_post']:.3f}")

            fr = p.get("failure_reason", None)
            fx = p.get("fix_suggestion", None)
            if not overall_success:
                if fr:
                    lines.append(f"  failure_reason: {fr}")
                if fx:
                    lines.append(f"  fix: {fx}")
                    recovery_patterns.append(str(fx))

            if steps:
                show_steps = (not overall_success) or (len(steps) <= 2)
                if show_steps:
                    max_show = 6
                    lines.append(f"  step_sequence (showing up to {max_show}):")
                    for s in steps[:max_show]:
                        idx = s.get("index", None)
                        desc = s.get("description", "")
                        exp = s.get("expected_outcome", "")
                        succ = s.get("success", None)
                        succ_str = "" if succ is None else ("[SUCCESS]" if bool(succ) else "[FAIL]")
                        lines.append(f"    - [{idx}] {succ_str} {desc} -> {exp}")
                    if len(steps) > max_show:
                        lines.append(f"    ... ({len(steps) - max_show} more steps omitted)")
                else:
                    lines.append(f"  step_sequence: omitted (success pattern, {len(steps)} steps)")

            lines.append("")

        def _emit_chunk_case(x: Dict[str, Any]) -> None:
            """
            Append one chunk fallback case to digest lines.
            """
            nonlocal case_counter
            bundle = x.get("bundle", {}) or {}
            cid = x.get("chunk_id")
            c = bundle.get("chunk", {}) or {}
            steps = bundle.get("steps", []) or []

            case_counter += 1
            ep_id = str(c.get("episode_id", ""))
            ch_index = int(c.get("chunk_index", -1))
            goal = str(c.get("macro_goal", ""))
            overall_success = bool(c.get("overall_success", False))
            failure_type = c.get("failure_type", None)
            failing_step_index = c.get("failing_step_index", None)
            post_state = str(c.get("post_chunk_state", "") or "")
            guidance = str(c.get("planner_guidance", "") or "")

            lines.append(f"- CASE {case_counter}: kind=CHUNK ep={ep_id} chunk={ch_index} chunk_id={cid}")
            lines.append("  reliability: LOW (raw chunk fallback; weaker transferability)")
            lines.append(f"  goal: {goal}")
            lines.append(
                f"  outcome: {'SUCCESS' if overall_success else 'FAIL'}"
                + (f" | failure_type={failure_type}" if failure_type is not None and str(failure_type) != "" else "")
                + (f" | failing_step={failing_step_index}" if failing_step_index is not None else "")
            )

            if post_state.strip():
                lines.append(f"  post_state: {post_state}")
            if guidance.strip():
                lines.append(f"  guidance: {guidance}")

            if overall_success is False:
                chosen_step: Optional[Dict[str, Any]] = None
                chosen_eval: Optional[Dict[str, Any]] = None

                if failing_step_index is not None:
                    idx = int(failing_step_index)
                    for s in steps:
                        if int(s.get("step_index", -1)) == idx:
                            chosen_step = s
                            chosen_eval = s.get("eval", None)
                            break

                if chosen_step is None:
                    for s in steps:
                        ev = s.get("eval", None)
                        if ev is not None and bool(ev.get("success", True)) is False:
                            chosen_step = s
                            chosen_eval = ev
                            break

                if chosen_step is not None:
                    lines.append(f"  failed_step[{chosen_step.get('step_index')}]: {chosen_step.get('description')}")
                    if chosen_eval is not None:
                        fr2 = chosen_eval.get("failure_reason", None)
                        fx2 = chosen_eval.get("fix_suggestion", None)
                        if fr2:
                            lines.append(f"    reason: {fr2}")
                        if fx2:
                            lines.append(f"    fix: {fx2}")
                            recovery_patterns.append(str(fx2))

            lines.append("")

        if useful_patterns or useful_chunks:
            lines.append("POTENTIALLY USEFUL CASES (PAST SUCCESSES):")
            for pc in useful_patterns:
                _emit_pattern_case(pc)
            for x in useful_chunks:
                _emit_chunk_case(x)

        if avoid_patterns or avoid_chunks:
            lines.append("SITUATIONS TO AVOID (PAST FAILURES):")
            for pc in avoid_patterns:
                _emit_pattern_case(pc)
            for x in avoid_chunks:
                _emit_chunk_case(x)

        # Keep low-reliability memory as a last-resort hint only.
        emit_low_reliability = (
            not useful_patterns
            and not avoid_patterns
            and not useful_chunks
            and not avoid_chunks
        )
        if low_reliability_patterns and emit_low_reliability:
            lines.append("LOW-RELIABILITY CASES (LAST-RESORT HINTS):")
            low_sorted = sorted(
                low_reliability_patterns,
                key=lambda c: (
                    float(c.get("reliability_ui_ref", 0.0)),
                    int((c.get("pattern") or {}).get("seen_count", 1) or 1),
                    int(c.get("last_seen_ts_ms", 0)),
                ),
                reverse=True,
            )
            for pc in low_sorted[:1]:
                _emit_pattern_case(pc)

        # anti-noise
        if case_counter == 0:
            return ""

        # Recovery patterns (dedup)
        if recovery_patterns:
            dedup_fx: List[str] = []
            seen_fx: Set[str] = set()
            for fx in recovery_patterns:
                fxs = fx.strip()
                if fxs == "" or fxs in seen_fx:
                    continue
                seen_fx.add(fxs)
                dedup_fx.append(fxs)

            lines.append("RECOVERY PATTERNS SEEN IN PAST CASES (USE AS IDEAS):")
            for fx in dedup_fx[:5]:
                lines.append(f"- {fx}")

        lines.append("")
        lines.append("OPERATIONAL NOTES:")
        lines.append("- Treat these as strategy hints only (NOT ground truth).")
        lines.append("- NEVER reuse old UI coordinates or ids; only use CURRENT ui_elements.")
        lines.append("- If current failure looks similar, prefer the 'fix' patterns above (WAIT, refocus, alternate trigger, etc.).")

        return "\n".join(lines)

    # -------------------------
    # Best-effort task_key (to avoid cross-task contamination)
    # -------------------------

    def _maybe_compute_task_key(self, *, instruction: str, system_info: Optional[Dict[str, Any]]) -> Optional[str]:
        """
        Compute a stable task key from instruction and system metadata.

        Parameters
        ----------
        instruction : str
            Task instruction text.
        system_info : Optional[Dict[str, Any]]
            Probe payload containing OS/de/WS information.

        Returns
        -------
        Optional[str]
            Deterministic task key when required fields are available;
            otherwise ``None``.
        """
        if system_info is None:
            return None

        try:
            os_name = system_info["os"]["pretty_name"]
            desktop_env = system_info["desktop_environment"]
            display_server = system_info["display_server"]
        except Exception:
            return None
        self.logger.info(
            "Planner task_key input | instruction='%s' | os_name='%s' | desktop_env='%s' | display_server='%s'",
            instruction,
            os_name,
            desktop_env,
            display_server,
        )

        try:
            key, _ = compute_task_key(
                instruction=instruction,
                os_name=os_name,
                desktop_env=desktop_env,
                display_server=display_server,
            )
            return key
        except Exception:
            return None

    def _build_tms_retrieval_query(
        self,
        *,
        task_instruction: str,
        last_chunk: Optional[ExecutedChunk],
        last_failing_step: str,
        judge_guidance: str,
    ) -> str:
        """
        Build a retrieval query anchored to the current local failure context.

        Parameters
        ----------
        task_instruction : str
            Original task instruction used as global fallback intent.
        last_chunk : Optional[ExecutedChunk]
            Last executed chunk, used to extract local context signals.
        last_failing_step : str
            Text of the failing step from judge output, when available.
        judge_guidance : str
            Judge suggestion string used to bias retrieval.

        Returns
        -------
        str
            Deduplicated query string for TMS retrieval.
        """
        parts: List[str] = []

        if last_chunk is not None:
            macro_goal = str(getattr(last_chunk, "macro_goal", "") or "").strip()
            if macro_goal:
                parts.append(macro_goal)

            if last_failing_step and last_failing_step != "(none)":
                parts.append(str(last_failing_step).strip())

            post_state = str(getattr(last_chunk, "post_chunk_state", "") or "").strip()
            if post_state:
                parts.append(post_state)

            if judge_guidance:
                parts.append(judge_guidance)

            failure_type = getattr(last_chunk, "failure_type", None)
            if failure_type is not None:
                ft = getattr(failure_type, "value", failure_type)
                ft_txt = str(ft).strip()
                if ft_txt:
                    parts.append(f"failure_type {ft_txt}")

        task_txt = str(task_instruction or "").strip()
        if task_txt:
            parts.append(task_txt)

        dedup_parts: List[str] = []
        seen: Set[str] = set()
        for p in parts:
            s = " ".join(str(p).split())
            if not s or s in seen:
                continue
            seen.add(s)
            dedup_parts.append(s)

        return " | ".join(dedup_parts) if dedup_parts else task_txt

    @staticmethod
    def _intent_key(*, action_type: Any, description: Any) -> str:
        """
        Build a normalized key representing a step intent.

        Parameters
        ----------
        action_type : Any
            Action type value associated with the step.
        description : Any
            Natural-language step description.

        Returns
        -------
        str
            Stable ``action|description`` key after lightweight text cleanup.
        """
        act = str(action_type or "").strip().lower()
        desc = str(description or "").strip().lower()
        if desc:
            desc = re.sub(r"(['\"`]).*?\1", " ", desc)
            desc = re.sub(r"\b\d+(\.\d+)?\b", " ", desc)
            desc = re.sub(r"\s+", " ", desc)
            desc = re.sub(r"[^\w\s\-\+\/:]", "", desc).strip()
        return f"{act}|{desc}" if (act or desc) else ""

    @staticmethod
    def _chunk_ui_changed(chunk: Optional[ExecutedChunk]) -> bool:
        """
        Check whether UI signature changed across a chunk execution.

        Parameters
        ----------
        chunk : Optional[ExecutedChunk]
            Executed chunk with first/last observations.

        Returns
        -------
        bool
            ``True`` when signatures differ or cannot be computed safely,
            otherwise ``False``.
        """
        if chunk is None:
            return True
        try:
            before_ui = getattr(getattr(chunk, "first_observation", None), "ui_elements", None) or {}
            after_ui = getattr(getattr(chunk, "last_observation", None), "ui_elements", None) or {}
            before_sig = build_ui_signature_from_elements(before_ui, stable=True, include_context=True)
            after_sig = build_ui_signature_from_elements(after_ui, stable=True, include_context=True)
            return before_sig != after_sig
        except Exception:
            # Conservative fallback: treat as changed to avoid forced pivots on missing data.
            return True

    def _compute_prompt_retry_metadata(
        self,
        *,
        history_manager: HistoryManager,
        last_chunk: Optional[ExecutedChunk],
    ) -> Dict[str, Any]:
        """
        Derive retry metadata used to adapt the next planner prompt.

        Parameters
        ----------
        history_manager : HistoryManager
            Run history with step/chunk timelines.
        last_chunk : Optional[ExecutedChunk]
            Most recent chunk result.

        Returns
        -------
        Dict[str, Any]
            Retry counters and guidance flags for anti-loop prompting.
        """
        steps = list(getattr(history_manager, "steps_history", None) or [])
        chunks = list(getattr(history_manager, "chunks_history", None) or [])

        # Consecutive repeated intent at step tail.
        same_intent_retry_count = 0
        if steps:
            last_step = steps[-1]
            last_key = self._intent_key(
                action_type=getattr(last_step, "action_type", None),
                description=getattr(last_step, "description", None),
            )
            if last_key:
                run_len = 0
                for st in reversed(steps):
                    k = self._intent_key(
                        action_type=getattr(st, "action_type", None),
                        description=getattr(st, "description", None),
                    )
                    if k != last_key:
                        break
                    run_len += 1
                same_intent_retry_count = max(0, run_len - 1)

        # Consecutive WAIT steps at step tail.
        consecutive_wait_steps = 0
        for st in reversed(steps):
            act = str(getattr(st, "action_type", "") or "").strip().lower()
            if act == "wait":
                consecutive_wait_steps += 1
                continue
            break

        # Consecutive chunks with no UI changes.
        no_ui_change_streak = 0
        for ch in reversed(chunks):
            if self._chunk_ui_changed(ch):
                break
            no_ui_change_streak += 1

        last_chunk_ui_changed = self._chunk_ui_changed(last_chunk)

        return {
            "same_intent_retry_count": int(same_intent_retry_count),
            "consecutive_wait_steps": int(consecutive_wait_steps),
            "no_ui_change_streak": int(no_ui_change_streak),
            "last_chunk_ui_changed": "true" if bool(last_chunk_ui_changed) else "false",
        }

