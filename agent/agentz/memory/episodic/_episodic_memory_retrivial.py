from __future__ import annotations

import re
import sqlite3
import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from agentz.constants import (
    DEFAULT_SEARCH_LIMIT,
    FTS_MAX_TOKENS,
    KIND_LIMIT_CHUNK_MAX,
    KIND_LIMIT_CHUNK_MIN,
    KIND_LIMIT_EPISODE_MAX,
    KIND_LIMIT_EPISODE_MIN,
    KIND_LIMIT_PATTERN_MAX,
    KIND_LIMIT_PATTERN_MIN,
    KIND_LIMIT_STEP_MAX,
    KIND_LIMIT_STEP_MIN,
    KIND_LIMIT_UI_DIVISOR,
    KIND_LIMIT_UI_MAX,
    KIND_LIMIT_UI_MIN,
    QUERY_SNIPPET_LEN,
)


class EpisodicMemoryRetriever:
    """
    Read-only retrieval module.

    Estensioni:
    - Search bilanciata per kind (evita che "ui" domini).
    - Supporto patterns persistenti (kind="pattern") + hydration get_pattern_bundle().
    - Filtri opzionali per task_key (fondamentale per non contaminare tra task diverse).
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        """
        Initialize class dependencies and runtime state.
        
        Parameters
        ----------
        conn : sqlite3.Connection
            Function argument.
        
        Returns
        -------
        None
            No return value.
        """
        self.conn = conn
        self.logger = logging.getLogger("EpisodicMemoryRetriever")

    # -------------------------
    # Basic helpers
    # -------------------------

    def _table_exists(self, name: str) -> bool:
        """
        Process table exists.
        
        Parameters
        ----------
        name : str
            Function argument.
        
        Returns
        -------
        bool
            True when the condition is satisfied, otherwise False.
        
        """
        row = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name=? LIMIT 1",
            (name,),
        ).fetchone()
        return row is not None

    def fts_ready(self) -> bool:
        # pattern è opzionale: se non c'è, si continua (fallback LIKE)
        """
        Run fts ready for the current workflow step.
        
        Returns
        -------
        bool
            Boolean outcome of the check.
        """
        base = all(self._table_exists(t) for t in ("fts_episode", "fts_chunk", "fts_step", "fts_ui"))
        pat = self._table_exists("fts_pattern")
        return base and pat

    def fts_ready_base(self) -> bool:
        """
        Run fts ready base for the current workflow step.
        
        Returns
        -------
        bool
            Boolean outcome of the check.
        """
        return all(self._table_exists(t) for t in ("fts_episode", "fts_chunk", "fts_step", "fts_ui"))

    # -------------------------
    # Query sanitization
    # -------------------------

    _FTS_RESERVED = {"and", "or", "not", "near"}

    def _to_safe_fts_query(self, text: str) -> str:
        """
        Convert to safe fts query.
        
        Parameters
        ----------
        text : str
            Input text.
        
        Returns
        -------
        str
            Resulting string value.
        """
        tokens = re.findall(r"\w+", (text or "").lower())
        tokens = [t for t in tokens if len(t) >= 2 and t not in self._FTS_RESERVED]
        tokens = tokens[:FTS_MAX_TOKENS]
        return " OR ".join(tokens)

    # -------------------------
    # Public search
    # -------------------------

    def search(
        self,
        query: str,
        *,
        limit: int = DEFAULT_SEARCH_LIMIT,
        require_fts: bool = False,
        task_key: Optional[str] = None,
        kind_limits: Optional[Dict[str, int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Restituisce hits bilanciati:
          - pattern (se esiste)
          - chunk/step (preferiti)
          - episode
          - ui (quota piccola)
        
        kind in output: "pattern" | "episode" | "chunk" | "step" | "ui"
        """
        q = (query or "").strip()
        if not q:
            return []

        # default quotas (anti-dominanza UI)
        if kind_limits is None:
            # euristica semplice: garantisci chunk/step/pattern; limita ui
            kind_limits = {
                "pattern": max(KIND_LIMIT_PATTERN_MIN, min(KIND_LIMIT_PATTERN_MAX, limit)),
                "chunk": max(KIND_LIMIT_CHUNK_MIN, min(KIND_LIMIT_CHUNK_MAX, limit)),
                "step": max(KIND_LIMIT_STEP_MIN, min(KIND_LIMIT_STEP_MAX, limit)),
                "episode": max(KIND_LIMIT_EPISODE_MIN, min(KIND_LIMIT_EPISODE_MAX, limit)),
                "ui": max(KIND_LIMIT_UI_MIN, min(KIND_LIMIT_UI_MAX, limit // KIND_LIMIT_UI_DIVISOR)),
            }

        use_fts = self.fts_ready() or self.fts_ready_base()
        self.logger.info(
            "EpisodicMemory search | q='%s' | limit=%d | task_key=%s | fts=%s",
            q[:QUERY_SNIPPET_LEN],
            limit,
            task_key or "(none)",
            use_fts,
        )

        if use_fts:
            try:
                safe_q = self._to_safe_fts_query(q)
                if safe_q:
                    out = self._search_fts_balanced(safe_q, limit=limit, task_key=task_key, kind_limits=kind_limits)
                    kind_counts = Counter([h.get("kind") for h in out])
                    self.logger.info("EpisodicMemory search | hits=%d | kinds=%s", len(out), dict(kind_counts))
                    if kind_counts and not any(k in kind_counts for k in ("pattern", "chunk", "step")):
                        self.logger.info("EpisodicMemory search | fts_only_episode | fallback_like=true")
                        out_like = self._search_like_balanced(q, limit=limit, task_key=task_key, kind_limits=kind_limits)
                        merged = out + out_like
                        # dedup
                        seen = set()
                        deduped = []
                        for h in merged:
                            key = (
                                h.get("kind"),
                                h.get("pattern_id"),
                                h.get("episode_id"),
                                h.get("chunk_id"),
                                h.get("step_id"),
                                h.get("observation_id"),
                                h.get("text"),
                            )
                            if key in seen:
                                continue
                            seen.add(key)
                            deduped.append(h)
                        out = deduped[:limit]
                        kind_counts = Counter([h.get("kind") for h in out])
                        self.logger.info("EpisodicMemory search | fallback_hits=%d | kinds=%s", len(out), dict(kind_counts))
                    return out
                # niente token => fallback LIKE
                out = self._search_like_balanced(q, limit=limit, task_key=task_key, kind_limits=kind_limits)
                kind_counts = Counter([h.get("kind") for h in out])
                self.logger.info("EpisodicMemory search | hits=%d | kinds=%s", len(out), dict(kind_counts))
                return out
            except sqlite3.OperationalError:
                out = self._search_like_balanced(q, limit=limit, task_key=task_key, kind_limits=kind_limits)
                kind_counts = Counter([h.get("kind") for h in out])
                self.logger.info("EpisodicMemory search | hits=%d | kinds=%s", len(out), dict(kind_counts))
                return out

        if require_fts:
            raise RuntimeError("FTS tables not found. Ensure EpisodeMemoryDB built the FTS index.")

        out = self._search_like_balanced(q, limit=limit, task_key=task_key, kind_limits=kind_limits)
        kind_counts = Counter([h.get("kind") for h in out])
        self.logger.info("EpisodicMemory search | hits=%d | kinds=%s", len(out), dict(kind_counts))
        return out

    # -------------------------
    # FTS balanced search
    # -------------------------

    def _search_fts_balanced(
        self,
        q: str,
        *,
        limit: int,
        task_key: Optional[str],
        kind_limits: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        """
        Process search fts balanced.
        
        Parameters
        ----------
        q : str
            Function argument.
        limit : int
            Function argument.
        task_key : Optional[str]
            Function argument.
        kind_limits : Dict[str, int]
            Function argument.
        
        Returns
        -------
        List[Dict[str, Any]]
            Dictionary with computed fields.
        
        """
        buckets: Dict[str, List[Dict[str, Any]]] = {
            "pattern": [],
            "episode": [],
            "chunk": [],
            "step": [],
            "ui": [],
        }

        # patterns (task_key diretto su fts_pattern.task_key)
        if self._table_exists("fts_pattern") and kind_limits.get("pattern", 0) > 0:
            buckets["pattern"] = self._search_fts_pattern(q, limit=kind_limits["pattern"], task_key=task_key)

        # episodes/chunks/steps/ui: filtro via JOIN episodes(task_key,retained)
        if self._table_exists("fts_episode") and kind_limits.get("episode", 0) > 0:
            buckets["episode"] = self._search_fts_episode(q, limit=kind_limits["episode"], task_key=task_key)

        if self._table_exists("fts_chunk") and kind_limits.get("chunk", 0) > 0:
            buckets["chunk"] = self._search_fts_chunk(q, limit=kind_limits["chunk"], task_key=task_key)

        if self._table_exists("fts_step") and kind_limits.get("step", 0) > 0:
            buckets["step"] = self._search_fts_step(q, limit=kind_limits["step"], task_key=task_key)

        if self._table_exists("fts_ui") and kind_limits.get("ui", 0) > 0:
            buckets["ui"] = self._search_fts_ui(q, limit=kind_limits["ui"], task_key=task_key)

        # merge round-robin con priorità: pattern -> chunk -> step -> episode -> ui
        order = ["pattern", "chunk", "step", "episode", "ui"]
        merged: List[Dict[str, Any]] = []
        idx = {k: 0 for k in order}

        # ordina ogni bucket per score asc (bm25 migliore = più basso)
        for k in order:
            buckets[k].sort(key=lambda x: (x.get("score") is None, x.get("score", 0.0)))

        while len(merged) < limit:
            progressed = False
            for k in order:
                i = idx[k]
                if i < len(buckets[k]):
                    merged.append(buckets[k][i])
                    idx[k] += 1
                    progressed = True
                    if len(merged) >= limit:
                        break
            if not progressed:
                break

        # dedup finale (per safety)
        seen = set()
        out = []
        for h in merged:
            key = (
                h.get("kind"),
                h.get("pattern_id"),
                h.get("episode_id"),
                h.get("chunk_id"),
                h.get("step_id"),
                h.get("observation_id"),
                h.get("text"),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(h)
        return out[:limit]

    def _search_fts_pattern(self, q: str, *, limit: int, task_key: Optional[str]) -> List[Dict[str, Any]]:
        """
        Process search fts pattern.
        
        Parameters
        ----------
        q : str
            Function argument.
        limit : int
            Function argument.
        task_key : Optional[str]
            Function argument.
        
        Returns
        -------
        List[Dict[str, Any]]
            Dictionary with computed fields.
        
        """
        out: List[Dict[str, Any]] = []
        if task_key:
            rows = self.conn.execute(
                """
                SELECT fts_pattern.pattern_id,
                       bm25(fts_pattern) AS score,
                       fts_pattern.macro_goal,
                       COALESCE(p.seen_count, 1) AS seen_count,
                       COALESCE(p.last_seen_ts_ms, 0) AS last_seen_ts_ms
                FROM fts_pattern
                LEFT JOIN patterns p ON p.pattern_id=fts_pattern.pattern_id
                WHERE fts_pattern MATCH ? AND fts_pattern.task_key=?
                ORDER BY score, seen_count DESC, last_seen_ts_ms DESC
                LIMIT ?
                """,
                (q, task_key, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT fts_pattern.pattern_id,
                       bm25(fts_pattern) AS score,
                       fts_pattern.macro_goal,
                       COALESCE(p.seen_count, 1) AS seen_count,
                       COALESCE(p.last_seen_ts_ms, 0) AS last_seen_ts_ms
                FROM fts_pattern
                LEFT JOIN patterns p ON p.pattern_id=fts_pattern.pattern_id
                WHERE fts_pattern MATCH ?
                ORDER BY score, seen_count DESC, last_seen_ts_ms DESC
                LIMIT ?
                """,
                (q, limit),
            ).fetchall()

        for r in rows:
            out.append({"kind": "pattern", "pattern_id": r[0], "score": float(r[1]), "text": r[2]})
        return out

    def _search_fts_episode(self, q: str, *, limit: int, task_key: Optional[str]) -> List[Dict[str, Any]]:
        """
        Process search fts episode.
        
        Parameters
        ----------
        q : str
            Function argument.
        limit : int
            Function argument.
        task_key : Optional[str]
            Function argument.
        
        Returns
        -------
        List[Dict[str, Any]]
            Dictionary with computed fields.
        
        """
        out: List[Dict[str, Any]] = []
        if task_key:
            rows = self.conn.execute(
                """
                SELECT f.episode_id, bm25(fts_episode) AS score, f.instruction
                FROM fts_episode f
                JOIN episodes e ON e.episode_id=f.episode_id
                WHERE fts_episode MATCH ? AND e.task_key=? AND e.retained=1
                ORDER BY score
                LIMIT ?
                """,
                (q, task_key, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT episode_id, bm25(fts_episode) AS score, instruction
                FROM fts_episode
                WHERE fts_episode MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (q, limit),
            ).fetchall()

        for r in rows:
            out.append({"kind": "episode", "episode_id": r[0], "score": float(r[1]), "text": r[2]})
        return out

    def _search_fts_chunk(self, q: str, *, limit: int, task_key: Optional[str]) -> List[Dict[str, Any]]:
        """
        Process search fts chunk.
        
        Parameters
        ----------
        q : str
            Function argument.
        limit : int
            Function argument.
        task_key : Optional[str]
            Function argument.
        
        Returns
        -------
        List[Dict[str, Any]]
            Dictionary with computed fields.
        
        """
        out: List[Dict[str, Any]] = []
        if task_key:
            rows = self.conn.execute(
                """
                SELECT f.episode_id, f.chunk_id, bm25(fts_chunk) AS score, f.macro_goal
                FROM fts_chunk f
                JOIN episodes e ON e.episode_id=f.episode_id
                WHERE fts_chunk MATCH ? AND e.task_key=? AND e.retained=1
                ORDER BY score
                LIMIT ?
                """,
                (q, task_key, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT episode_id, chunk_id, bm25(fts_chunk) AS score, macro_goal
                FROM fts_chunk
                WHERE fts_chunk MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (q, limit),
            ).fetchall()

        for r in rows:
            out.append({"kind": "chunk", "episode_id": r[0], "chunk_id": r[1], "score": float(r[2]), "text": r[3]})
        return out

    def _search_fts_step(self, q: str, *, limit: int, task_key: Optional[str]) -> List[Dict[str, Any]]:
        """
        Process search fts step.
        
        Parameters
        ----------
        q : str
            Function argument.
        limit : int
            Function argument.
        task_key : Optional[str]
            Function argument.
        
        Returns
        -------
        List[Dict[str, Any]]
            Dictionary with computed fields.
        
        """
        out: List[Dict[str, Any]] = []
        if task_key:
            rows = self.conn.execute(
                """
                SELECT f.episode_id, f.chunk_id, f.step_id, bm25(fts_step) AS score, f.description
                FROM fts_step f
                JOIN episodes e ON e.episode_id=f.episode_id
                WHERE fts_step MATCH ? AND e.task_key=? AND e.retained=1
                ORDER BY score
                LIMIT ?
                """,
                (q, task_key, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT episode_id, chunk_id, step_id, bm25(fts_step) AS score, description
                FROM fts_step
                WHERE fts_step MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (q, limit),
            ).fetchall()

        for r in rows:
            out.append(
                {"kind": "step", "episode_id": r[0], "chunk_id": r[1], "step_id": r[2], "score": float(r[3]), "text": r[4]}
            )
        return out

    def _search_fts_ui(self, q: str, *, limit: int, task_key: Optional[str]) -> List[Dict[str, Any]]:
        """
        Process search fts ui.
        
        Parameters
        ----------
        q : str
            Function argument.
        limit : int
            Function argument.
        task_key : Optional[str]
            Function argument.
        
        Returns
        -------
        List[Dict[str, Any]]
            Dictionary with computed fields.
        
        """
        out: List[Dict[str, Any]] = []
        if task_key:
            rows = self.conn.execute(
                """
                SELECT f.episode_id, f.observation_id, bm25(fts_ui) AS score,
                    f.label, f.app_name, f.window_name
                FROM fts_ui f
                JOIN episodes e ON e.episode_id=f.episode_id
                WHERE fts_ui MATCH ? AND e.task_key=? AND e.retained=1
                ORDER BY score
                LIMIT ?
                """,
                (q, task_key, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT episode_id, observation_id, bm25(fts_ui) AS score,
                    label, app_name, window_name
                FROM fts_ui
                WHERE fts_ui MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (q, limit),
            ).fetchall()

        for r in rows:
            label = r[3] or ""
            app = r[4] or ""
            win = r[5] or ""
            suffix = ""
            if app or win:
                suffix = f" [{app} | {win}]".strip()
            out.append({"kind": "ui", "episode_id": r[0], "observation_id": r[1], "score": float(r[2]), "text": (label + suffix)})
        return out


    # -------------------------
    # LIKE balanced search (fallback)
    # -------------------------

    def _search_like_balanced(
        self,
        q: str,
        *,
        limit: int,
        task_key: Optional[str],
        kind_limits: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        """
        Process search like balanced.
        
        Parameters
        ----------
        q : str
            Function argument.
        limit : int
            Function argument.
        task_key : Optional[str]
            Function argument.
        kind_limits : Dict[str, int]
            Function argument.
        
        Returns
        -------
        List[Dict[str, Any]]
            Dictionary with computed fields.
        
        """
        like = f"%{q}%"

        buckets: Dict[str, List[Dict[str, Any]]] = {
            "pattern": [],
            "episode": [],
            "chunk": [],
            "step": [],
            "ui": [],
        }

        # patterns
        if self._table_exists("patterns") and kind_limits.get("pattern", 0) > 0:
            if task_key:
                rows = self.conn.execute(
                    """
                    SELECT pattern_id,
                           macro_goal,
                           COALESCE(seen_count, 1) AS seen_count,
                           COALESCE(last_seen_ts_ms, 0) AS last_seen_ts_ms
                    FROM patterns
                    WHERE task_key=?
                      AND (
                        COALESCE(macro_goal,'') LIKE ?
                        OR COALESCE(planner_guidance,'') LIKE ?
                        OR COALESCE(post_chunk_state,'') LIKE ?
                        OR COALESCE(steps_json,'') LIKE ?
                        OR COALESCE(failure_reason,'') LIKE ?
                        OR COALESCE(fix_suggestion,'') LIKE ?
                      )
                    ORDER BY seen_count DESC, last_seen_ts_ms DESC
                    LIMIT ?
                    """,
                    (task_key, like, like, like, like, like, like, kind_limits["pattern"]),
                ).fetchall()
            else:
                rows = self.conn.execute(
                    """
                    SELECT pattern_id,
                           macro_goal,
                           COALESCE(seen_count, 1) AS seen_count,
                           COALESCE(last_seen_ts_ms, 0) AS last_seen_ts_ms
                    FROM patterns
                    WHERE COALESCE(macro_goal,'') LIKE ?
                       OR COALESCE(planner_guidance,'') LIKE ?
                       OR COALESCE(post_chunk_state,'') LIKE ?
                       OR COALESCE(steps_json,'') LIKE ?
                       OR COALESCE(failure_reason,'') LIKE ?
                       OR COALESCE(fix_suggestion,'') LIKE ?
                    ORDER BY seen_count DESC, last_seen_ts_ms DESC
                    LIMIT ?
                    """,
                    (like, like, like, like, like, like, kind_limits["pattern"]),
                ).fetchall()

            for r in rows:
                buckets["pattern"].append({"kind": "pattern", "pattern_id": r[0], "score": None, "text": r[1]})

        # episodes/chunks/steps/ui solo su retained
        if kind_limits.get("episode", 0) > 0:
            if task_key:
                rows = self.conn.execute(
                    """
                    SELECT episode_id, instruction
                    FROM episodes
                    WHERE retained=1 AND task_key=? AND instruction LIKE ?
                    LIMIT ?
                    """,
                    (task_key, like, kind_limits["episode"]),
                ).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT episode_id, instruction FROM episodes WHERE retained=1 AND instruction LIKE ? LIMIT ?",
                    (like, kind_limits["episode"]),
                ).fetchall()
            for r in rows:
                buckets["episode"].append({"kind": "episode", "episode_id": r[0], "score": None, "text": r[1]})

        if kind_limits.get("chunk", 0) > 0:
            if task_key:
                rows = self.conn.execute(
                    """
                    SELECT c.episode_id, c.chunk_id, c.macro_goal
                    FROM chunks c
                    JOIN episodes e ON e.episode_id=c.episode_id
                    WHERE e.retained=1 AND e.task_key=?
                      AND (
                        c.macro_goal LIKE ?
                        OR COALESCE(c.planner_guidance,'') LIKE ?
                        OR COALESCE(c.post_chunk_state,'') LIKE ?
                      )
                    LIMIT ?
                    """,
                    (task_key, like, like, like, kind_limits["chunk"]),
                ).fetchall()
            else:
                rows = self.conn.execute(
                    """
                    SELECT episode_id, chunk_id, macro_goal
                    FROM chunks
                    WHERE macro_goal LIKE ?
                       OR COALESCE(planner_guidance,'') LIKE ?
                       OR COALESCE(post_chunk_state,'') LIKE ?
                    LIMIT ?
                    """,
                    (like, like, like, kind_limits["chunk"]),
                ).fetchall()
            for r in rows:
                buckets["chunk"].append({"kind": "chunk", "episode_id": r[0], "chunk_id": r[1], "score": None, "text": r[2]})

        if kind_limits.get("step", 0) > 0:
            if task_key:
                rows = self.conn.execute(
                    """
                    SELECT e.episode_id, s.chunk_id, s.step_id, s.description
                    FROM steps s
                    JOIN chunks c ON c.chunk_id=s.chunk_id
                    JOIN episodes e ON e.episode_id=c.episode_id
                    LEFT JOIN step_evals ev ON ev.step_id=s.step_id
                    WHERE e.retained=1 AND e.task_key=?
                      AND (
                        s.description LIKE ?
                        OR s.expected_outcome LIKE ?
                        OR COALESCE(ev.evidence,'') LIKE ?
                        OR COALESCE(ev.fix_suggestion,'') LIKE ?
                      )
                    LIMIT ?
                    """,
                    (task_key, like, like, like, like, kind_limits["step"]),
                ).fetchall()
            else:
                rows = self.conn.execute(
                    """
                    SELECT c.episode_id, s.chunk_id, s.step_id, s.description
                    FROM steps s
                    JOIN chunks c ON c.chunk_id=s.chunk_id
                    LEFT JOIN step_evals e ON e.step_id=s.step_id
                    WHERE s.description LIKE ?
                       OR s.expected_outcome LIKE ?
                       OR COALESCE(e.evidence,'') LIKE ?
                       OR COALESCE(e.fix_suggestion,'') LIKE ?
                    LIMIT ?
                    """,
                    (like, like, like, like, kind_limits["step"]),
                ).fetchall()
            for r in rows:
                buckets["step"].append({"kind": "step", "episode_id": r[0], "chunk_id": r[1], "step_id": r[2], "score": None, "text": r[3]})

        if kind_limits.get("ui", 0) > 0:
            if task_key:
                rows = self.conn.execute(
                    """
                    SELECT o.episode_id, o.observation_id, COALESCE(u.label,'')
                    FROM ui_elements u
                    JOIN observations o ON o.observation_id=u.observation_id
                    JOIN episodes e ON e.episode_id=o.episode_id
                    WHERE e.retained=1 AND e.task_key=?
                      AND (
                        COALESCE(u.label,'') LIKE ?
                        OR COALESCE(u.value,'') LIKE ?
                        OR COALESCE(u.a11y_role,'') LIKE ?
                        OR COALESCE(u.kind,'') LIKE ?
                      )
                    LIMIT ?
                    """,
                    (task_key, like, like, like, like, kind_limits["ui"]),
                ).fetchall()
            else:
                rows = self.conn.execute(
                    """
                    SELECT o.episode_id, o.observation_id, COALESCE(u.label,'')
                    FROM ui_elements u
                    JOIN observations o ON o.observation_id=u.observation_id
                    WHERE COALESCE(u.label,'') LIKE ?
                       OR COALESCE(u.value,'') LIKE ?
                       OR COALESCE(u.a11y_role,'') LIKE ?
                       OR COALESCE(u.kind,'') LIKE ?
                    LIMIT ?
                    """,
                    (like, like, like, like, kind_limits["ui"]),
                ).fetchall()
            for r in rows:
                buckets["ui"].append({"kind": "ui", "episode_id": r[0], "observation_id": r[1], "score": None, "text": r[2]})

        # merge
        order = ["pattern", "chunk", "step", "episode", "ui"]
        merged: List[Dict[str, Any]] = []
        idx = {k: 0 for k in order}

        while len(merged) < limit:
            progressed = False
            for k in order:
                i = idx[k]
                if i < len(buckets[k]):
                    merged.append(buckets[k][i])
                    idx[k] += 1
                    progressed = True
                    if len(merged) >= limit:
                        break
            if not progressed:
                break

        # dedup finale
        seen = set()
        out = []
        for h in merged:
            key = (
                h.get("kind"),
                h.get("pattern_id"),
                h.get("episode_id"),
                h.get("chunk_id"),
                h.get("step_id"),
                h.get("observation_id"),
                h.get("text"),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(h)
        return out[:limit]

    # -------------------------
    # Hydration
    # -------------------------

    def get_chunk_bundle(self, *, chunk_id: int) -> Dict[str, Any]:
        """
        Return get chunk bundle.
        
        Parameters
        ----------
        chunk_id : int
            Identifier value.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with computed fields.
        """
        chunk = self.conn.execute(
            """
            SELECT chunk_id, episode_id, chunk_index, macro_goal, decision,
                   overall_success, failure_type, failing_step_index, planner_guidance, post_chunk_state,
                   first_observation_id, last_observation_id
            FROM chunks
            WHERE chunk_id=?
            """,
            (chunk_id,),
        ).fetchone()
        if chunk is None:
            raise KeyError(f"chunk_id not found: {chunk_id}")

        steps = self.conn.execute(
            """
            SELECT s.step_id, s.step_index, s.description, s.expected_outcome, s.action_type, s.command, s.pause,
                   e.success, e.confidence, e.evidence, e.failure_reason, e.fix_suggestion
            FROM steps s
            LEFT JOIN step_evals e ON e.step_id=s.step_id
            WHERE s.chunk_id=?
            ORDER BY s.step_index ASC
            """,
            (chunk_id,),
        ).fetchall()

        history = self.conn.execute(
            """
            SELECT seq_index, kind, observation_id, step_id
            FROM chunk_history
            WHERE chunk_id=?
            ORDER BY seq_index ASC
            """,
            (chunk_id,),
        ).fetchall()

        return {
            "chunk": {
                "chunk_id": chunk[0],
                "episode_id": chunk[1],
                "chunk_index": chunk[2],
                "macro_goal": chunk[3],
                "decision": chunk[4],
                "overall_success": chunk[5],
                "failure_type": chunk[6],
                "failing_step_index": chunk[7],
                "planner_guidance": chunk[8],
                "post_chunk_state": chunk[9],
                "first_observation_id": chunk[10],
                "last_observation_id": chunk[11],
            },
            "steps": [
                {
                    "step_id": r[0],
                    "step_index": r[1],
                    "description": r[2],
                    "expected_outcome": r[3],
                    "action_type": r[4],
                    "command": r[5],
                    "pause": r[6],
                    "eval": None
                    if r[7] is None
                    else {
                        "success": r[7],
                        "confidence": r[8],
                        "evidence": r[9],
                        "failure_reason": r[10],
                        "fix_suggestion": r[11],
                    },
                }
                for r in steps
            ],
            "history": [{"seq_index": r[0], "kind": r[1], "observation_id": r[2], "step_id": r[3]} for r in history],
        }

    def get_pattern_bundle(self, *, pattern_id: int) -> Dict[str, Any]:
        """
        Hydration dei pattern persistenti (anche se l'episodio originale è stato espulso).
        """
        row = self.conn.execute(
            """
            SELECT pattern_id, task_key, signature,
                   source_episode_id, source_chunk_index,
                   macro_goal, decision, overall_success,
                   failure_type, failing_step_index,
                   planner_guidance, post_chunk_state,
                   steps_json, failure_reason, fix_suggestion,
                   pre_ui_signature, post_ui_signature,
                   pre_ui_json, post_ui_json,
                   first_seen_ts_ms, last_seen_ts_ms, seen_count
            FROM patterns
            WHERE pattern_id=?
            """,
            (pattern_id,),
        ).fetchone()
        if not row:
            raise KeyError(f"pattern_id not found: {pattern_id}")

        steps = []
        try:
            import json
            steps = json.loads(row[12]) if row[12] else []
        except Exception:
            steps = []

        return {
            "pattern": {
                "pattern_id": row[0],
                "task_key": row[1],
                "signature": row[2],
                "source_episode_id": row[3],
                "source_chunk_index": row[4],
                "macro_goal": row[5],
                "decision": row[6],
                "overall_success": bool(row[7]),
                "failure_type": row[8],
                "failing_step_index": row[9],
                "planner_guidance": row[10],
                "post_chunk_state": row[11],
                "failure_reason": row[13],
                "fix_suggestion": row[14],
                "pre_ui_signature": row[15],
                "post_ui_signature": row[16],
                "pre_ui_json": row[17],
                "post_ui_json": row[18],
                "first_seen_ts_ms": row[19],
                "last_seen_ts_ms": row[20],
                "seen_count": row[21],
            },
            "steps": steps,
        }
