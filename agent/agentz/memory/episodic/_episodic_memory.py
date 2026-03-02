from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agentz.pydantic_models import Episode, ExecutedChunk, MemorySettings, Observation, UIElement
from agentz.constants import (
    EPISODIC_SIGNATURE_MAX_LEN,
    FINGERPRINT_SIZE,
    GRAY_B,
    GRAY_G,
    GRAY_R,
    HEX_HASH_MIN_LEN,
    OBS_UI_COMPACT_MAX_ITEMS,
    PNG_CLIP_MAX,
    PNG_CLIP_MIN,
    SCRUB_MAX_WORDS,
    STABLE_ACTION_MAX_STEPS,
    STABLE_UI_SIG_MAX_ITEMS,
    STEPS_TEXT_MAX_CHARS,
    MEMORY_PATTERN_INGEST_MIN_STEPS,
    MEMORY_PATTERN_INGEST_MIN_UI_TOKENS,
)
from ..tms._tms_online import OnlineTMS
from ..core._history_manager import HistoryManager
from ._episodic_memory_retrivial import EpisodicMemoryRetriever
from ..utils._signatures import (
    build_ui_signature_from_elements,
    build_stable_signature_from_string,
    compute_task_key,
    normalize_text,
    sha1_hex,
)

# =======================
# Helpers: image handling
# =======================

def _fingerprint_image(arr: np.ndarray, size: int = FINGERPRINT_SIZE) -> str:
    """
    Process fingerprint image.
        
        Parameters
        ----------
        arr : np.ndarray
            Function argument.
        size : Optional[int]
            Function argument.
        
        Returns
        -------
        str
            Resulting string value.
        
    """
    if arr is None:
        return ""
    a = arr
    if a.ndim == 3:
        rgb = a.astype(np.float32)
        gray = GRAY_R * rgb[..., 0] + GRAY_G * rgb[..., 1] + GRAY_B * rgb[..., 2]
    else:
        gray = a.astype(np.float32)

    h, w = gray.shape[:2]
    ys = (np.linspace(0, h - 1, size)).astype(np.int32)
    xs = (np.linspace(0, w - 1, size)).astype(np.int32)
    small = gray[np.ix_(ys, xs)]
    m = float(small.mean())
    bits = (small > m).astype(np.uint8).flatten()

    out = []
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | int(bits[i + j])
        out.append(f"{byte:02x}")
    return "".join(out)


def _encode_png_bytes(arr: np.ndarray) -> bytes:
    """
    Process encode png bytes.
        
        Parameters
        ----------
        arr : np.ndarray
            Function argument.
        
        Returns
        -------
        bytes
            Function result.
        
    """
    if arr is None:
        return b""
    a = arr
    if a.dtype != np.uint8:
        a = np.clip(a, PNG_CLIP_MIN, PNG_CLIP_MAX).astype(np.uint8)

    try:
        from PIL import Image
        from io import BytesIO

        mode = "RGB" if (a.ndim == 3 and a.shape[-1] == 3) else "L"
        img = Image.fromarray(a, mode=mode)
        bio = BytesIO()
        img.save(bio, format="PNG", optimize=True)
        return bio.getvalue()
    except Exception:
        import imageio.v3 as iio
        return iio.imwrite("<bytes>", a, extension=".png")


def _json_dumps(x: Any) -> str:
    """
    Process json dumps.
        
        Parameters
        ----------
        x : Any
            Function argument.
        
        Returns
        -------
        str
            Resulting string value.
        
    """
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def _score_value(score_obj: Optional[dict]) -> float:
    """
    Estrae uno score numerico in modo robusto.
    Supporta:
      - {"score": <float>}
      - {"metric": <float>}
      - {"metric": {"score": <float>}}
      - {"result": {"metric": <...>}}  (fallback)
    """
    if not score_obj:
        return 0.0
    try:
        if isinstance(score_obj.get("score", None), (int, float)):
            return float(score_obj["score"])
        m = score_obj.get("metric", None)
        if isinstance(m, (int, float)):
            return float(m)
        if isinstance(m, dict) and isinstance(m.get("score", None), (int, float)):
            return float(m["score"])
        # fallback su forme annidate
        r = score_obj.get("result", None)
        if isinstance(r, dict):
            mm = r.get("metric", None)
            if isinstance(mm, (int, float)):
                return float(mm)
            if isinstance(mm, dict) and isinstance(mm.get("score", None), (int, float)):
                return float(mm["score"])
    except Exception:
        return 0.0
    return 0.0


def _success_flag(score_obj: Optional[dict], status: Optional[str]) -> int:
    """
    Flag di successo per tie-breaker.
    """
    try:
        if score_obj and isinstance(score_obj.get("success", None), (bool, int)):
            return int(bool(score_obj["success"]))
    except Exception:
        pass
    if status and str(status).upper() == "DONE":
        return 1
    return 0


def _ui_signature_from_obs(obs: Observation) -> str:
    """
    Process ui signature from obs.
        
        Parameters
        ----------
        obs : Observation
            Function argument.
        
        Returns
        -------
        str
            Resulting string value.
        
    """
    return build_ui_signature_from_elements(obs.ui_elements, stable=False, include_context=True)


def _ui_compact_json_from_obs(obs: Observation, max_items: int = OBS_UI_COMPACT_MAX_ITEMS) -> str:
    """
    Process ui compact json from obs.
        
        Parameters
        ----------
        obs : Observation
            Function argument.
        max_items : Optional[int]
            Function argument.
        
        Returns
        -------
        str
            Resulting string value.
        
    """
    ui_list = list((obs.ui_elements or {}).values())
    out = []
    for el in ui_list[:max_items]:
        out.append(
            {
                "id": el.id,
                "label": el.label,
                "value": el.value,
                "role": el.a11y_role,
                "kind": el.kind,
                "visible": el.visible,
                "enabled": el.enabled,
                "actionable": el.actionable,
                "focused": el.focused,
                "selected": el.selected,
                "checked": el.checked,
                "expanded": el.expanded,
                "actions": el.actions,
                "states": el.states,
                "score": el.score,

                # NEW attribution (safe)
                "app_name": getattr(el, "app_name", None),
                "window_name": getattr(el, "window_name", None),
                "window_active": getattr(el, "window_active", None),
            }
        )
    return _json_dumps(out)


def _scrub_step_text(s: str, *, max_words: Optional[int] = SCRUB_MAX_WORDS) -> str:
    """
    Normalizza una descrizione step eliminando parti altamente variabili:
    - stringhe tra apici/virgolette/backticks
    - numeri (id, coordinate, timestamp, ecc.)
    - token esadecimali lunghi / hash-like
    - limita a max_words per stabilità
    """
    s = (s or "").strip().lower()
    if not s:
        return ""

    # rimuovi contenuti quoted (tipicamente label dinamiche)
    s = re.sub(r"(['\"`]).*?\1", " ", s)

    # rimuovi numeri (inclusi decimali)
    s = re.sub(r"\b\d+(\.\d+)?\b", " ", s)

    # rimuovi hex / hash-like
    s = re.sub(r"\b0x[0-9a-f]+\b", " ", s)
    s = re.sub(rf"\b[a-f0-9]{{{HEX_HASH_MIN_LEN},}}\b", " ", s)

    # normalizza e ripulisci punteggiatura rumorosa
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s\-\+\/:]", "", s).strip()

    if not s:
        return ""

    words = s.split()
    if max_words is not None:
        words = words[:max_words]
    return " ".join(words)


def _stable_ui_signature(ui_sig: str, *, max_items: int = STABLE_UI_SIG_MAX_ITEMS) -> str:
    """
    Stabilizza una ui_signature tipo "role:token|role:token|..."
    rimuovendo numeri e parti troppo variabili. Serve SOLO per dedup keys.
    """
    sig = build_stable_signature_from_string(ui_sig, max_items=max_items * 2)
    if not sig:
        return ""
    parts = sig.split("|")
    return "|".join(parts[:max_items])


def _signature_item_count(sig: str) -> int:
    """
    Count atomic items in a pipe-separated signature string.
    """
    if not sig:
        return 0
    return sum(1 for x in str(sig).split("|") if x.strip() != "")


def _action_type_str(x: Any) -> str:
    """
    Process action type str.
        
        Parameters
        ----------
        x : Any
            Function argument.
        
        Returns
        -------
        str
            Resulting string value.
        
    """
    if x is None:
        return ""
    if hasattr(x, "value"):
        try:
            return str(x.value).strip().lower()
        except Exception:
            pass
    return str(x).strip().lower()


def _failure_type_str(ft: Any) -> str:
    """
    Process failure type str.
        
        Parameters
        ----------
        ft : Any
            Function argument.
        
        Returns
        -------
        str
            Resulting string value.
        
    """
    if ft is None:
        return ""
    if hasattr(ft, "value"):
        try:
            return str(ft.value).strip().lower()
        except Exception:
            pass
    return str(ft).strip().lower()


def _stable_action_skeleton(steps: List[Any], *, max_steps: int = STABLE_ACTION_MAX_STEPS) -> str:
    """
    Skeleton stabile della sequenza: action_type + scrub(desc).
    Limita a max_steps per non rendere la signature sensibile alla lunghezza.
    """
    parts: List[str] = []
    for st in (steps or [])[:max_steps]:
        act = _action_type_str(getattr(st, "action_type", None))
        desc = _scrub_step_text(getattr(st, "description", "") or "")
        if act and desc:
            parts.append(f"{act}:{desc}")
        elif act:
            parts.append(act)
        elif desc:
            parts.append(desc)
    return "|".join(parts)



# =======================
# EpisodicMemory
# =======================

class EpisodicMemory:
    """
    Persistent, end-of-episode memory store.

    Nuove policy:
    - Per task_key: conserva al massimo 3 episodi "retained" (top-3 per score).
    - Estrae SEMPRE patterns (success e fail) da ogni episodio.
    - I patterns sono disaccoppiati dagli episodi: se un episodio viene espulso,
      i patterns NON vengono cancellati.
    - Dedup patterns via (task_key, signature), con seen_count/last_seen_ts_ms.
    """

    SCHEMA_VERSION = 3

    def __init__(self, settings: MemorySettings) -> None:
        """
        Initialize `EpisodicMemory` dependencies and runtime state.
        
        Parameters
        ----------
        settings : MemorySettings
            Runtime settings for this component.
        
        Returns
        -------
        None
            No return value.
        """
        self.enable_fts: bool = True
        self.store_ui_raw_json: bool = True
        self.store_tms: bool = True
        self.store_requests: bool = True
        self.logger = logging.getLogger("EpisodicMemory")

        self.settings = settings
        self.root = Path(settings.root)
        self.root.mkdir(parents=True, exist_ok=True)

        self.blobs_dir = self.root / "blobs"
        self.blobs_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.root / f"{settings.memory_name}.sqlite"

        if settings.initialize_memory and self.db_path.exists():
            self.db_path.unlink()
            for p in sorted(self.blobs_dir.glob("**/*"), reverse=True):
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    try:
                        p.rmdir()
                    except OSError:
                        pass

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.logger.info("EpisodicMemory initialized | db=%s", self.db_path)
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self.conn.execute("PRAGMA synchronous = NORMAL;")

        self.create_schema()
        self._migrate_schema_if_needed()

        self.retriever = EpisodicMemoryRetriever(self.conn)

        if self.enable_fts:
            self.ensure_fts_schema()

    def close(self) -> None:
        """
        Run close for the current workflow step.
        
        Returns
        -------
        None
            No return value.
        """
        try:
            self.conn.close()
        except Exception:
            pass

    # =======================
    # Schema + migration
    # =======================

    def create_schema(self) -> None:
        """
        Run create schema for the current workflow step.
        
        Returns
        -------
        None
            No return value.
        """
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                task_json TEXT,
                instruction TEXT NOT NULL,
                instruction_norm TEXT,
                task_key TEXT,
                status TEXT NOT NULL,
                started_ts_ms INTEGER NOT NULL,
                finished_ts_ms INTEGER,
                os_name TEXT,
                desktop_env TEXT,
                display_server TEXT,
                score_json TEXT,
                retained INTEGER DEFAULT 1
            );

            CREATE INDEX IF NOT EXISTS idx_episodes_task_key ON episodes(task_key);
            CREATE INDEX IF NOT EXISTS idx_episodes_retained ON episodes(retained);

            CREATE TABLE IF NOT EXISTS screenshots (
                screenshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id TEXT NOT NULL,
                obs_index INTEGER NOT NULL,
                observation_uid INTEGER,
                path TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                UNIQUE(episode_id, obs_index),
                FOREIGN KEY (episode_id) REFERENCES episodes(episode_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS observations (
                observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id TEXT NOT NULL,
                obs_index INTEGER NOT NULL,
                observation_uid INTEGER,
                screenshot_id INTEGER NOT NULL,
                ui_signature TEXT NOT NULL,
                ui_raw_json TEXT,
                UNIQUE(episode_id, obs_index),
                FOREIGN KEY (episode_id) REFERENCES episodes(episode_id) ON DELETE CASCADE,
                FOREIGN KEY (screenshot_id) REFERENCES screenshots(screenshot_id)
            );

            CREATE TABLE IF NOT EXISTS ui_elements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                observation_id INTEGER NOT NULL,
                el_id TEXT,
                kind TEXT,
                label TEXT,
                value TEXT,
                a11y_role TEXT,

                -- NEW: attribution
                app_name TEXT,
                window_name TEXT,
                window_active INTEGER,

                visible INTEGER,
                enabled INTEGER,
                actionable INTEGER,
                focused INTEGER,
                selected INTEGER,
                checked INTEGER,
                expanded INTEGER,
                actions_json TEXT,
                states_json TEXT,
                x1 REAL, y1 REAL, x2 REAL, y2 REAL,
                cx REAL, cy REAL,
                score REAL,
                FOREIGN KEY (observation_id) REFERENCES observations(observation_id) ON DELETE CASCADE
            );


            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                macro_goal TEXT NOT NULL,
                decision TEXT NOT NULL,
                overall_success INTEGER NOT NULL,
                failure_type TEXT,
                failing_step_index INTEGER,
                planner_guidance TEXT,
                post_chunk_state TEXT,
                first_observation_id INTEGER NOT NULL,
                last_observation_id INTEGER NOT NULL,
                UNIQUE(episode_id, chunk_index),
                FOREIGN KEY (episode_id) REFERENCES episodes(episode_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS steps (
                step_id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id INTEGER NOT NULL,
                step_index INTEGER NOT NULL,
                description TEXT NOT NULL,
                expected_outcome TEXT NOT NULL,
                action_type TEXT,
                command TEXT,
                pause REAL NOT NULL,
                UNIQUE(chunk_id, step_index),
                FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS step_evals (
                step_id INTEGER PRIMARY KEY,
                success INTEGER,
                confidence REAL,
                evidence TEXT,
                failure_reason TEXT,
                fix_suggestion TEXT,
                FOREIGN KEY (step_id) REFERENCES steps(step_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS chunk_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id INTEGER NOT NULL,
                seq_index INTEGER NOT NULL,
                kind TEXT NOT NULL,
                observation_id INTEGER,
                step_id INTEGER,
                UNIQUE(chunk_id, seq_index),
                FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                FOREIGN KEY (observation_id) REFERENCES observations(observation_id),
                FOREIGN KEY (step_id) REFERENCES steps(step_id)
            );

            -- =======================
            -- Patterns: persistono indipendentemente dagli episodi
            -- =======================

            CREATE TABLE IF NOT EXISTS patterns (
                pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_key TEXT NOT NULL,
                signature TEXT NOT NULL,

                source_episode_id TEXT,
                source_chunk_index INTEGER,

                macro_goal TEXT,
                decision TEXT,
                overall_success INTEGER,
                failure_type TEXT,
                failing_step_index INTEGER,
                planner_guidance TEXT,
                post_chunk_state TEXT,

                steps_json TEXT,
                failure_reason TEXT,
                fix_suggestion TEXT,

                pre_ui_signature TEXT,
                post_ui_signature TEXT,
                pre_ui_json TEXT,
                post_ui_json TEXT,

                first_seen_ts_ms INTEGER,
                last_seen_ts_ms INTEGER,
                seen_count INTEGER DEFAULT 1,

                UNIQUE(task_key, signature)
            );

            CREATE INDEX IF NOT EXISTS idx_patterns_task_key ON patterns(task_key);
            CREATE INDEX IF NOT EXISTS idx_patterns_success ON patterns(overall_success);
            CREATE INDEX IF NOT EXISTS idx_patterns_failure_type ON patterns(failure_type);
            """
        )
        self.conn.commit()
        cur.close()

    def _migrate_schema_if_needed(self) -> None:
        """
        Migrazione soft per DB pre-esistenti:
        aggiunge colonne mancanti in episodes.
        """
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(episodes)").fetchall()}
        alters: List[str] = []
        if "instruction_norm" not in cols:
            alters.append("ALTER TABLE episodes ADD COLUMN instruction_norm TEXT;")
        if "task_key" not in cols:
            alters.append("ALTER TABLE episodes ADD COLUMN task_key TEXT;")
        if "retained" not in cols:
            alters.append("ALTER TABLE episodes ADD COLUMN retained INTEGER DEFAULT 1;")

        if alters:
            with self.conn:
                for stmt in alters:
                    self.conn.execute(stmt)

        # indici (idempotenti)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_task_key ON episodes(task_key);")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_retained ON episodes(retained);")

        # ---- NEW: migrate ui_elements columns (soft) ----
        ui_cols = {r[1] for r in self.conn.execute("PRAGMA table_info(ui_elements)").fetchall()}
        ui_alters: List[str] = []
        if "app_name" not in ui_cols:
            ui_alters.append("ALTER TABLE ui_elements ADD COLUMN app_name TEXT;")
        if "window_name" not in ui_cols:
            ui_alters.append("ALTER TABLE ui_elements ADD COLUMN window_name TEXT;")
        if "window_active" not in ui_cols:
            ui_alters.append("ALTER TABLE ui_elements ADD COLUMN window_active INTEGER;")

        if ui_alters:
            with self.conn:
                for stmt in ui_alters:
                    self.conn.execute(stmt)

        self.conn.commit()

    # =======================
    # Ingest policy
    # =======================

    def ingest_end_of_episode(
        self,
        episode: Episode,
        history_manager: "HistoryManager",
        tms: Optional[OnlineTMS] = None,
        max_episodes_per_task: int = 3,
        pattern_step_window: Optional[int] = None,  # se None -> salva tutti gli step del chunk
    ) -> None:
        """
        Pipeline:
        1) calcola task_key
        2) salva patterns SEMPRE (dedup by signature)
        3) decide se trattenere episodio "full" (top-3 per score)
        4) se retained -> salva episodio + canonical tables + FTS, poi applica cap
        """
        try:
            obs_count = len(history_manager.observations_history or [])
            chunk_count = len(history_manager.chunks_history or [])
            step_count = len(history_manager.steps_history or [])
        except Exception:
            obs_count = 0
            chunk_count = 0
            step_count = 0

        task_key, instruction_norm = compute_task_key(
            instruction=episode.instruction,
            os_name=episode.os_name,
            desktop_env=episode.desktop_env,
            display_server=episode.display_server,
        )
        self.logger.info(
            "EpisodicMemory task_key input | instruction='%s' | os_name='%s' | desktop_env='%s' | display_server='%s'",
            episode.instruction,
            episode.os_name,
            episode.desktop_env,
            episode.display_server,
        )
        self.logger.info(
            "EpisodicMemory ingest | episode_id=%s | task_key=%s | obs=%d | chunks=%d | steps=%d",
            episode.episode_id,
            task_key,
            obs_count,
            chunk_count,
            step_count,
        )

        # 1) patterns sempre, indipendentemente da retention episodi
        self._ingest_patterns_from_history(
            task_key=task_key,
            episode=episode,
            history_manager=history_manager,
            step_window=pattern_step_window,
        )

        # 2) decisione retention
        retain = self._should_retain_episode(task_key=task_key, episode=episode, keep=max_episodes_per_task)
        if not retain:
            self.logger.info("EpisodicMemory retain=false | episode_id=%s", episode.episode_id)
            return
        self.logger.info("EpisodicMemory retain=true | episode_id=%s", episode.episode_id)

        # Idempotenza su episode_id: rimuovi l'eventuale vecchia versione (cascading)
        self._delete_episode(episode.episode_id)

        # 3) salva episodio completo
        self._upsert_episode(episode, task_key=task_key, instruction_norm=instruction_norm, retained=1)

        obs_row_by_uid = self._insert_observations(
            episode_id=episode.episode_id,
            observations=history_manager.observations_history,
        )

        self._insert_chunks_with_history(
            episode_id=episode.episode_id,
            chunks=history_manager.chunks_history,
            obs_row_by_uid=obs_row_by_uid,
        )

        if self.enable_fts:
            self.rebuild_fts_for_episode(episode_id=episode.episode_id)

        # 4) enforce cap
        self._enforce_episode_cap(task_key=task_key, keep=max_episodes_per_task)

    def _should_retain_episode(self, *, task_key: str, episode: Episode, keep: int) -> bool:
        """
        Policy: conserva top-keep episodi per task_key basandosi su:
          (score_value, success_flag, finished_ts_ms/started_ts_ms)
        """
        new_score = _score_value(episode.score)
        new_succ = _success_flag(episode.score, episode.status)
        new_ts = int(episode.finished_ts_ms or episode.started_ts_ms or 0)

        existing = self.conn.execute(
            """
            SELECT episode_id, score_json, status, finished_ts_ms, started_ts_ms
            FROM episodes
            WHERE task_key=? AND retained=1
            """,
            (task_key,),
        ).fetchall()

        if len(existing) < keep:
            return True

        # trova il worst tra i retained attuali
        def key_of(row: Tuple[Any, ...]) -> Tuple[float, int, int]:
            """
            Process key of.
                        
                        Parameters
                        ----------
                        row : Tuple[Any, ...]
                            Function argument.
                        
                        Returns
                        -------
                        Tuple[float, int, int]
                            Tuple with computed values.
                        
            """
            score_json = row[1]
            status = row[2]
            finished = row[3]
            started = row[4]
            try:
                score_obj = json.loads(score_json) if score_json else None
            except Exception:
                score_obj = None
            sc = _score_value(score_obj)
            su = _success_flag(score_obj, status)
            ts = int(finished or started or 0)
            return (sc, su, ts)

        existing_sorted = sorted(existing, key=key_of, reverse=True)
        worst = existing_sorted[-1]
        worst_sc, worst_su, worst_ts = key_of(worst)

        # retieni se "strettamente migliore" o pari ma più recente e/o più successful
        if (new_score, new_succ, new_ts) > (worst_sc, worst_su, worst_ts):
            return True
        return False

    def _enforce_episode_cap(self, *, task_key: str, keep: int) -> None:
        """
        Mantiene solo top-keep episodi retained per task_key.
        Gli altri vengono cancellati (patterns restano).
        """
        rows = self.conn.execute(
            """
            SELECT episode_id, score_json, status, finished_ts_ms, started_ts_ms
            FROM episodes
            WHERE task_key=? AND retained=1
            """,
            (task_key,),
        ).fetchall()

        if len(rows) <= keep:
            return

        def key_of(row: Tuple[Any, ...]) -> Tuple[float, int, int]:
            """
            Process key of.
                        
                        Parameters
                        ----------
                        row : Tuple[Any, ...]
                            Function argument.
                        
                        Returns
                        -------
                        Tuple[float, int, int]
                            Tuple with computed values.
                        
            """
            score_json = row[1]
            status = row[2]
            finished = row[3]
            started = row[4]
            try:
                score_obj = json.loads(score_json) if score_json else None
            except Exception:
                score_obj = None
            sc = _score_value(score_obj)
            su = _success_flag(score_obj, status)
            ts = int(finished or started or 0)
            return (sc, su, ts)

        rows_sorted = sorted(rows, key=key_of, reverse=True)
        to_keep = {r[0] for r in rows_sorted[:keep]}
        to_drop = [r[0] for r in rows_sorted[keep:] if r[0] not in to_keep]

        for ep_id in to_drop:
            self._delete_episode(ep_id)

    def _delete_episode(self, episode_id: str) -> None:
        """
        Cancella episodio e tutte le sue tabelle canonical (CASCADE).
        Pulisce anche righe FTS e blobs su disco.
        NON tocca patterns.
        """
        if not episode_id:
            return

        with self.conn:
            # pulizia FTS (non cascaded)
            if self.enable_fts:
                for tbl in ("fts_episode", "fts_chunk", "fts_step", "fts_ui"):
                    try:
                        self.conn.execute(f"DELETE FROM {tbl} WHERE episode_id=?", (episode_id,))
                    except sqlite3.OperationalError:
                        pass

            # delete canonical
            self.conn.execute("DELETE FROM episodes WHERE episode_id=?", (episode_id,))

        # cleanup blobs folder
        ep_dir = self.blobs_dir / episode_id
        if ep_dir.exists() and ep_dir.is_dir():
            try:
                shutil.rmtree(ep_dir)
            except Exception:
                pass

    # =======================
    # Patterns ingestion
    # =======================

    def _ingest_patterns_from_history(
        self,
        *,
        task_key: str,
        episode: Episode,
        history_manager: "HistoryManager",
        step_window: Optional[int] = None,
    ) -> None:
        """
        Estrae patterns chunk-level dal history_manager.
        Dedup via (task_key, signature).
        
        FIX:
        - Signature v2 stabile: NON include campi variabili (planner_guidance/post_chunk_state/failing_step_index).
        - Upsert merge-aware: aggiorna contenuti (steps_json, failure_reason, fix, ui json, ecc.).
        """
        now_ts = int(episode.finished_ts_ms or episode.started_ts_ms or 0)
        chunks_total = len(history_manager.chunks_history or [])
        patterns_upserted = 0
        skipped_counts: Dict[str, int] = {
            "no_steps": 0,
            "sparse_ui": 0,
            "weak_success": 0,
            "weak_failure": 0,
            "transient_failure": 0,
        }

        # evita chiamate ripetute dentro al loop
        if self.enable_fts:
            self.ensure_fts_schema()

        for ch_index, ch in enumerate(history_manager.chunks_history or []):
            # ----- steps + eval map (robusto) -----
            eval_by_idx: Dict[int, Any] = {}
            for ev in (getattr(ch, "steps_eval", None) or []):
                try:
                    if ev is None or ev.index is None:
                        continue
                    eval_by_idx[int(ev.index)] = ev
                except Exception:
                    continue

            steps_all = list(getattr(ch, "steps", None) or [])
            if len(steps_all) < int(MEMORY_PATTERN_INGEST_MIN_STEPS):
                skipped_counts["no_steps"] += 1
                continue

            # window per signature (contenuto: salviamo comunque steps_json completo)
            if step_window is not None and step_window > 0 and len(steps_all) > step_window:
                steps_for_sig = steps_all[-step_window:]
            else:
                steps_for_sig = steps_all

            # ----- failure info (contenuto, non chiave) -----
            fr = None
            fx = None
            if bool(getattr(ch, "overall_success", False)) is False and getattr(ch, "failing_step_index", None) is not None:
                try:
                    ev = eval_by_idx.get(int(ch.failing_step_index))
                except Exception:
                    ev = None
                if ev is not None:
                    fr = getattr(ev, "failure_reason", None)
                    fx = getattr(ev, "fix_suggestion", None)

            # ----- UI snapshots -----
            try:
                pre_ui_sig_raw = _ui_signature_from_obs(ch.first_observation)
                post_ui_sig_raw = _ui_signature_from_obs(ch.last_observation)
                pre_ui_json = _ui_compact_json_from_obs(ch.first_observation) if self.store_ui_raw_json else None
                post_ui_json = _ui_compact_json_from_obs(ch.last_observation) if self.store_ui_raw_json else None
            except Exception:
                pre_ui_sig_raw = ""
                post_ui_sig_raw = ""
                pre_ui_json = None
                post_ui_json = None

            # ----- steps_json (contenuto completo) -----
            steps_payload: List[Dict[str, Any]] = []
            for st in steps_all:
                try:
                    idx = int(getattr(st, "index", -1))
                except Exception:
                    idx = -1
                ev = eval_by_idx.get(idx) if idx >= 0 else None

                steps_payload.append(
                    {
                        "index": idx,
                        "description": getattr(st, "description", None),
                        "expected_outcome": getattr(st, "expected_outcome", None),
                        "action_type": getattr(st, "action_type", None),
                        "pause": float(getattr(st, "pause", 0.0) or 0.0),
                        "success": None if ev is None else bool(getattr(ev, "success", False)),
                        "failure_reason": None if ev is None else getattr(ev, "failure_reason", None),
                        "fix_suggestion": None if ev is None else getattr(ev, "fix_suggestion", None),
                    }
                )
            steps_json = _json_dumps(steps_payload)

            # ----- Signature v2 (stabile) -----
            overall_success_flag = int(bool(getattr(ch, "overall_success", False)))

            failure_type = _failure_type_str(getattr(ch, "failure_type", None))

            # UI key (stabilizzata + hash)
            pre_ui_sig_stable = _stable_ui_signature(pre_ui_sig_raw)
            if _signature_item_count(pre_ui_sig_stable) < int(MEMORY_PATTERN_INGEST_MIN_UI_TOKENS):
                skipped_counts["sparse_ui"] += 1
                continue
            pre_ui_key = sha1_hex(pre_ui_sig_stable)  # pieno, così collisioni bassissime

            # sequence key (stabile + hash)
            # ulteriore cap per evitare sensibilità alla lunghezza
            seq_skeleton = _stable_action_skeleton(steps_for_sig, max_steps=STABLE_ACTION_MAX_STEPS)
            seq_key = sha1_hex(seq_skeleton)

            # Restrictive ingestion policy:
            # - success patterns need at least one observed successful step when evaluations exist.
            # - failure patterns require actionable diagnostics and skip transient noise.
            has_eval = any(s.get("success") is not None for s in steps_payload)
            success_eval_count = sum(1 for s in steps_payload if s.get("success") is True)
            if overall_success_flag == 1:
                if has_eval and success_eval_count <= 0:
                    skipped_counts["weak_success"] += 1
                    continue
            else:
                failure_type_upper = str(failure_type or "").strip().upper()
                has_failure_detail = bool((fr or "").strip()) or bool((fx or "").strip()) or (
                    getattr(ch, "failing_step_index", None) is not None
                )
                if not has_failure_detail and failure_type_upper == "":
                    skipped_counts["weak_failure"] += 1
                    continue
                if failure_type_upper in {"ENV_LIMITATION", "UI_NOT_READY"} and not bool((fx or "").strip()):
                    skipped_counts["transient_failure"] += 1
                    continue

            # failing step key (solo per fail)
            fail_step_key = ""
            if overall_success_flag == 0:
                failing_idx = getattr(ch, "failing_step_index", None)
                failing_step = None
                if failing_idx is not None:
                    try:
                        fidx = int(failing_idx)
                        for st in steps_all:
                            try:
                                if int(getattr(st, "index", -1)) == fidx:
                                    failing_step = st
                                    break
                            except Exception:
                                continue
                    except Exception:
                        failing_step = None

                if failing_step is not None:
                    fs = _stable_action_skeleton([failing_step], max_steps=1)
                    fail_step_key = sha1_hex(fs)

            sig_material = "|".join(
                [
                    "v2",
                    task_key,
                    pre_ui_key,
                    seq_key,
                    str(overall_success_flag),
                    failure_type,
                    fail_step_key,
                ]
            )
            signature = "v2:" + sha1_hex(sig_material)

            # ----- Upsert merge-aware -----
            with self.conn:
                self.conn.execute(
                    """
                    INSERT INTO patterns(
                        task_key, signature,
                        source_episode_id, source_chunk_index,
                        macro_goal, decision, overall_success,
                        failure_type, failing_step_index,
                        planner_guidance, post_chunk_state,
                        steps_json, failure_reason, fix_suggestion,
                        pre_ui_signature, post_ui_signature,
                        pre_ui_json, post_ui_json,
                        first_seen_ts_ms, last_seen_ts_ms, seen_count
                    )
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(task_key, signature) DO UPDATE SET
                        -- tracking
                        last_seen_ts_ms = excluded.last_seen_ts_ms,
                        seen_count      = patterns.seen_count + 1,

                        -- debug provenance: tieni l'ultimo
                        source_episode_id  = excluded.source_episode_id,
                        source_chunk_index = excluded.source_chunk_index,

                        -- campi core (dovrebbero essere coerenti con signature v2, ma meglio sync)
                        macro_goal         = CASE
                                            WHEN excluded.macro_goal IS NOT NULL AND excluded.macro_goal <> ''
                                            THEN excluded.macro_goal
                                            ELSE patterns.macro_goal
                                            END,
                        decision           = CASE
                                            WHEN excluded.decision IS NOT NULL AND excluded.decision <> ''
                                            THEN excluded.decision
                                            ELSE patterns.decision
                                            END,
                        overall_success    = excluded.overall_success,
                        failure_type       = excluded.failure_type,
                        failing_step_index = excluded.failing_step_index,

                        -- campi “variabili”: aggiorna solo se non vuoti
                        planner_guidance   = CASE
                                            WHEN excluded.planner_guidance IS NOT NULL AND excluded.planner_guidance <> ''
                                            THEN excluded.planner_guidance
                                            ELSE patterns.planner_guidance
                                            END,
                        post_chunk_state   = CASE
                                            WHEN excluded.post_chunk_state IS NOT NULL AND excluded.post_chunk_state <> ''
                                            THEN excluded.post_chunk_state
                                            ELSE patterns.post_chunk_state
                                            END,

                        -- steps_json: scegli la versione più ricca (più lunga)
                        steps_json         = CASE
                                            WHEN excluded.steps_json IS NOT NULL
                                                    AND length(excluded.steps_json) > length(COALESCE(patterns.steps_json,''))
                                            THEN excluded.steps_json
                                            ELSE patterns.steps_json
                                            END,

                        -- failure/fix: preferisci non-vuoti
                        failure_reason     = CASE
                                            WHEN excluded.failure_reason IS NOT NULL AND excluded.failure_reason <> ''
                                            THEN excluded.failure_reason
                                            ELSE patterns.failure_reason
                                            END,
                        fix_suggestion     = CASE
                                            WHEN excluded.fix_suggestion IS NOT NULL AND excluded.fix_suggestion <> ''
                                            THEN excluded.fix_suggestion
                                            ELSE patterns.fix_suggestion
                                            END,

                        -- UI: aggiorna solo se presenti
                        pre_ui_signature   = CASE
                                            WHEN excluded.pre_ui_signature IS NOT NULL AND excluded.pre_ui_signature <> ''
                                            THEN excluded.pre_ui_signature
                                            ELSE patterns.pre_ui_signature
                                            END,
                        post_ui_signature  = CASE
                                            WHEN excluded.post_ui_signature IS NOT NULL AND excluded.post_ui_signature <> ''
                                            THEN excluded.post_ui_signature
                                            ELSE patterns.post_ui_signature
                                            END,
                        pre_ui_json        = CASE
                                            WHEN excluded.pre_ui_json IS NOT NULL AND excluded.pre_ui_json <> ''
                                            THEN excluded.pre_ui_json
                                            ELSE patterns.pre_ui_json
                                            END,
                        post_ui_json       = CASE
                                            WHEN excluded.post_ui_json IS NOT NULL AND excluded.post_ui_json <> ''
                                            THEN excluded.post_ui_json
                                            ELSE patterns.post_ui_json
                                            END,

                        -- first_seen: preserva il più vecchio
                        first_seen_ts_ms   = CASE
                                            WHEN patterns.first_seen_ts_ms IS NULL OR patterns.first_seen_ts_ms = 0
                                            THEN excluded.first_seen_ts_ms
                                            ELSE patterns.first_seen_ts_ms
                                            END
                    """,
                    (
                        task_key,
                        signature,
                        episode.episode_id,
                        int(ch_index),
                        getattr(ch, "macro_goal", None),
                        getattr(ch, "decision", None),
                        overall_success_flag,
                        failure_type if failure_type != "" else None,
                        getattr(ch, "failing_step_index", None),
                        getattr(ch, "planner_guidance", None),
                        getattr(ch, "post_chunk_state", None),
                        steps_json,
                        fr,
                        fx,
                        pre_ui_sig_raw,
                        post_ui_sig_raw,
                        pre_ui_json,
                        post_ui_json,
                        now_ts,
                        now_ts,
                        1,
                    ),
                )

                # FTS pattern refresh (idempotente)
                if self.enable_fts:
                    row = self.conn.execute(
                        "SELECT pattern_id FROM patterns WHERE task_key=? AND signature=?",
                        (task_key, signature),
                    ).fetchone()
                    if row:
                        self._upsert_fts_pattern(pattern_id=int(row[0]))
            patterns_upserted += 1

        self.logger.info(
            "Patterns ingested | episode_id=%s | task_key=%s | chunks=%d | upserted=%d",
            episode.episode_id,
            task_key,
            chunks_total,
            patterns_upserted,
        )
        if any(v > 0 for v in skipped_counts.values()):
            self.logger.info(
                "Patterns ingest filter | no_steps=%d | sparse_ui=%d | weak_success=%d | weak_failure=%d | transient_failure=%d",
                int(skipped_counts["no_steps"]),
                int(skipped_counts["sparse_ui"]),
                int(skipped_counts["weak_success"]),
                int(skipped_counts["weak_failure"]),
                int(skipped_counts["transient_failure"]),
            )


    def _upsert_fts_pattern(self, *, pattern_id: int) -> None:
        """
        (Ri)inserisce una riga in fts_pattern per un pattern_id.
        """
        pat = self.conn.execute(
            """
            SELECT pattern_id, task_key, macro_goal, planner_guidance, post_chunk_state,
                   steps_json, failure_reason, fix_suggestion, pre_ui_json, post_ui_json
            FROM patterns
            WHERE pattern_id=?
            """,
            (pattern_id,),
        ).fetchone()
        if not pat:
            return

        pid, task_key, goal, guide, post_state, steps_json, fr, fx, pre_ui_json, post_ui_json = pat

        # steps_text: concat description/expected_outcome (no comandi)
        steps_text = ""
        try:
            arr = json.loads(steps_json) if steps_json else []
            parts = []
            for s in arr:
                parts.append(str(s.get("description", "")))
                parts.append(str(s.get("expected_outcome", "")))
            steps_text = " ".join([p for p in parts if p.strip()])
            if STEPS_TEXT_MAX_CHARS is not None:
                steps_text = steps_text[:STEPS_TEXT_MAX_CHARS]
        except Exception:
            steps_text = ""

        ui_text = ""
        try:
            ui_text = " ".join(
                [
                    normalize_text(pre_ui_json or ""),
                    normalize_text(post_ui_json or ""),
                ]
            )
            if STEPS_TEXT_MAX_CHARS is not None:
                ui_text = ui_text[:STEPS_TEXT_MAX_CHARS]
        except Exception:
            ui_text = ""

        # Semplice strategia: delete+insert (evita problemi se schema FTS cambia)
        self.conn.execute("DELETE FROM fts_pattern WHERE pattern_id=?", (int(pid),))
        self.conn.execute(
            """
            INSERT INTO fts_pattern(
                pattern_id, task_key,
                macro_goal, planner_guidance, post_chunk_state,
                steps_text, failure_reason, fix_suggestion, ui_text
            )
            VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (
                int(pid),
                str(task_key or ""),
                str(goal or ""),
                str(guide or ""),
                str(post_state or ""),
                str(steps_text or ""),
                str(fr or ""),
                str(fx or ""),
                str(ui_text or ""),
            ),
        )

    # =======================
    # Canonical inserts (episodi retained)
    # =======================

    def _upsert_episode(self, episode: Episode, *, task_key: str, instruction_norm: str, retained: int = 1) -> None:
        """
        Process upsert episode.
        
        Parameters
        ----------
        episode : Episode
            Function argument.
        task_key : str
            Function argument.
        instruction_norm : str
            Function argument.
        retained : Optional[int]
            Function argument.
        
        Returns
        -------
        None
            No return value.
        
        """
        task_json = None if episode.task is None else _json_dumps(episode.task)
        score_json = None if episode.score is None else _json_dumps(episode.score)
        finished = None if episode.finished_ts_ms is None else int(episode.finished_ts_ms)

        self.conn.execute(
            """
            INSERT INTO episodes(
                episode_id, task_json, instruction, instruction_norm, task_key, status,
                started_ts_ms, finished_ts_ms,
                os_name, desktop_env, display_server, score_json, retained
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(episode_id) DO UPDATE SET
                task_json=excluded.task_json,
                instruction=excluded.instruction,
                instruction_norm=excluded.instruction_norm,
                task_key=excluded.task_key,
                status=excluded.status,
                started_ts_ms=excluded.started_ts_ms,
                finished_ts_ms=excluded.finished_ts_ms,
                os_name=excluded.os_name,
                desktop_env=excluded.desktop_env,
                display_server=excluded.display_server,
                score_json=excluded.score_json,
                retained=excluded.retained
            """,
            (
                episode.episode_id,
                task_json,
                episode.instruction,
                instruction_norm,
                task_key,
                episode.status,
                int(episode.started_ts_ms),
                finished,
                episode.os_name,
                episode.desktop_env,
                episode.display_server,
                score_json,
                int(retained),
            ),
        )
        self.conn.commit()

    def _insert_observations(self, episode_id: str, observations: list[Observation]) -> Dict[int, int]:
        """
        Process insert observations.
        
        Parameters
        ----------
        episode_id : str
            Identifier value.
        observations : list[Observation]
            Function argument.
        
        Returns
        -------
        Dict[int, int]
            Dictionary with computed fields.
        
        """
        obs_row_by_uid: Dict[int, int] = {}
        ep_dir = self.blobs_dir / episode_id
        ep_dir.mkdir(parents=True, exist_ok=True)
        inserted = 0

        with self.conn:
            for obs_index, obs in enumerate(observations):
                uid = obs.observation_id
                img = obs.screenshot

                if uid is None or img is None:
                    continue

                png = _encode_png_bytes(img)
                sha = hashlib.sha256(png).hexdigest()
                fp = _fingerprint_image(img)

                path = ep_dir / f"obs_{obs_index:05d}.png"
                path.write_bytes(png)

                cur = self.conn.execute(
                    """
                    INSERT INTO screenshots(
                        episode_id, obs_index, observation_uid,
                        path, width, height, sha256, fingerprint
                    )
                    VALUES(?,?,?,?,?,?,?,?)
                    """,
                    (
                        episode_id,
                        obs_index,
                        int(uid),
                        f"{episode_id}/{path.name}",
                        int(img.shape[1]),
                        int(img.shape[0]),
                        sha,
                        fp,
                    ),
                )
                screenshot_id = int(cur.lastrowid)

                ui_dict = obs.ui_elements
                ui_list = list(ui_dict.values())

                ui_sig = build_ui_signature_from_elements(
                    ui_dict,
                    stable=False,
                    include_context=False,
                    max_len=EPISODIC_SIGNATURE_MAX_LEN or 10_000_000,
                )

                ui_raw_json = None
                if self.store_ui_raw_json:
                    ui_raw_json = _json_dumps(
                        [{"id": el.id, "label": el.label, "value": el.value, "role": el.a11y_role, "kind": el.kind} for el in ui_list]
                    )

                cur2 = self.conn.execute(
                    """
                    INSERT INTO observations(
                        episode_id, obs_index, observation_uid,
                        screenshot_id, ui_signature, ui_raw_json
                    )
                    VALUES(?,?,?,?,?,?)
                    """,
                    (
                        episode_id,
                        obs_index,
                        int(uid),
                        screenshot_id,
                        ui_sig,
                        ui_raw_json,
                    ),
                )
                obs_row_id = int(cur2.lastrowid)
                obs_row_by_uid[int(uid)] = obs_row_id
                inserted += 1

                ui_rows = [self._ui_el_to_row(obs_row_id, el) for el in ui_list]
                if ui_rows:
                    self.conn.executemany(
                        """
                        INSERT INTO ui_elements(
                            observation_id,
                            el_id, kind, label, value, a11y_role,
                            app_name, window_name, window_active,
                            visible, enabled, actionable, focused, selected, checked, expanded,
                            actions_json, states_json,
                            x1, y1, x2, y2, cx, cy,
                            score
                        )
                        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        ui_rows,
                    )

        self.logger.info(
            "EpisodicMemory observations stored | episode_id=%s | total=%d | inserted=%d",
            episode_id,
            len(observations or []),
            inserted,
        )
        return obs_row_by_uid

    def _insert_chunks_with_history(self, episode_id: str, chunks: list[ExecutedChunk], obs_row_by_uid: Dict[int, int]) -> None:
        """
        Process insert chunks with history.
        
        Parameters
        ----------
        episode_id : str
            Identifier value.
        chunks : list[ExecutedChunk]
            Function argument.
        obs_row_by_uid : Dict[int, int]
            Function argument.
        
        Returns
        -------
        None
            No return value.
        
        """
        inserted = 0
        with self.conn:
            for chunk_index, ch in enumerate(chunks):
                first_uid = ch.first_observation.observation_id
                last_uid = ch.last_observation.observation_id

                if first_uid is None or last_uid is None:
                    continue
                if first_uid not in obs_row_by_uid or last_uid not in obs_row_by_uid:
                    continue

                first_obs_id = obs_row_by_uid[first_uid]
                last_obs_id = obs_row_by_uid[last_uid]

                ft = getattr(ch, "failure_type", None)
                if ft is not None and hasattr(ft, "value"):
                    ft = ft.value

                cur = self.conn.execute(
                    """
                    INSERT INTO chunks(
                        episode_id, chunk_index,
                        macro_goal, decision,
                        overall_success, failure_type,
                        failing_step_index, planner_guidance,
                        post_chunk_state,
                        first_observation_id, last_observation_id
                    )
                    VALUES(?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        episode_id,
                        chunk_index,
                        ch.macro_goal,
                        ch.decision,
                        int(bool(ch.overall_success)),
                        ft,
                        ch.failing_step_index,
                        ch.planner_guidance,
                        ch.post_chunk_state,
                        first_obs_id,
                        last_obs_id,
                    ),
                )
                chunk_id = int(cur.lastrowid)
                inserted += 1

                step_id_by_index: Dict[int, int] = {}
                eval_by_index = {int(ev.index): ev for ev in ch.steps_eval}

                for st in ch.steps:
                    st_i = st.index
                    cur2 = self.conn.execute(
                        """
                        INSERT INTO steps(
                            chunk_id, step_index,
                            description, expected_outcome,
                            action_type, command, pause
                        )
                        VALUES(?,?,?,?,?,?,?)
                        """,
                        (
                            chunk_id,
                            st_i,
                            st.description,
                            st.expected_outcome,
                            st.action_type,
                            st.command,
                            float(st.pause),
                        ),
                    )
                    step_id = int(cur2.lastrowid)
                    step_id_by_index[st_i] = step_id

                    ev = eval_by_index.get(st_i)
                    if ev is not None:
                        self.conn.execute(
                            """
                            INSERT INTO step_evals(
                                step_id, success, confidence,
                                evidence, failure_reason, fix_suggestion
                            )
                            VALUES(?,?,?,?,?,?)
                            """,
                            (
                                step_id,
                                int(bool(ev.success)),
                                float(ev.confidence),
                                ev.evidence,
                                ev.failure_reason,
                                ev.fix_suggestion,
                            ),
                        )

                for seq_i, item in enumerate(ch.history):
                    if hasattr(item, "observation_id"):
                        uid = item.observation_id
                        if uid is None or uid not in obs_row_by_uid:
                            continue
                        self.conn.execute(
                            """
                            INSERT INTO chunk_history(chunk_id, seq_index, kind, observation_id, step_id)
                            VALUES(?,?,?,?,NULL)
                            """,
                            (chunk_id, int(seq_i), "observation", obs_row_by_uid[int(uid)]),
                        )
                    else:
                        st_i = item.index
                        if st_i not in step_id_by_index:
                            continue
                        self.conn.execute(
                            """
                            INSERT INTO chunk_history(chunk_id, seq_index, kind, observation_id, step_id)
                            VALUES(?,?,?,NULL,?)
                            """,
                            (chunk_id, int(seq_i), "step", step_id_by_index[st_i]),
                        )
        self.logger.info(
            "EpisodicMemory chunks stored | episode_id=%s | total=%d | inserted=%d",
            episode_id,
            len(chunks or []),
            inserted,
        )

    def _ui_el_to_row(self, observation_id: int, el: UIElement) -> Tuple[Any, ...]:
        """
        Process ui el to row.
        
        Parameters
        ----------
        observation_id : int
            Identifier value.
        el : UIElement
            Function argument.
        
        Returns
        -------
        Tuple[Any, ...]
            Tuple with computed values.
        
        """
        bb = el.bb_coords
        cc = el.center_coords

        def b(v: Any) -> Optional[int]:
            """
            Process b.
                        
                        Parameters
                        ----------
                        v : Any
                            Function argument.
                        
                        Returns
                        -------
                        Optional[int]
                            Function result.
                        
            """
            if v is None:
                return None
            return int(bool(v))

        actions = el.actions
        states = el.states

        # NEW attribution (safe)
        app_name = getattr(el, "app_name", None)
        window_name = getattr(el, "window_name", None)
        window_active = getattr(el, "window_active", None)

        return (
            observation_id,
            el.id,
            el.kind,
            el.label,
            el.value,
            el.a11y_role,

            app_name,
            window_name,
            b(window_active),

            b(el.visible),
            b(el.enabled),
            b(el.actionable),
            b(el.focused),
            b(el.selected),
            b(el.checked),
            b(el.expanded),
            None if actions is None else _json_dumps(actions),
            None if states is None else _json_dumps(states),
            None if bb is None else float(bb.x_1),
            None if bb is None else float(bb.y_1),
            None if bb is None else float(bb.x_2),
            None if bb is None else float(bb.y_2),
            None if cc is None else float(cc.x),
            None if cc is None else float(cc.y),
            el.score,
        )


    # =======================
    # FTS schema
    # =======================

    def ensure_fts_schema(self) -> None:
        """
        Run ensure fts schema for the current workflow step.
        
        Returns
        -------
        None
            No return value.
        """
        if not getattr(self, "enable_fts", False):
            return
        if not self._fts5_available():
            raise RuntimeError("SQLite build does not support FTS5")

        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_episode USING fts5(
                episode_id UNINDEXED,
                instruction,
                tokenize='unicode61'
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunk USING fts5(
                episode_id UNINDEXED,
                chunk_id UNINDEXED,
                macro_goal,
                planner_guidance,
                post_chunk_state,
                tokenize='unicode61'
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS fts_step USING fts5(
                episode_id UNINDEXED,
                chunk_id UNINDEXED,
                step_id UNINDEXED,
                description,
                expected_outcome,
                evidence,
                fix_suggestion,
                tokenize='unicode61'
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS fts_ui USING fts5(
                episode_id UNINDEXED,
                observation_id UNINDEXED,
                label,
                value,
                a11y_role,
                kind,
                app_name,
                window_name,
                tokenize='unicode61'
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS fts_pattern USING fts5(
                pattern_id UNINDEXED,
                task_key UNINDEXED,
                macro_goal,
                planner_guidance,
                post_chunk_state,
                steps_text,
                failure_reason,
                fix_suggestion,
                ui_text,
                tokenize='unicode61'
            );
            """
        )
        self.conn.commit()
        cur.close()

    def rebuild_fts_for_episode(self, episode_id: str) -> None:
        """
        Process rebuild fts for episode.
        
        Parameters
        ----------
        episode_id : str
            Identifier value.
        
        Returns
        -------
        None
            No return value.
        
        """
        if not self.enable_fts:
            return

        self.ensure_fts_schema()

        with self.conn:
            self.conn.execute("DELETE FROM fts_episode WHERE episode_id=?", (episode_id,))
            self.conn.execute("DELETE FROM fts_chunk WHERE episode_id=?", (episode_id,))
            self.conn.execute("DELETE FROM fts_step WHERE episode_id=?", (episode_id,))
            self.conn.execute("DELETE FROM fts_ui WHERE episode_id=?", (episode_id,))

            row = self.conn.execute(
                "SELECT instruction FROM episodes WHERE episode_id=?",
                (episode_id,),
            ).fetchone()
            if row and row[0]:
                self.conn.execute(
                    "INSERT INTO fts_episode(episode_id, instruction) VALUES(?,?)",
                    (episode_id, row[0]),
                )

            for (chunk_id, macro_goal, planner_guidance, post_chunk_state) in self.conn.execute(
                """
                SELECT chunk_id,
                       macro_goal,
                       COALESCE(planner_guidance,''),
                       COALESCE(post_chunk_state,'')
                FROM chunks
                WHERE episode_id=?
                """,
                (episode_id,),
            ):
                self.conn.execute(
                    """
                    INSERT INTO fts_chunk(episode_id, chunk_id, macro_goal, planner_guidance, post_chunk_state)
                    VALUES(?,?,?,?,?)
                    """,
                    (episode_id, chunk_id, macro_goal or "", planner_guidance or "", post_chunk_state or ""),
                )

            for (chunk_id, step_id, desc, exp, evidence, fix) in self.conn.execute(
                """
                SELECT s.chunk_id,
                       s.step_id,
                       s.description,
                       s.expected_outcome,
                       COALESCE(e.evidence,''),
                       COALESCE(e.fix_suggestion,'')
                FROM steps s
                JOIN chunks c ON c.chunk_id=s.chunk_id
                LEFT JOIN step_evals e ON e.step_id=s.step_id
                WHERE c.episode_id=?
                """,
                (episode_id,),
            ):
                self.conn.execute(
                    """
                    INSERT INTO fts_step(
                        episode_id, chunk_id, step_id,
                        description, expected_outcome, evidence, fix_suggestion
                    )
                    VALUES(?,?,?,?,?,?,?)
                    """,
                    (episode_id, chunk_id, step_id, desc or "", exp or "", evidence or "", fix or ""),
                )

            for (observation_id, label, value, role, kind, app_name, window_name) in self.conn.execute(
                """
                SELECT o.observation_id,
                    COALESCE(u.label,''),
                    COALESCE(u.value,''),
                    COALESCE(u.a11y_role,''),
                    COALESCE(u.kind,''),
                    COALESCE(u.app_name,''),
                    COALESCE(u.window_name,'')
                FROM observations o
                JOIN ui_elements u ON u.observation_id=o.observation_id
                WHERE o.episode_id=?
                """,
                (episode_id,),
            ):

                self.conn.execute(
                    """
                    INSERT INTO fts_ui(episode_id, observation_id, label, value, a11y_role, kind, app_name, window_name)
                    VALUES(?,?,?,?,?,?,?,?)
                    """,
                    (episode_id, observation_id, label, value, role, kind, app_name, window_name),
                )


    def _fts5_available(self) -> bool:
        """
        Process fts5 available.
        
        Returns
        -------
        bool
            True when the condition is satisfied, otherwise False.
        
        """
        try:
            self.conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS __fts5_probe USING fts5(x);")
            self.conn.execute("DROP TABLE IF EXISTS __fts5_probe;")
            return True
        except sqlite3.OperationalError:
            return False
