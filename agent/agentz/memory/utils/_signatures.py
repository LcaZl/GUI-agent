from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Set, Tuple

from agentz.constants import (
    HASH_MIN_LEN,
    MAX_SIGNATURE_LEN,
    MAX_SIGNATURE_TOKENS,
    MAX_STABLE_ITEMS,
    MAX_TOKEN_LEN,
    SHA1_SHORT_LEN,
)


def _normalize_token(tok: Any) -> str:
    """
    Normalize token.
        
        Parameters
        ----------
        tok : Any
            Function argument.
        
        Returns
        -------
        str
            Resulting string value.
        
    """
    s = str(tok or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s\-\+\/:]", "", s)
    return s.strip()


def _stable_token(tok: Any) -> str:
    """
    Process stable token.
        
        Parameters
        ----------
        tok : Any
            Function argument.
        
        Returns
        -------
        str
            Resulting string value.
        
    """
    s = str(tok or "").strip().lower()
    s = re.sub(r"\b\d+(\.\d+)?\b", " ", s)
    s = re.sub(rf"\b[a-f0-9]{{{HASH_MIN_LEN},}}\b", " ", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s\-\+\/:]", "", s)
    return s.strip()


def build_ui_signature_from_elements(
    ui_elements: Dict[str, Any],
    *,
    stable: bool = False,
    include_context: bool = True,
    max_len: int = MAX_SIGNATURE_LEN,
) -> str:
    """
    Build a UI signature from ui_elements.
    stable=False keeps more raw signal; stable=True removes numeric noise and dedups.
    """
    parts: List[str] = []
    if not ui_elements:
        return ""

    for el in ui_elements.values():
        if stable:
            role = _stable_token(getattr(el, "a11y_role", None) or getattr(el, "kind", None) or "")
            token = _stable_token(getattr(el, "label", None) or getattr(el, "value", None) or "")
            app = _stable_token(getattr(el, "app_name", None) or "")
            win = _stable_token(getattr(el, "window_name", None) or "")
        else:
            role = _normalize_token(getattr(el, "a11y_role", None) or getattr(el, "kind", None) or "")
            token = _normalize_token(getattr(el, "label", None) or getattr(el, "value", None) or "")
            app = _normalize_token(getattr(el, "app_name", None) or "")
            win = _normalize_token(getattr(el, "window_name", None) or "")

        if include_context:
            context = win or app
            if context:
                token = f"{context}:{token}" if token else ""

        if token:
            parts.append(f"{role}:{token}" if role else token)

    if stable:
        parts = sorted(set(parts))
    else:
        parts.sort()

    return "|".join(parts)[:max_len]


def build_stable_signature_from_string(sig: str, *, max_items: int = MAX_STABLE_ITEMS) -> str:
    """
    Stabilize a raw signature string (role:token|...) for similarity scoring/dedup.
    """
    if not sig:
        return ""

    out: List[str] = []
    for part in (sig or "").split("|"):
        part = part.strip()
        if not part:
            continue
        role = ""
        token = part
        if ":" in part:
            role, token = part.split(":", 1)
        role = _normalize_token(role)
        token = _stable_token(token)
        if len(token) > MAX_TOKEN_LEN:
            token = token[:MAX_TOKEN_LEN]
        if not token:
            continue
        out.append(f"{role}:{token}" if role else token)
        if len(out) >= max_items:
            break

    out = sorted(set(out))
    return "|".join(out)


def signature_tokens(sig: str, *, max_tokens: int = MAX_SIGNATURE_TOKENS) -> Set[str]:
    """
    Process signature tokens.
        
        Parameters
        ----------
        sig : str
            Function argument.
        max_tokens : Optional[int]
            Function argument.
        
        Returns
        -------
        Set[str]
            Function result.
        
    """
    if not sig:
        return set()
    toks = [t.strip() for t in sig.split("|") if t.strip()]
    out: List[str] = []
    for t in toks:
        t = _stable_token(t)
        if t:
            out.append(t)
        if len(out) >= max_tokens:
            break
    return set(out)


def normalize_text(s: Any) -> str:
    """
    Normalize text.
        
        Parameters
        ----------
        s : Any
            Function argument.
        
        Returns
        -------
        str
            Resulting string value.
        
    """
    return _normalize_token(s)


def sha1_hex(s: str) -> str:
    """
    Process sha1 hex.
        
        Parameters
        ----------
        s : str
            Function argument.
        
        Returns
        -------
        str
            Resulting string value.
        
    """
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()


def sha1_short(s: str, *, length: int = SHA1_SHORT_LEN) -> str:
    """
    Process sha1 short.
        
        Parameters
        ----------
        s : str
            Function argument.
        length : Optional[int]
            Function argument.
        
        Returns
        -------
        str
            Resulting string value.
        
    """
    return sha1_hex(s)[: max(1, int(length))]


def compute_task_key(
    *,
    instruction: str,
    os_name: str,
    desktop_env: str,
    display_server: str,
) -> Tuple[str, str]:
    """
    Stable task_key: normalize(instruction) + OS profile.
    Returns (task_key_hex, instruction_norm).
    """
    import logging
    instruction_norm = _normalize_token(instruction)
    profile = "|".join(
        [
            _normalize_token(os_name),
            _normalize_token(desktop_env),
            _normalize_token(display_server),
        ]
    )
    raw = f"{instruction_norm}||{profile}"
    task_key = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    logging.getLogger("TaskKey").info(
        "compute_task_key | instruction_norm='%s' | profile='%s' | raw='%s' | key=%s",
        instruction_norm,
        profile,
        raw,
        task_key,
    )
    return task_key, instruction_norm
