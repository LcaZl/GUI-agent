from __future__ import annotations

from typing import Set

from agentz.constants import MIN_WORD_LEN


def jaccard_set(a: Set[str], b: Set[str]) -> float:
    """
    Process jaccard set.
        
        Parameters
        ----------
        a : Set[str]
            Function argument.
        b : Set[str]
            Function argument.
        
        Returns
        -------
        float
            Floating-point result value.
        
    """
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter) / float(union) if union else 0.0


def jaccard_words(a: str, b: str, *, min_len: int = MIN_WORD_LEN) -> float:
    """
    Process jaccard words.
        
        Parameters
        ----------
        a : str
            Function argument.
        b : str
            Function argument.
        min_len : Optional[int]
            Function argument.
        
        Returns
        -------
        float
            Floating-point result value.
        
    """
    sa = set(w for w in (a or "").split() if len(w) >= min_len)
    sb = set(w for w in (b or "").split() if len(w) >= min_len)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return float(inter) / float(union) if union else 0.0
