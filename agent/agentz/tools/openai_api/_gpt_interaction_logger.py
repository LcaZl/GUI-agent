import os
import csv
import json
import logging
import threading
from datetime import datetime
from typing import Any, Dict, Optional

import tiktoken

from agentz.constants import (
    API_TOKENS_DEFAULT,
    ELAPSED_TIME_DEFAULT,
    ELAPSED_TIME_ROUND_DIGITS,
    LOG_FIELDNAMES,
    PROMPT_SNIPPET_LEN,
)


class GPTInteractionLogger:
    """
    CSV logger for GPTClient interactions.
    - Thread-safe CSV appends
    - Ensures header exists and matches LOG_FIELDNAMES
    - Computes prompt token estimate for text (tiktoken)
    - Extracts usage from API responses (when available)
    - Logs image metadata without persisting image content
    """

    BASE_ENCODING_MODEL = "cl100k_base"

    def __init__(self, log_file: str, encoding_model_name: str, logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize class dependencies and runtime state.
        
        Parameters
        ----------
        log_file : str
            Function argument.
        encoding_model_name : str
            Function argument.
        logger : Optional[logging.Logger]
            Function argument.
        
        Returns
        -------
        None
            No return value.
        """
        self.log_file = log_file
        self.encoding_model_name = encoding_model_name
        self.logger = logger or logging.getLogger("GPTInteractionLogger")
        self._lock = threading.Lock()
        self._ensure_header()

    # -----------------------------
    # Header / file management
    # -----------------------------

    def _ensure_header(self) -> None:
        """
        Ensure header.
        
        Returns
        -------
        None
            No return value.
        
        """
        needs_header = (not os.path.exists(self.log_file)) or (os.path.getsize(self.log_file) == 0)

        if not needs_header:
            try:
                with open(self.log_file, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                existing = [h.strip() for h in first_line.split(",")] if first_line else []
                if existing != LOG_FIELDNAMES:
                    needs_header = True
            except Exception:
                needs_header = True

        if needs_header:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True) if os.path.dirname(self.log_file) else None
            with open(self.log_file, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
                writer.writeheader()

    # -----------------------------
    # Token counting (text)
    # -----------------------------

    def count_tokens(self, text: str, model_name: Optional[str] = None) -> int:
        """
        Process count tokens.
        
        Parameters
        ----------
        text : str
            Input text.
        model_name : Optional[str]
            Function argument.
        
        Returns
        -------
        int
            Integer result value.
        
        """
        model = model_name or self.encoding_model_name
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding(self.BASE_ENCODING_MODEL)
        return len(encoding.encode(text or ""))

    # -----------------------------
    # API usage extraction
    # -----------------------------

    @staticmethod
    def _extract_api_usage(response: Any) -> Dict[str, int]:
        """
        Best-effort extraction from OpenAI/Azure SDK response usage.
        Returns -1 when unavailable.
        """
        api_prompt_tokens = API_TOKENS_DEFAULT
        api_completion_tokens = API_TOKENS_DEFAULT
        api_total_tokens = API_TOKENS_DEFAULT
        api_cached_tokens = API_TOKENS_DEFAULT

        try:
            u = getattr(response, "usage", None)
            if u is None:
                return {
                    "api_prompt_tokens": api_prompt_tokens,
                    "api_completion_tokens": api_completion_tokens,
                    "api_total_tokens": api_total_tokens,
                    "api_cached_tokens": api_cached_tokens,
                }

            api_prompt_tokens = int(getattr(u, "prompt_tokens", -1) or -1)
            api_completion_tokens = int(getattr(u, "completion_tokens", -1) or -1)
            api_total_tokens = int(getattr(u, "total_tokens", -1) or -1)

            # cached tokens often appear under prompt_tokens_details.cached_tokens
            ptd = getattr(u, "prompt_tokens_details", None)
            if ptd is not None:
                api_cached_tokens = int(getattr(ptd, "cached_tokens", -1) or -1)

        except Exception:
            pass

        return {
            "api_prompt_tokens": api_prompt_tokens,
            "api_completion_tokens": api_completion_tokens,
            "api_total_tokens": api_total_tokens,
            "api_cached_tokens": api_cached_tokens,
        }

    # -----------------------------
    # Main logging entrypoint
    # -----------------------------

    def log_interaction(
        self,
        *,
        prompt: str,
        response: Any,
        model_name: str,
        tool_name: str = "",
        tool_args: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error: Optional[str] = None,
        elapsed_time: Optional[float] = None,
        image_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Ensure header exists even if file got truncated mid-run
        """
        Process log interaction.
        
        Parameters
        ----------
        prompt : str
            Function argument.
        response : Any
            Response payload.
        model_name : str
            Function argument.
        tool_name : Optional[str]
            Function argument.
        tool_args : Optional[Dict[str, Any]]
            Function argument.
        success : Optional[bool]
            Function argument.
        error : Optional[str]
            Function argument.
        elapsed_time : Optional[float]
            Function argument.
        image_meta : Optional[Dict[str, Any]]
            Function argument.
        
        Returns
        -------
        None
            No return value.
        
        """
        if (not os.path.exists(self.log_file)) or (os.path.getsize(self.log_file) == 0):
            self._ensure_header()

        # Prompt sanitization
        prompt_clean = (prompt or "").replace("\n", " ").replace("\r", " ").strip()

        # Error sanitization
        error_clean = ""
        if error:
            error_clean = str(error).replace("\n", " ").replace("\r", " ").strip()

        # Tool args serialization
        tool_args_str = ""
        if tool_args is not None:
            try:
                tool_args_str = json.dumps(tool_args, ensure_ascii=False)
            except Exception:
                tool_args_str = str(tool_args)

        # Token counts (text prompt estimation)
        prompt_tokens_est = self.count_tokens(prompt_clean, model_name=model_name)

        # Completion tokens from response usage (best-effort; historically used in your class)
        completion_tokens_est = 0
        try:
            if response is not None and getattr(response, "usage", None) is not None:
                completion_tokens_est = int(getattr(response.usage, "completion_tokens", 0) or 0)
        except Exception:
            completion_tokens_est = 0

        # API usage (actual if present)
        usage = self._extract_api_usage(response)

        # Image meta defaults
        image_count = 0
        image_detail = ""
        image_total_bytes = 0
        image_est_low = -1
        image_est_high = -1

        if image_meta:
            image_count = int(image_meta.get("count", 0) or 0)
            image_detail = str(image_meta.get("detail", "") or "")
            image_total_bytes = int(image_meta.get("total_bytes", 0) or 0)
            image_est_low = int(image_meta.get("est_tokens_low", -1) or -1)
            image_est_high = int(image_meta.get("est_tokens_high", -1) or -1)

        row = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt_clean[:PROMPT_SNIPPET_LEN],
            "tool_name": tool_name or "",
            "tool_args": tool_args_str,
            "model_name": model_name,
            "token_count_prompt": prompt_tokens_est,
            "token_count_completion": completion_tokens_est,
            "success": success,
            "error_message": error_clean,
            "elapsed_time_sec": round(elapsed_time, ELAPSED_TIME_ROUND_DIGITS)
            if elapsed_time is not None
            else ELAPSED_TIME_DEFAULT,
            "image_count": image_count,
            "image_detail": image_detail,
            "image_total_bytes": image_total_bytes,
            "image_est_tokens_low": image_est_low,
            "image_est_tokens_high": image_est_high,
            "api_prompt_tokens": usage["api_prompt_tokens"],
            "api_completion_tokens": usage["api_completion_tokens"],
            "api_total_tokens": usage["api_total_tokens"],
            "api_cached_tokens": usage["api_cached_tokens"],
        }

        with self._lock:
            with open(self.log_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
                writer.writerow(row)

        self.logger.info(
            f"Interaction stored. model={model_name} prompt_tokens_est={prompt_tokens_est} "
            f"completion_tokens_est={completion_tokens_est} success={success}"
        )
