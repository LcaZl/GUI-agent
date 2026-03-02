import os
import json
import logging
import time
import base64
import io
import math
from mimetypes import guess_type
from urllib.parse import urlparse
from typing import Dict, Any, List, Optional, Type, Tuple

from pydantic import BaseModel, ValidationError
from openai import AzureOpenAI

from agentz.pydantic_models import (
    GPTClientRequest,
    ExperimentConfiguration,
    DefaultGPTSettings,
)
from agentz.constants import (
    ALLOWED_OVERRIDES,
    AZURE_API_VERSION,
    AZURE_ENDPOINT,
    BASE_TILE_4O_MINI,
    BASE_TILE_COMPUTER_USE,
    BASE_TILE_GPT5,
    BASE_TILE_O1,
    CLIP_MAX,
    CLIP_MIN,
    DEFAULT_IMAGE_FORMAT,
    DEFAULT_JPEG_QUALITY,
    DEFAULT_MAX_TOKENS,
    FLOAT_NORM_MAX,
    MAX_DIM_STAGE1,
    MAX_PATCHES,
    MAX_SHORTEST_DIM,
    MAX_VISION_IMAGES,
    MODEL_PROMPT_MULTIPLIERS,
    MODEL_PROMPT_RATIO_DEFAULT,
    PATCH_SIZE,
    TILE_SIZE,
)

from ._gpt_interaction_logger import GPTInteractionLogger


def _model_supports_reasoning(model_name: str) -> bool:
    """
    Process model supports reasoning.
        
        Parameters
        ----------
        model_name : str
            Function argument.
        
        Returns
        -------
        bool
            True when the condition is satisfied, otherwise False.
        
    """
    name = (model_name or "").lower()
    return name.startswith("o4") or name.startswith("o3") or name.startswith("gpt-5")


class GPTClient:
    """
    Azure OpenAI client wrapper with:
    - Pydantic validation
    - Retry logic (with nudging)
    - CSV logging via GPTInteractionLogger
    - Vision support (paths/urls/data-urls + numpy arrays)
    """

    def __init__(self, settings: ExperimentConfiguration) -> None:
        """
        Initialize `GPTClient` dependencies and runtime state.
        
        Parameters
        ----------
        settings : ExperimentConfiguration
            Runtime settings for this component.
        
        Returns
        -------
        None
            No return value.
        """
        self.settings: DefaultGPTSettings = settings.gpt_client_settings

        self.model = self.settings.model
        self.temperature = self.settings.temperature
        self.max_retries = self.settings.max_retries

        self.logger = logging.getLogger("GPTClient")

        # Logging to separate module
        self.log_file = self.settings.gpt_log_path + ".csv"
        self.interaction_logger = GPTInteractionLogger(
            log_file=self.log_file,
            encoding_model_name=self.model,
            logger=self.logger,
        )
        self._configure_third_party_logging()

        subscription_key = os.environ["OPENAI_AZURE_API_KEY"]

        self.client = AzureOpenAI(
            api_key=subscription_key,
            azure_endpoint=AZURE_ENDPOINT,
            api_version=AZURE_API_VERSION,
        )

        self.logger.info(f"GPTClient initialized (Azure). model={self.model}")

    # ------------------------------------------------------------------
    # IMAGE HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _is_probably_url(s: str) -> bool:
        """
        Return whether is probably url.
        
        Parameters
        ----------
        s : str
            Function argument.
        
        Returns
        -------
        bool
            True when the condition is satisfied, otherwise False.
        """
        if not s:
            return False
        if s.startswith("data:"):
            return True
        try:
            p = urlparse(s)
            return p.scheme in ("http", "https") and bool(p.netloc)
        except Exception:
            return False

    @staticmethod
    def _local_image_to_data_url(image_path: str) -> Dict[str, Any]:
        """
        Process local image to data url.
        
        Parameters
        ----------
        image_path : str
            Filesystem path.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with computed fields.
        
        """
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = "application/octet-stream"

        with open(image_path, "rb") as f:
            raw = f.read()

        b64 = base64.b64encode(raw).decode("utf-8")
        return {
            "url": f"data:{mime_type};base64,{b64}",
            "bytes": len(raw),
            "width": None,
            "height": None,
            "source": "path",
            "label": os.path.basename(image_path) or "path",
        }

    @staticmethod
    def _probe_image_size_if_possible(image_path: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Best-effort dimension extraction for local files. If PIL is missing, return (None, None).
        """
        try:
            from PIL import Image
        except Exception:
            return (None, None)

        try:
            with Image.open(image_path) as im:
                w, h = im.size
            return (int(w), int(h))
        except Exception:
            return (None, None)

    @staticmethod
    def _numpy_to_data_url(
        arr: Any,
        fmt: str = DEFAULT_IMAGE_FORMAT,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    ) -> Dict[str, Any]:
        """
        Accepts numpy arrays shaped (H,W,3) or (H,W,4). Expects uint8 or float.
        Converts to data URL (base64).
        """
        try:
            import numpy as np
        except Exception as e:
            raise RuntimeError("numpy is required to pass ndarray images.") from e

        try:
            from PIL import Image
        except Exception as e:
            raise RuntimeError("Pillow (PIL) is required to convert numpy arrays to images.") from e

        if not hasattr(arr, "shape"):
            raise TypeError("Expected a numpy ndarray-like object with .shape")

        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise ValueError(f"Expected ndarray shape (H,W,3) or (H,W,4). Got {arr.shape}")

        h, w = int(arr.shape[0]), int(arr.shape[1])

        a = arr
        if a.dtype != np.uint8:
            a = np.clip(a, CLIP_MIN, CLIP_MAX)
            if np.issubdtype(a.dtype, np.floating) and float(a.max()) <= FLOAT_NORM_MAX:
                a = (a * float(CLIP_MAX)).round()
            a = a.astype(np.uint8)

        mode = "RGB" if a.shape[2] == 3 else "RGBA"
        img = Image.fromarray(a, mode=mode)

        buf = io.BytesIO()
        fmt_up = (fmt or DEFAULT_IMAGE_FORMAT).upper()
        if fmt_up in ("JPG", DEFAULT_IMAGE_FORMAT):
            if mode == "RGBA":
                img = img.convert("RGB")
            img.save(buf, format=DEFAULT_IMAGE_FORMAT, quality=int(jpeg_quality), optimize=True)
            mime = "image/jpeg"
        else:
            img.save(buf, format="PNG")
            mime = "image/png"

        raw = buf.getvalue()
        b64 = base64.b64encode(raw).decode("utf-8")
        return {
            "url": f"data:{mime};base64,{b64}",
            "bytes": len(raw),
            "width": w,
            "height": h,
            "source": "numpy",
            "label": f"numpy[{h}x{w}]",
        }

    @staticmethod
    def _estimate_image_tokens(model_name: str, width: int, height: int, detail: str) -> int:
        """
        Best-effort estimate for "image tokens" based on published tokenization rules.
        This is for logging/cost tracking only; actual metering may differ on Azure.
        """
        name = (model_name or "").lower()
        d = (detail or "auto").lower()

        # Patch-based models (32px patches + multiplier; capped)
        patch_models = MODEL_PROMPT_MULTIPLIERS
        for k, mult in patch_models.items():
            if k in name:
                patches_w = math.ceil(width / PATCH_SIZE)
                patches_h = math.ceil(height / PATCH_SIZE)
                raw_patches = patches_w * patches_h

                if raw_patches > MAX_PATCHES:
                    r = math.sqrt((PATCH_SIZE * PATCH_SIZE * MAX_PATCHES) / (width * height))
                    new_w = max(1, int(math.floor(width * r)))
                    new_h = max(1, int(math.floor(height * r)))
                else:
                    new_w, new_h = width, height

                image_tokens = math.ceil(new_w / PATCH_SIZE) * math.ceil(new_h / PATCH_SIZE)
                image_tokens = min(MAX_PATCHES, image_tokens)

                return int(math.ceil(image_tokens * mult))

        # Tile-based models (base + tiles). We map family -> base/tile tokens (best-effort).
        def base_tile_for(n: str) -> Tuple[int, int]:
            """
            Process base tile for.
                        
                        Parameters
                        ----------
                        n : str
                            Function argument.
                        
                        Returns
                        -------
                        Tuple[int, int]
                            Tuple with computed values.
                        
            """
            if "gpt-5" in n:
                return BASE_TILE_GPT5
            if "4o-mini" in n:
                return BASE_TILE_4O_MINI
            if "o1" in n or "o3" in n or "o1-pro" in n:
                return BASE_TILE_O1
            if "computer-use" in n:
                return BASE_TILE_COMPUTER_USE
            return MODEL_PROMPT_RATIO_DEFAULT  # default 4o/4.1/4.5-like

        base, tile = base_tile_for(name)

        if d == "low":
            return base

        # high detail tokenization:
        w, h = float(width), float(height)
        s1 = min(MAX_DIM_STAGE1 / w, MAX_DIM_STAGE1 / h, 1.0)
        w1, h1 = w * s1, h * s1

        shortest = min(w1, h1)
        s2 = MAX_SHORTEST_DIM / shortest if shortest > 0 else 1.0
        w2, h2 = w1 * s2, h1 * s2

        tiles = math.ceil(w2 / TILE_SIZE) * math.ceil(h2 / TILE_SIZE)
        return int(base + tile * tiles)

    @staticmethod
    def _summarize_images_for_prompt(norm: List[Dict[str, Any]]) -> str:
        """
        Short, non-sensitive summary of image inputs for logging prompt.
        """
        parts = []
        for it in norm:
            src = it.get("source")
            label = it.get("label", src or "image")
            w = it.get("width")
            h = it.get("height")
            if w and h:
                parts.append(f"{label}({w}x{h})")
            else:
                parts.append(f"{label}")
        return ", ".join(parts)

    # ------------------------------------------------------------------
    # CORE CALL (TEXT ONLY)
    # ------------------------------------------------------------------

    def _configure_third_party_logging(self) -> None:
        """
        Process configure third party logging.
        
        Returns
        -------
        None
            No return value.
        
        """
        level_name = os.getenv("LLM_THIRD_PARTY_LOG_LEVEL", "WARNING").strip().upper()
        if level_name in {"OFF", "SILENT", "NONE"}:
            level = logging.CRITICAL
        else:
            level = getattr(logging, level_name, logging.WARNING)

        for logger_name in [
            "langchain",
            "langchain_core",
            "langchain_openai",
            "langchain_google_genai",
            "openai",
            "httpx",
            "httpcore",
            "urllib3",
            "google",
            "google.api_core",
            "google.auth",
            "google_genai",
            "grpc",
        ]:
            logging.getLogger(logger_name).setLevel(level)

    def chat_with_tool(self, request: GPTClientRequest) -> BaseModel:
        """
        Ask the model to return JSON conforming to tool_schema.
        Adds retry nudging + richer logging.
        """
        tool_schema: Type[BaseModel] = request.tool_schema
        overrides = request.overrides or {}
        override_parameters = {k: v for k, v in overrides.items() if k in ALLOWED_OVERRIDES and v is not None}

        model = override_parameters.get("model", self.model)
        try:
            temperature = float(override_parameters.get("temperature", self.temperature))
        except Exception:
            temperature = float(self.temperature)
        max_retries = int(override_parameters.get("max_retries", self.max_retries))
        reasoning = override_parameters.get("reasoning", None)
        if "temperature" in override_parameters:
            self.logger.info("temperature override applied | temp=%s", temperature)

        schema_json = json.dumps(tool_schema.model_json_schema(), indent=2, ensure_ascii=False)

        base_prompt = f"""
You must respond ONLY with valid JSON.
The JSON must conform exactly to the following schema:

{schema_json}

User request:
{request.prompt}
""".strip()

        current_prompt = base_prompt
        last_error: Optional[Exception] = None
        raw_text: str = ""
        response = None
        elapsed: Optional[float] = None

        for attempt in range(1, max_retries + 1):
            self.logger.info(f"Prompt to LLM model={model} (attempt {attempt})")
            start = time.perf_counter()

            try:
                req_kwargs: Dict[str, Any] = {
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": DEFAULT_MAX_TOKENS,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": "You are a planner agent."},
                        {"role": "user", "content": current_prompt},
                    ],
                }

                # Optional reasoning left conservative
                # if reasoning is not None and _model_supports_reasoning(model):
                #     req_kwargs["reasoning"] = reasoning

                response = self.client.chat.completions.create(**req_kwargs)
                elapsed = time.perf_counter() - start

                raw_text = (response.choices[0].message.content or "").strip()

                parsed = json.loads(raw_text)
                validated = tool_schema(**parsed)

                self.interaction_logger.log_interaction(
                    prompt=current_prompt,
                    response=response,
                    model_name=model,
                    tool_name=getattr(tool_schema, "__name__", "tool_schema"),
                    tool_args=parsed,
                    success=True,
                    elapsed_time=elapsed,
                    image_meta=None,
                )
                return validated

            except (json.JSONDecodeError, ValidationError, RuntimeError) as e:
                elapsed = time.perf_counter() - start
                last_error = e

                self.interaction_logger.log_interaction(
                    prompt=current_prompt,
                    response=response,
                    model_name=model,
                    tool_name=getattr(tool_schema, "__name__", "tool_schema"),
                    tool_args=None,
                    success=False,
                    error=str(e),
                    elapsed_time=elapsed,
                    image_meta=None,
                )

                current_prompt = (
                    "WARNING: respond only with valid JSON matching the schema. "
                    "Last answer was:\n"
                    f"{raw_text}\n"
                    "It causes the following error:\n"
                    f"{e}\n\n"
                    "Here the same previous prompt:\n"
                    f"{base_prompt}"
                )

        raise RuntimeError(
            f"Azure OpenAI call failed after {max_retries} attempts. Last error: {last_error}"
        )

    # ------------------------------------------------------------------
    # CORE CALL (TEXT + IMAGES)
    # ------------------------------------------------------------------

    def chat_with_tool_and_images(
        self,
        request: GPTClientRequest,
        images: List[Any],
        image_detail: str = "auto",
        numpy_image_format: str = DEFAULT_IMAGE_FORMAT,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    ) -> BaseModel:
        """
        Like chat_with_tool(), but supports 1+ images as inputs.
        
        images can contain:
          - local filesystem paths (str)
          - http(s) URLs (str)
          - already-formed data: URLs (str)
          - numpy.ndarray (H,W,3) or (H,W,4)
        """
        tool_schema: Type[BaseModel] = request.tool_schema
        overrides = request.overrides or {}
        override_parameters = {k: v for k, v in overrides.items() if k in ALLOWED_OVERRIDES and v is not None}

        model = override_parameters.get("model", self.model)
        try:
            temperature = float(override_parameters.get("temperature", self.temperature))
        except Exception:
            temperature = float(self.temperature)
        max_retries = int(override_parameters.get("max_retries", self.max_retries))
        reasoning = override_parameters.get("reasoning", None)
        if "temperature" in override_parameters:
            self.logger.info("temperature override applied | temp=%s", temperature)

        if images is None:
            images = []
        if len(images) == 0:
            return self.chat_with_tool(request)
        if len(images) > MAX_VISION_IMAGES:
            raise ValueError("Azure vision chat requests support up to 10 images per request.")

        # Normalize images -> data/url payloads (do NOT store image content in logs)
        norm: List[Dict[str, Any]] = []
        for img in images:
            if isinstance(img, str):
                if self._is_probably_url(img):
                    norm.append({
                        "url": img,
                        "bytes": 0,
                        "width": None,
                        "height": None,
                        "source": "url",
                        "label": "url",
                    })
                else:
                    item = self._local_image_to_data_url(img)
                    w, h = self._probe_image_size_if_possible(img)
                    item["width"], item["height"] = w, h
                    norm.append(item)
            else:
                norm.append(self._numpy_to_data_url(img, fmt=numpy_image_format, jpeg_quality=jpeg_quality))

        # Build schema + prompt
        schema_json = json.dumps(tool_schema.model_json_schema(), indent=2, ensure_ascii=False)
        base_prompt = f"""
You must respond ONLY with valid JSON.
The JSON must conform exactly to the following schema:

{schema_json}

User request:
{request.prompt}
""".strip()

        current_prompt = base_prompt
        last_error: Optional[Exception] = None
        raw_text: str = ""
        response = None
        elapsed: Optional[float] = None

        # Estimate image token cost (range for auto; single value for low/high)
        est_low_total = 0
        est_high_total = 0
        total_bytes = 0
        counted_any_dims = False

        for it in norm:
            total_bytes += int(it.get("bytes", 0) or 0)
            w = it.get("width")
            h = it.get("height")
            if w is None or h is None:
                continue

            counted_any_dims = True
            low = self._estimate_image_tokens(model, int(w), int(h), "low")
            high = self._estimate_image_tokens(model, int(w), int(h), "high")

            d = (image_detail or "auto").lower()
            if d == "low":
                est_low_total += low
                est_high_total += low
            elif d == "high":
                est_low_total += high
                est_high_total += high
            else:
                est_low_total += low
                est_high_total += high

        image_meta = {
            "count": len(norm),
            "detail": image_detail,
            "total_bytes": total_bytes,
            "est_tokens_low": est_low_total if counted_any_dims else -1,
            "est_tokens_high": est_high_total if counted_any_dims else -1,
        }

        self.logger.info(
            f"model={model} temp={temperature} reasoning={reasoning} "
            f"max_retries={max_retries} images={len(norm)} detail={image_detail}"
        )

        for attempt in range(1, max_retries + 1):
            self.logger.info(f"attempt={attempt} model={model}")
            start = time.perf_counter()

            try:
                content_parts: List[Dict[str, Any]] = [{"type": "text", "text": current_prompt}]
                for it in norm:
                    part = {"type": "image_url", "image_url": {"url": it["url"]}}
                    if image_detail is not None:
                        part["image_url"]["detail"] = image_detail
                    content_parts.append(part)

                req_kwargs: Dict[str, Any] = {
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": DEFAULT_MAX_TOKENS,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": "You are a planner agent."},
                        {"role": "user", "content": content_parts},
                    ],
                }

                # Optional reasoning left conservative
                # if reasoning is not None and _model_supports_reasoning(model):
                #     req_kwargs["reasoning"] = reasoning

                response = self.client.chat.completions.create(**req_kwargs)
                elapsed = time.perf_counter() - start

                msg = response.choices[0].message
                if msg.content is None:
                    raise RuntimeError(f"Empty content. finish_reason={response.choices[0].finish_reason}, tool_calls={getattr(msg,'tool_calls', None)}")

                raw_text = msg.content.strip()
                parsed = json.loads(raw_text)
                validated = tool_schema(**parsed)

                log_prompt = current_prompt + " [IMAGES: " + self._summarize_images_for_prompt(norm) + "]"

                self.interaction_logger.log_interaction(
                    prompt=log_prompt,
                    response=response,
                    model_name=model,
                    tool_name=getattr(tool_schema, "__name__", "tool_schema"),
                    tool_args=parsed,
                    success=True,
                    elapsed_time=elapsed,
                    image_meta=image_meta,
                )
                return validated

            except (json.JSONDecodeError, ValidationError, RuntimeError, OSError, ValueError, TypeError, Exception) as e:
                elapsed = time.perf_counter() - start
                last_error = e

                log_prompt = current_prompt + " [IMAGES: " + self._summarize_images_for_prompt(norm) + "]"

                self.interaction_logger.log_interaction(
                    prompt=log_prompt,
                    response=response,
                    model_name=model,
                    tool_name=getattr(tool_schema, "__name__", "tool_schema"),
                    tool_args=None,
                    success=False,
                    error=str(e),
                    elapsed_time=elapsed,
                    image_meta=image_meta,
                )

                current_prompt = (
                    "WARNING: respond only with valid JSON matching the schema. "
                    "Last answer was:\n"
                    f"{raw_text}\n"
                    "It causes the following error:\n"
                    f"{e}\n\n"
                    "Here the same previous prompt:\n"
                    f"{base_prompt}"
                )

        raise RuntimeError(
            f"Azure OpenAI call failed after {max_retries} attempts. Last error: {last_error}"
        )

    # Optional convenience wrapper (if other code relied on GPTClient.count_tokens)
    def count_tokens(self, text: str) -> int:
        """
        Process count tokens.
        
        Parameters
        ----------
        text : str
            Input text.
        
        Returns
        -------
        int
            Integer result value.
        
        """
        return self.interaction_logger.count_tokens(text, model_name=self.model)
