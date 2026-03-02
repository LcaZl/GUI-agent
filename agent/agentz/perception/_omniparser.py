import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import logging

import cv2
import easyocr
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import ToPILImage

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Blip2ForConditionalGeneration,
)
from ultralytics import YOLO

from agentz.pydantic_models import OmniParserSettings
from agentz.constants import (
    CAPTION_ICON_MIN_CONF,
    CAPTION_MAX_CHARS,
    CAPTION_MAX_WORDS,
    ICON_COLOR,
    OCR_MAX_LEN,
    OCR_MIN_CONF,
    OCR_MIN_LEN,
    TEXT_COLOR,
)

_RE_MOSTLY_NONALNUM = re.compile(r"^[\W_]+$", re.UNICODE)
_RE_MOSTLY_NUM = re.compile(r"^[0-9\W_]+$", re.UNICODE)
_RE_WHITESPACE = re.compile(r"\s+", re.UNICODE)

class OmniParserLocal:
    def __init__(self, settings: OmniParserSettings):
        """
        Initialize `OmniParserLocal` dependencies and runtime state.
        
        Parameters
        ----------
        settings : OmniParserSettings
            Runtime settings for this component.
        
        Returns
        -------
        None
            No return value.
        """
        self.settings = settings
        self.logger = logging.getLogger("OmniParser")

        detection_model_path = Path(self.settings.icon_detect_model_path)
        caption_model_dir = Path(self.settings.icon_caption_model_dir)

        self.device = torch.device(self.settings.device if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        if not caption_model_dir.exists():
            raise FileNotFoundError(f"Caption model directory not found: {caption_model_dir}")
        if not detection_model_path.exists():
            raise FileNotFoundError(f"Detection model path not found: {detection_model_path}")

        self.logger.info(f"--- OmniParser initialized on {self.device} ---")
        self.logger.info(f"Loading Detection Model: {detection_model_path}")
        self.yolo = YOLO(str(detection_model_path))

        # ----------------------------
        # Captioner init: Florence vs BLIP-2
        # ----------------------------
        if self.settings.caption_backend == "florence":
            caption_model_id = str(self.settings.icon_caption_model_id)
            self.logger.info(f"Caption backend: florence | processor={caption_model_id} | weights={caption_model_dir}")

            # Processor from HF ID (reliable), model from local fine-tuned weights
            self.processor = AutoProcessor.from_pretrained(caption_model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                str(caption_model_dir),
                torch_dtype=self.dtype,
                trust_remote_code=True,
                attn_implementation="eager",
            ).to(self.device)
            self._caption_backend = "florence"

        elif self.settings.caption_backend == "blip2":
            base_id = str(self.settings.blip2_base_model_id)
            self.logger.info(f"Caption backend: blip2 | processor={base_id} | weights={caption_model_dir}")

            # For BLIP-2: processor from base model, model from local fine-tuned weights
            self.processor = AutoProcessor.from_pretrained(base_id)  # base provides image processor + tokenizer
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                str(caption_model_dir),
                torch_dtype=self.dtype,
            ).to(self.device)
            self._caption_backend = "blip2"
            self._normalize_blip2_image_tokens()

        else:
            raise ValueError(f"Unknown caption_backend: {self.settings.caption_backend}")

        self.model.eval()

        self.logger.info("Initializing OCR...")
        self.ocr = easyocr.Reader(self.settings.ocr_languages, gpu=(self.device.type == "cuda"))

        # ---- noise reduction knobs ----
        self._min_ocr_conf = OCR_MIN_CONF
        self._min_ocr_len = OCR_MIN_LEN
        self._max_ocr_len = OCR_MAX_LEN

        self._min_caption_icon_conf = CAPTION_ICON_MIN_CONF
        self._max_caption_chars = CAPTION_MAX_CHARS
        self._max_caption_words = CAPTION_MAX_WORDS

    def _normalize_blip2_image_tokens(self) -> None:
        """
        Backward-compatibility for BLIP-2 checkpoints saved with older
        transformers versions where image token metadata may be missing.
        """
        image_token_id = getattr(self.model.config, "image_token_id", None)
        image_token_index = getattr(self.model.config, "image_token_index", None)

        if image_token_id is None:
            tokenizer = getattr(self.processor, "tokenizer", None)
            tok_image_id = getattr(tokenizer, "image_token_id", None) if tokenizer is not None else None
            image_token_id = image_token_index if image_token_index is not None else tok_image_id

        if image_token_id is None and image_token_index is None:
            raise ValueError(
                "BLIP2 config is missing image token metadata (image_token_id/image_token_index). "
                "Use a processor/checkpoint that includes an image token id."
            )

        if image_token_id is None:
            image_token_id = image_token_index
        if image_token_index is None:
            image_token_index = image_token_id

        self.model.config.image_token_id = int(image_token_id)
        self.model.config.image_token_index = int(image_token_index)

    def _move_inputs_to_device(self, inputs):
        """
        Move inputs to device.
        
        Parameters
        ----------
        inputs : Any
            Function argument.
        
        Returns
        -------
        Any
            Function result.
        
        """
        moved = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                if value.is_floating_point():
                    moved[key] = value.to(self.device, dtype=self.dtype)
                else:
                    moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def parse(
        self,
        image_np: np.ndarray,
        box_threshold: float | None = None,
        iou_threshold: float | None = None,
        min_icon_area_ratio: float | None = None,
        max_icon_area_ratio: float | None = None,
    ) -> pd.DataFrame:
        """
        Parse vision detections and return standardized UI rows.
        
        Parameters
        ----------
        image_np : np.ndarray
            Function argument.
        box_threshold : Optional[float | None]
            Function argument.
        iou_threshold : Optional[float | None]
            Function argument.
        min_icon_area_ratio : Optional[float | None]
            Function argument.
        max_icon_area_ratio : Optional[float | None]
            Function argument.
        
        Returns
        -------
        pd.DataFrame
            Function result.
        
        """
        if image_np is None:
            return pd.DataFrame()

        height, width = image_np.shape[:2]

        box_threshold = self.settings.box_threshold if box_threshold is None else box_threshold
        iou_threshold = self.settings.iou_threshold if iou_threshold is None else iou_threshold
        min_icon_area_ratio = self.settings.min_icon_area_ratio if min_icon_area_ratio is None else min_icon_area_ratio
        max_icon_area_ratio = self.settings.max_icon_area_ratio if max_icon_area_ratio is None else max_icon_area_ratio

        ocr_rows = self._run_ocr(image_np)
        icon_rows = self._run_yolo(
            image_np,
            height=height,
            width=width,
            box_threshold=box_threshold,
            iou_threshold=iou_threshold,
            min_icon_area_ratio=min_icon_area_ratio,
            max_icon_area_ratio=max_icon_area_ratio,
        )

        icons_final = self._filter_icon_overlap(icon_rows, ocr_rows, iou_threshold=iou_threshold)
        self._caption_icons(image_np, icons_final)

        # final: drop any rows that ended up with empty content (rare but happens)
        all_rows = []
        for r in (ocr_rows + icons_final):
            if r.get("type") == "text":
                if not str(r.get("content") or "").strip():
                    continue
            all_rows.append(r)

        return pd.DataFrame(all_rows)

    # ----------------------------
    # OCR
    # ----------------------------

    def _sanitize_ocr_text(self, text: str) -> str:
        """
        Process sanitize ocr text.
        
        Parameters
        ----------
        text : str
            Input text.
        
        Returns
        -------
        str
            Resulting string value.
        
        """
        t = (text or "").strip()
        t = _RE_WHITESPACE.sub(" ", t)
        return t

    def _keep_ocr(self, text: str, conf: Optional[float]) -> bool:
        """
        Keep ocr.
        
        Parameters
        ----------
        text : str
            Input text.
        conf : Optional[float]
            Function argument.
        
        Returns
        -------
        bool
            True when the condition is satisfied, otherwise False.
        
        """
        if conf is None or conf < self._min_ocr_conf:
            return False
        t = (text or "").strip()
        if not t:
            return False
        if len(t) < self._min_ocr_len:
            return False
        if len(t) > self._max_ocr_len:
            return False
        if _RE_MOSTLY_NONALNUM.match(t) or _RE_MOSTLY_NUM.match(t):
            return False
        return True

    def _run_ocr(self, image_np: np.ndarray) -> List[dict]:
        """
        Execute run ocr.
        
        Parameters
        ----------
        image_np : np.ndarray
            Function argument.
        
        Returns
        -------
        List[dict]
            Dictionary with computed fields.
        """
        rows: List[dict] = []
        for bbox_raw, text, conf in self.ocr.readtext(image_np):
            try:
                conf_f = float(conf)
            except Exception:
                conf_f = None

            txt = self._sanitize_ocr_text(str(text))
            if not self._keep_ocr(txt, conf_f):
                continue

            x1, y1 = map(int, bbox_raw[0])
            x2, y2 = map(int, bbox_raw[2])
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            rows.append(
                {
                    "source": "vision",
                    "type": "text",
                    "role": "text",
                    "content": txt,
                    "value": None,
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "cx": float(cx),
                    "cy": float(cy),
                    "score": conf_f,
                    "vision_score": conf_f,
                }
            )
        return rows

    # ----------------------------
    # YOLO icons
    # ----------------------------

    def _run_yolo(
        self,
        image_np: np.ndarray,
        *,
        height: int,
        width: int,
        box_threshold: float,
        iou_threshold: float,
        min_icon_area_ratio: float,
        max_icon_area_ratio: float,
    ) -> List[dict]:
        """
        Execute run yolo.
        
        Parameters
        ----------
        image_np : np.ndarray
            Function argument.
        height : int
            Function argument.
        width : int
            Function argument.
        box_threshold : float
            Function argument.
        iou_threshold : float
            Function argument.
        min_icon_area_ratio : float
            Function argument.
        max_icon_area_ratio : float
            Function argument.
        
        Returns
        -------
        List[dict]
            Dictionary with computed fields.
        """
        yolo_out = self.yolo.predict(
            image_np,
            imgsz=self.settings.yolo_image_size,
            conf=box_threshold,
            iou=iou_threshold,
            verbose=False,
        )[0]

        rows: List[dict] = []
        for bbox, score in zip(
            yolo_out.boxes.xyxy.cpu().numpy(),
            yolo_out.boxes.conf.cpu().numpy(),
        ):
            x1, y1, x2, y2 = map(float, bbox)
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(width - 1), x2), min(float(height - 1), y2)
            if x2 <= x1 or y2 <= y1:
                continue

            area_ratio = (x2 - x1) * (y2 - y1) / float(width * height)
            if not (min_icon_area_ratio <= area_ratio <= max_icon_area_ratio):
                continue

            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            sc = float(score) if score is not None else None
            rows.append(
                {
                    "source": "vision",
                    "type": "icon",
                    "role": "icon",
                    "content": None,
                    "value": None,
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "cx": float(cx),
                    "cy": float(cy),
                    "score": sc,
                    "vision_score": sc,
                }
            )

        return rows

    def _filter_icon_overlap(self, icon_rows: List[dict], ocr_rows: List[dict], *, iou_threshold: float) -> List[dict]:
        """
        Filter icon overlap.
        
        Parameters
        ----------
        icon_rows : List[dict]
            Function argument.
        ocr_rows : List[dict]
            Function argument.
        iou_threshold : float
            Function argument.
        
        Returns
        -------
        List[dict]
            Dictionary with computed fields.
        
        """
        ocr_boxes = [(o["x1"], o["y1"], o["x2"], o["y2"]) for o in ocr_rows if o.get("x1") is not None]
        icons_final: List[dict] = []
        for icon in icon_rows:
            bx = (icon["x1"], icon["y1"], icon["x2"], icon["y2"])
            if any(self._compute_iou(bx, ob) > iou_threshold for ob in ocr_boxes):
                continue
            icons_final.append(icon)
        return icons_final

    # ----------------------------
    # Captioning / caption hygiene
    # ----------------------------

    def _sanitize_caption(self, caption: str) -> Optional[str]:
        """
        Reduce 'caption noise':
        - drop long sentences / verbose captions
        - drop captions that look like descriptions instead of UI labels
        """
        t = (caption or "").strip()
        t = _RE_WHITESPACE.sub(" ", t)

        if not t:
            return None

        # If it looks like a descriptive sentence, drop it.
        # Common: ends with '.' or contains many words.
        words = [w for w in t.split(" ") if w]
        if self._max_caption_chars is not None and len(t) > self._max_caption_chars:
            return None
        if self._max_caption_words is not None and len(words) > self._max_caption_words:
            return None
        if t.endswith("."):
            return None

        # Avoid purely numeric/punct
        if _RE_MOSTLY_NONALNUM.match(t) or _RE_MOSTLY_NUM.match(t):
            return None

        return t

    def _caption_icons(self, image_np: np.ndarray, icons: List[dict]) -> None:
        """
        Caption icons.
        
        Parameters
        ----------
        image_np : np.ndarray
            Function argument.
        icons : List[dict]
            Function argument.
        
        Returns
        -------
        None
            No return value.
        
        """
        targets = [
            el for el in icons
            if el.get("content") is None and (el.get("score") or 0.0) >= self._min_caption_icon_conf
        ]
        if not targets:
            return

        crops = []
        cropped_targets = []
        for el in targets:
            x1, y1, x2, y2 = map(int, [el["x1"], el["y1"], el["x2"], el["y2"]])
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image_np[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, self.settings.icon_crop_size)
            crops.append(ToPILImage()(crop))
            cropped_targets.append(el)

        if not crops:
            return

        prompt = self.settings.caption_prompt

        if self._caption_backend == "florence":
            inputs = self.processor(
                text=[prompt] * len(crops),
                images=crops,
                return_tensors="pt",
                do_resize=False,
            )
        elif self._caption_backend == "blip2":
            # BLIP-2 generate() builds its own image placeholder tokens when input_ids is omitted.
            inputs = self.processor(
                images=crops,
                return_tensors="pt",
                do_resize=False,
            )
        else:
            raise ValueError(f"Unknown caption backend: {self._caption_backend}")

        inputs = self._move_inputs_to_device(inputs)

        with torch.no_grad():
            ids = self.model.generate(
                **inputs,
                max_new_tokens=self.settings.caption_max_new_tokens,
                num_beams=self.settings.caption_num_beams,
                use_cache=False,
            )

        texts = self.processor.batch_decode(ids, skip_special_tokens=True)
        for el, txt in zip(cropped_targets, texts):
            raw = txt.replace(prompt, "").strip() if self._caption_backend == "florence" else txt.strip()
            cleaned = self._sanitize_caption(raw)
            el["content"] = cleaned

    # ----------------------------
    # Utils
    # ----------------------------

    def _compute_iou(self, A, B) -> float:
        """
        Compute iou.
        
        Parameters
        ----------
        A : Any
            Function argument.
        B : Any
            Function argument.
        
        Returns
        -------
        float
            Floating-point result value.
        
        """
        xA, yA = max(A[0], B[0]), max(A[1], B[1])
        xB, yB = min(A[2], B[2]), min(A[3], B[3])
        inter = max(0.0, xB - xA) * max(0.0, yB - yA)
        areaA = max(0.0, (A[2] - A[0])) * max(0.0, (A[3] - A[1]))
        areaB = max(0.0, (B[2] - B[0])) * max(0.0, (B[3] - B[1]))
        denom = areaA + areaB - inter
        return 0.0 if denom <= 0.0 else float(inter / (denom + self.settings.iou_eps))

    def save_visualization(
        self,
        image_np: np.ndarray,
        df: pd.DataFrame,
        out_dir: str | os.PathLike = ".",
        prefix: str | None = None,
    ) -> str:
        """
        Persist save visualization.
        
        Parameters
        ----------
        image_np : np.ndarray
            Function argument.
        df : pd.DataFrame
            Function argument.
        out_dir : Optional[str | os.PathLike]
            Function argument.
        prefix : Optional[str | None]
            Function argument.
        
        Returns
        -------
        str
            Resulting string value.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)

        if prefix is None:
            prefix = self.settings.default_vis_prefix

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"{prefix}_{ts}.png"

        img = image_np.copy()
        for _, el in df.iterrows():
            try:
                x1, y1, x2, y2 = map(int, [el["x1"], el["y1"], el["x2"], el["y2"]])
            except Exception:
                continue
            color = ICON_COLOR if el.get("type") == "icon" else TEXT_COLOR
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        cv2.imwrite(str(out_path), img)
        return str(out_path)
