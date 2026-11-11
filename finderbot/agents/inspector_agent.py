"""
agents/inspector_agent.py
=========================
Agent 1 — Item Inspector (IBM watsonx.ai integrated)

Responsibilities:
  - Validate JPG/JPEG/PNG image bytes with Pillow.
  - Correct EXIF orientation, resize oversized images, convert to RGB.
  - Perform basic quality checks (brightness, contrast, minimum size).
  - Extract approximate dominant colours with Pillow (heuristic, not AI).
  - Encode image as Base64 JPEG data URI for multimodal foundation models.
  - Call IBM watsonx.ai via ibm-watsonx-ai SDK using ModelInference.chat().
  - Validate the JSON response with Pydantic (via GuardrailsManager).
  - Return a validated InspectionResult on success.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ExifTags, ImageStat

from guardrails.safety import GuardrailsManager, InspectionResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_IMAGE_BYTES: int = 10 * 1024 * 1024   # 10 MB limit
MAX_DIMENSION:   int = 1600               # Scale down longest edge
MIN_DIMENSION:   int = 50                 # Minimum dimension
SUPPORTED_FORMATS = ("JPEG", "PNG")

_SYSTEM_PROMPT = (
    "You are an expert forensic item analyst. "
    "Examine the image and extract structured physical attributes about the visible object. "
    "Be precise and factual. Do not invent features that are not visible. "
    "If a feature cannot be determined, state it in hidden_or_obscured_features. "
    "Return ONLY a valid JSON object — no commentary, no markdown fences."
)

_USER_PROMPT = """\
Analyse the object in this image and return a single JSON object.

Required schema (return ONLY the raw JSON, nothing else):
{
  "is_valid": true,
  "category": "<general category e.g. Electronics, Clothing, Accessories, Bags>",
  "item_type": "<specific item type e.g. Wireless headphones, Leather wallet>",
  "primary_color": "<dominant colour>",
  "secondary_color": "<secondary colour or empty string>",
  "material": "<visible material e.g. Plastic, Metal, Leather>",
  "physical_traits": ["<trait1>", "<trait2>"],
  "distinctive_marks": ["<scratch, sticker, engraving, logo, serial number>"],
  "hidden_or_obscured_features": ["<features not visible in image>"],
  "rejection_reason": "",
  "image_quality_notes": "<brief note on image quality>"
}

Rules:
1. Set is_valid to false only if the image is too blurry, dark, cut off, or cannot be reliably analysed.
2. When is_valid is false, set rejection_reason to an explanation and leave unknown fields empty.
3. Record visible scratches, stickers, engravings, logos, and labels in distinctive_marks.
4. Return ONLY valid JSON — no surrounding conversational text.
"""

# ---------------------------------------------------------------------------
# Pillow image utilities
# ---------------------------------------------------------------------------

def _correct_orientation(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation correction."""
    try:
        exif_data = image._getexif()  # type: ignore[attr-defined]
        if exif_data is None:
            return image
        orientation_key = next(
            (k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None
        )
        if orientation_key is None or orientation_key not in exif_data:
            return image
        rotations = {3: 180, 6: 270, 8: 90}
        angle = rotations.get(exif_data[orientation_key])
        if angle:
            image = image.rotate(angle, expand=True)
    except Exception:
        pass
    return image


def _resize_if_needed(image: Image.Image) -> Image.Image:
    """Scale down so the longest edge does not exceed MAX_DIMENSION."""
    w, h = image.size
    longest = max(w, h)
    if longest <= MAX_DIMENSION:
        return image
    ratio = MAX_DIMENSION / longest
    return image.resize((max(1, int(w * ratio)), max(1, int(h * ratio))), Image.LANCZOS)


def _to_rgb(image: Image.Image) -> Image.Image:
    """Convert any mode to RGB."""
    if image.mode == "RGBA":
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        return bg
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def _check_quality(image: Image.Image) -> Tuple[bool, str]:
    """Basic quality check."""
    w, h = image.size
    if w < MIN_DIMENSION or h < MIN_DIMENSION:
        return False, f"Image too small ({w}×{h} px). Minimum is {MIN_DIMENSION}×{MIN_DIMENSION} px."
    notes: List[str] = []
    try:
        stat = ImageStat.Stat(image)
        mean_brightness = sum(stat.mean[:3]) / 3.0
        stddev = sum(stat.stddev[:3]) / 3.0
        if mean_brightness < 20:
            notes.append("Image appears very dark.")
        elif mean_brightness > 240:
            notes.append("Image appears overexposed.")
        if stddev < 8:
            notes.append("Very low contrast — may be blurry.")
    except Exception:
        pass
    return True, " ".join(notes) if notes else "OK"


def _extract_dominant_colors(image: Image.Image, n: int = 3) -> List[str]:
    """Heuristic dominant colour extraction using Pillow palette quantization."""
    try:
        small = image.copy()
        small.thumbnail((100, 100), Image.LANCZOS)
        quantized = small.quantize(colors=16, method=Image.Quantize.MEDIANCUT)
        palette = quantized.getpalette()
        counts: Dict[int, int] = {}
        for idx in quantized.getdata():
            counts[idx] = counts.get(idx, 0) + 1
        top = sorted(counts, key=lambda i: counts[i], reverse=True)[:n]
        colors: List[str] = []
        for idx in top:
            if palette and idx * 3 + 2 < len(palette):
                colors.append(_rgb_to_name(palette[idx * 3], palette[idx * 3 + 1], palette[idx * 3 + 2]))
        return colors if colors else ["Unknown"]
    except Exception:
        return ["Unknown"]


def _rgb_to_name(r: int, g: int, b: int) -> str:
    """Map an RGB triple to a simple colour family name."""
    rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
    mx, mn = max(rf, gf, bf), min(rf, gf, bf)
    lightness = (mx + mn) / 2.0
    if lightness < 0.15:
        return "Black"
    if lightness > 0.90:
        return "White"
    diff = mx - mn
    saturation = 0.0
    if diff:
        saturation = diff / (2.0 - mx - mn) if lightness > 0.5 else diff / (mx + mn)
    if saturation < 0.12:
        return "Dark Grey" if lightness < 0.45 else ("Grey" if lightness < 0.70 else "Light Grey")
    if not diff:
        return "Grey"
    hue = ((gf - bf) / diff % 6 if mx == rf
           else (bf - rf) / diff + 2 if mx == gf
           else (rf - gf) / diff + 4) * 60
    if hue < 0:
        hue += 360
    if hue < 30:  return "Red"
    if hue < 50:  return "Orange"
    if hue < 75:  return "Yellow"
    if hue < 150: return "Green"
    if hue < 195: return "Cyan"
    if hue < 265: return "Blue"
    if hue < 290: return "Purple"
    if hue < 330: return "Pink"
    return "Red"


def _image_to_data_uri(image: Image.Image) -> str:
    """Encode a PIL Image to a Base64 JPEG data URI."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def validate_and_prepare_image(raw: bytes) -> Tuple[Optional[Image.Image], str, str]:
    """Validate raw bytes and return a prepared PIL Image."""
    if len(raw) > MAX_IMAGE_BYTES:
        size_mb = len(raw) / (1024 * 1024)
        return None, "", f"Image is too large ({size_mb:.1f} MB). Maximum is 10 MB."
    try:
        img = Image.open(io.BytesIO(raw))
        img_format = img.format
    except Exception as exc:
        return None, "", f"Invalid or unreadable image file: {exc}"

    if img_format not in SUPPORTED_FORMATS:
        return None, "", f"Unsupported format '{img_format}'. Please upload JPG or PNG."

    img = _correct_orientation(img)
    img = _resize_if_needed(img)
    img = _to_rgb(img)
    passes, notes = _check_quality(img)
    if not passes:
        return None, "", notes
    return img, notes, ""

# ---------------------------------------------------------------------------
# IBM SDK response parser
# ---------------------------------------------------------------------------

def _parse_sdk_response(response: Any) -> str:
    """Extract text content from an ibm-watsonx-ai ModelInference.chat() response."""
    try:
        choices = response.get("choices") or response.get("results") or []
        if choices:
            first = choices[0]
            msg = first.get("message") or {}
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            clean = str(content).strip()
            # Unescape markdown code fences if wrapped
            if clean.startswith("```"):
                clean = re.sub(r"^```(?:json)?\n?", "", clean)
                clean = re.sub(r"\n?```$", "", clean)
            return clean.strip()
    except Exception:
        pass
    return str(response).strip()


def _classify_api_error(exc: Exception) -> str:
    """User-friendly message for common IBM watsonx SDK errors."""
    msg = str(exc).lower()
    if "not supported" in msg or "multimodal" in msg or "vision" in msg:
        return (
            "The configured model does not support multimodal vision on this endpoint. "
            "Select meta-llama/llama-4-maverick-17b-128e-instruct-fp8 in the sidebar."
        )
    if "401" in msg or "403" in msg or "authentication" in msg or "unauthorized" in msg:
        return "Authentication failed. Check your IBM Cloud API key and Project ID."
    if "404" in msg or "not found" in msg:
        return "Model not found. Please verify the Vision Model ID and Dallas regional URL."
    if "timeout" in msg or "timed out" in msg:
        return "Request timed out. Please check network connection and try again."
    if "quota" in msg or "rate limit" in msg or "429" in msg:
        return "API rate limit or quota exceeded. Wait a few seconds and try again."
    return f"IBM watsonx.ai API error: {exc}"

# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class ItemInspectorAgent:
    """
    Agent 1 — Visual Intake & Attribute Extraction.
    Uses watsonx.ai multimodal models via the ibm-watsonx-ai SDK.
    """

    AGENT_NAME = "Item Inspector"

    def __init__(
        self,
        api_key: str,
        project_id: str,
        url: str,
        model_id: str = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
        guardrails: Optional[GuardrailsManager] = None,
    ) -> None:
        self.api_key = api_key
        self.project_id = project_id
        self.url = url
        self.model_id = model_id
        self.guardrails = guardrails or GuardrailsManager()
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-init ibm-watsonx-ai ModelInference client."""
        if self._client is None:
            try:
                from ibm_watsonx_ai import Credentials
                from ibm_watsonx_ai.foundation_models import ModelInference
            except ImportError as exc:
                raise RuntimeError(
                    "ibm-watsonx-ai is not installed. Run: pip install ibm-watsonx-ai"
                ) from exc
            credentials = Credentials(url=self.url, api_key=self.api_key)
            self._client = ModelInference(
                model_id=self.model_id,
                credentials=credentials,
                project_id=self.project_id,
            )
        return self._client

    def _call_vision_model(self, data_uri: str) -> str:
        """Send multimodal chat request to watsonx.ai."""
        client = self._get_client()
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": _USER_PROMPT},
                ],
            },
        ]
        try:
            response = client.chat(
                messages=messages,
                params={"max_new_tokens": 800, "temperature": 0.1},
            )
        except Exception as exc:
            raise RuntimeError(_classify_api_error(exc)) from exc
        return _parse_sdk_response(response)

    def inspect(
        self,
        raw_image_bytes: bytes,
        verification_notes: str = "",
    ) -> Tuple[Optional[InspectionResult], str, float]:
        """Full Agent 1 inspection pipeline."""
        start = time.perf_counter()

        image, quality_notes, err = validate_and_prepare_image(raw_image_bytes)
        if image is None:
            return None, err, time.perf_counter() - start

        dominant_colors = _extract_dominant_colors(image, n=3)
        data_uri = _image_to_data_uri(image)

        try:
            raw_response = self._call_vision_model(data_uri)
        except RuntimeError as exc:
            return None, str(exc), time.perf_counter() - start

        result, val_err = self.guardrails.validate_inspection_output(raw_response)
        latency = time.perf_counter() - start

        if result is None:
            return None, val_err, latency

        result.image_width = image.width
        result.image_height = image.height
        result.dominant_colors = dominant_colors
        result.image_quality_notes = quality_notes or result.image_quality_notes
        result.verification_notes = verification_notes.strip()

        return result, "", latency

    def test_connection(self) -> Tuple[bool, str]:
        """Test watsonx.ai client initialisation."""
        try:
            self._client = None
            self._get_client()
            return True, f"Vision model ready: {self.model_id}"
        except Exception as exc:
            return False, f"Vision model connection failed: {exc}"
         