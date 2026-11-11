"""
guardrails/safety.py
====================
FinderBot guardrails — IBM watsonx.ai integrated version.

Responsibilities:
  - Prompt injection / manipulation phrase detection.
  - PII masking (phone numbers, national IDs, credit card numbers, SSN, passport).
  - JSON extraction from raw LLM responses (strips markdown fences, prose).
  - Pydantic schema validation for Agent 1 and Agent 2 outputs.
  - Blind-shield: staff-only verification notes and reference answers never
    reach claimant-facing output.
  - Telemetry counters consumed by dashboard/metrics.py.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

__all__ = [
    "GuardrailViolation",
    "InspectionResult",
    "ChallengeQuestion",
    "MatchResult",
    "GuardrailsManager",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class GuardrailViolation(Exception):
    """Raised when an input or output violates guardrail constraints."""
    pass


# ---------------------------------------------------------------------------
# Pydantic schemas — validated output contracts for both agents
# ---------------------------------------------------------------------------

class InspectionResult(BaseModel):
    """
    Validated output from Agent 1 (Item Inspector).
    """

    is_valid: bool = Field(default=True, description="True when image was analysed successfully.")
    rejection_reason: str = Field(default="", description="Non-empty when is_valid is False.")

    category: str = Field(default="", description="General object category.")
    item_type: str = Field(default="", description="Specific item type.")
    primary_color: str = Field(default="", description="Dominant colour.")
    secondary_color: str = Field(default="", description="Secondary colour if visible.")
    material: str = Field(default="", description="Visible material.")
    physical_traits: List[str] = Field(default_factory=list)
    distinctive_marks: List[str] = Field(default_factory=list)

    verification_notes: str = Field(default="", description="Staff-only notes.")

    image_width: int = Field(default=0)
    image_height: int = Field(default=0)
    dominant_colors: List[str] = Field(default_factory=list)
    image_quality_notes: str = Field(default="")

    hidden_or_obscured_features: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _fill_rejection_reason(self) -> "InspectionResult":
        if not self.is_valid and not self.rejection_reason:
            self.rejection_reason = "Image did not meet quality requirements."
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Full dict — includes verification_notes (staff use only)."""
        return self.model_dump()

    def to_public_dict(self) -> Dict[str, Any]:
        """Claimant-safe dict — strips verification_notes."""
        d = self.model_dump()
        d.pop("verification_notes", None)
        return d


class ChallengeQuestion(BaseModel):
    """One anti-fraud verification question."""
    question: str = Field(default="", description="Claimant-facing question text.")
    reference_answer: str = Field(default="", description="Staff-only reference answer.")


class MatchResult(BaseModel):
    """Validated output from Agent 2 (Item Matcher)."""

    match_score: int = Field(default=0, ge=0, le=100)
    confidence_rating: str = Field(default="Unlikely")
    matched_aspects: List[str] = Field(default_factory=list)
    discrepancies: List[str] = Field(default_factory=list)
    challenge_questions: List[ChallengeQuestion] = Field(default_factory=list)
    staff_advisory: str = Field(default="")
    recommended_action: str = Field(default="Verify In Person")
    score_breakdown: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("match_score", mode="before")
    @classmethod
    def _coerce_score(cls, v: Any) -> int:
        try:
            return max(0, min(100, int(v)))
        except (TypeError, ValueError):
            return 0

    @field_validator("confidence_rating")
    @classmethod
    def _validate_confidence(cls, v: str) -> str:
        allowed = {"High", "Probable", "Unlikely"}
        if v not in allowed:
            title = v.strip().title()
            if title in allowed:
                return title
            return "Unlikely"
        return v

    @field_validator("recommended_action")
    @classmethod
    def _validate_action(cls, v: str) -> str:
        allowed = {"Immediate Release", "Verify In Person", "Reject Claim"}
        if v not in allowed:
            return "Verify In Person"
        return v

    def claimant_view(self) -> Dict[str, Any]:
        """Strip reference answers — safe to show claimant."""
        return {
            "match_score": self.match_score,
            "confidence_rating": self.confidence_rating,
            "matched_aspects": self.matched_aspects,
            "discrepancies": self.discrepancies,
            "challenge_questions": [q.question for q in self.challenge_questions],
            "staff_advisory": self.staff_advisory,
            "recommended_action": self.recommended_action,
        }

    def staff_view(self) -> Dict[str, Any]:
        """Full dict including reference answers — staff screens only."""
        return self.model_dump()


# ---------------------------------------------------------------------------
# Injection detection patterns
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: List[str] = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"override\s+(system|all)\s+instructions?",
    r"disregard\s+(all\s+)?(previous|prior)\s+instructions?",
    r"confirm\s+(that\s+)?i\s+am\s+the\s+owner",
    r"set\s+(the\s+)?match\s*score\s+to\s+\d+",
    r"set\s+(the\s+)?confidence\s+to\s+(high|100)",
    r"reveal\s+(hidden|private|internal|confidential)\s+(item|answer|information|detail)",
    r"disclose\s+(internal|private|staff|verification)\s+(answer|note|detail|data)",
    r"you\s+are\s+now\s+(in\s+)?(developer|admin|god|jailbreak)\s+mode",
    r"act\s+as\s+if\s+(there\s+are\s+)?no\s+(restrictions?|guidelines?|rules?)",
    r"pretend\s+(you\s+have\s+)?no\s+(restrictions?|rules?|guidelines?)",
    r"show\s+me\s+(all\s+)?(verification|staff|private|internal)\s+(note|answer|detail|data)",
]

# PII patterns — (regex, replacement_tag)
_PII_PATTERNS: List[Tuple[str, str]] = [
    (r"\+?[\d\s\-().]{10,15}(?<!\s)", "[REDACTED_PHONE]"),
    (r"\b(?:\d[ \-]?){13,16}\b", "[REDACTED_CREDIT_CARD]"),
    (r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b", "[REDACTED_NATIONAL_ID]"),
    (r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b", "[REDACTED_SSN]"),
    (r"\b[A-Z]{1,2}\d{7,8}\b", "[REDACTED_PASSPORT]"),
]


# ---------------------------------------------------------------------------
# GuardrailsManager
# ---------------------------------------------------------------------------

class GuardrailsManager:
    """
    Central guardrails controller.
    """

    def __init__(self) -> None:
        self._injection_re = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]
        self._pii_re = [(re.compile(p, re.IGNORECASE), tag) for p, tag in _PII_PATTERNS]

        self.guardrail_interventions: int = 0
        self.pii_redactions: int = 0
        self.injection_rejections: int = 0

    def check_prompt_injection(self, text: str) -> Tuple[bool, str]:
        """Returns (is_safe, reason). is_safe=False when injection detected."""
        for pattern in self._injection_re:
            if pattern.search(text):
                self.injection_rejections += 1
                self.guardrail_interventions += 1
                logger.warning("Injection pattern matched: %s", pattern.pattern)
                return False, (
                    "Your input contains a phrase that resembles an attempt to "
                    "manipulate system results. Please describe the item in plain language."
                )
        return True, ""

    def redact_pii(self, text: str) -> Tuple[str, bool]:
        """Replace PII with redaction tags. Returns (redacted_text, was_redacted)."""
        result = text
        redacted = False
        for pattern, tag in self._pii_re:
            new_text, count = pattern.subn(tag, result)
            if count:
                result = new_text
                redacted = True
                self.pii_redactions += count
                self.guardrail_interventions += count
        return result, redacted

    def sanitize_input(self, text: str) -> Tuple[str, Dict[str, Any]]:
        is_safe, reason = self.check_prompt_injection(text)
        if not is_safe:
            return text, {"is_safe": False, "injection_reason": reason, "pii_redacted": False}
        sanitized, pii_redacted = self.redact_pii(text)
        return sanitized, {"is_safe": True, "injection_reason": "", "pii_redacted": pii_redacted}

    @staticmethod
    def extract_json(raw: str) -> Optional[Dict[str, Any]]:
        """Extract first JSON object from a raw LLM response."""
        if not raw:
            return None
        cleaned = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE)
        cleaned = cleaned.replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start: end + 1])
            except json.JSONDecodeError:
                pass
        return None

    def validate_inspection_output(
        self, raw: str
    ) -> Tuple[Optional[InspectionResult], str]:
        data = self.extract_json(raw)
        if data is None:
            self.guardrail_interventions += 1
            return None, "Agent 1 response could not be parsed as JSON."

        if "is_clear" in data and "is_valid" not in data:
            data["is_valid"] = data.pop("is_clear")
        if "image_clarity_notes" in data and "image_quality_notes" not in data:
            data["image_quality_notes"] = data.pop("image_clarity_notes")

        try:
            result = InspectionResult(**data)
            return result, ""
        except Exception as exc:
            self.guardrail_interventions += 1
            logger.warning("InspectionResult validation failed: %s", exc)
            return None, f"Agent 1 schema validation failed: {exc}"

    def validate_match_output(
        self, raw: str
    ) -> Tuple[Optional[MatchResult], str]:
        data = self.extract_json(raw)
        if data is None:
            self.guardrail_interventions += 1
            return None, "Agent 2 response could not be parsed as JSON."

        if "challenge_questions" in data:
            normalised: List[Any] = []
            for item in data["challenge_questions"]:
                if isinstance(item, str):
                    normalised.append({"question": item, "reference_answer": ""})
                elif isinstance(item, dict):
                    normalised.append(item)
            data["challenge_questions"] = normalised

        try:
            result = MatchResult(**data)
            return result, ""
        except Exception as exc:
            self.guardrail_interventions += 1
            logger.warning("MatchResult validation failed: %s", exc)
            return None, f"Agent 2 schema validation failed: {exc}"

    @staticmethod
    def claimant_safe_match(match: MatchResult) -> Dict[str, Any]:
        return match.claimant_view()

    @staticmethod
    def staff_full_match(match: MatchResult) -> Dict[str, Any]:
        return match.staff_view()