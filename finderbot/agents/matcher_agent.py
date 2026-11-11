"""
agents/matcher_agent.py
=======================
Agent 2 — Item Matcher (IBM watsonx.ai integrated)

Responsibilities:
  - Receive the validated InspectionResult from Agent 1 (explicit handoff).
  - Accept claimant's lost-item description and location.
  - Run a transparent Python weighted scoring algorithm first.
  - Pass Agent 1's structured JSON and the Python pre-scores to
    IBM watsonx.ai foundation models via the ibm-watsonx-ai SDK.
  - The LLM refines the score, identifies matched/unmatched aspects,
    generates anti-fraud challenge questions, and writes the staff advisory.
  - Validate the LLM response with Pydantic (via GuardrailsManager).
  - Fall back to Python-only result if the LLM response is malformed.
  - Separate claimant-facing output from staff-only reference answers.
  - Return a validated MatchResult — never simulate AI output.

Matching weights (used by the Python pre-scorer):
  Category / type          20%
  Colour                   15%
  Material                 10%
  Distinctive marks        25%
  Location context         15%
  Description overlap      15%

Confidence tiers:
  High     = 85–100
  Probable = 60–84
  Unlikely = 0–59
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from guardrails.safety import (
    ChallengeQuestion,
    GuardrailViolation,
    GuardrailsManager,
    InspectionResult,
    MatchResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Matching weights  (must sum to 1.0)
# ---------------------------------------------------------------------------

WEIGHTS: Dict[str, float] = {
    "category_type":     0.20,
    "color":             0.15,
    "material":          0.10,
    "distinctive_marks": 0.25,
    "location_context":  0.15,
    "description":       0.15,
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a professional lost-and-found reconciliation officer.
You receive AI-extracted visual attributes of a found item, pre-calculated
Python matching sub-scores, and a claimant's verbal description.

Your responsibilities:
1. Review the Python pre-scores and apply semantic adjustments (±10 pts each) where warranted.
2. Calculate a final weighted match_score (0–100).
3. List matched aspects and discrepancies with clear explanations.
4. Generate 2–3 anti-fraud verification questions based ONLY on visible attributes.
5. Write a staff advisory and choose a recommended action.

Rules:
- NEVER reveal reference answers in claimant-facing question text.
- Do not invent hidden or unverifiable features.
- Always treat "Immediate Release" as requiring mandatory staff review.
- Return ONLY a valid JSON object — no commentary, no markdown fences.
"""

_USER_PROMPT_TEMPLATE = """\
## Found Item (AI-extracted visual attributes)
{inspection_json}

## Claimant Description
Description: {claimant_description}
Reported lost location: {lost_location}
Additional details: {additional_details}

## Found Location (registered)
{found_location}

## Python Pre-Scores (keyword overlap — use as starting point)
{score_breakdown}

Instructions:
- Adjust each sub-score by up to ±10 points for semantic nuance (synonyms, paraphrasing).
- Calculate the weighted match_score using the weights shown.
- Populate all fields in the schema below.

## Required Output Schema (return ONLY this JSON, nothing else):
{{
  "match_score": <integer 0–100>,
  "confidence_rating": "<High|Probable|Unlikely>",
  "matched_aspects": ["<aspect>"],
  "discrepancies": ["<discrepancy>"],
  "challenge_questions": [
    {{
      "question": "<claimant-facing question — do NOT reveal the answer>",
      "reference_answer": "<staff-only — the expected answer>"
    }}
  ],
  "staff_advisory": "<detailed advisory for staff>",
  "recommended_action": "<Immediate Claim In Person|Reject Release|Verify>",
  "score_breakdown": {{
    "category_type": <0–100>,
    "color": <0–100>,
    "material": <0–100>,
    "distinctive_marks": <0–100>,
    "location_context": <0–100>,
    "description": <0–100>,
    "weighted_total": <0–100>
  }}
}}

Anti-fraud question rules:
- Base questions ONLY on attributes present in the found item JSON.
- Use non-obvious details where possible — never ask yes/no questions that reveal the answer.
- Correct example: "Can you describe any marks or imperfections on the device?"

Confidence:  High = 85–100  |  Probable = 60–84  |  Unlikely = 0–59
Scores are preliminary estimates. Label them as such in the staff_advisory.
"""


# ---------------------------------------------------------------------------
# Python pre-scorer
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def _word_overlap(a: str, b: str) -> float:
    """Jaccard word overlap (0.0–1.0). Returns 0.0 if either string is empty."""
    if not a or not b:
        return 0.0
    wa = set(_normalise(a).split())
    wb = set(_normalise(b).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _list_vs_text(items: List[str], text: str) -> float:
    """Fraction of list phrases that have a significant word in text."""
    if not items or not text:
        return 0.0
    norm = _normalise(text)
    matched = sum(
        1 for phrase in items
        if any(w in norm for w in _normalise(phrase).split() if len(w) > 3)
    )
    return matched / len(items)


def calculate_scores(
    inspection: InspectionResult,
    claimant_description: str,
    lost_location: str,
    found_location: str,
    additional_details: str,
) -> Dict[str, Any]:
    """Compute transparent 0–100 sub-scores using Python keyword overlap."""
    combined = f"{claimant_description} {additional_details}".strip()
    found_type = f"{inspection.category} {inspection.item_type}".strip()
    found_color = f"{inspection.primary_color} {inspection.secondary_color}".strip()

    breakdown: Dict[str, Any] = {
        "category_type":     round(_word_overlap(found_type, combined) * 100),
        "color":             round(_word_overlap(found_color, combined) * 100),
        "material":          round(_word_overlap(inspection.material, combined) * 100),
        "distinctive_marks": round(_list_vs_text(inspection.distinctive_marks, combined) * 100),
        "location_context":  round(_word_overlap(found_location, lost_location) * 100),
        "description":       round(_word_overlap(found_type + " " + found_color, combined) * 100),
    }
    breakdown["weighted_total"] = round(
        sum(breakdown[k] * WEIGHTS[k] for k in WEIGHTS)
    )
    return breakdown


def classify_confidence(score: int) -> str:
    if score >= 85: return "High"
    if score >= 60: return "Probable"
    return "Unlikely"


def _python_fallback(
    inspection: InspectionResult,
    breakdown: Dict[str, Any],
    val_err: str,
) -> MatchResult:
    """Construct a minimal MatchResult from Python scores when LLM validation fails."""
    score = breakdown["weighted_total"]
    conf = classify_confidence(score)
    questions: List[ChallengeQuestion] = []
    if inspection.verification_notes:
        questions.append(ChallengeQuestion(
            question="Can you describe any unique identifying features of the item?",
            reference_answer=f"Staff note: {inspection.verification_notes}",
        ))
    if inspection.distinctive_marks:
        questions.append(ChallengeQuestion(
            question="Please describe the condition and surface markings of the item.",
            reference_answer=f"Expected mention of: {inspection.distinctive_marks[0]}",
        ))
    if not questions:
        questions.append(ChallengeQuestion(
            question="Can you describe the item in detail including colour and any marks?",
            reference_answer="No specific verification notes available.",
        ))
    action = "Immediate Release" if score >= 85 else ("Verify In Person" if score >= 60 else "Reject Claim")
    return MatchResult(
        match_score=score,
        confidence_rating=conf,
        matched_aspects=["Python keyword-overlap scoring (LLM response fallback)"],
        discrepancies=[f"LLM analysis unavailable: {val_err}"] if val_err else [],
        challenge_questions=questions,
        staff_advisory=(
            f"IMPORTANT: LLM analysis was unavailable ({val_err}). "
            f"Score {score}/100 ({conf}) is based on Python keyword overlap only. "
            "Manual review is strongly recommended before any action."
        ),
        recommended_action=action,
        score_breakdown=breakdown,
    )


# ---------------------------------------------------------------------------
# IBM SDK helpers
# ---------------------------------------------------------------------------

def _parse_sdk_response(response: Any) -> str:
    """Extract text content from an ibm-watsonx-ai response and strip markdown fences."""
    try:
        choices = response.get("choices") or response.get("results") or []
        if choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            raw_text = str(content).strip()
            # Remove ```json and ``` if present
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)
            return raw_text.strip()
    except Exception:
        pass
    clean = str(response).strip()
    clean = re.sub(r"^```(?:json)?\s*", "", clean)
    clean = re.sub(r"\s*```$", "", clean)
    return clean.strip()


def _classify_api_error(exc: Exception) -> str:
    msg = str(exc).lower()
    if "401" in msg or "403" in msg or "authentication" in msg or "unauthorized" in msg:
        return "Authentication failed. Check your IBM Cloud API key and Project ID."
    if "404" in msg or "not found" in msg:
        return "Model not found. Verify the Text Model ID and Dallas regional URL."
    if "timeout" in msg or "timed out" in msg:
        return "Request timed out. Check your network connection and try again."
    if "quota" in msg or "rate limit" in msg or "429" in msg:
        return "API rate limit or quota exceeded. Wait a moment and try again."
    return f"IBM watsonx.ai API error: {exc}"


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class MatcherAgent:
    """
    Agent 2 — Item Matcher (IBM watsonx.ai integrated).
    """

    AGENT_NAME = "Item Matcher"

    def __init__(
        self,
        api_key: str,
        project_id: str,
        url: str,
        model_id: str = "meta-llama/llama-3-3-70b-instruct",
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

    def _call_text_model(self, user_content: str) -> str:
        """Send chat request to watsonx.ai foundation model."""
        client = self._get_client()
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]
        try:
            response = client.chat(
                messages=messages,
                params={"max_new_tokens": 1200, "temperature": 0.1},
            )
        except Exception as exc:
            raise RuntimeError(_classify_api_error(exc)) from exc
        return _parse_sdk_response(response)

    def match(
        self,
        inspection: InspectionResult,
        found_location: str,
        claimant_description: str,
        lost_location: str,
        additional_details: str = "",
    ) -> Tuple[MatchResult, str, float]:
        """Full Agent 2 pipeline."""
        start = time.perf_counter()

        # 1. Agent 1 validation check
        if not inspection.is_valid:
            result = MatchResult(
                match_score=0,
                confidence_rating="Unlikely",
                staff_advisory=f"Agent 1 rejected item: {inspection.rejection_reason}",
                recommended_action="Reject Claim",
            )
            return result, "Agent 1 rejected the item image.", time.perf_counter() - start

        # 2. Input sanitization
        combined = f"{claimant_description} {lost_location} {additional_details}"
        _, flags = self.guardrails.sanitize_input(combined)
        if not flags["is_safe"]:
            result = MatchResult(
                match_score=0,
                confidence_rating="Unlikely",
                staff_advisory=f"Claim rejected by guardrails: {flags['injection_reason']}",
                recommended_action="Reject Claim",
            )
            return result, flags["injection_reason"], time.perf_counter() - start

        # 3. PII redaction
        claimant_description, _ = self.guardrails.redact_pii(claimant_description)
        lost_location, _        = self.guardrails.redact_pii(lost_location)
        additional_details, _   = self.guardrails.redact_pii(additional_details)

        # 4. Python pre-scoring
        breakdown = calculate_scores(
            inspection=inspection,
            claimant_description=claimant_description,
            lost_location=lost_location,
            found_location=found_location,
            additional_details=additional_details,
        )

        # 5. Build prompt
        inspection_json = json.dumps(inspection.to_public_dict(), indent=2)
        score_text = "\n".join(
            f"  {k}: {v}  (weight: {WEIGHTS.get(k, 'N/A')})"
            for k, v in breakdown.items()
        )
        user_content = _USER_PROMPT_TEMPLATE.format(
            inspection_json=inspection_json,
            claimant_description=claimant_description,
            lost_location=lost_location,
            additional_details=additional_details or "None",
            found_location=found_location,
            score_breakdown=score_text,
        )

        # 6. Execute LLM call
        try:
            raw_response = self._call_text_model(user_content)
        except RuntimeError as exc:
            latency = time.perf_counter() - start
            fallback = _python_fallback(inspection, breakdown, str(exc))
            return fallback, str(exc), latency

        # 7. Validate output schema
        result, val_err = self.guardrails.validate_match_output(raw_response)
        latency = time.perf_counter() - start

        if result is None:
            logger.warning("Agent 2 LLM validation failed: %s", val_err)
            fallback = _python_fallback(inspection, breakdown, val_err)
            return fallback, f"Warning: LLM output invalid — {val_err}", latency

        if not result.score_breakdown:
            result.score_breakdown = breakdown

        return result, "", latency

    def test_connection(self) -> Tuple[bool, str]:
        """Test watsonx.ai client initialisation."""
        try:
            self._client = None
            self._get_client()
            return True, f"Text model ready: {self.model_id}"
        except Exception as exc:
            return False, f"Text model connection failed: {exc}"


# Alias for backward compatibility with app.py imports
ItemMatcherAgent = MatcherAgent