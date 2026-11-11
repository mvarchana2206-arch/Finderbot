"""
dashboard/metrics.py
====================
Session telemetry for FinderBot — IBM watsonx.ai integrated version.

Tracks per-session:
  - Image inspection requests (Agent 1).
  - Matching requests (Agent 2).
  - Successful/failed counts for each agent.
  - Explicit handoff success rate.
  - Per-agent latency lists.
  - Guardrail intervention counts (synced from GuardrailsManager).
  - Item registration count (in-session).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, MutableMapping, Optional

__all__ = ["AgentMetrics", "get_metrics"]


@dataclass
class AgentMetrics:
    """
    Per-session telemetry counters.
    Stored in st.session_state["finderbot_metrics"].
    """

    # --- Request counts ---
    total_inspection_requests: int = 0
    total_matching_requests: int = 0

    # --- Success / failure ---
    successful_inspections: int = 0
    failed_inspections: int = 0
    successful_evaluations: int = 0
    failed_evaluations: int = 0

    # --- Handoff tracking ---
    eligible_for_handoff: int = 0
    successful_handoffs: int = 0

    # --- Latency (seconds) ---
    agent1_latencies: List[float] = field(default_factory=list)
    agent2_latencies: List[float] = field(default_factory=list)

    # --- Guardrail events (synced from GuardrailsManager) ---
    guardrail_interventions: int = 0
    pii_redactions: int = 0
    injection_rejections: int = 0

    # --- API failures ---
    api_failure_count: int = 0

    # --- Item registrations (in-session) ---
    items_registered: int = 0

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------

    def record_inspection_start(self) -> None:
        self.total_inspection_requests += 1

    def record_inspection_success(self, latency: float) -> None:
        self.successful_inspections += 1
        self.eligible_for_handoff += 1
        self.agent1_latencies.append(round(latency, 3))

    def record_inspection_failure(self) -> None:
        self.failed_inspections += 1

    def record_matching_start(self) -> None:
        self.total_matching_requests += 1

    def record_matching_success(self, latency: float) -> None:
        self.successful_evaluations += 1
        self.successful_handoffs += 1
        self.agent2_latencies.append(round(latency, 3))

    def record_matching_failure(self) -> None:
        self.failed_evaluations += 1

    def record_api_failure(self) -> None:
        self.api_failure_count += 1

    def record_item_registered(self) -> None:
        self.items_registered += 1

    def sync_from_guardrails(self, guardrails_manager: Any) -> None:
        """Copy current counters from the shared GuardrailsManager instance."""
        if guardrails_manager is not None:
            self.guardrail_interventions = getattr(
                guardrails_manager, "guardrail_interventions", self.guardrail_interventions
            )
            self.pii_redactions = getattr(
                guardrails_manager, "pii_redactions", self.pii_redactions
            )
            self.injection_rejections = getattr(
                guardrails_manager, "injection_rejections", self.injection_rejections
            )

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def handoff_success_rate(self) -> float:
        """
        Percentage of eligible Agent 1 outputs that Agent 2 consumed successfully.
        Returns 0.0 if no eligible handoffs have occurred.
        """
        if self.eligible_for_handoff == 0:
            return 0.0
        return round((self.successful_handoffs / self.eligible_for_handoff) * 100.0, 1)

    @property
    def avg_agent1_latency(self) -> float:
        if not self.agent1_latencies:
            return 0.0
        return round(sum(self.agent1_latencies) / len(self.agent1_latencies), 2)

    @property
    def avg_agent2_latency(self) -> float:
        if not self.agent2_latencies:
            return 0.0
        return round(sum(self.agent2_latencies) / len(self.agent2_latencies), 2)

    @property
    def avg_total_latency(self) -> float:
        all_lat = self.agent1_latencies + self.agent2_latencies
        if not all_lat:
            return 0.0
        return round(sum(all_lat) / len(all_lat), 2)

    def to_summary_dict(self) -> Dict[str, Any]:
        """Return a flat dict of all session metrics for the dashboard table."""
        return {
            "Total Inspection Requests":    self.total_inspection_requests,
            "Total Matching Requests":      self.total_matching_requests,
            "Successful Inspections":       self.successful_inspections,
            "Failed Inspections":           self.failed_inspections,
            "Successful Evaluations":       self.successful_evaluations,
            "Failed Evaluations":           self.failed_evaluations,
            "Eligible for Handoff":         self.eligible_for_handoff,
            "Successful Handoffs":          self.successful_handoffs,
            "Handoff Success Rate (%)":     self.handoff_success_rate,
            "Avg Agent 1 Latency (s)":      self.avg_agent1_latency,
            "Avg Agent 2 Latency (s)":      self.avg_agent2_latency,
            "Avg Total Latency (s)":        self.avg_total_latency,
            "Guardrail Interventions":      self.guardrail_interventions,
            "PII Redactions":               self.pii_redactions,
            "Injection Rejections":         self.injection_rejections,
            "API Failures":                 self.api_failure_count,
            "Items Registered (session)":   self.items_registered,
        }


# ---------------------------------------------------------------------------
# Session-state helper
# ---------------------------------------------------------------------------

def get_metrics(session_state: MutableMapping[str, Any]) -> AgentMetrics:
    """
    Retrieve (or initialise) the AgentMetrics object from Streamlit session_state.
    Compatible with st.session_state mapping.
    """
    if "finderbot_metrics" not in session_state:
        session_state["finderbot_metrics"] = AgentMetrics()
    return session_state["finderbot_metrics"]