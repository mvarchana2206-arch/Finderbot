"""
FinderBot: Smart Lost & Found — IBM watsonx.ai Integrated Application

Three tabs:
  Tab 1: 📦 Intake Found Item   — image upload → Agent 1 (Multimodal Vision) → register
  Tab 2: 🔎 Match & Claim       — select item → Agent 2 (Reasoning LLM) → advisory
  Tab 3: 📊 Operations          — real telemetry dashboard from session + database

Sidebar:
  - IBM watsonx.ai credential panel (API key, Project ID, URL, model IDs).
  - Connection test for both models.
  - Administrator login / logout.

Security:
  - API keys entered via password-type inputs — never logged or displayed.
  - Sidebar credentials override .env at runtime.
  - Staff-only verification notes and reference answers never reach claimants.
  - Every match result displays the HITL disclaimer.
  - Release requires administrator password authentication.
"""

from __future__ import annotations

import io
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from agents.inspector_agent import ItemInspectorAgent
from agents.matcher_agent import ItemMatcherAgent
from dashboard.metrics import AgentMetrics, get_metrics
from guardrails.safety import GuardrailsManager, InspectionResult
from storage import (
    DatabaseManager,
    STATUS_FOUND,
    STATUS_UNDER_VERIFICATION,
    STATUS_CLAIM_APPROVED,
    STATUS_RELEASED,
    STATUS_REJECTED,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)
logger = logging.getLogger("finderbot.app")

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="FinderBot — Smart Lost & Found",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS — minimal professional styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .main .block-container { max-width: 1100px; padding-top: 1.5rem; }
    h1, h2, h3 { color: #0f172a; }
    .stMetric > label { font-size: 0.8rem !important; color: #57606a !important; }
    .stMetric > div  { font-size: 1.4rem !important; font-weight: 700 !important; }
    .hitl-banner {
        background: #fef3c7; border: 1px solid #d97706;
        border-radius: 6px; padding: 0.6rem 1rem;
        font-size: 0.85rem; color: #92400e; font-weight: 600;
        margin-bottom: 1rem;
    }
    .question-card {
        background: #f0f7ff; border-left: 4px solid #3b82d4;
        padding: 0.75rem 1rem; border-radius: 4px; margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Supported Models Configuration (Dallas us-south instance)
# ---------------------------------------------------------------------------

SUPPORTED_VISION_MODELS = [
    "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
    "meta-llama/llama-3-1-8b",
    "ibm/granite-4-h-small",
]

SUPPORTED_TEXT_MODELS = [
    "meta-llama/llama-3-3-70b-instruct",
    "mistralai/mistral-small-3-1-24b-instruct-2503",
    "meta-llama/llama-3-1-8b",
    "ibm/granite-4-h-small",
    "ibm/granite-3-1-8b-base",
]

# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults: Dict[str, Any] = {
        "admin_logged_in":      False,
        "inspection_result":    None,
        "inspection_image":     None,
        "inspection_location":  "",
        "inspection_notes":     "",
        "match_result":         None,
        "current_item_id":      None,
        "current_eval_id":      None,
        "connection_status":    None,
        "connection_message":   "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()

# ---------------------------------------------------------------------------
# Shared singleton objects
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_db() -> DatabaseManager:
    db_path = os.getenv("DATABASE_PATH", "data/finderbot.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return DatabaseManager(db_path=db_path)


@st.cache_resource
def _get_guardrails() -> GuardrailsManager:
    return GuardrailsManager()


def _get_metrics() -> AgentMetrics:
    return get_metrics(st.session_state)


# ---------------------------------------------------------------------------
# Credential resolution
# ---------------------------------------------------------------------------

def _resolve_credentials() -> Dict[str, str]:
    """Sidebar values override .env. Keys never appear in logs or UI text."""
    return {
        "api_key":          st.session_state.get("sb_api_key")      or os.getenv("WATSONX_API_KEY", ""),
        "project_id":       st.session_state.get("sb_project_id")   or os.getenv("WATSONX_PROJECT_ID", ""),
        "url":              st.session_state.get("sb_url")          or os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
        "vision_model_id":  st.session_state.get("sb_vision_model") or os.getenv("WATSONX_VISION_MODEL_ID", SUPPORTED_VISION_MODELS[0]),
        "text_model_id":    st.session_state.get("sb_text_model")   or os.getenv("WATSONX_TEXT_MODEL_ID", SUPPORTED_TEXT_MODELS[0]),
    }


def _credentials_available(creds: Dict[str, str]) -> bool:
    return bool(creds["api_key"] and creds["project_id"] and creds["url"])


def _build_inspector(creds: Dict[str, str]) -> ItemInspectorAgent:
    return ItemInspectorAgent(
        api_key=creds["api_key"],
        project_id=creds["project_id"],
        url=creds["url"],
        model_id=creds["vision_model_id"],
        guardrails=_get_guardrails(),
    )


def _build_matcher(creds: Dict[str, str]) -> ItemMatcherAgent:
    return ItemMatcherAgent(
        api_key=creds["api_key"],
        project_id=creds["project_id"],
        url=creds["url"],
        model_id=creds["text_model_id"],
        guardrails=_get_guardrails(),
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🔍 FinderBot")
        st.markdown("**Smart Lost & Found**  \n*IBM watsonx.ai Powered*")
        st.divider()

        # ---- IBM credentials ----
        st.subheader("🔑 IBM watsonx.ai")

        st.text_input(
            "API Key",
            type="password",
            key="sb_api_key",
            placeholder="Paste your IBM Cloud API key",
            help="IBM Cloud API key. Overrides WATSONX_API_KEY in .env.",
        )
        st.text_input(
            "Project ID",
            key="sb_project_id",
            placeholder="watsonx.ai project UUID",
            help="Your watsonx.ai project ID. Overrides WATSONX_PROJECT_ID in .env.",
        )
        st.text_input(
            "Endpoint URL",
            key="sb_url",
            value=os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
            placeholder="https://us-south.ml.cloud.ibm.com",
            help="Regional watsonx.ai endpoint. Overrides WATSONX_URL in .env.",
        )

        st.selectbox(
            "Vision Model ID (Agent 1)",
            options=SUPPORTED_VISION_MODELS,
            index=0,
            key="sb_vision_model",
            help="Agent 1 model for visual extraction.",
        )

        st.selectbox(
            "Text Model ID (Agent 2)",
            options=SUPPORTED_TEXT_MODELS,
            index=0,
            key="sb_text_model",
            help="Agent 2 model for semantic claim matching.",
        )

        if st.button("🔌 Test Connection", use_container_width=True):
            creds = _resolve_credentials()
            if not _credentials_available(creds):
                st.session_state["connection_status"]  = False
                st.session_state["connection_message"] = (
                    "Credentials incomplete. Enter API key, Project ID, and URL."
                )
            else:
                with st.spinner("Testing connection…"):
                    try:
                        ok1, msg1 = _build_inspector(creds).test_connection()
                        ok2, msg2 = _build_matcher(creds).test_connection()
                        if ok1 and ok2:
                            st.session_state["connection_status"]  = True
                            st.session_state["connection_message"] = (
                                f"✅ Vision: {creds['vision_model_id']}\n"
                                f"✅ Text:   {creds['text_model_id']}"
                            )
                        else:
                            st.session_state["connection_status"]  = False
                            st.session_state["connection_message"] = (
                                f"Vision: {msg1}\nText: {msg2}"
                            )
                    except Exception as exc:
                        st.session_state["connection_status"]  = False
                        st.session_state["connection_message"] = str(exc)

        status = st.session_state.get("connection_status")
        if status is True:
            st.success("Connected")
            st.caption(st.session_state.get("connection_message", ""))
        elif status is False:
            st.error("Not connected")
            st.caption(st.session_state.get("connection_message", ""))
        else:
            st.info("Connection not tested yet.")

        st.divider()

        # ---- Admin login ----
        st.subheader("🔐 Administrator")
        if st.session_state.get("admin_logged_in"):
            st.success("✅ Logged in as Administrator")
            if st.button("Logout", use_container_width=True):
                st.session_state["admin_logged_in"] = False
                st.rerun()
        else:
            admin_pw = st.text_input(
                "Admin Password", type="password",
                key="sb_admin_pw",
                placeholder="Enter admin password",
                help="Set ADMIN_PASSWORD in your .env file.",
            )
            if st.button("Login as Admin", use_container_width=True):
                expected = os.getenv("ADMIN_PASSWORD", "CHANGE_THIS_PASSWORD")
                if admin_pw and admin_pw == expected:
                    st.session_state["admin_logged_in"] = True
                    st.rerun()
                else:
                    st.error("Incorrect password.")

        st.divider()
        db = _get_db()
        st.markdown(
            "**FinderBot v3.0** — IBM watsonx.ai  \n"
            "Meta Llama 4 / 3.3 + Granite  \n\n"
            "⚠️ Prototype — not production-ready."
        )
        st.caption(f"📦 {db.count_items()} items  |  🔎 {db.count_evaluations()} evaluations")


# ---------------------------------------------------------------------------
# Tab 1: Intake Found Item
# ---------------------------------------------------------------------------

def render_intake_tab() -> None:
    db    = _get_db()
    gm    = _get_guardrails()
    metr  = _get_metrics()

    st.header("📦 Intake Found Item")
    st.markdown(
        "Upload a photograph. **Agent 1** will analyse the image "
        "and extract visual attributes automatically."
    )

    # ---- Image capture ---------------------------------------------------
    st.subheader("1️⃣  Image")
    upload_method = st.radio(
        "Capture method", ["Upload file", "Camera"],
        horizontal=True, label_visibility="collapsed",
    )
    raw_bytes: Optional[bytes] = None
    col_left, col_right = st.columns([1, 1])

    with col_left:
        if upload_method == "Upload file":
            uploaded = st.file_uploader(
                "Choose a JPG or PNG", type=["jpg", "jpeg", "png"], help="Maximum 10 MB.",
            )
            if uploaded:
                raw_bytes = uploaded.getvalue()
        else:
            cam = st.camera_input("Take a photo")
            if cam:
                raw_bytes = cam.getvalue()

    with col_right:
        if raw_bytes:
            try:
                preview = Image.open(io.BytesIO(raw_bytes))
                st.image(preview, caption="Preview", use_container_width=True)
                st.caption(
                    f"Size: {len(raw_bytes)//1024} KB  |  "
                    f"{preview.width}×{preview.height} px  |  "
                    f"Format: {preview.format or 'Unknown'}"
                )
            except Exception:
                st.warning("Could not display preview.")
        else:
            st.info("No image selected yet.")

    # ---- Found location --------------------------------------------------
    st.subheader("2️⃣  Found Location")
    found_location = st.text_input(
        "Where was this item found?",
        placeholder="e.g. Building A, Reception desk",
        key="intake_location",
    )

    # ---- Private staff notes (staff-only — NOT sent to AI) ---------------
    st.subheader("3️⃣  Private Verification Notes (Staff Only)")
    st.caption(
        "These notes are used to generate anti-fraud challenge questions. "
        "They are **never** sent to the AI model or shown to claimants."
    )
    verification_notes = st.text_area(
        "Verification notes",
        placeholder="e.g. Owner's initials 'MK' scratched on underside. Serial starts with SN7.",
        height=80,
        key="intake_notes",
    )

    st.divider()

    # ---- Analyse button --------------------------------------------------
    if st.button("🔬 Analyse with Agent 1", type="primary", use_container_width=True):
        if raw_bytes is None:
            st.error("Please upload or capture an image first.")
            return
        if not found_location.strip():
            st.error("Please enter the found location.")
            return

        creds = _resolve_credentials()
        if not _credentials_available(creds):
            st.error(
                "IBM watsonx.ai credentials are not configured. "
                "Enter your API key, Project ID, and URL in the sidebar."
            )
            return

        inspector = _build_inspector(creds)
        metr.record_inspection_start()

        with st.spinner("🤖 Agent 1 analysing image…"):
            result, err, latency = inspector.inspect(
                raw_image_bytes=raw_bytes,
                verification_notes=verification_notes,
            )
            metr.sync_from_guardrails(gm)

        if result is None:
            metr.record_inspection_failure()
            metr.record_api_failure()
            st.error(f"❌ Agent 1 failed: {err}")
            return

        metr.record_inspection_success(latency)

        if result.image_quality_notes and result.image_quality_notes != "OK":
            st.warning(f"⚠️ Image quality: {result.image_quality_notes}")

        if not result.is_valid:
            st.warning(
                f"⚠️ Agent 1 rejected this image: {result.rejection_reason}  \n"
                "Please upload a clearer photograph."
            )
            st.session_state["inspection_result"] = result
            return

        st.session_state["inspection_result"]   = result
        st.session_state["inspection_image"]    = raw_bytes
        st.session_state["inspection_location"] = found_location.strip()
        st.session_state["inspection_notes"]    = verification_notes.strip()
        st.success(f"✅ Agent 1 completed in {latency:.2f}s")

    # ---- Show results and save button ------------------------------------
    if st.session_state.get("inspection_result"):
        result = st.session_state["inspection_result"]
        _render_inspection_result(result)

        if result.is_valid:
            st.divider()
            if st.button("💾 Save Item to Database", type="primary", use_container_width=True):
                location = st.session_state.get("inspection_location", "")
                if not location:
                    st.error("Found location is missing. Please re-run Agent 1.")
                    return
                r: InspectionResult = st.session_state["inspection_result"]
                raw = st.session_state.get("inspection_image")
                try:
                    item_id = db.register_item(
                        category=r.category,
                        item_type=r.item_type,
                        primary_color=r.primary_color,
                        secondary_color=r.secondary_color,
                        material=r.material,
                        physical_traits=r.physical_traits,
                        distinctive_marks=r.distinctive_marks,
                        verification_notes=r.verification_notes,
                        dominant_colors=r.dominant_colors,
                        image_quality_notes=r.image_quality_notes,
                        image_width=r.image_width,
                        image_height=r.image_height,
                        found_location=location,
                        image_data=raw,
                    )
                    metr.record_item_registered()
                    st.success(f"✅ Item saved!  \n**Item ID:** `{item_id}`")
                    st.session_state["inspection_result"]   = None
                    st.session_state["inspection_image"]    = None
                    st.session_state["inspection_location"] = ""
                    st.session_state["inspection_notes"]    = ""
                    st.rerun()
                except Exception as exc:
                    st.error(f"Database error: {exc}")


def _render_inspection_result(result: InspectionResult) -> None:
    """Display Agent 1's InspectionResult in the UI."""
    st.subheader("🧪 Agent 1 — Feature Extraction")

    if not result.is_valid:
        st.error(f"❌ Image rejected: {result.rejection_reason}")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Category",         result.category or "—")
        st.metric("Item Type",        result.item_type or "—")
        st.metric("Primary Colour",   result.primary_color or "—")
        st.metric("Secondary Colour", result.secondary_color or "—")
    with col2:
        st.metric("Material",         result.material or "—")
        st.metric("Image Size",       f"{result.image_width}×{result.image_height} px")
        if result.dominant_colors:
            st.metric("Pillow Colour Estimate", ", ".join(result.dominant_colors))

    if result.physical_traits:
        st.markdown("**Physical Traits:**")
        for t in result.physical_traits:
            st.markdown(f"- {t}")

    if result.distinctive_marks:
        st.markdown("**Distinctive Marks:**")
        for m in result.distinctive_marks:
            st.markdown(f"- {m}")

    if result.hidden_or_obscured_features:
        st.markdown("**Hidden / Obscured Features:**")
        for f in result.hidden_or_obscured_features:
            st.markdown(f"- {f}")

    if result.verification_notes:
        st.markdown("**🔒 Verification Notes (Staff Only):**")
        st.info(result.verification_notes)

    with st.expander("🔍 Full Inspection JSON"):
        st.json(result.to_dict())


# ---------------------------------------------------------------------------
# Tab 2: Match & Claim Property
# ---------------------------------------------------------------------------

def render_match_tab() -> None:
    db    = _get_db()
    gm    = _get_guardrails()
    metr  = _get_metrics()

    st.header("🔎 Match & Claim Property")
    st.markdown(
        '<div class="hitl-banner">⚠️ AI-Assisted Preliminary Match — '
        'Requires Manual Handover Verification. '
        'This system does not automatically release items.</div>',
        unsafe_allow_html=True,
    )

    items = db.list_items()
    if not items:
        st.info("No found items registered yet. Go to **📦 Intake Found Item** to register one.")
        return

    item_options = {
        f"{it['item_id'][:8]}… | {it.get('item_type') or it.get('category') or 'Unknown'} "
        f"| {it['found_location']} | {it['status']}": it
        for it in items
    }
    selected_label = st.selectbox("Select a registered found item", list(item_options.keys()))
    selected_item = item_options[selected_label]

    with st.expander("📋 Registered Item Summary", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Item ID:** `{selected_item['item_id']}`")
            st.markdown(f"**Status:** {selected_item['status']}")
            st.markdown(f"**Found Location:** {selected_item['found_location']}")
            st.markdown(f"**Registered:** {selected_item['registered_at'][:19]}")
        with col2:
            st.markdown(f"**Category:** {selected_item.get('category') or '—'}")
            st.markdown(f"**Item Type:** {selected_item.get('item_type') or '—'}")
            st.markdown(f"**Primary Colour:** {selected_item.get('primary_color') or '—'}")
            st.markdown(f"**Material:** {selected_item.get('material') or '—'}")

        img_bytes = db.get_item_image(selected_item["item_id"])
        if img_bytes:
            try:
                st.image(Image.open(io.BytesIO(img_bytes)), caption="Registered image", width=250)
            except Exception:
                pass

    st.divider()
    st.subheader("Claimant Information")

    claimant_desc = st.text_area(
        "Describe the lost item",
        placeholder="Describe the item you lost in as much detail as possible…",
        height=120, key="match_claimant_desc",
    )
    lost_location = st.text_input(
        "📍 Where did you lose it?",
        placeholder="e.g. Building A cafeteria", key="match_lost_location",
    )
    additional_details = st.text_input(
        "Additional details (optional)",
        placeholder="e.g. Lost on Monday, had a case on it", key="match_additional",
    )

    if st.button("🤝 Run Matching (Agent 2)", type="primary", use_container_width=True):
        if not claimant_desc.strip():
            st.error("Please describe the lost item.")
            return
        if not lost_location.strip():
            st.error("Please enter where you lost the item.")
            return

        creds = _resolve_credentials()
        if not _credentials_available(creds):
            st.error(
                "IBM watsonx.ai credentials are not configured. "
                "Enter your credentials in the sidebar."
            )
            return

        inspection = _item_to_inspection_result(selected_item)
        matcher = _build_matcher(creds)
        metr.record_matching_start()

        with st.spinner("🤖 Agent 2 evaluating match…"):
            match_result, warning, latency = matcher.match(
                inspection=inspection,
                found_location=selected_item["found_location"],
                claimant_description=claimant_desc,
                lost_location=lost_location,
                additional_details=additional_details,
            )
            metr.sync_from_guardrails(gm)

        is_hard_failure = (
            warning
            and match_result.match_score == 0
            and match_result.recommended_action == "Reject Claim"
            and not match_result.matched_aspects
        )
        if is_hard_failure:
            metr.record_matching_failure()
            metr.record_api_failure()
            st.error(f"❌ Matching failed: {warning}")
            return

        if warning and ("API error" in warning or "authentication" in warning.lower()):
            metr.record_api_failure()

        metr.record_matching_success(latency)
        if warning:
            st.warning(f"⚠️ {warning}")

        st.session_state["match_result"]    = match_result
        st.session_state["current_item_id"] = selected_item["item_id"]

        try:
            eval_id = db.save_evaluation(
                item_id=selected_item["item_id"],
                match_score=match_result.match_score,
                confidence_tier=match_result.confidence_rating,
                matched_aspects=match_result.matched_aspects,
                discrepancies=match_result.discrepancies,
                recommended_action=match_result.recommended_action,
                score_breakdown=match_result.score_breakdown,
                raw_result=gm.staff_full_match(match_result),
                latency_seconds=latency,
            )
            st.session_state["current_eval_id"] = eval_id
            db.update_item_status(
                selected_item["item_id"],
                STATUS_UNDER_VERIFICATION,
                "Claim evaluation submitted",
            )
        except Exception as exc:
            st.warning(f"Could not save evaluation: {exc}")

        st.success(f"✅ Agent 2 completed in {latency:.2f}s")

    if st.session_state.get("match_result"):
        _render_match_result(
            mr=st.session_state["match_result"],
            item_id=st.session_state.get("current_item_id", ""),
            eval_id=st.session_state.get("current_eval_id", ""),
            db=db,
        )


def _item_to_inspection_result(item: Dict[str, Any]) -> InspectionResult:
    """Reconstruct an InspectionResult from a database row for Agent 2 handoff."""
    def _safe_list(val: Any) -> List[str]:
        if isinstance(val, list): return val
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
                return parsed if isinstance(parsed, list) else []
            except Exception: return []
        return []

    full = _get_db().get_item(item["item_id"])
    vn = full.get("verification_notes", "") if full else ""

    return InspectionResult(
        is_valid=True,
        category=item.get("category", ""),
        item_type=item.get("item_type", ""),
        primary_color=item.get("primary_color", ""),
        secondary_color=item.get("secondary_color", ""),
        material=item.get("material", ""),
        physical_traits=_safe_list(item.get("physical_traits", [])),
        distinctive_marks=_safe_list(item.get("distinctive_marks", [])),
        verification_notes=vn,
        image_width=item.get("image_width", 0),
        image_height=item.get("image_height", 0),
        dominant_colors=_safe_list(item.get("dominant_colors", [])),
        image_quality_notes=item.get("image_quality_notes", ""),
    )


def _render_match_result(
    mr: Any,
    item_id: str,
    eval_id: str,
    db: DatabaseManager,
) -> None:
    """Render Agent 2 results — claimant-safe view + staff-only admin section."""
    st.divider()
    st.subheader("🤝 Matching Result — Analysis")

    st.markdown(
        '<div class="hitl-banner">⚠️ AI-Assisted Preliminary Match — '
        'Requires Manual Handover Verification</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Match Score", f"{mr.match_score} / 100")
    with col2:
        badge_color = {"High": "#dcfce7", "Probable": "#fef9c3", "Unlikely": "#fee2e2"}.get(mr.confidence_rating, "#f3f4f6")
        text_color  = {"High": "#15803d", "Probable": "#854d0e", "Unlikely": "#991b1b"}.get(mr.confidence_rating, "#374151")
        st.markdown(
            f"**Confidence**  \n"
            f'<span style="background:{badge_color};color:{text_color};'
            f'padding:3px 12px;border-radius:12px;font-weight:600;">{mr.confidence_rating}</span>',
            unsafe_allow_html=True,
        )
    with col3:
        action_bg = {
            "Immediate Release": "#dcfce7",
            "Verify In Person":  "#fef3c7",
            "Reject Claim":      "#fee2e2",
        }.get(mr.recommended_action, "#f3f4f6")
        st.markdown(
            f"**Recommended Action**  \n"
            f'<span style="background:{action_bg};padding:3px 10px;'
            f'border-radius:10px;font-weight:600;">{mr.recommended_action}</span>',
            unsafe_allow_html=True,
        )

    st.progress(mr.match_score / 100)
    st.caption(
        "⚠️ Scores are preliminary AI estimates — not statistically calibrated probabilities."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if mr.matched_aspects:
            st.markdown("**✅ Matching Evidence**")
            for a in mr.matched_aspects:
                st.markdown(f"- {a}")
    with col_b:
        if mr.discrepancies:
            st.markdown("**⚠️ Discrepancies**")
            for d in mr.discrepancies:
                st.markdown(f"- {d}")

    if mr.score_breakdown:
        with st.expander("📐 Score Breakdown"):
            for k, v in mr.score_breakdown.items():
                if k != "weighted_total":
                    st.write(f"**{k.replace('_', ' ').title()}:** {v}/100")
            st.write(f"**Weighted Total:** {mr.score_breakdown.get('weighted_total', '—')}/100")

    if mr.challenge_questions:
        st.markdown("**🔐 Ownership Verification Questions**")
        st.caption("Please answer these when you attend in person.")
        for i, q in enumerate(mr.challenge_questions, 1):
            qt = q.question if hasattr(q, "question") else str(q)
            st.markdown(
                f'<div class="question-card"><strong>Q{i}:</strong> {qt}</div>',
                unsafe_allow_html=True,
            )

    if mr.staff_advisory:
        st.markdown("**📋 Staff Advisory**")
        st.info(mr.staff_advisory)

    advisory_txt = _build_advisory_text(mr, item_id, eval_id)
    st.download_button(
        label="📥 Download Advisory (.txt)",
        data=advisory_txt,
        file_name=f"advisory_{item_id[:8]}_{eval_id[:8] if eval_id else 'eval'}.txt",
        mime="text/plain",
    )

    # ---- Admin approval section ----
    st.divider()
    st.subheader("🔐 Administrator Actions")

    if not st.session_state.get("admin_logged_in"):
        st.info("Log in as Administrator (sidebar) to approve or reject this claim.")
        return

    st.warning(
        "⚠️ Verify claimant identity and challenge question answers before "
        "changing item status. All actions are recorded in the audit log."
    )

    with st.expander("🔒 Reference Answers (Admin Only)", expanded=False):
        for i, q in enumerate(mr.challenge_questions, 1):
            if hasattr(q, "question"):
                st.markdown(f"**Q{i}:** {q.question}")
                ref = q.reference_answer if q.reference_answer else "Not available"
                st.markdown(f"*Reference answer:* {ref}")
                st.divider()

    col_app, col_rel, col_rej = st.columns(3)
    with col_app:
        if st.button("✅ Approve Claim", use_container_width=True):
            try:
                db.update_item_status(item_id, STATUS_CLAIM_APPROVED,
                                      "Administrator approved claim after manual verification")
                st.success("Claim approved — status: 'Claim Approved'.")
            except Exception as exc:
                st.error(str(exc))
    with col_rel:
        if st.button("🚀 Mark Released", use_container_width=True):
            try:
                db.update_item_status(item_id, STATUS_RELEASED,
                                      "Administrator confirmed physical handover")
                st.success("Item marked as Released.")
            except Exception as exc:
                st.error(str(exc))
    with col_rej:
        if st.button("❌ Reject Claim", use_container_width=True):
            try:
                db.update_item_status(item_id, STATUS_REJECTED,
                                      "Administrator rejected claim — insufficient evidence")
                st.warning("Claim rejected.")
            except Exception as exc:
                st.error(str(exc))


def _build_advisory_text(mr: Any, item_id: str, eval_id: str) -> str:
    """Claimant-safe plain-text advisory. Reference answers are excluded."""
    lines = [
        "=" * 60,
        "FINDERBOT — MATCH ADVISORY REPORT",
        "AI-Assisted Preliminary Match — Requires Manual Handover Verification",
        "=" * 60,
        f"Item ID:          {item_id}",
        f"Evaluation ID:    {eval_id}",
        f"Generated:        {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Match Score:      {mr.match_score} / 100",
        f"Confidence:       {mr.confidence_rating}",
        f"Recommended:      {mr.recommended_action}",
        "",
        "NOTE: Scores are preliminary AI estimates — not calibrated probabilities.",
        "",
        "--- Matching Evidence ---",
    ]
    for a in (mr.matched_aspects or []):
        lines.append(f"  + {a}")
    lines += ["", "--- Discrepancies ---"]
    for d in (mr.discrepancies or []):
        lines.append(f"  - {d}")
    lines += [
        "", "--- Ownership Verification Questions ---",
        "Please answer the following when you attend in person:",
    ]
    for i, q in enumerate(mr.challenge_questions or [], 1):
        qt = q.question if hasattr(q, "question") else str(q)
        lines.append(f"  Q{i}: {qt}")
    lines += [
        "", "--- Staff Advisory ---",
        mr.staff_advisory or "—",
        "",
        "=" * 60,
        "NOTE: This report does not confirm ownership.",
        "All releases require manual identity verification by staff.",
        "=" * 60,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tab 3: Operations & Telemetry
# ---------------------------------------------------------------------------

def render_dashboard_tab() -> None:
    db   = _get_db()
    metr = _get_metrics()

    st.header("📊 Operations & Telemetry Dashboard")
    st.caption("All metrics are sourced from real session activity and database records.")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Registered Items",     db.count_items())
    col2.metric("Total Evaluations",    db.count_evaluations())
    col3.metric("Avg Match Score",      f"{db.avg_match_score():.0f}")
    col4.metric("Avg Latency (s)",      f"{db.avg_latency():.2f}")
    col5.metric("Handoff Success",      f"{metr.handoff_success_rate:.0f}%")

    col6, col7, col8, col9 = st.columns(4)
    col6.metric("Found (Pending)",       db.count_items_by_status(STATUS_FOUND))
    col7.metric("Under Verification",    db.count_items_by_status(STATUS_UNDER_VERIFICATION))
    col8.metric("Released",              db.count_items_by_status(STATUS_RELEASED))
    col9.metric("Rejected",              db.count_items_by_status(STATUS_REJECTED))

    st.divider()

    st.subheader("🛡️ Guardrail Activity (Session)")
    cg1, cg2, cg3, cg4 = st.columns(4)
    cg1.metric("Interventions",   metr.guardrail_interventions)
    cg2.metric("PII Redactions",  metr.pii_redactions)
    cg3.metric("Injections Blocked", metr.injection_rejections)
    cg4.metric("API Failures",    metr.api_failure_count)

    st.divider()

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.subheader("Match Score Distribution")
        scores = db.match_score_distribution()
        if scores:
            buckets = {"0–19": 0, "20–39": 0, "40–59": 0, "60–79": 0, "80–100": 0}
            for s in scores:
                if s < 20:   buckets["0–19"] += 1
                elif s < 40: buckets["20–39"] += 1
                elif s < 60: buckets["40–59"] += 1
                elif s < 80: buckets["60–79"] += 1
                else:        buckets["80–100"] += 1
            st.bar_chart(buckets)
        else:
            st.info("No evaluation data yet.")

    with col_c2:
        st.subheader("Confidence Tier Distribution")
        tier_counts = db.confidence_tier_counts()
        if tier_counts:
            st.bar_chart(tier_counts)
        else:
            st.info("No evaluation data yet.")

    if metr.agent1_latencies or metr.agent2_latencies:
        st.subheader("Per-Request Latencies (Session)")
        max_len = max(
            len(metr.agent1_latencies) if metr.agent1_latencies else 0,
            len(metr.agent2_latencies) if metr.agent2_latencies else 0,
        )
        chart_data: Dict[str, list] = {}
        if metr.agent1_latencies:
            chart_data["Agent 1"] = metr.agent1_latencies + [None] * (max_len - len(metr.agent1_latencies))
        if metr.agent2_latencies:
            chart_data["Agent 2"] = metr.agent2_latencies + [None] * (max_len - len(metr.agent2_latencies))
        st.line_chart(chart_data)

    st.divider()

    st.subheader("Agent Performance (Session)")
    cp1, cp2, cp3 = st.columns(3)
    cp1.metric("Agent 1 Requests",  metr.total_inspection_requests)
    cp1.metric("Agent 1 Successes", metr.successful_inspections)
    cp1.metric("Avg Latency (s)",   f"{metr.avg_agent1_latency:.2f}")
    cp2.metric("Agent 2 Requests",  metr.total_matching_requests)
    cp2.metric("Agent 2 Successes", metr.successful_evaluations)
    cp2.metric("Avg Latency (s)",   f"{metr.avg_agent2_latency:.2f}")
    cp3.metric("Eligible Handoffs", metr.eligible_for_handoff)
    cp3.metric("Successful Handoffs", metr.successful_handoffs)
    cp3.metric("Handoff Rate",      f"{metr.handoff_success_rate:.0f}%")

    st.divider()

    st.subheader("Registered Found Items")
    items = db.list_items()
    if items:
        rows = []
        for it in items:
            rows.append({
                "Item ID":    it["item_id"][:8] + "…",
                "Category":  it.get("category") or "—",
                "Type":      it.get("item_type") or "—",
                "Colour":    it.get("primary_color") or "—",
                "Location":  it["found_location"],
                "Status":    it["status"],
                "Registered": it["registered_at"][:10],
            })
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("No items registered yet.")

    st.subheader("Recent Evaluations")
    evals = db.list_all_evaluations()
    if evals:
        eval_rows = []
        for ev in evals[:20]:
            eval_rows.append({
                "Eval ID":    ev["eval_id"][:8] + "…",
                "Item ID":    ev["item_id"][:8] + "…",
                "Score":      ev["match_score"],
                "Confidence": ev["confidence_tier"],
                "Action":     ev["recommended_action"],
                "Latency(s)": f"{ev.get('latency_seconds', 0):.2f}",
                "Date":       ev["evaluated_at"][:10],
            })
        st.dataframe(eval_rows, use_container_width=True)
    else:
        st.info("No evaluations yet.")

    st.subheader("Audit Log")
    audit = db.get_audit_log()
    if audit:
        audit_rows = []
        for a in audit[:20]:
            audit_rows.append({
                "Item ID":   a["item_id"][:8] + "…",
                "Action":    a["admin_action"],
                "From":      a.get("previous_status") or "—",
                "To":        a.get("new_status") or "—",
                "By":        a.get("performed_by", "admin"),
                "Timestamp": a["timestamp"][:19],
            })
        st.dataframe(audit_rows, use_container_width=True)
    else:
        st.info("No audit entries yet.")

    with st.expander("📋 Full Session Metrics"):
        rows = [{"Metric": k, "Value": str(v)} for k, v in metr.to_summary_dict().items()]
        st.dataframe(rows, use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    render_sidebar()

    tab1, tab2, tab3 = st.tabs([
        "📦 Intake Found Item",
        "🔎 Match & Claim Property",
        "📊 Operations & Telemetry",
    ])

    with tab1:
        render_intake_tab()

    with tab2:
        render_match_tab()

    with tab3:
        render_dashboard_tab()


if __name__ == "__main__":
    main()