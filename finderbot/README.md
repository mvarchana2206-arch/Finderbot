# 🔍 FinderBot: Smart Lost & Found

> A production-ready, beginner-friendly Streamlit web application powered by **IBM watsonx.ai** foundation models — featuring a two-agent forensic workflow, enterprise guardrails, and a real-time operations dashboard.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Agent 1 — Item Inspector** | Vision model (`ibm/granite-vision-3-2-2b`) extracts category, colour, material, distinctive marks, and hidden features from a photo |
| **Agent 2 — Reconciliation Officer** | Text model (`ibm/granite-4-h-small`) cross-examines visual attributes vs claimant description and produces a match score, anti-fraud questions, and a recommended action |
| **Enterprise Guardrails** | Prompt-injection protection, PII masking (SSN/Aadhaar/passport/phone/credit card), schema validation, and anti-fraud blind shield |
| **Telemetry Dashboard** | Latency tracking, handoff success rate, match score history, guardrail trigger counters, model topology |
| **Human-in-the-Loop** | Every output is labelled as AI-assisted and item release requires manual administrator approval |

---

## 🗂️ Repository Structure

```
finderbot/
├── .env.example           ← Copy to .env and fill in your credentials
├── requirements.txt
├── README.md
├── app.py                 ← Main Streamlit application entry point
├── guardrails/
│   ├── __init__.py
│   └── safety.py          ← GuardrailsManager
├── agents/
│   ├── __init__.py
│   ├── inspector_agent.py  ← Agent 1 (vision)
│   └── matcher_agent.py    ← Agent 2 (text)
└── dashboard/
    ├── __init__.py
    └── metrics.py          ← MetricsDashboard
```

---

## 🚀 Quick Start

### 1. Clone / navigate to the project

```bash
cd finderbot
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure credentials

```bash
cp .env.example .env
```

Edit `.env` and fill in your IBM Cloud credentials:

```dotenv
WATSONX_API_KEY="your-ibm-cloud-api-key"
WATSONX_PROJECT_ID="your-watsonx-project-id"
WATSONX_URL="https://us-south.ml.cloud.ibm.com"
WATSONX_TEXT_MODEL_ID="ibm/granite-4-h-small"
WATSONX_VISION_MODEL_ID="ibm/granite-vision-3-2-2b"
```

> **Never commit your `.env` file.** It is included in `.gitignore` by convention.

### 5. Run the application

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🔑 Getting IBM watsonx.ai Credentials

1. Log in to [IBM Cloud](https://cloud.ibm.com).
2. Create or open a **watsonx.ai** service instance.
3. Go to **Manage → Access (IAM)** and create an **API key**.
4. Open your watsonx.ai project and copy the **Project ID** from the project settings.
5. Confirm the endpoint URL for your region (default: `https://us-south.ml.cloud.ibm.com`).

---

## 🖥️ Application Tabs

### 📦 Tab 1 — Intake Found Item
1. Upload a JPG/JPEG/PNG photo of the found item (or use your camera).
2. Enter the location where it was found.
3. Click **Run Visual Analysis (Agent 1)**.
4. Review the extracted JSON attributes and register the item.

### 🔎 Tab 2 — Match & Claim Property
1. Select a registered found item from the dropdown.
2. Enter the claimant's free-text description and reported lost location.
3. Click **Assess Claim (Agent 2)**.
4. Review the match score, anti-fraud questions, and staff advisory.
5. Download the advisory as a `.txt` file for your records.

### 📊 Tab 3 — Operations & Telemetry
- Average latency per agent.
- Multi-agent handoff success rate.
- Match score history chart.
- Guardrail intervention counters.
- Registered items table.

---

## 🛡️ Guardrails Reference

| Guardrail | Behaviour |
|---|---|
| **Prompt Injection** | Detects override commands ("ignore previous instructions", "set score to 100", etc.) and blocks the request |
| **PII Masking** | Scrubs credit cards, SSN, Aadhaar, passports, phone numbers with `[REDACTED_*]` tokens |
| **Schema Validation** | Parses and repairs malformed LLM JSON (markdown fences, trailing prose) |
| **Anti-Fraud Blind Shield** | Scans challenge questions for embedded answer leaks and redacts offending questions |
| **HITL Disclaimer** | Every result is labelled "AI-Assisted Preliminary Match — Requires Manual Handover Verification" |

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | ≥1.35.0 | Web UI framework |
| `ibm-watsonx-ai` | ≥1.1.0 | IBM watsonx.ai SDK |
| `pydantic` | ≥2.5.0 | Typed output schemas |
| `pillow` | ≥10.0.0 | Image loading and encoding |
| `python-dotenv` | ≥1.0.0 | `.env` credential loading |

---

## ⚠️ Important Policies

- **Item Release**: No item should be marked as "Released" without explicit **administrator sign-off**. The AI advisory is informational only.
- **Security**: Credentials are loaded from `.env` or the sidebar password fields. They are never logged, displayed in plaintext, or embedded in source code.
- **Privacy**: PII detected in user inputs is automatically redacted before any LLM call.

---

## 📄 License

This project is provided for educational and demonstration purposes.
