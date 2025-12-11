import os
import io
import time
import json
import re
from typing import Any, Dict, List

import streamlit as st
import pandas as pd
import pdfplumber

# Optional providers; import only if installed on your environment
try:
    from groq import Groq
except Exception:
    Groq = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="HR ATS — Resume Screener",
    page_icon="🧾",
    layout="wide",
)

# ---------------------------
# Professional styling 
# ---------------------------
st.markdown(
    """
    <style>
    :root {
        --bg: #0b1220;
        --card: #0f1724;
        --muted: #9aa4b2;
        --accent: #5661f9;
        --accent-2: #07b59b;
        --surface-2: #111827;
        --white: #eef2ff;
    }

    .stApp {
        background: linear-gradient(180deg, var(--bg) 0%, #041023 100%);
        color: var(--white);
        font-family: Inter, "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont;
    }

    .header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        padding: 8px 6px 18px 6px;
    }
    .brand {
        display:flex;
        align-items: center;
        gap: 12px;
    }
    .brand-logo {
        width:48px;
        height:48px;
        border-radius:10px;
        display:flex;
        align-items:center;
        justify-content:center;
        background: linear-gradient(135deg, rgba(86,97,249,0.18), rgba(7,181,155,0.08));
        font-weight:700;
        color: var(--white);
        box-shadow: 0 6px 18px rgba(7,11,20,0.6);
    }
    .brand-title {
        font-size:20px;
        font-weight:700;
        margin:0;
        color: var(--white);
    }
    .brand-sub {
        font-size:12px;
        color: var(--muted);
        margin-top:2px;
    }

    .panel {
        background: var(--card);
        border-radius: 12px;
        padding: 18px;
        border: 1px solid rgba(255,255,255,0.03);
        box-shadow: 0 10px 30px rgba(2,6,23,0.6);
    }
    .input-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border-radius: 10px;
        padding: 14px;
        border: 1px solid rgba(255,255,255,0.03);
    }
    .card-title {
        font-weight:700;
        color:var(--white);
        margin-bottom:6px;
        font-size:15px;
    }
    .card-desc {
        margin:0 0 10px 0;
        color: var(--muted);
        font-size:13px;
    }

    .helper {
        color: var(--muted);
        font-size:12px;
        margin-top:6px;
    }

    .stButton>button {
        background: linear-gradient(90deg, var(--accent), #15b1e6);
        color: white;
        font-weight: 600;
        padding: 8px 14px;
        border-radius: 10px;
        box-shadow: 0 8px 26px rgba(86,97,249,0.16);
    }

    [data-testid="stExpander"] {
        border-radius: 10px;
        background: var(--surface-2) !important;
        color: var(--white);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Header
# ---------------------------
with st.container():
    st.markdown(
        """
        <div class="header">
          <div class="brand">
            <div class="brand-logo">HR</div>
            <div>
              <div class="brand-title">HR ATS — Resume Screener</div>
              <div class="brand-sub">AI-assisted shortlisting • Fast, auditable, exportable</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------
# Utility functions
# ---------------------------
def clean_text(inp: str) -> str:
    if not isinstance(inp, str):
        return ""
    s = inp.replace("\x00", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def extract_text_from_pdf_file(file_obj) -> str:
    text = ""
    try:
        with pdfplumber.open(file_obj) as pdf:
            for p in pdf.pages:
                page_text = p.extract_text() or ""
                text += page_text + "\n"
    except Exception:
        try:
            raw = file_obj.read()
            text = raw.decode("utf-8", errors="ignore")
        except Exception:
            text = ""
    return text


SCHEMA_KEYS = [
    "match_score",
    "candidate_name",
    "current_organization",
    "total_experience_years",
    "relevant_experience_years",
    "organization_names",
    "top_strengths",
    "critical_gaps",
    "relevant_experience_summary",
    "rationale_summary",
    "education",
    "academic_percentage",
    "phone_number",
]


def llm_evaluate(provider: str, client: Any, model: str, jd: str, resume: str) -> Dict[str, Any]:

    system_msg = (
        "You are an expert HR resume screening assistant. Return STRICT JSON only with keys: " 
        + ", ".join(SCHEMA_KEYS)
        + ". If any field is missing, return empty string. "
        "If timeline missing, set relevant_experience_years exactly to 'Specific timeline not mentioned'."
    )

    user_msg = f"Job Description:\n{jd}\n\nResume:\n{resume}\n\nReturn JSON only."

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    last_err = ""

    for _ in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.06,
                max_tokens=700,
            )
            text = resp.choices[0].message.content.strip()

            if text.startswith("```"):
                text = text.strip("`").replace("json", "").strip()

            data = json.loads(text)

            # fill missing keys
            for k in SCHEMA_KEYS:
                if k not in data:
                    data[k] = ""

            return data

        except Exception as e:
            last_err = str(e)
            time.sleep(0.4)

    return {"error": last_err}


# ---------------------------
# UI Layout
# ---------------------------
col_left, col_right = st.columns([2.4, 1])

# --- Right panel ---
with col_right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### Configuration", unsafe_allow_html=True)

    provider = st.selectbox("Provider", ["Groq (Llama)", "OpenAI (ChatGPT)"])

    # API key field switches based on provider
    env_var = "GROQ_API_KEY" if provider.startswith("Groq") else "OPENAI_API_KEY"
    api_key = st.text_input("API Key", value=os.getenv(env_var, ""), type="password")

    # Model dropdown based on provider
    if provider.startswith("Groq"):
        model_choice = st.selectbox("Model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"])
    else:
        model_choice = st.selectbox("Model", ["gpt-4.1-mini", "gpt-4o-mini"])

    threshold = st.slider("Suitability threshold", 0, 100, 70, 5)

    st.markdown("</div>", unsafe_allow_html=True)

# --- Left panel ---
with col_left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.markdown(
        '<div style="display:flex;gap:16px;align-items:stretch;">',
        unsafe_allow_html=True,
    )

    # JD upload
    st.markdown('<div style="flex:1">', unsafe_allow_html=True)
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Job Description</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-desc">Upload JD (PDF or TXT)</div>', unsafe_allow_html=True)
    jd_file = st.file_uploader("", type=["pdf", "txt"], key="jd_upload", label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Resume upload
    st.markdown('<div style="width:360px">', unsafe_allow_html=True)
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Candidate Resumes</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-desc">Upload multiple resume PDFs</div>', unsafe_allow_html=True)
    resumes = st.file_uploader("", type=["pdf"], accept_multiple_files=True, key="resumes", label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    run_btn = st.button("Run screening")

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------
# Run ATS Logic
# ---------------------------
if run_btn:

    if not api_key:
        st.error("Missing API key.")
        st.stop()

    if jd_file is None:
        st.error("Upload a Job Description.")
        st.stop()

    if not resumes:
        st.error("Upload at least one resume.")
        st.stop()

    # Create provider client
    if provider.startswith("Groq"):
        if Groq is None:
            st.error("Groq SDK missing.")
            st.stop()
        client_obj = Groq(api_key=api_key)
    else:
        if OpenAI is None:
            st.error("OpenAI SDK missing.")
            st.stop()
        client_obj = OpenAI(api_key=api_key)

    # Read JD
    if jd_file.type == "application/pdf":
        jd_raw = extract_text_from_pdf_file(jd_file)
    else:
        jd_raw = jd_file.read().decode("utf-8", errors="ignore")

    jd_text = clean_text(jd_raw)

    results = []
    progress = st.progress(0.0)
    status = st.empty()

    total = len(resumes)

    for idx, r in enumerate(resumes, 1):
        status.info(f"Processing {idx}/{total}: {r.name}")

        resume_raw = extract_text_from_pdf_file(r)
        resume_text = clean_text(resume_raw)

        data = llm_evaluate(provider, client_obj, model_choice, jd_text, resume_text)

        if "error" in data:
            st.error(f"Error on {r.name}: {data['error']}")
            score = 0
            suitable = "No"
        else:
            score = float(data.get("match_score") or 0)
            suitable = "Yes" if score >= threshold else "No"

        orgs = data.get("organization_names", [])
        if isinstance(orgs, list):
            orgs = "; ".join(orgs)

        results.append({
            "resume_file": r.name,
            "candidate_name": data.get("candidate_name", ""),
            "current_organization": data.get("current_organization", ""),
            "total_experience_years": data.get("total_experience_years", ""),
            "relevant_experience_years": data.get("relevant_experience_years", ""),
            "organization_names": orgs,
            "match_score": score,
            "suitable": suitable,
            "top_strengths": "; ".join(data.get("top_strengths") or []),
            "critical_gaps": "; ".join(data.get("critical_gaps") or []),
            "relevant_experience_summary": data.get("relevant_experience_summary", ""),
            "rationale_summary": data.get("rationale_summary", ""),
            "education": data.get("education", ""),
            "academic_percentage": data.get("academic_percentage", ""),
            "phone_number": data.get("phone_number", ""),
        })

        progress.progress(idx / total)

    status.success("Screening complete.")
    progress.empty()

    df = pd.DataFrame(results)
    df_sorted = df.sort_values("match_score", ascending=False).reset_index(drop=True)

    st.subheader("Screening Summary")

    st.dataframe(
        df_sorted[
            [
                "resume_file",
                "candidate_name",
                "current_organization",
                "total_experience_years",
                "organization_names",
                "match_score",
                "suitable",
            ]
        ],
        use_container_width=True,
        height=300
    )

    st.subheader("Candidate Details")
    for _, row in df_sorted.iterrows():
        with st.expander(f"{row['candidate_name']} — Score {row['match_score']}"):
            st.write(row)

    # Export Excel
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_sorted.to_excel(writer, index=False)

    st.download_button(
        "Download Excel",
        data=buf.getvalue(),
        file_name="ats_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
