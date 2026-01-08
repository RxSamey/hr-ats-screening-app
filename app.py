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

# DOCX support
try:
    import docx
except Exception:
    docx = None


# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="HR ATS — Resume Screener",
    page_icon="🧾",
    layout="wide",
)

# ---------------------------
# Styling
# ---------------------------
st.markdown(
    """
    <style>
    .panel {
        background: #0f1724;
        border-radius: 12px;
        padding: 18px;
        border: 1px solid rgba(255,255,255,0.03);
        box-shadow: 0 10px 30px rgba(2,6,23,0.6);
    }
    .stButton>button {
        background: linear-gradient(90deg, #5661f9, #15b1e6);
        color: white;
        font-weight: 600;
        padding: 8px 14px;
        border-radius: 10px;
    }
    </style>
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


def extract_text_from_docx(file_obj) -> str:
    if docx is None:
        return ""
    try:
        document = docx.Document(file_obj)
        return "\n".join([para.text for para in document.paragraphs])
    except Exception:
        return ""


def extract_text_generic(file_obj, file_type: str) -> str:
    if file_type == "application/pdf":
        return extract_text_from_pdf_file(file_obj)
    elif file_type in [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    ]:
        return extract_text_from_docx(file_obj)
    elif file_type == "text/plain":
        try:
            return file_obj.read().decode("utf-8", errors="ignore")
        except Exception:
            return ""
    else:
        return ""


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

            for k in SCHEMA_KEYS:
                if k not in data:
                    data[k] = ""

            return data

        except Exception as e:
            last_err = str(e)
            time.sleep(0.4)

    return {"error": last_err}


def apply_critical_skill_penalty(score: float, resume_text: str, critical_skills: List[str]):
    missing = []
    resume_lower = resume_text.lower()

    for skill in critical_skills:
        if skill.lower() not in resume_lower:
            missing.append(skill)

    if missing:
        score = max(0, score - 40)
        return score, False, missing

    return score, True, []


# ---------------------------
# UI Layout
# ---------------------------
col_left, col_right = st.columns([2.4, 1])

with col_right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### Configuration")

    provider = st.selectbox("Provider", ["Groq (Llama)", "OpenAI (ChatGPT)"])
    env_var = "GROQ_API_KEY" if provider.startswith("Groq") else "OPENAI_API_KEY"
    api_key = st.text_input("API Key", value=os.getenv(env_var, ""), type="password")

    if provider.startswith("Groq"):
        model_choice = st.selectbox("Model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"])
    else:
        model_choice = st.selectbox("Model", ["gpt-4.1-mini", "gpt-4o-mini"])

    threshold = st.slider("Suitability threshold", 0, 100, 70, 5)
    st.markdown("</div>", unsafe_allow_html=True)


with col_left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.markdown("### Job Description")
    jd_input_mode = st.radio("JD Input Mode", ["Upload File", "Paste Text"], horizontal=True)

    jd_file = None
    jd_text_input = ""

    if jd_input_mode == "Upload File":
        jd_file = st.file_uploader("Upload JD", type=["pdf", "doc", "docx", "txt"])
    else:
        jd_text_input = st.text_area("Paste Job Description here", height=200)

    st.markdown("### Candidate Resumes")
    resumes = st.file_uploader("Upload resumes", type=["pdf", "doc", "docx", "txt"], accept_multiple_files=True)

    st.markdown("### Critical / Must-Have Skills")
    critical_skills_input = st.text_input("Add the skills (e.g., Kubernetes, Docker, GCP)")

    run_btn = st.button("Run screening")
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------
# Run ATS Logic
# ---------------------------
if run_btn:

    if not api_key:
        st.error("Missing API key.")
        st.stop()

    if jd_input_mode == "Upload File" and jd_file is None:
        st.error("Upload a Job Description file or paste text.")
        st.stop()

    if jd_input_mode == "Paste Text" and not jd_text_input.strip():
        st.error("Paste the Job Description text.")
        st.stop()

    if not resumes:
        st.error("Upload at least one resume.")
        st.stop()

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

    if jd_input_mode == "Upload File":
        jd_raw = extract_text_generic(jd_file, jd_file.type)
    else:
        jd_raw = jd_text_input

    jd_text = clean_text(jd_raw)
    critical_skills = [s.strip() for s in critical_skills_input.split(",") if s.strip()]

    results = []

    for r in resumes:
        resume_raw = extract_text_generic(r, r.type)
        resume_text = clean_text(resume_raw)

        data = llm_evaluate(provider, client_obj, model_choice, jd_text, resume_text)

        if "error" in data:
            raw_score = 0.0
            suitable = "No"
        else:
            raw_score = float(data.get("match_score") or 0)
            suitable = "Yes" if raw_score >= threshold else "No"

        # Apply Critical Skill penalty
        if critical_skills:
            raw_score, meets_critical, missing_skills = apply_critical_skill_penalty(
                raw_score, resume_text, critical_skills
            )
            if not meets_critical:
                suitable = "No"
                data["critical_gaps"] = list(set((data.get("critical_gaps") or []) + missing_skills))

        # Final formatted score
        score = round(raw_score, 2)
        matching_percentage = f"{score:.2f}%"

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
            "matching_percentage": matching_percentage,
            "suitable": suitable,
            "top_strengths": "; ".join(data.get("top_strengths") or []),
            "critical_gaps": "; ".join(data.get("critical_gaps") or []),
            "relevant_experience_summary": data.get("relevant_experience_summary", ""),
            "rationale_summary": data.get("rationale_summary", ""),
            "education": data.get("education", ""),
            "academic_percentage": data.get("academic_percentage", ""),
            "phone_number": data.get("phone_number", ""),
        })

    df_sorted = pd.DataFrame(results).sort_values("match_score", ascending=False).reset_index(drop=True)

    # ---------------------------
    # SUMMARY TABLE
    # ---------------------------
    st.subheader("Screening Summary")

    summary_df = df_sorted[
        [
            "resume_file",
            "candidate_name",
            "current_organization",
            "total_experience_years",
            "organization_names",
            "match_score",
            "matching_percentage",
            "suitable",
        ]
    ].rename(columns={
        "resume_file": "Resume File",
        "candidate_name": "Candidate",
        "current_organization": "Organization",
        "total_experience_years": "Total Exp",
        "organization_names": "Organizations",
        "match_score": "Resume Score",
        "matching_percentage": "Matching %",
        "suitable": "Suitable",
    })

    st.table(summary_df)

    # ---------------------------
    # CANDIDATE DETAILS
    # ---------------------------
    st.subheader("Candidate Details")

    for _, row in df_sorted.iterrows():
        st.markdown(f"## {row['candidate_name']} — Score {row['match_score']:.2f}")

        details_md = f"""
| Field | Value |
|-------|-------|
| Resume File | {row['resume_file']} |
| Current Organization | {row['current_organization']} |
| Total Experience | {row['total_experience_years']} |
| Relevant Experience | {row['relevant_experience_years']} |
| Organizations | {row['organization_names']} |
| Resume Score | {row['match_score']:.2f} |
| Matching Percentage | {row['matching_percentage']} |
| Suitable | {row['suitable']} |
| Education | {row['education']} |
| Academic Percentage | {row['academic_percentage']} |
| Phone Number | {row['phone_number']} |
"""
        st.markdown(details_md)

        st.markdown("### Top Strengths")
        st.markdown(row["top_strengths"] if row["top_strengths"] else "Not specified")

        st.markdown("### Critical Gaps")
        st.markdown(row["critical_gaps"] if row["critical_gaps"] else "None")

        st.markdown("### Relevant Experience Summary")
        st.markdown(row["relevant_experience_summary"])

        # ---------------------------
        # FINAL RECRUITER EVALUATION
        # ---------------------------
        st.markdown("### 🎯 Final Recruiter Evaluation")

        strengths = row["top_strengths"] if row["top_strengths"] else "Not specified"
        gaps = row["critical_gaps"] if row["critical_gaps"] else "None"

        if row["suitable"] == "Yes":
            recommendation = "→ Proceed to Technical Round"
        elif row["match_score"] >= 50:
            recommendation = "→ Hold for Review"
        else:
            recommendation = "→ Not Recommended"

        evaluation_md = f"""
**Final Score:** {row["match_score"]:.2f}  
**Matching Percentage:** {row["matching_percentage"]}

**Summary:**  
{row["rationale_summary"]}

**Key Strengths:**  
{strengths}

**Gaps Identified:**  
{gaps}

**Recommendation:**  
{recommendation}
"""
        st.markdown(evaluation_md)
        st.markdown("---")

    # ---------------------------
    # EXPORT
    # ---------------------------
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_sorted.to_excel(writer, index=False)

    st.download_button(
        "Download Excel",
        data=buf.getvalue(),
        file_name="ats_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
