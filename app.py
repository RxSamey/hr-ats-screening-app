import os
import io
import time
import json
import re
import streamlit as st
import pandas as pd
import pdfplumber
from groq import Groq



st.set_page_config(
    page_title="AI Resume Screening",
    page_icon="🧠",
    layout="wide"
)

st.markdown(
    "<h2 style='font-family:Segoe UI;'>🧠 AI-Powered Resume Screening Tool</h2>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='color:#475569;'>Upload a Job Description and multiple resumes. "
    "The system will automatically evaluate, score, and extract key HR insights.</p>",
    unsafe_allow_html=True
)


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_from_pdf_file(file_obj) -> str:
    text = ""
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
    return text


def groq_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


def generate_report(client: Groq, model_name: str, jd: str, resume: str) -> dict:
    """
    LLM must return JSON with schema:

    {
      "match_score": 0-100,
      "candidate_name": "string or null",
      "current_organization": "string or null",
      "total_experience_years": number,
      "relevant_experience_years": "string",
      "organization_names": ["string"],
      "top_strengths": ["string"],
      "critical_gaps": ["string"],
      "relevant_experience_summary": "string",
      "rationale_summary": "string",
      "education": "string",
      "academic_percentage": "float",
      "phone_number": "string"
    }
    """

    system_instruction = """
You are an expert HR Resume Screening Agent.  
Return your output STRICTLY as valid JSON with this schema:

{
  "match_score": 0-100,
  "candidate_name": "string or null",
  "current_organization": "string or null",
  "total_experience_years": number,
  "relevant_experience_years": "string",
  "organization_names": ["string"],
  "top_strengths": ["string"],
  "critical_gaps": ["string"],
  "relevant_experience_summary": "string",
  "rationale_summary": "string",
  "education": "string",
  "academic_percentage": "float",
  "phone_number": "string"
}

Rules:
- If resume lacks timeline: "relevant_experience_years" MUST be exactly: "Specific timeline not mentioned"
- "organization_names": must list only real companies mentioned.
- No markdown. No comments. JSON only.
"""

    user_prompt = f"""
Evaluate this candidate:

--- JOB DESCRIPTION ---
{jd}

--- RESUME TEXT ---
{resume}

Output JSON only.
"""

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_prompt},
    ]

    required_keys = [
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
        "phone_number"
    ]

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=700,
            )

            content = response.choices[0].message.content.strip()

            # Remove accidental markdown fences
            if content.startswith("```"):
                content = content.strip("`").replace("json", "").strip()

            report = json.loads(content)

            # Validate keys
            if not all(k in report for k in required_keys):
                raise ValueError("Missing required keys in JSON response")

            return report

        except Exception as e:
            if attempt == 2:
                return {"error": str(e)}
            time.sleep(1)



with st.sidebar:
    st.header("⚙️ Settings")

    default_key = os.getenv("GROQ_API_KEY", "")

    api_key = st.text_input(
        "Groq API Key",
        value=default_key,
        type="password",
        help="Set GROQ_API_KEY in environment for auto-fill."
    )

    if api_key:
        st.success("API key is set.")
    else:
        st.error("No API key provided.")

    model_name = st.selectbox(
        "Model",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
    )

    suitable_threshold = st.slider(
        "Suitability threshold (match score)",
        min_value=0,
        max_value=100,
        value=70,
        step=5,
        help="Candidates with match_score ≥ threshold are marked Suitable = Yes."
    )

    st.markdown("---")
    st.caption("HR can paste the key here; on servers use environment variables.")



col1, col2 = st.columns(2)

with col1:
    jd_file = st.file_uploader("📄 Upload Job Description (PDF or TXT)", type=["pdf", "txt"])

with col2:
    resume_files = st.file_uploader(
        "👥 Upload Resume PDFs (Multiple Allowed)",
        type=["pdf"],
        accept_multiple_files=True
    )

run_button = st.button("🚀 Run Screening")





if run_button:

    if not api_key:
        st.error("Please provide a Groq API key.")
        st.stop()

    if not jd_file:
        st.error("Upload a Job Description first.")
        st.stop()

    if not resume_files:
        st.error("Upload at least one resume.")
        st.stop()

    client = groq_client(api_key)

    # Read JD
    if jd_file.name.lower().endswith(".pdf"):
        jd_raw = extract_text_from_pdf_file(jd_file)
    else:
        jd_raw = jd_file.read().decode("utf-8", errors="ignore")
    jd_text = clean_text(jd_raw)

    results = []
    progress = st.progress(0)
    status = st.empty()

    total = len(resume_files)

    for idx, file in enumerate(resume_files, start=1):

        status.text(f"Processing {idx}/{total}: {file.name}")
        progress.progress(idx / total)

        resume_text = clean_text(extract_text_from_pdf_file(file))

        report = generate_report(client, model_name, jd_text, resume_text)

        if "error" in report:
            st.error(f"Error in {file.name}: {report['error']}")
            continue

        match_score = report.get("match_score", 0)
        suitable = "Yes" if match_score >= suitable_threshold else "No"

        results.append({
            "resume_file": file.name,
            "candidate_name": report["candidate_name"],
            "current_organization": report["current_organization"],
            "total_experience_years": report["total_experience_years"],
            "relevant_experience_years": report["relevant_experience_years"],
            "organization_names": "; ".join(report.get("organization_names", [])),
            "match_score": match_score,
            "suitable": suitable,
            "top_strengths": "; ".join(report.get("top_strengths", [])),
            "critical_gaps": "; ".join(report.get("critical_gaps", [])),
            "relevant_experience_summary": report["relevant_experience_summary"],
            "rationale_summary": report["rationale_summary"],
            "education": report["education"],
            "academic_percentage": report["academic_percentage"],
            "phone_number": report["phone_number"]
        })

    progress.empty()
    status.text("Screening complete!")

    if not results:
        st.warning("No results to display.")
        st.stop()

    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by="match_score", ascending=False).reset_index(drop=True)


    
    st.subheader("📊 Ranked Screening Results (Summary)")

    summary_cols = [
        "resume_file",
        "candidate_name",
        "current_organization",
        "total_experience_years",
        "relevant_experience_years",
        "organization_names",
        "match_score",
        "suitable",
    ]

    st.table(df_sorted[summary_cols])


    
    
    st.subheader("🔍 Candidate Details")

    for _, row in df_sorted.iterrows():
        label = row["candidate_name"] or row["resume_file"]

        with st.expander(f"{label}  |  Score: {row['match_score']}  |  Suitable: {row['suitable']}"):

            st.markdown(f"**Resume file:** {row['resume_file']}")
            st.markdown(f"**Suitable:** {row['suitable']}")
            st.markdown(f"**Current organization:** {row['current_organization']}")
            st.markdown(f"**Total experience (years):** {row['total_experience_years']}")
            st.markdown(f"**Relevant experience (years):** {row['relevant_experience_years']}")
            st.markdown(f"**Organizations:** {row['organization_names']}")
            st.markdown(f"**Top strengths:** {row['top_strengths']}")
            st.markdown(f"**Critical gaps:** {row['critical_gaps']}")
            st.markdown(f"**Relevant experience summary:** {row['relevant_experience_summary']}")
            st.markdown(f"**Rationale summary:** {row['rationale_summary']}")
            st.markdown(f"**Education:** {row['education']}")
            st.markdown(f"**Academic Percentage:** {row['academic_percentage']}")
            st.markdown(f"**Phone Number:** {row['phone_number']}")


   
    
    
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_sorted.to_excel(writer, index=False)

    st.download_button(
        label="⬇️ Download Excel",
        data=buf.getvalue(),
        file_name="hr_screening_results.xlsx",
        mime="application/vnd.ms-excel"
    )
