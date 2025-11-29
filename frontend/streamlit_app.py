import streamlit as st
import requests
import pandas as pd
from io import BytesIO
import time

st.set_page_config(page_title="Resume Screening Agent", layout="wide")
st.title("Resume Screening Agent")
st.markdown("Upload a job description and candidate resumes (PDF/DOCX/TXT).")

with st.form("rank_form"):
    job_desc = st.text_area("Job Description", height=200)
    files = st.file_uploader(
        "Upload resumes", accept_multiple_files=True, type=["pdf", "docx", "txt"]
    )
    submitted = st.form_submit_button("Rank Resumes")

if submitted:
    if not job_desc.strip():
        st.error("Please enter a job description")
    elif not files:
        st.error("Please upload at least one resume")
    else:
        # Your live backend URL
        BACKEND_URL = "https://ai-resume-screening-agent.onrender.com"

        files_payload = []
        for f in files:
            try:
                f.seek(0)  # ensure pointer reset
            except Exception:
                pass
            content = f.read()
            files_payload.append(("files", (f.name, BytesIO(content), f.type)))

        data = {"job_description": job_desc}

        start_time = time.time()

        try:
            with st.spinner("Sending to backend and scoring..."):
                resp = requests.post(
                    BACKEND_URL + "/rank",
                    data=data,
                    files=files_payload,
                    timeout=200
                )
                resp.raise_for_status()
                payload = resp.json()

            elapsed = time.time() - start_time
            results = payload.get("results", [])

            if not results:
                st.warning("No results returned from backend.")
            else:
                st.success(f"Ranking complete — {len(results)} candidate(s) — {elapsed:.1f}s")

                for i, r in enumerate(results):
                    llm = r.get("llm", {})
                    score = r.get("final_score", 0)
                    similarity = r.get("similarity", 0.0)

                    # Score color formatting
                    if score >= 75:
                        score_color = "#16a34a"  # green
                    elif score >= 45:
                        score_color = "#f59e0b"  # orange
                    else:
                        score_color = "#dc2626"  # red

                    st.markdown(f"### {i+1}. **{r.get('filename','Unnamed')}**")
                    st.markdown(
                        f"**Final Score:** <span style='color:{score_color}; font-weight:700;'>{score}</span> "
                        f"&nbsp;&nbsp; **Similarity:** `{round(similarity, 3)}`",
                        unsafe_allow_html=True,
                    )

                    # Strengths
                    with st.expander("Strengths"):
                        strengths = llm.get("strengths", [])
                        if strengths:
                            for s in strengths:
                                st.markdown(f"- {s}")
                        else:
                            st.markdown("- No strengths available.")

                    # Weaknesses
                    with st.expander("Weaknesses"):
                        weaknesses = llm.get("weaknesses", [])
                        if weaknesses:
                            for w in weaknesses:
                                st.markdown(f"- {w}")
                        else:
                            st.markdown("- No weaknesses available.")

                    # Preview resume text (first 400 chars)
                    with st.expander("Resume Preview (first 400 chars)"):
                        preview = r.get("text", "")[:400]
                        if preview:
                            st.code(preview)
                        else:
                            st.markdown("No preview available.")
                    st.markdown("---")

                # Summary table
                summary_rows = []
                for r in results:
                    llm = r.get("llm", {})
                    summary_rows.append(
                        {
                            "filename": r.get("filename", ""),
                            "final_score": r.get("final_score", 0),
                            "similarity": round(r.get("similarity", 0.0), 3),
                            "strengths": " | ".join(llm.get("strengths", [])),
                            "weaknesses": " | ".join(llm.get("weaknesses", [])),
                        }
                    )

                summary_df = pd.DataFrame(summary_rows).sort_values(
                    "final_score", ascending=False
                )

                st.markdown("### 📊 Summary Table")
                st.dataframe(summary_df, use_container_width=True)

                csv_bytes = summary_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download CSV report",
                    data=csv_bytes,
                    file_name="resume_ranking_report.csv",
                    mime="text/csv",
                )

        except requests.exceptions.RequestException as e:
            st.error(f"Network/Backend Error: {e}")

        except Exception as e:
            st.error(f"Unexpected Error: {e}")
