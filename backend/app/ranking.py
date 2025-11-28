import json
import google.generativeai as genai
from app.embeddings import embed_text, cosine_sim
from app.config import GEMINI_API_KEY, SIMILARITY_WEIGHT, LLM_WEIGHT

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)


def llm_score_and_explain(job_desc: str, resume_text: str, sim_score: float) -> dict:
    """
    Uses Google Gemini to evaluate the resume.
    Returns a JSON containing:
    - score (0–100)
    - strengths (list)
    - weaknesses (list)
    """

    prompt = f"""
You are an AI Resume Screening Agent.

Given the Job Description and the Resume:

1. Analyze candidate fit.
2. Score the candidate from 0 to 100.
3. Provide exactly 3 strengths.
4. Provide exactly 2 weaknesses.

Return ONLY a valid JSON in this format:

{{
 "score": 0-100,
 "strengths": ["point1", "point2", "point3"],
 "weaknesses": ["point1", "point2"]
}}

Do not include any extra text.

---

Job Description:
{job_desc}

Resume Text (trimmed):
{resume_text[:4000]}

Embedding similarity score: {sim_score:.3f}
"""

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)

        text = response.text.strip()

        # Extract JSON only
        import re
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group(0))

    except Exception as e:
        print("Gemini error:", e)

    # ----------- FALLBACK (when no Gemini or error) -----------
    base_score = int(sim_score * 100)
    return {
        "score": base_score,
        "strengths": [
            "Relevant skills detected",
            "General experience is related",
            "Readable formatting"
        ],
        "weaknesses": [
            "Lacks measurable achievements",
            "Could be more detailed"
        ]
    }


def rank_resumes(job_description: str, resumes: list):
    """
    Main ranking pipeline:
    - Compute embeddings
    - Compute similarity
    - Call Gemini for explanation & scoring
    - Compute final ranking score
    """

    job_vec = embed_text(job_description)
    ranked_output = []

    for r in resumes:
        text = r["text"]
        r_vec = embed_text(text)

        # cosine similarity (0–1)
        sim = float(cosine_sim(job_vec, r_vec))

        # Gemini score + strengths/weaknesses
        llm_result = llm_score_and_explain(job_description, text, sim)
        llm_score = llm_result.get("score", 0)

        # Combined score
        final_score = int(
            SIMILARITY_WEIGHT * sim * 100 +
            LLM_WEIGHT * llm_score
        )

        ranked_output.append({
            "filename": r["filename"],
            "similarity": sim,
            "final_score": final_score,
            "llm": llm_result,
            "text": text
        })

    # Sort by score (descending)
    ranked_output = sorted(ranked_output, key=lambda x: x["final_score"], reverse=True)
    return ranked_output
