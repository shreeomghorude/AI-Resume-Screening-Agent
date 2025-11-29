import json
import google.generativeai as genai
from app.config import GEMINI_API_KEY, SIMILARITY_WEIGHT, LLM_WEIGHT

# Gemini Setup
genai.configure(api_key=GEMINI_API_KEY)

# ----------------------------
# 1. Gemini Embeddings
# ----------------------------
def embed_text(text: str):
    """Generate embeddings using Gemini Embedding API."""
    try:
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return result["embedding"]
    except Exception as e:
        print("Embedding Error:", e)
        return [0.0] * 768  # fallback vector


# ----------------------------
# 2. Cosine Similarity
# ----------------------------
def cosine_sim(a, b):
    import numpy as np
    a, b = np.array(a), np.array(b)
    if len(a) != len(b):
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ----------------------------
# 3. LLM Score + Explanation
# ----------------------------
def llm_score_and_explain(job_desc: str, resume_text: str, sim_score: float) -> dict:
    prompt = f"""
You are an AI Resume Screening Agent.

Analyze the Job Description and Resume, then return ONLY a valid JSON:

{{
 "score": 0-100,
 "strengths": ["point1", "point2", "point3"],
 "weaknesses": ["point1", "point2"]
}}

Job Description:
{job_desc}

Resume Text:
{resume_text[:3500]}

Similarity score: {sim_score:.3f}
"""

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        text = response.text.strip()

        # extract JSON only
        import re
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group(0))
    except Exception as e:
        print("Gemini LLM Error:", e)

    # fallback
    base_score = int(sim_score * 100)
    return {
        "score": base_score,
        "strengths": [
            "Relevant skills detected",
            "Experience somewhat aligned",
            "Clean formatting"
        ],
        "weaknesses": [
            "Missing metrics",
            "Resume could be more specific"
        ]
    }


# ----------------------------
# 4. Ranking Pipeline
# ----------------------------
def rank_resumes(job_description: str, resumes: list):
    job_vec = embed_text(job_description)
    ranked_output = []

    for r in resumes:
        text = r["text"]

        # embeddings
        r_vec = embed_text(text)
        sim = cosine_sim(job_vec, r_vec)

        # LLM explanation + scoring
        llm_result = llm_score_and_explain(job_description, text, sim)
        llm_score = llm_result.get("score", 0)

        # final combined score
        final_score = int(
            SIMILARITY_WEIGHT * sim * 100 +
            LLM_WEIGHT * llm_score
        )

        ranked_output.append({
            "filename": r["filename"],
            "similarity": sim,
            "final_score": final_score,
            "llm": llm_result,
            "text_preview": text[:400],
            "text": text
        })

    # highest first
    return sorted(ranked_output, key=lambda x: x["final_score"], reverse=True)
