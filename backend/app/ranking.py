# backend/app/ranking.py
from typing import List, Dict
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import SIMILARITY_WEIGHT, HEURISTIC_WEIGHT, PREVIEW_CHARS

# simple text cleaning
def normalize_text(text: str) -> str:
    text = text or ""
    text = text.lower()
    # remove multiple whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Use TF-IDF vectors (lightweight and memory friendly)
_global_vectorizer = None

def _ensure_vectorizer(corpus: List[str]):
    global _global_vectorizer
    if _global_vectorizer is None:
        # create a new TF-IDF vectorizer for the current batch
        _global_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
        _global_vectorizer.fit(corpus)
    return _global_vectorizer

def compute_tfidf_vectors(texts: List[str]):
    vec = _ensure_vectorizer(texts)
    return vec.transform(texts)

# a simple heuristic to extract strengths/weaknesses based on keywords
COMMON_SKILLS = [
    "python", "java", "c++", "javascript", "react", "node", "fastapi", "django",
    "sql", "mysql", "postgres", "mongodb", "aws", "gcp", "docker", "kubernetes",
    "linux", "html", "css", "git", "rest", "api", "tensorflow", "pytorch"
]

def heuristic_strengths_weaknesses(job_desc: str, resume_text: str):
    jd = normalize_text(job_desc)
    rt = normalize_text(resume_text)

    strengths = []
    weaknesses = []

    # detect matched skills (present in resume and in JD)
    for skill in COMMON_SKILLS:
        if skill in jd and skill in rt:
            strengths.append(f"{skill} (matches job requirement)")
        elif skill in rt and skill not in jd:
            # resume has extra skill not specifically required — still a strength
            strengths.append(f"{skill}")
        elif skill in jd and skill not in rt:
            weaknesses.append(f"Missing skill: {skill}")

    # Limit to top 3 strengths and top 2 weaknesses
    # Keep uniqueness and preserve some priority
    strengths = list(dict.fromkeys(strengths))[:3]
    weaknesses = list(dict.fromkeys(weaknesses))[:2]

    # If none found, provide generic outputs
    if not strengths:
        strengths = ["Relevant keywords detected", "Readable formatting", "General experience"]
    if not weaknesses:
        weaknesses = ["Lacks measurable achievements", "Could provide more specific metrics"]

    return strengths, weaknesses

def rank_resumes(job_description: str, resumes: List[Dict]) -> List[Dict]:
    """
    Input:
      job_description: str
      resumes: list of { "filename": str, "text": str }
    Output:
      ranked list with fields:
        filename, similarity (0..1), final_score (0..100), llm(dict: strengths, weaknesses), text, text_preview
    """

    # Normalize inputs
    job_description = normalize_text(job_description)
    texts = [job_description] + [normalize_text(r.get("text", "")) for r in resumes]

    # If all resumes are empty, return fallback with zeros
    if all(not t.strip() for t in texts[1:]):
        output = []
        for r in resumes:
            strengths, weaknesses = heuristic_strengths_weaknesses(job_description, r.get("text", ""))
            output.append({
                "filename": r.get("filename", "unknown"),
                "similarity": 0.0,
                "final_score": 0,
                "llm": {"strengths": strengths, "weaknesses": weaknesses},
                "text": r.get("text", ""),
                "text_preview": (r.get("text", "") or "")[:PREVIEW_CHARS]
            })
        return output

    # Compute TF-IDF vectors
    try:
        tfidf_matrix = compute_tfidf_vectors(texts)
        # first vector is job_description
        job_vec = tfidf_matrix[0]
        res_vecs = tfidf_matrix[1:]
        sims = cosine_similarity(res_vecs, job_vec).reshape(-1)
    except Exception as e:
        # fallback: if vectorization fails, set zero similarities
        print("TF-IDF/vectorization error:", e)
        sims = np.zeros(len(resumes))

    ranked_output = []
    for idx, r in enumerate(resumes):
        sim = float(sims[idx]) if idx < len(sims) else 0.0

        strengths, weaknesses = heuristic_strengths_weaknesses(job_description, r.get("text", ""))

        # heuristic score (simple function based on matches)
        heuristic_score = 0
        # count exact keyword matches from job desc
        for word in set(job_description.split()):
            if len(word) > 2 and word in normalize_text(r.get("text", "")):
                heuristic_score += 1
        # scale heuristic_score to 0..100 (but we will combine with similarity)
        if heuristic_score > 0:
            # Normalize by a reasonable constant to avoid domination
            heuristic_score = min(100, int((heuristic_score / 10.0) * 100))
        else:
            heuristic_score = 0

        # Final score: combine TF-IDF similarity and heuristic signals
        final_score = int(
            SIMILARITY_WEIGHT * sim * 100 +
            HEURISTIC_WEIGHT * heuristic_score
        )
        final_score = max(0, min(100, final_score))

        ranked_output.append({
            "filename": r.get("filename", "unknown"),
            "similarity": round(sim, 3),
            "final_score": final_score,
            "llm": {"strengths": strengths, "weaknesses": weaknesses},
            "text": r.get("text", ""),
            "text_preview": (r.get("text", "") or "")[:PREVIEW_CHARS]
        })

    # Sort descending by final_score
    ranked_output = sorted(ranked_output, key=lambda x: x["final_score"], reverse=True)
    return ranked_output
