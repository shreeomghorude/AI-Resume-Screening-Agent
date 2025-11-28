import os


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# If you prefer to keep things free: leave OPENAI_API_KEY unset and the app will use a rule-based fallback for explanations.
import os
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")



EMBEDDING_MODEL = "all-MiniLM-L6-v2" # sentence-transformers


# Weighting between embedding similarity and LLM score (if LLM available)
SIMILARITY_WEIGHT = 0.6
LLM_WEIGHT = 0.4