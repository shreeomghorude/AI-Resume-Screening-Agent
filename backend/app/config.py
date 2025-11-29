# backend/app/config.py

# Weights for combining similarity and heuristic score
# Adjust as needed (sum not required to be 1)
SIMILARITY_WEIGHT = 0.7   # weight for cosine similarity contribution
HEURISTIC_WEIGHT = 0.3    # weight for simple keyword/heuristic contribution

# Preview chars shown on frontend
PREVIEW_CHARS = 400
