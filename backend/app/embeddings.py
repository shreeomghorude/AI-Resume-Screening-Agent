from sentence_transformers import SentenceTransformer
import numpy as np

model = None


def get_model():
    global model
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


def embed_text(text: str):
    m = get_model()
    vec = m.encode([text], show_progress_bar=False)[0]
    return vec


def cosine_sim(a, b):
    a = np.array(a)
    b = np.array(b)

    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    num = np.dot(a, b.T)
    denom = (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)).reshape(-1, 1)

    denom[denom == 0] = 1e-9  # avoid divide by zero

    return (num / denom).squeeze()
