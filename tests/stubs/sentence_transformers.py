"""Offline stand-in for sentence-transformers (no HuggingFace access in the sandbox)."""
import hashlib, re
import numpy as np

def _vec(text, dim=384):
    v = np.zeros(dim)
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        v[h % dim] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v

class SentenceTransformer:
    def __init__(self, *a, **k): pass
    def encode(self, texts, normalize_embeddings=False, show_progress_bar=False, **k):
        if isinstance(texts, str): texts = [texts]
        return np.array([_vec(t) for t in texts])

class CrossEncoder:
    def __init__(self, *a, **k): pass
    def predict(self, pairs):
        out = []
        for q, c in pairs:
            qs, cs = set(re.findall(r"[a-z0-9]+", q.lower())), set(re.findall(r"[a-z0-9]+", c.lower()))
            out.append(len(qs & cs) / (len(qs) or 1))
        return np.array(out)
