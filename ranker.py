import numpy as np
from sentence_transformers import SentenceTransformer, util

# you can share the same embedding model instance
_EMB = SentenceTransformer("all-MiniLM-L6-v2")

def rank_sections(sections, persona_emb, top_k=None):
    """
    sections: list of {"text":str, "page":int, ...}
    persona_emb: np.ndarray
    returns: same list sorted by similarity
    """
    texts = [sec["text"] for sec in sections]
    sec_embs = _EMB.encode(texts, normalize_embeddings=True)
    sims = util.cos_sim(persona_emb, sec_embs)[0].cpu().tolist()
    for sec, score in zip(sections, sims):
        sec["score"] = float(score)
    sections.sort(key=lambda x: x["score"], reverse=True)
    if top_k:
        return sections[:top_k]
    return sections
