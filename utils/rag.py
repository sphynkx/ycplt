"""Optional RAG: relevant-context search via FAISS + sentence-transformers.

If the libraries aren't installed or the index (data/faiss_index.bin,
data/meta.pkl) hasn't been built, load_rag() simply returns False and the
app keeps working as a plain chat, without RAG.
"""
import os
import pickle
from typing import List

from utils import config

_embed_model = None
_faiss_index = None
_meta: list = []


def load_rag() -> bool:
    """Tries to load the index. Never raises an exception."""
    global _embed_model, _faiss_index, _meta

    if not (os.path.exists(config.INDEX_PATH) and os.path.exists(config.META_PATH)):
        print("No RAG index found — running without RAG (fine for plain chat).")
        return False

    try:
        import faiss
        from sentence_transformers import SentenceTransformer

        _embed_model = SentenceTransformer(config.EMBED_MODEL)
        _faiss_index = faiss.read_index(config.INDEX_PATH)
        with open(config.META_PATH, "rb") as f:
            _meta = pickle.load(f)
        print(f"RAG index loaded: {len(_meta)} chunks")
        return True
    except Exception as e:
        _embed_model = None
        _faiss_index = None
        _meta = []
        print("RAG unavailable (not fatal):", e)
        return False


def is_available() -> bool:
    return _faiss_index is not None and _embed_model is not None


def retrieve_context(query: str, top_k: int = config.TOP_K) -> List[str]:
    if not is_available():
        return []
    import faiss

    q_emb = _embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = _faiss_index.search(q_emb, top_k)
    results = []
    for idx in I[0]:
        if 0 <= idx < len(_meta):
            results.append(_meta[idx]["text"])
    return results


def build_prompt(query: str, contexts: List[str]) -> str:
    if not contexts:
        return query
    ctx_text = "\n\n---\n\n".join(contexts)
    return (
        "Use the context below if it's relevant to the question.\n\n"
        f"Context:\n{ctx_text}\n\nQuestion: {query}\n\nAnswer:"
    )
