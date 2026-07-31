"""Optional RAG: relevant-context search via FAISS + sentence-transformers.

If the libraries aren't installed or the index (data/faiss_index.bin,
data/meta.pkl) hasn't been built, load_rag() simply returns False and the
app keeps working as a plain chat, without RAG.

Two things beyond plain top-k similarity retrieval, both driven by
metadata build_index.py attaches to each chunk (topic, always_include):

1. "Methodology" documents (named "*_methodology.txt/.pdf" — see
   build_index.py) describe HOW to combine facts into a conclusion rather
   than being a fact themselves, so they rarely resemble a specific
   question closely enough to rank in an ordinary top-k similarity search
   — a real risk for a synthesis task like astrological interpretation,
   where the individual planet/house/aspect facts retrieve fine but the
   rules for combining them into something new might not. retrieve_context
   works around this: once ordinary similarity search has found chunks
   belonging to a given topic, every "always_include" chunk from that same
   topic is added too, regardless of its own similarity rank.

2. When any always-include (methodology) context is present, build_prompt
   switches from a plain "answer using this context" prompt to one that
   asks the model to reason step by step — enumerate the relevant facts,
   think through how they interact per the methodology, then synthesize —
   instead of a one-shot lookup-style answer. This is deliberately just a
   prompting change on the existing chat model, not a separate reasoning
   model; see the project README for the fuller design discussion and what
   a dedicated reasoning model would add on top of this.
"""
import os
import pickle
from typing import Any, Dict, List

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


def _similarity_search(query: str, top_k: int) -> List[Dict[str, Any]]:
    """The plain top-k similarity lookup shared by retrieve_context (below)
    and retrieve_similarity_only — no always-include expansion here, just
    embed the query and return the nearest chunks."""
    import faiss

    q_emb = _embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = _faiss_index.search(q_emb, top_k)

    results: List[Dict[str, Any]] = []
    for idx in I[0]:
        if 0 <= idx < len(_meta):
            results.append(_meta[idx])
    return results


def retrieve_similarity_only(query: str, top_k: int = 2) -> List[Dict[str, Any]]:
    """Plain top-k similarity search, no always-include expansion — used
    for targeted, per-fact retrieval (utils/interpret.py) where the "query"
    is something like "Юпитер в 12 доме" rather than the user's actual
    question. Pulling in a topic's entire methodology document again for
    every one of a dozen per-fact queries would be pure waste (it's already
    included once via retrieve_context/build_prompt) — this function exists
    specifically to avoid that, returning just the handful of chunks that
    are actually about this one specific fact."""
    if not is_available():
        return []
    return _similarity_search(query, top_k)


def retrieve_context(query: str, top_k: int = config.TOP_K) -> List[Dict[str, Any]]:
    """Returns a list of chunk dicts (text/topic/always_include), not just
    plain strings — build_prompt needs always_include to decide whether to
    switch into reasoning mode, and topic is what drives the expansion
    below.

    Two passes: first ordinary top-k similarity search, then — for every
    topic represented among those hits — pull in any "always_include"
    (methodology) chunks for that topic that didn't already make the
    top-k cut, so they aren't at the mercy of similarity ranking. Chunks
    with no topic (files directly in rag_data/, not in a subfolder) never
    trigger this expansion, since there's no topic to match against.

    The always_include expansion stops once it's added
    config.RAG_ALWAYS_INCLUDE_MAX_CHARS worth of text, in whatever order
    the chunks happen to appear in _meta (build_index.py's chunking order
    — roughly document order). A long methodology document would
    otherwise be included *in full* the instant any chunk from its topic
    is retrieved, regardless of length, which can overrun a small model's
    whole context window on its own — this was confirmed in practice, not
    theoretical (see the astro tool's answer path in routes/chat.py).
    """
    if not is_available():
        return []

    results = _similarity_search(query, top_k)
    seen_ids = {chunk.get("id") for chunk in results}
    topics_hit = {chunk["topic"] for chunk in results if chunk.get("topic")}

    if topics_hit:
        always_include_chars = 0
        for chunk in _meta:
            if (
                chunk.get("always_include")
                and chunk.get("topic") in topics_hit
                and chunk.get("id") not in seen_ids
            ):
                if always_include_chars >= config.RAG_ALWAYS_INCLUDE_MAX_CHARS:
                    break
                results.append(chunk)
                seen_ids.add(chunk.get("id"))
                always_include_chars += len(chunk.get("text", ""))

    return results


def build_prompt(query: str, contexts: List[Dict[str, Any]]) -> str:
    if not contexts:
        return query

    ctx_text = "\n\n---\n\n".join(c["text"] for c in contexts)
    has_methodology = any(c.get("always_include") for c in contexts)

    if has_methodology:
        return (
            "Context below includes both reference facts and an "
            "interpretation methodology (how to combine facts into a "
            "conclusion) relevant to the question.\n\n"
            f"Context:\n{ctx_text}\n\nQuestion: {query}\n\n"
            "First, reason step by step: list the specific facts from the "
            "context that matter for this question, then think through how "
            "they interact and what their combination suggests according to "
            "the methodology — not just what each fact means in isolation. "
            "Write this under \"Рассуждение:\". Then, under \"Ответ:\", give "
            "the final synthesized answer for the user — natural and "
            "conversational, in the same language the question was asked "
            "in, not a restatement of the reasoning steps. The Ответ must "
            "stay consistent with your own Рассуждение above: if you used "
            "specific facts from the context there, treat them as given "
            "and known in the Ответ too — never claim in the Ответ that "
            "data is missing or wasn't provided if you already reasoned "
            "over it just above; that contradiction is a mistake, not "
            "appropriate caution."
        )

    return (
        "Use the context below if it's relevant to the question.\n\n"
        f"Context:\n{ctx_text}\n\nQuestion: {query}\n\nAnswer:"
    )
