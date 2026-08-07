"""Optional RAG: relevant-context search via FAISS + sentence-transformers.

If the libraries aren't installed or no index has been built, load_rag()
simply returns False and the app keeps working as a plain chat, without RAG.

Per-corpus index shards:
  build_index.py writes one faiss_index.bin+meta.pkl pair per corpus (a
  rag_data/ topic subfolder, or the loose files directly in rag_data/)
  under config.INDEX_DIR/<topic>/ — not one single combined index. This
  module loads EVERY corpus found there at startup and treats them as
  shards of one logical index: _similarity_search queries each shard
  separately and merges the results by score before taking the global
  top-k, and retrieve_context's always-include expansion (below) scans
  across every shard's metadata the same way it would scan one combined
  list. This preserves the exact same retrieval behavior as a single
  combined index (similarity search here never was topic-scoped — see
  build_index.py's module docstring, "Organizing documents by topic") —
  splitting the on-disk index by corpus only changed how build_index.py
  builds and rebuilds things, not what a query sees at runtime. If
  config.INDEX_DIR doesn't exist (e.g. an index built before per-corpus
  indexing existed), this falls back to loading the single legacy
  config.INDEX_PATH/config.META_PATH pair as one shard, so an
  already-deployed server keeps working without an immediate rebuild.

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
import itertools
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

from utils import config

_embed_model = None
# One (faiss_index, meta_list) pair per corpus shard — see module docstring.
_shards: List[Tuple[Any, list]] = []


def _discover_index_shards() -> List[Tuple[str, str]]:
    """Every (index_path, meta_path) pair to load. New layout:
    config.INDEX_DIR/<topic-or-"_root">/faiss_index.bin + meta.pkl — one
    pair per corpus, built independently by build_index.py. Falls back to
    the single legacy config.INDEX_PATH/config.META_PATH pair if INDEX_DIR
    doesn't exist yet."""
    shards: List[Tuple[str, str]] = []
    if os.path.isdir(config.INDEX_DIR):
        for name in sorted(os.listdir(config.INDEX_DIR)):
            corpus_dir = os.path.join(config.INDEX_DIR, name)
            index_path = os.path.join(corpus_dir, "faiss_index.bin")
            meta_path = os.path.join(corpus_dir, "meta.pkl")
            if os.path.isfile(index_path) and os.path.isfile(meta_path):
                shards.append((index_path, meta_path))
    if shards:
        return shards
    if os.path.exists(config.INDEX_PATH) and os.path.exists(config.META_PATH):
        return [(config.INDEX_PATH, config.META_PATH)]
    return []


def load_rag() -> bool:
    """Tries to load every corpus shard. Never raises an exception."""
    global _embed_model, _shards

    shard_paths = _discover_index_shards()
    if not shard_paths:
        print("No RAG index found — running without RAG (fine for plain chat).")
        return False

    try:
        import faiss
        from sentence_transformers import SentenceTransformer

        _embed_model = SentenceTransformer(config.EMBED_MODEL)
        loaded: List[Tuple[Any, list]] = []
        for index_path, meta_path in shard_paths:
            idx = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            loaded.append((idx, meta))
        _shards = loaded
        total_chunks = sum(len(meta) for _, meta in _shards)
        print(f"RAG index loaded: {len(_shards)} corpus/corpora, {total_chunks} chunks total")
        return True
    except Exception as e:
        _embed_model = None
        _shards = []
        print("RAG unavailable (not fatal):", e)
        return False


def is_available() -> bool:
    return bool(_shards) and _embed_model is not None


def _similarity_search(query: str, top_k: int) -> List[Dict[str, Any]]:
    """The plain top-k similarity lookup shared by retrieve_context (below)
    and retrieve_similarity_only — no always-include expansion here, just
    embed the query and return the nearest chunks. Queries every corpus
    shard separately (each is its own FAISS index) and merges the results
    by score before taking the global top-k, so this behaves exactly like
    querying one combined index across every corpus, regardless of how
    many separate index files build_index.py actually wrote."""
    import faiss

    q_emb = _embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)

    candidates: List[Tuple[float, Dict[str, Any]]] = []
    for idx, meta in _shards:
        if idx.ntotal == 0:
            continue
        k = min(top_k, idx.ntotal)
        D, I = idx.search(q_emb, k)
        for score, pos in zip(D[0], I[0]):
            if 0 <= pos < len(meta):
                candidates.append((float(score), meta[pos]))

    candidates.sort(key=lambda pair: pair[0], reverse=True)  # inner product on normalized vectors = cosine — higher is more similar
    return [chunk for _, chunk in candidates[:top_k]]


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


def retrieve_context(
    query: str, top_k: int = config.TOP_K, topic_hint: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Returns a list of chunk dicts (text/topic/always_include), not just
    plain strings — build_prompt needs always_include to decide whether to
    switch into reasoning mode, and topic is what drives the expansion
    below.

    Two passes: first ordinary top-k similarity search, then — for every
    topic represented among those hits (plus topic_hint, if given — see
    below) — pull in any "always_include" (methodology) chunks for that
    topic that didn't already make the top-k cut, so they aren't at the
    mercy of similarity ranking. Chunks with no topic (files directly in
    rag_data/, not in a subfolder) never trigger this expansion, since
    there's no topic to match against.

    topic_hint: when the caller already KNOWS which technique's answer
    this is for (every astro_* tool does — decision.tool_name is exact,
    not a guess), pass its rag_data/ subfolder name here to guarantee that
    topic's methodology gets included even if plain similarity search
    against the free-text query didn't happen to surface any chunk from
    it. Found necessary in practice: a query like "на какой день лучше
    планировать уборку в комнате" shares essentially no vocabulary with
    electional_methodology.txt's astrological terms (кверент,
    сигнификатор, планетарный час...), so it can lose the similarity race
    entirely to some other, unrelated topic's chunks — and that unrelated
    topic's own always_include methodology would get pulled in instead,
    with no warning (this is how a "natal chart" reference ended up in an
    electional answer even with rag_data/astro_elect fully indexed: the
    similarity search's top-k simply never touched that topic at all, so
    the old topics_hit-only expansion had nothing to include from it).
    This does NOT topic-scope the similarity search itself — that still
    searches the whole index, which is correct and intentional (see
    build_index.py's module docstring, "Organizing documents by topic");
    it only guarantees the ONE topic the caller already knows is relevant
    is never silently skipped.

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
    if topic_hint:
        topics_hit.add(topic_hint)

    if topics_hit:
        always_include_chars = 0
        # Flattened across every corpus shard — a topic can in principle
        # only live in one shard's metadata (each corpus has one topic),
        # but chaining them keeps this loop identical to the pre-sharding
        # single-list version rather than needing a nested break.
        for chunk in itertools.chain.from_iterable(meta for _, meta in _shards):
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
