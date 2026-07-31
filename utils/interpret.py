"""Multi-stage RAG for astrological interpretation: a "digest" pass
between raw retrieval and final answer synthesis.

Why this exists: plain RAG (retrieve chunks similar to the user's
question, paste them into the prompt) works fine for simple lookup
questions, but this app's astro-interpretation use case surfaced a
different failure mode in real testing — the model correctly used a
computed chart's own facts (verified against the user's own reference
chart software) but attached the wrong or generic MEANING to a correct
fact (e.g. describing a Cancer Sun as "leadership", not a characteristic
association for that sign), and used the same flat "усиливает связь"
gloss for a tense square exactly like a harmonious trine. The likely
cause: reference material about what a specific placement or aspect
*means* is written and organized very differently from the user's
free-text birth-data question ("Юпитер в 12 доме" vs. "составь описание
человека, родившегося 5 июля..."), so a single top-k similarity search
against that question rarely surfaces the right handful of paragraphs out
of a large indexed corpus, however good that corpus is.

The fix has three parts:
  1. utils/astro.py's get_significant_facts() ranks a chart's placements
     and aspects by the same qualitative priority rules the methodology
     document already states in prose (angularity, orb precision,
     retrogradation) — reimplemented in plain, deterministic Python rather
     than left for an LLM to apply on the fly under time/context pressure.
  2. For each significant fact, one or two TARGETED retrieval queries
     ("Юпитер в 12 доме", "квадрат Сатурн и Уран") pull in whatever
     reference material actually exists about that specific placement —
     something a single generic search essentially never surfaces.
  3. Rather than pasting those raw fragments straight into the final
     answer prompt (which just relocates the "comprehension + narration in
     one pass" problem to a slightly different place), one additional LLM
     call here first "digests" them: for each fact, given its raw
     fragments plus the concrete degree/orb/house data, it produces a
     short (2-3 sentence) note that already applies the priority/orb rules
     to reach a conclusion — not just quotes or summarizes the source
     text. Only the resulting digested notes (not the raw fragments) then
     go into the final answer-synthesis prompt (routes/chat.py), which
     just has to weave already-reasoned material into a narrative instead
     of doing retrieval-comprehension-synthesis all in one generation.

Cost: one more LLM call per astro answer (this digest pass), on top of the
existing reasoning-mode answer generation — a real latency increase, which
is the deliberate trade being made here for interpretive accuracy. Facts
are capped (see astro.get_significant_facts's top_n) specifically to keep
this bounded to one extra call total, not one call per fact.
"""
from typing import Dict, List, Optional

from utils import llm as llm_utils
from utils import rag as rag_utils

# How many chunks retrieve_similarity_only returns per individual query
# string (a planet fact contributes up to two queries — sign and house —
# so this is per-query, not per-fact).
_CHUNKS_PER_QUERY = 2


def _gather_fact_fragments(facts: List[Dict]) -> Dict[int, List[str]]:
    """For each fact (by its index in `facts`), runs its query strings
    through retrieve_similarity_only and collects the matched chunk texts,
    deduplicated by chunk id within that fact (a planet's sign-query and
    house-query can otherwise legitimately return the same chunk twice)."""
    fragments_by_fact: Dict[int, List[str]] = {}
    for i, fact in enumerate(facts):
        seen_ids = set()
        texts: List[str] = []
        for query in fact["queries"]:
            for chunk in rag_utils.retrieve_similarity_only(query, top_k=_CHUNKS_PER_QUERY):
                cid = chunk.get("id")
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)
                texts.append(chunk["text"])
        fragments_by_fact[i] = texts
    return fragments_by_fact


def _build_digest_prompt(facts: List[Dict], fragments_by_fact: Dict[int, List[str]]) -> str:
    blocks = []
    for i, fact in enumerate(facts):
        fragments = fragments_by_fact.get(i) or []
        frag_text = (
            "\n".join(f"  - {f}" for f in fragments)
            if fragments
            else "  (специфичных справочных материалов не найдено — рассуждай по общим принципам методологии)"
        )
        blocks.append(f"{i + 1}. Факт: {fact['text']}\nСправочные материалы:\n{frag_text}")

    facts_block = "\n\n".join(blocks)
    return (
        "Ниже — список конкретных фактов натальной карты одного человека, и "
        "для каждого — найденные справочные материалы о том, что означает "
        "именно такое положение или аспект (если материалы не найдены — "
        "рассуждай по общим принципам).\n\n"
        f"{facts_block}\n\n"
        "Для КАЖДОГО факта по отдельности напиши короткую (2-3 предложения) "
        "осмысленную заметку: что это конкретно значит для этого человека, "
        "используя найденные материалы там, где они есть, и применяя правила "
        "силы/приоритета (угловые дома важнее кадентных, точный орбис важнее "
        "широкого, ретроградность усиливает влияние). Не пересказывай "
        "справочный текст дословно — переосмысли его применительно именно к "
        "этому факту с этими конкретными значениями. Не путай знак с домом, "
        "не путай факты между собой. Пронумеруй заметки так же, как факты выше "
        "(1, 2, 3, ...), без лишних вступлений."
    )


async def digest_facts_async(facts: List[Dict], max_tokens: Optional[int] = None) -> str:
    """Runs the whole digest pass: per-fact targeted retrieval, then one
    LLM call producing a numbered set of short reinterpreted notes.
    Returns "" (never raises) if there are no facts or anything in here
    fails — callers should treat that as "no digest available, fall back
    to the plain computed-data + methodology prompt" rather than let a
    digest failure block the whole answer."""
    if not facts or not rag_utils.is_available():
        return ""
    try:
        fragments_by_fact = _gather_fact_fragments(facts)
        prompt = _build_digest_prompt(facts, fragments_by_fact)
        # Lower temperature than the final answer's 0.5 — this step is
        # meant to be literal and rule-applying, not creatively phrased;
        # the final synthesis call is where narrative latitude belongs.
        return await llm_utils.generate_async(prompt, max_tokens=max_tokens, temperature=0.3)
    except Exception as e:
        print(f"[interpret] digest step failed, continuing without it: {e}")
        return ""
