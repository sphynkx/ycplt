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
  1. utils/astro.py's get_planet_profiles() ranks a chart's significant
     points and bundles each one's sign, house, retrograde state, its OWN
     aspects to other points (each carrying the other point's sign/house
     too, so the digest step can judge how strong/relevant that aspecting
     influence is), and any fixed-star conjunction — reimplemented in
     plain, deterministic Python rather than left for an LLM to apply on
     the fly under time/context pressure. This replaced an earlier
     flat-fact design (separate, unconnected "planet", "aspect", and
     "house" facts) after real end-to-end testing showed its actual
     failure mode: aspects were digested in total isolation from the
     placement they modified, so a hard square from a personal planet was
     never distinguished from a supportive trine, and a 12th-house
     placement's normal muting effect was never registered at all — the
     per-planet profile above is what lets a single digest note actually
     synthesize "sign + house + these specific aspects" into one
     characterization instead of three disconnected notes. The Sun, Moon,
     Part of Fortune, and any star-conjunct point are always included
     regardless of score — rare/fundamental points that a pure
     angularity/aspect-count score would otherwise starve out entirely on
     a chart with several fixed-star conjunctions (a real tested case).
  2. For each profiled point, one or more TARGETED retrieval queries
     ("Юпитер в 12 доме", "квадрат Сатурн и Уран", "Сатурн соединение
     звезда Фомальгаут") pull in whatever reference material actually
     exists about that specific placement or aspect — something a single
     generic search essentially never surfaces.
  3. Rather than pasting those raw fragments straight into the final
     answer prompt (which just relocates the "comprehension + narration in
     one pass" problem to a slightly different place), one additional LLM
     call here first "digests" them: for each profile, given its raw
     fragments plus the concrete sign/house/aspect data, it produces one
     synthesized note (not a list) that already applies the priority/orb
     rules AND weighs each aspect's favorable-vs-tense nature and the
     aspecting planet's own strength — not just quotes or summarizes the
     source text. Only the resulting digested notes (not the raw
     fragments) then go into the final answer-synthesis prompt
     (routes/chat.py), which just has to weave already-reasoned material
     into a narrative instead of doing retrieval-comprehension-synthesis
     all in one generation.

Cost: one more LLM call per astro answer (this digest pass), on top of the
existing reasoning-mode answer generation — a real latency increase, which
is the deliberate trade being made here for interpretive accuracy. Profiles
are capped (see astro.get_planet_profiles's top_n) specifically to keep
this bounded to one extra call total, not one call per profile — a
one-call-per-profile design was considered for even deeper per-aspect
synthesis, but ruled out for now given how much it would multiply an
already multi-minute CPU-only generation; worth revisiting if
hardware/latency allow later.
"""
from typing import Any, Dict, List, Optional

from utils import llm as llm_utils
from utils import rag as rag_utils

# How many chunks retrieve_similarity_only returns per individual query
# string (a planet fact contributes up to two queries — sign and house —
# so this is per-query, not per-fact).
_CHUNKS_PER_QUERY = 2


def _gather_fact_fragments(profiles: List[Dict]) -> Dict[int, List[str]]:
    """For each profile (by its index in `profiles`), runs its query
    strings through retrieve_similarity_only and collects the matched
    chunk texts, deduplicated by chunk id within that profile (a planet's
    sign-query, house-query, and several aspect-queries can otherwise
    legitimately return the same chunk more than once). Works unchanged on
    astro.get_planet_profiles' output — a profile is just a dict with a
    "queries" list, same shape the old flat get_significant_facts facts
    had, so nothing about the retrieval step itself needed to change."""
    fragments_by_fact: Dict[int, List[str]] = {}
    for i, profile in enumerate(profiles):
        seen_ids = set()
        texts: List[str] = []
        for query in profile["queries"]:
            for chunk in rag_utils.retrieve_similarity_only(query, top_k=_CHUNKS_PER_QUERY):
                cid = chunk.get("id")
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)
                texts.append(chunk["text"])
        fragments_by_fact[i] = texts
    return fragments_by_fact


def _format_aspect_line(aspect: Dict) -> str:
    other_place = ""
    if aspect.get("other_sign") or aspect.get("other_house"):
        parts = [aspect.get("other_sign") or "", f"{aspect['other_house']} дом" if aspect.get("other_house") else ""]
        other_place = " (" + ", ".join(p for p in parts if p) + ")"
    # Leads with astro.get_planet_profiles' pre-formatted, grammatically
    # correct "phrase" ("трин Солнца и Луны") rather than assembling
    # "<aspect> к <planet>" here — "к" demands the dative case, which none
    # of this module's point labels are in, and a real answer showed what
    # happens when the model has to paper over that mismatch itself
    # ("Лунный аспект", "Марский аспект", "Юпитерианский аспект" —
    # invented, sometimes ungrammatical adjectives). "phrase" sidesteps the
    # whole problem by never needing "к" at all.
    phrase = aspect.get("phrase") or f"{aspect['aspect_ru']} и {aspect['other_label']}"
    # nature_ru ("гармоничный"/"напряжённый"/...) is astro.py's own
    # pre-computed classification (_ASPECT_NATURE), included here for the
    # exact same reason "phrase" exists: a real, reported failure showed a
    # trine and a semisextile — both conventionally harmonious, never
    # tense — labeled "точка напряжения" in a generated answer, even
    # though the digest prompt's own prose rules already listed which
    # aspect types are harmonious vs. tense. Spelling the answer out here,
    # in the data itself, removes the need for the model to re-derive or
    # remember that classification under generation pressure — same
    # "compute it once in Python, hand over the answer" principle already
    # used for aspect-naming grammar.
    return (
        f"{phrase}{other_place}, орбис {aspect['orb']:.1f}° "
        f"[орбис только для оценки силы аспекта здесь, не для итогового текста], "
        f"{aspect['movement_ru']}, природа: {aspect.get('nature_ru', 'нейтральный')}"
    )


def _build_digest_prompt(
    profiles: List[Dict], fragments_by_fact: Dict[int, List[str]], chart_kind: str = "natal"
) -> str:
    """Builds the digest prompt from a list of profile bundles —
    astro.get_planet_profiles' (sign+house+retrograde+this point's own
    aspects+any fixed-star conjunction, chart_kind="natal"),
    astro.get_transit_profiles'/get_dual_chart_profiles' (a transiting
    planet's sign+NATAL house+its own cross-chart aspects to natal
    points, chart_kind="transit"), or astro.get_synastry_profiles' (one
    person's own point, the OTHER person's house it overlays, and its
    synastric cross-aspects to that other person's points,
    chart_kind="synastry", with both people's profiles passed together in
    one combined list — see routes/chat.py) — instead of the older flat,
    disconnected planet/aspect/house facts a single-chart design used to
    produce. All three shapes carry the same dict keys ("text", "aspects",
    "stars", ...), so only the framing/intro text and a couple of
    instruction lines below actually differ by chart_kind; the fact-block
    rendering loop is shared.

    This is the actual fix for a real failure found in end-to-end testing
    (natal case): the old design digested "Юпитер в Овне, 12 дом" and each
    of its aspects as entirely separate, unrelated notes, so the final
    answer never once let an aspect color a planet's description (a hard
    square from a personal planet was never distinguished from a
    supportive trine), and a 12th-house placement's normal muting/hiding
    effect was ignored even when explicitly asked about. Explicitly
    instructing the model to weigh aspects by the aspecting planet's own
    strength (its house/angularity) and by orb tightness, and to apply
    favorable-vs-tense distinctions per aspect type, is meant to produce a
    genuinely synthesized characterization instead of a list of isolated
    facts — equally true whether the aspecting planet is natal or
    transiting."""
    blocks = []
    for i, profile in enumerate(profiles):
        fragments = fragments_by_fact.get(i) or []
        frag_text = (
            "\n".join(f"    - {f}" for f in fragments)
            if fragments
            else "    (специфичных справочных материалов не найдено — рассуждай по общим принципам методологии)"
        )
        lines = [f"{i + 1}. {profile['text']}"]
        if profile.get("aspects"):
            if chart_kind == "transit":
                aspects_header = "   Аспекты этой точки к натальным точкам:"
            elif chart_kind == "synastry":
                aspects_header = "   Синастрические аспекты этой точки к точкам партнёра:"
            else:
                aspects_header = "   Аспекты этой точки к другим:"
            lines.append(aspects_header)
            for a in profile["aspects"]:
                lines.append(f"    - {_format_aspect_line(a)}")
        if profile.get("stars"):
            lines.append("   Соединения с неподвижными звёздами:")
            for s in profile["stars"]:
                # s["text"] already starts with the point's own name
                # ("Сатурн ♄ — соединение..."), redundant right after its
                # own numbered heading above — strip that repeated prefix.
                star_tail = s["text"].split(" — ", 1)[-1]
                lines.append(f"    - {star_tail}")
        lines.append("   Справочные материалы:")
        lines.append(frag_text)
        blocks.append("\n".join(lines))

    facts_block = "\n\n".join(blocks)

    if chart_kind == "transit":
        intro = (
            "Ниже — транзитные (текущие) положения значимых планет одного "
            "человека: для каждой указан её текущий знак и НАТАЛЬНЫЙ дом "
            "(дом карты рождения, через который эта планета сейчас "
            "проходит — не дом, вычисленный отдельно для текущего "
            "момента), её аспекты к натальным точкам (со знаком/домом "
            "натальной точки — чтобы можно было оценить, насколько сильна "
            "и уместна эта транзитная связь), и найденные справочные "
            "материалы (если материалы не найдены — рассуждай по общим "
            "принципам транзитной астрологии)."
        )
        house_rule = (
            "— как натальный дом, через который сейчас проходит эта "
            "транзитная планета, окрашивает, В КАКОЙ ОБЛАСТИ ЖИЗНИ "
            "ощущается её текущее влияние (12-й дом — скрыто, "
            "внутренне; угловые дома 1/4/7/10 — заметно для окружающих "
            "и по внешним обстоятельствам);\n"
        )
        aspect_rule = (
            "— как КАЖДЫЙ аспект к натальной точке окрашивает влияние "
            "транзитной планеты: используй ГОТОВУЮ пометку «природа: ...» "
            "у каждого аспекта выше, НЕ переопределяй её сам по типу "
            "аспекта — «гармоничный» создаёт лёгкую, поддерживающую "
            "активацию натальной точки; «напряжённый» — трение, давление, "
            "требующее сознательного усилия; «неоднозначный» — не "
            "конфликт, а необходимость подстроиться; «нейтральный "
            "(зависит от планет)» (соединение) — итог зависит от природы "
            "обеих планет, не от самого аспекта;\n"
        )
        movement_rule = (
            "— направление движения ключевое для транзитов: "
            "«сходящийся, усиливается» значит влияние ещё нарастает, "
            "«расходящийся, ослабевает» — уже проходит и слабеет; "
            "учитывай это при оценке актуальности каждого аспекта прямо "
            "сейчас.\n"
        )
        closing = (
            "Пиши только на русском языке, без английских слов и без "
            "иероглифов. Не придумывай имя человеку, чья это карта, — "
            "данные не содержат имени; называй его «этот человек» / "
            "«он» / «она». Слова «натальный»/«транзитный» — термины, а "
            "не чьи-то имена.\n\n"
        )
    elif chart_kind == "synastry":
        intro = (
            "Ниже — синастрические профили ДВУХ людей вместе (различить их "
            "можно по имени в скобках рядом с названием точки, например "
            "«Солнце (Человек A)»): для каждой точки одного человека "
            "указан её собственный знак и дом ДРУГОГО человека, в который "
            "она попадает при наложении карт («N дом у <имя>» — это дом "
            "ИМЕННО ДРУГОГО человека, не дом самой этой точки в её "
            "собственной карте), её синастрические аспекты к точкам "
            "другого человека (со знаком/домом этой другой точки в ЕЁ "
            "СОБСТВЕННОЙ карте), и найденные справочные материалы (если "
            "материалы не найдены — рассуждай по общим принципам "
            "синастрии)."
        )
        house_rule = (
            "— как дом ДРУГОГО человека, в который попадает эта точка при "
            "наложении карт, окрашивает, в какой сфере жизни ДРУГОГО "
            "человека ощущается влияние этой точки и этого человека на "
            "него (12-й дом партнёра — скрыто, подсознательно; угловые "
            "дома 1/4/7/10 партнёра — прямо и заметно влияет на партнёра "
            "и на отношения в целом);\n"
        )
        aspect_rule = (
            "— как КАЖДЫЙ синастрический аспект окрашивает связь между "
            "этой точкой и точкой партнёра: используй ГОТОВУЮ пометку "
            "«природа: ...» у каждого аспекта выше — она уже правильно "
            "классифицирует именно этот тип аспекта, НЕ переопределяй её "
            "сам и не полагайся на память о том, какой аспект гармоничный, "
            "а какой напряжённый (реальная, найденная тестированием ошибка: "
            "трин и полусекстиль — оба гармоничные аспекты без "
            "стандартного конфликтного значения — были неверно названы "
            "«точками напряжения» в готовом ответе). «Гармоничный» "
            "создаёт лёгкость, взаимную поддержку и естественное "
            "понимание по темам обеих планет; «напряжённый» — трение, "
            "притяжение через конфликт или взаимное раздражение, "
            "требующее сознательной работы, и должен называться прямо "
            "(трение/конфликт/сложность), а не смягчаться до «динамики» "
            "или «роста»; «неоднозначный» — не конфликт, а необходимость "
            "подстроиться друг под друга; «нейтральный (зависит от "
            "планет)» (соединение) — итог зависит от природы обеих "
            "планет, может быть как усиливающим, так и довлеющим;\n"
        )
        movement_rule = ""
        closing = (
            "Пиши только на русском языке, без английских слов и без "
            "иероглифов. Используй имена (или обозначения «Человек A»/"
            "«Человек B»), указанные в данных, для каждого из двух людей "
            "— не путай, чья это точка и чей это дом.\n\n"
        )
    else:
        intro = (
            "Ниже — натальные точки одного человека, для каждой: её знак, дом, "
            "её собственные аспекты к другим точкам (со знаком/домом ДРУГОЙ "
            "точки — чтобы можно было оценить, насколько сильна и уместна её "
            "аспектирующая роль), соединения с неподвижными звёздами (если "
            "есть), и найденные справочные материалы (если материалы не "
            "найдены — рассуждай по общим принципам)."
        )
        house_rule = (
            "— как дом видоизменяет проявление точки (например, 12-й дом "
            "приглушает и скрывает, угловые дома 1/4/7/10 усиливают и делают "
            "заметным для окружающих);\n"
        )
        aspect_rule = (
            "— как КАЖДЫЙ аспект окрашивает точку: используй ГОТОВУЮ "
            "пометку «природа: ...» у каждого аспекта выше, НЕ "
            "переопределяй её сам по типу аспекта — «гармоничный» "
            "усиливает только те качества точки, что созвучны природе "
            "аспектирующей планеты, не все её качества подряд; "
            "«напряжённый» создаёт трение и внутреннее противоречие между "
            "природой этой точки и аспектирующей; «неоднозначный» — не "
            "конфликт, а необходимость приспособления; «нейтральный "
            "(зависит от планет)» (соединение) — зависит от природы "
            "соединяющейся планеты;\n"
        )
        movement_rule = ""
        closing = (
            "Пиши только на русском языке, без английских слов и без "
            "иероглифов. Не придумывай имя человеку, чья это карта, — данные "
            "не содержат имени; называй его «этот человек» / «он» / «она». "
            "Слово «натальный» — термин («карта рождения»), а не чьё-то имя.\n\n"
        )

    return (
        f"{intro}\n\n"
        f"{facts_block}\n\n"
        "Для КАЖДОЙ точки по отдельности напиши одну синтезированную "
        "заметку (3-5 предложений) — не список фактов, а итоговую "
        "характеристику ЭТОЙ конкретной точки, обязательно учитывая:\n"
        f"{house_rule}"
        f"{aspect_rule}"
        "— масштаб влияния аспекта зависит от точности орбиса (чем меньше "
        "орбис, тем сильнее) и от собственной силы аспектирующей планеты "
        "(её дом/угловатость — та же логика, что и для самой точки) — "
        "используй само число орбиса ТОЛЬКО чтобы решить, насколько силён "
        "аспект, но не упоминай его числовое значение (например «орбис "
        "1.4°») в тексте самой заметки — заметка описывает влияние аспекта "
        "словами (например «плотный, ощутимый аспект» вместо цифры), а не "
        "техническую точность его расчёта;\n"
        f"{movement_rule}"
        "— соединение с неподвижной звездой, если есть, — как её "
        "традиционное значение накладывается на природу точки.\n"
        "Используй найденные материалы там, где они есть, но не "
        "пересказывай их дословно — переосмысли применительно именно к "
        "этой точке с этими конкретными значениями. Не путай знак с домом, "
        "не путай точки между собой, не приписывай точке дом или аспект, "
        "которых нет в данных выше. Если один и тот же тезис или отсылка "
        "(например, к конкретному автору/психологу/традиции) встречается в "
        "справочных материалах НЕСКОЛЬКИХ точек подряд, не повторяй её "
        "дословно в каждой заметке — упомяни не более одного-двух раз во "
        "всей карте, там, где это действительно наиболее уместно, а в "
        "остальных заметках сформулируй мысль своими словами без этой "
        "конкретной отсылки.\n\n"
        "Как называть аспект: используй готовую фразу из списка выше "
        "дословно («трин Солнца и Луны», «квадрат Меркурия и Марса» и "
        "т.п.) или перефразируй её только заменой порядка слов, сохраняя "
        "падежи — никогда не изобретай прилагательные вроде «Лунный», "
        "«Марсианский», «Юпитерианский» вместо названия аспекта: это "
        "неинформативно (непонятно, какой именно аспект) и часто "
        "грамматически неверно.\n\n"
        f"{closing}"
        "Пронумеруй заметки так же, как точки выше (1, 2, 3, ...), без "
        "лишних вступлений."
    )


async def digest_facts_async(
    profiles: List[Dict], max_tokens: Optional[int] = None, chart_kind: str = "natal"
) -> str:
    """Runs the whole digest pass: per-profile targeted retrieval, then one
    LLM call producing a numbered set of short reinterpreted notes — one
    per profile bundle (astro.get_planet_profiles' natal bundles,
    astro.get_transit_profiles' transiting-planet bundles when
    chart_kind="transit", or astro.get_synastry_profiles' two combined
    profile lists when chart_kind="synastry"), not per isolated fact.
    Returns "" (never raises) if there are no profiles or anything in here
    fails — callers should treat that as "no digest available, fall back
    to the plain computed-data + methodology prompt" rather than let a
    digest failure block the whole answer."""
    if not profiles or not rag_utils.is_available():
        return ""
    try:
        fragments_by_fact = _gather_fact_fragments(profiles)
        # Diagnostic, not decorative: whether the corpus actually covers a
        # given point (a specific placement, an aspect, a fixed-star
        # conjunction, ...) is otherwise invisible — "the interpretation
        # doesn't mention this" could mean the corpus lacks that material,
        # or that it's there but phrased in a way these queries don't match,
        # or that the model just ignored good material it was given. This
        # print is what makes those three cases distinguishable from the
        # server console instead of guessed at.
        for i, profile in enumerate(profiles):
            hit_count = len(fragments_by_fact.get(i, []))
            status = f"{hit_count} fragment(s)" if hit_count else "NO fragments found"
            print(
                f"[interpret] profile {profile['text']!r} "
                f"(aspects={len(profile.get('aspects', []))}, stars={len(profile.get('stars', []))}) "
                f"-> {status}"
            )
        prompt = _build_digest_prompt(profiles, fragments_by_fact, chart_kind=chart_kind)
        # Lower temperature than the final answer's 0.5 — this step is
        # meant to be literal and rule-applying, not creatively phrased;
        # the final synthesis call is where narrative latitude belongs.
        return await llm_utils.generate_async(prompt, max_tokens=max_tokens, temperature=0.3)
    except Exception as e:
        print(f"[interpret] digest step failed, continuing without it: {e}")
        return ""


# Section headers for the final natal-chart answer (build_sectioned_
# answer_prompt below) — a proposed, not fixed, breakdown; easy to add/
# remove/rename since it's just a prompt template, not a schema anything
# else depends on. Chosen to mirror the traditional houses/topics a natal
# reading conventionally covers (identity, home/emotion, mind, love, work,
# growth), so each section maps onto a recognizable, coherent chunk of the
# chart rather than being an arbitrary split.
ASTRO_ANSWER_SECTIONS = [
    "Личность и общий характер",
    "Эмоциональный мир, дом и семья",
    "Разум, общение и обучение",
    "Любовь и отношения",
    "Работа, призвание и статус",
    "Рост, вызовы и внутренние противоречия",
    "Итог",
]


def build_sectioned_answer_prompt(
    query: str, computed_text: str, digested_notes: str, general_contexts: List[Dict[str, Any]]
) -> str:
    """Astro-natal-chart-specific final-answer prompt — used by
    routes/chat.py INSTEAD OF rag_utils.build_prompt()'s generic reasoning-
    mode template for this one path, after real, repeated testing showed
    the generic template's final answer kept collapsing into a single
    short paragraph regardless of how emphatically it was told to "write
    in detail" — an abstract length/detail instruction, however strongly
    worded, wasn't a strong enough forcing function on its own. Explicit
    named section headers are: each one is a concrete slot the model has
    to fill in turn, which is harder to skip past than a general
    instruction to elaborate.

    No separate "Рассуждение:"/"Ответ:" split here (unlike
    rag_utils.build_prompt's reasoning mode) — the digest step
    (digest_facts_async) already did the per-fact reasoning; this call's
    only job is organizing already-reasoned material into sections, so
    there's no earlier reasoning trace left for a final answer to
    contradict (the self-consistency failure that split was originally
    built to prevent doesn't apply to a single-pass sectioned write-up).

    Instructions below were extended after review of a real full answer:
    markdown headers are now requested (the "no markdown" instruction was
    the opposite of what turned out to be wanted — plain unformatted
    section names in a long wall of text were hard to scan), a rule
    against non-Russian/non-Latin script leakage was added after a real
    answer contained stray Chinese characters mid-sentence (a small
    model's known failure mode under repetitive phrasing, not something
    pulled from the corpus), and an explicit instruction to always reuse a
    point's sign/house exactly as given rather than re-deriving it was
    added after a real answer stated two different houses for the same
    planet in two different sections."""
    context_parts = [f"ДАННЫЕ ДЛЯ ЭТОГО ЗАПРОСА:\n{computed_text}"]
    if digested_notes:
        context_parts.append(f"ОСМЫСЛЕННЫЕ ЗАМЕТКИ ПО ЗНАЧИМЫМ ТОЧКАМ КАРТЫ:\n{digested_notes}")
    if general_contexts:
        general_text = "\n\n---\n\n".join(c["text"] for c in general_contexts)
        context_parts.append(f"ОБЩАЯ МЕТОДОЛОГИЯ И СПРАВОЧНЫЕ МАТЕРИАЛЫ:\n{general_text}")
    context_block = "\n\n===\n\n".join(context_parts)

    sections_list = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(ASTRO_ANSWER_SECTIONS))

    return (
        f"{context_block}\n\n"
        f"Вопрос пользователя: {query}\n\n"
        "Напиши развёрнутую астрологическую интерпретацию этого человека, "
        "разбитую строго на следующие разделы — используй эти названия "
        "дословно как markdown-заголовки (## Название раздела), каждый "
        "раздел — несколько предложений связного текста, а не одна фраза:\n\n"
        f"{sections_list}\n\n"
        "В каждом разделе опирайся на конкретные точки карты и осмысленные "
        "заметки выше, относящиеся к этой теме — не перечисляй их списком, "
        "а свяжи в связное описание, объясняя, ПОЧЕМУ эти черты характерны "
        "именно для этого человека (через конкретные аспекты и дома), а не "
        "просто общие свойства знака самого по себе. Не описывай дом сам "
        "по себе в отрыве от того, что в нём находится — дом это контекст "
        "для планеты, а не отдельная тема для абзаца. Не повторяй одну и "
        "ту же точку карты в нескольких разделах без необходимости. Если "
        "одна и та же отсылка или формулировка (например, к конкретному "
        "автору/традиции) напрашивается для нескольких разных точек, "
        "используй её не больше одного-двух раз на весь ответ, а не в "
        "каждом разделе подряд.\n\n"
        "Важно для точности: каждый раз, когда упоминаешь конкретную точку "
        "(планету, узел, Асцендент и т.д.), используй ТОЛЬКО её знак и дом "
        "ровно так, как указано в блоке ДАННЫЕ и ОСМЫСЛЕННЫЕ ЗАМЕТКИ выше "
        "— не пересчитывай и не переформулируй их по памяти; если та же "
        "точка упоминается в другом разделе, её знак и дом должны совпадать "
        "с первым упоминанием слово в слово.\n\n"
        "Если материала по какому-то разделу мало — так и напиши коротко, "
        "не выдумывай недостающее. Не добавляй отдельный мини-«Итог» "
        "внутри каждого раздела — итоговый вывод пишется только один раз, "
        "в последнем разделе «Итог» из списка выше: 2-3 предложения общего "
        "вывода и, если есть, явно назови внутреннее противоречие между "
        "разделами.\n\n"
        "Как называть аспект между планетами: используй готовую фразу из "
        "осмысленных заметок дословно («трин Солнца и Луны», «квадрат "
        "Меркурия и Марса») — никогда не изобретай прилагательные вроде "
        "«Лунный», «Марсианский», «Юпитерианский» вместо названия аспекта, "
        "это неинформативно и часто грамматически неверно.\n\n"
        "Не упоминай числовое значение орбиса (например «орбис 1.4°») в "
        "тексте ответа — это техническая точность расчёта, а не то, о чём "
        "пишут в самой интерпретации; вместо цифры описывай силу аспекта "
        "словами, если это вообще нужно (например «выраженный», "
        "«ощутимый»).\n\n"
        "Избегай расплывчатых, ничего не говорящих формулировок вроде "
        "«структурные качества», «социальные качества», «генетические "
        "качества» — вместо абстрактной категории называй конкретное "
        "качество или тему, которая реально следует из данных (например "
        "не «социальные качества», а «лёгкость в общении на публике» или "
        "«потребность в признании в коллективе», если это подтверждается "
        "конкретным домом/аспектом).\n\n"
        "Форматирование: заголовки разделов — markdown (## Заголовок), "
        "при необходимости можно **выделять жирным** ключевые термины; "
        "юникод-значки планет/знаков/аспектов (☉ ♋ ☌ △ и т.п.) уже "
        "используются в данных выше — можешь использовать их и в своём "
        "тексте рядом с названием для краткости, но не обязательно "
        "везде. Пиши только на русском языке: никаких иероглифов, "
        "английских слов или заголовков-переводов (не дублируй русский "
        "заголовок раздела его английским эквивалентом) и никаких других "
        "языков вообще — если почувствуешь, что вот-вот повторишь то же "
        "слово ещё раз или собираешься переключиться на другой язык, "
        "перефразируй по-русски вместо этого.\n\n"
        "Не придумывай имя человеку: birth-данные не содержат имени, если "
        "оно не было явно названо в вопросе пользователя — называй "
        "человека «этот человек» / «он» / «она» (грамматический род бери "
        "из формулировки вопроса, например «родившегося» = мужской, "
        "«родившейся» = женский), а не выдуманным именем вроде «Наталья» "
        "или похожим. Слово «натальный» в данных — термин ( = «карта "
        "рождения»), а не чьё-то имя."
    )


# Section headers for the final transit-chart answer (build_transit_
# answer_prompt below) — deliberately a DIFFERENT breakdown from
# ASTRO_ANSWER_SECTIONS, not a reuse of it: a transit reading is about
# what's currently activated and how long it lasts, not a static
# personality portrait, so "Любовь и отношения"/"Работа, призвание и
# статус"-style life-domain sections would either sit mostly empty (most
# transits don't touch every life domain at once) or force material into
# a section it doesn't really belong in. This breakdown instead follows
# what a transit reading conventionally organizes around: the overall
# theme, what's supportive, what's tense, and — the piece unique to
# transits and not applicable to natal charts at all — timing (each
# profiled aspect already carries "сходящийся, усиливается" / "расходящийся,
# ослабевает" from astro.get_transit_profiles, which this section is
# built to make use of).
TRANSIT_ANSWER_SECTIONS = [
    "Общая картина текущего периода",
    "Возможности и поддерживающие тенденции",
    "Вызовы и внутреннее напряжение",
    "Сроки: что нарастает, что уже проходит",
    "Итог",
]


def build_transit_answer_prompt(
    query: str, computed_text: str, digested_notes: str, general_contexts: List[Dict[str, Any]]
) -> str:
    """Transit-chart counterpart to build_sectioned_answer_prompt — same
    overall mechanism (fixed named markdown sections as a concrete forcing
    function, consumed after digest_facts_async(..., chart_kind="transit")
    already did the per-planet reasoning), but with TRANSIT_ANSWER_SECTIONS
    instead of ASTRO_ANSWER_SECTIONS and instructions reframed around
    transiting-vs-natal relationships rather than single-chart placements.

    Reuses essentially the same hard-won guardrails build_sectioned_
    answer_prompt already has (markdown headers, no CJK/English leakage, no
    invented personal name, no per-section repeated stock references, no
    invented aspect-naming adjectives) — those failure modes are properties
    of the underlying small model's generation behavior, not specific to
    natal charts, so there's no reason to expect a transit answer would be
    immune to any of them."""
    context_parts = [f"ДАННЫЕ ДЛЯ ЭТОГО ЗАПРОСА:\n{computed_text}"]
    if digested_notes:
        context_parts.append(f"ОСМЫСЛЕННЫЕ ЗАМЕТКИ ПО ТРАНЗИТНЫМ ПЛАНЕТАМ:\n{digested_notes}")
    if general_contexts:
        general_text = "\n\n---\n\n".join(c["text"] for c in general_contexts)
        context_parts.append(f"ОБЩАЯ МЕТОДОЛОГИЯ И СПРАВОЧНЫЕ МАТЕРИАЛЫ:\n{general_text}")
    context_block = "\n\n===\n\n".join(context_parts)

    sections_list = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(TRANSIT_ANSWER_SECTIONS))

    return (
        f"{context_block}\n\n"
        f"Вопрос пользователя: {query}\n\n"
        "Напиши развёрнутую астрологическую интерпретацию ТЕКУЩЕГО ПЕРИОДА "
        "(транзитов) для этого человека, разбитую строго на следующие "
        "разделы — используй эти названия дословно как markdown-заголовки "
        "(## Название раздела), каждый раздел — несколько предложений "
        "связного текста, а не одна фраза:\n\n"
        f"{sections_list}\n\n"
        "В каждом разделе опирайся на конкретные транзитные аспекты и "
        "осмысленные заметки выше, относящиеся к этой теме — не "
        "перечисляй их списком, а свяжи в связное описание, объясняя, "
        "ПОЧЕМУ именно сейчас активирована та или иная тема (через "
        "конкретный транзитный аспект и натальный дом/точку, которую он "
        "затрагивает), а не общие свойства планеты-транзитёра самой по "
        "себе. Не описывай натальный дом сам по себе в отрыве от того, "
        "какая транзитная планета через него сейчас проходит. Не повторяй "
        "одну и ту же транзитную планету в нескольких разделах без "
        "необходимости — за исключением раздела «Сроки», где допустимо "
        "вернуться к уже упомянутым аспектам именно ради их временной "
        "динамики. Если одна и та же отсылка или формулировка напрашивается "
        "для нескольких разных аспектов, используй её не больше "
        "одного-двух раз на весь ответ, а не в каждом разделе подряд.\n\n"
        "Важно для точности: каждый раз, когда упоминаешь транзитную или "
        "натальную точку, используй ТОЛЬКО её знак и дом ровно так, как "
        "указано в блоке ДАННЫЕ и ОСМЫСЛЕННЫЕ ЗАМЕТКИ выше (натальный дом "
        "транзитной планеты — это дом натальной карты, через который она "
        "сейчас проходит, а не отдельно вычисленный дом момента) — не "
        "пересчитывай и не переформулируй их по памяти; если та же точка "
        "упоминается в другом разделе, её знак и дом должны совпадать с "
        "первым упоминанием слово в слово.\n\n"
        "Раздел «Сроки: что нарастает, что уже проходит» — используй "
        "именно то, что указано в данных для каждого аспекта "
        "(«сходящийся, усиливается» значит влияние ещё растёт; "
        "«расходящийся, ослабевает» — уже проходит), не придумывай сроки "
        "от себя.\n\n"
        "Если материала по какому-то разделу мало — так и напиши коротко, "
        "не выдумывай недостающее. Не добавляй отдельный мини-«Итог» "
        "внутри каждого раздела — итоговый вывод пишется только один раз, "
        "в последнем разделе «Итог» из списка выше: 2-3 предложения общего "
        "вывода.\n\n"
        "Как называть аспект между планетами: используй готовую фразу из "
        "осмысленных заметок дословно («квадрат Сатурна и Солнца», «трин "
        "Юпитера и Нептуна») — никогда не изобретай прилагательные вроде "
        "«Сатурнианский», «Юпитерианский» вместо названия аспекта, это "
        "неинформативно и часто грамматически неверно.\n\n"
        "Не упоминай числовое значение орбиса (например «орбис 1.4°») в "
        "тексте ответа — используй его только чтобы понять силу аспекта, "
        "а описывай эту силу словами, если нужно, а не цифрой.\n\n"
        "Избегай расплывчатых формулировок вроде «структурные качества», "
        "«социальные качества», «генетические качества» — называй "
        "конкретное качество или тему, реально следующую из данных, а не "
        "абстрактную категорию.\n\n"
        "Форматирование: заголовки разделов — markdown (## Заголовок), "
        "при необходимости можно **выделять жирным** ключевые термины; "
        "юникод-значки планет/знаков/аспектов (☉ ♋ ☌ △ и т.п.) уже "
        "используются в данных выше — можешь использовать их и в своём "
        "тексте рядом с названием для краткости, но не обязательно "
        "везде. Пиши только на русском языке: никаких иероглифов, "
        "английских слов или заголовков-переводов (не дублируй русский "
        "заголовок раздела его английским эквивалентом) и никаких других "
        "языков вообще — если почувствуешь, что вот-вот повторишь то же "
        "слово ещё раз или собираешься переключиться на другой язык, "
        "перефразируй по-русски вместо этого.\n\n"
        "Не придумывай имя человеку: birth-данные не содержат имени, если "
        "оно не было явно названо в вопросе пользователя — называй "
        "человека «этот человек» / «он» / «она» (грамматический род бери "
        "из формулировки вопроса), а не выдуманным именем. Слова "
        "«натальный»/«транзитный» в данных — термины, а не чьё-то имя."
    )


# Section headers for the final synastry answer (build_synastry_answer_
# prompt below) — a third, deliberately different breakdown from both
# ASTRO_ANSWER_SECTIONS (single-person personality) and TRANSIT_ANSWER_
# SECTIONS (single-person current period): synastry is about the
# RELATIONSHIP between two specific people, so every section here is
# framed around the connection, not either person alone. Seven sections
# (matching ASTRO_ANSWER_SECTIONS' own count) rather than the original
# five — a real, reported failure with the five-section version: the
# answer covered only a handful of aspects total across the whole
# response even though get_synastry_profiles() had produced far more
# material, reading as thinner and more one-sidedly harmonious than the
# actual chart data supported. More named section slots is the same
# forcing-function lever that already worked for natal charts (repeated
# testing there showed the model reliably fills however many concrete
# slots it's given, rather than self-regulating how much to write from an
# abstract "be thorough" instruction alone) — splitting what was one
# "Эмоциональная и близкая связь" section into separate emotional and
# romantic/attraction sections, and one "Общение и вызовы" pairing into
# separate communication/values and tension/conflict sections, plus a new
# long-term/practical section, gives the richer aspect set produced by
# top_n_each more places to actually land instead of being compressed
# away.
SYNASTRY_ANSWER_SECTIONS = [
    "Общая картина совместимости",
    "Эмоциональная связь и взаимопонимание",
    "Романтика, влечение и близость",
    "Общение, ценности и совместные цели",
    "Точки напряжения и конфликты",
    "Долгосрочный потенциал и практическая совместимость",
    "Итог",
]


def build_synastry_answer_prompt(
    query: str,
    computed_text: str,
    digested_notes: str,
    general_contexts: List[Dict[str, Any]],
    name_a: str,
    name_b: str,
) -> str:
    """Synastry counterpart to build_sectioned_answer_prompt/
    build_transit_answer_prompt — same overall mechanism (fixed named
    markdown sections, consumed after digest_facts_async(...,
    chart_kind="synastry") already reasoned over each profile), with
    SYNASTRY_ANSWER_SECTIONS and instructions reframed around the
    RELATIONSHIP between two people instead of one person's own
    placements or current transits.

    name_a/name_b are astro.get_synastry_profiles' own subject names —
    each person's actual name, or a role word like "Мужчина"/"Женщина",
    if astro._extract_person_label found one right before that person's
    birth date in the free text (or if the router supplied explicit
    name_a=/name_b= key=value fields); "Человек A"/"Человек B" only as a
    last-resort fallback when neither is available. Since name_a/name_b
    already reflect however the user themselves referred to each person
    in the common case, the model mostly just has to use them consistently
    — the instruction below is a safety net for the remaining case where
    extraction found nothing but the user's own question still names both
    people some other way (e.g. "Иван и Мария" phrased differently from
    how their birth data was written), keeping first-mentioned = name_a's
    data and second-mentioned = name_b's data (matching how
    astro._split_two_person_text itself orders the two people — first
    date found in the text becomes person A, second becomes person B)."""
    context_parts = [f"ДАННЫЕ ДЛЯ ЭТОГО ЗАПРОСА:\n{computed_text}"]
    if digested_notes:
        context_parts.append(f"ОСМЫСЛЕННЫЕ ЗАМЕТКИ ПО СИНАСТРИЧЕСКИМ ТОЧКАМ ОБОИХ ЛЮДЕЙ:\n{digested_notes}")
    if general_contexts:
        general_text = "\n\n---\n\n".join(c["text"] for c in general_contexts)
        context_parts.append(f"ОБЩАЯ МЕТОДОЛОГИЯ И СПРАВОЧНЫЕ МАТЕРИАЛЫ:\n{general_text}")
    context_block = "\n\n===\n\n".join(context_parts)

    sections_list = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(SYNASTRY_ANSWER_SECTIONS))

    return (
        f"{context_block}\n\n"
        f"Вопрос пользователя: {query}\n\n"
        f"В данных двое людей обозначены как «{name_a}» и «{name_b}» — "
        "используй именно эти обозначения последовательно во всём "
        "ответе. Если в вопросе пользователя оба человека названы каким-"
        "то ДРУГИМ способом (например по-другому написанным именем), "
        f"используй то, как их назвал пользователь, вместо «{name_a}»/"
        f"«{name_b}», считая, что «{name_a}» — это тот человек, что "
        "упомянут в вопросе ПЕРВЫМ, а второй по порядку — это "
        f"«{name_b}».\n\n"
        "Напиши развёрнутую астрологическую интерпретацию СОВМЕСТИМОСТИ "
        "и динамики отношений между этими двумя людьми (синастрия), "
        "разбитую строго на следующие разделы — используй эти названия "
        "дословно как markdown-заголовки (## Название раздела), каждый "
        "раздел — НЕСКОЛЬКО предложений связного текста, а не одна "
        "фраза, и опирается на НЕСКОЛЬКО разных синастрических аспектов "
        "и наложений из осмысленных заметок и данных выше (не один-два "
        "самых ярких, а как можно более полное освещение значимых "
        "связей между картами, относящихся к теме раздела):\n\n"
        f"{sections_list}\n\n"
        "В каждом разделе опирайся на конкретные синастрические аспекты "
        "и осмысленные заметки выше, относящиеся к этой теме — не "
        "перечисляй их списком, а свяжи в связное описание, объясняя, "
        "ПОЧЕМУ именно эта черта отношений характерна именно для этой "
        "пары (через конкретный синастрический аспект и то, в чей дом "
        "попадает чья точка), а не общие свойства планет самих по себе. "
        "Пиши именно про ОТНОШЕНИЯ и взаимодействие двух людей, а не про "
        "характер одного из них в отрыве от другого. Не повторяй один и "
        "тот же синастрический аспект в нескольких разделах без "
        "необходимости.\n\n"
        "Не рисуй искусственно гармоничную картину: синастрия — это в "
        "первую очередь исследование того, ГДЕ в паре реальное трение, а "
        "не просто перечисление того, что и так хорошо сочетается. Если "
        "аспект помечен в данных как «природа: гармоничный» — упомяни это "
        "просто как факт, одной-двумя фразами, не нужно его расписывать "
        "подробнее остального; а если аспект помечен как «природа: "
        "напряжённый» или «неоднозначный» — раскрой его подробно и "
        "внимательно (особенно в разделе «Точки напряжения и конфликты»): "
        "в чём именно трение, между какими конкретно потребностями двух "
        "людей, как оно может проявляться в быту, и что с этим можно "
        "сделать — не сворачивай его в одну общую фразу и не смягчай до "
        "уровня гармоничного. НИКОГДА не называй гармоничный аспект "
        "(трин, секстиль, полусекстиль, квинтиль, биквинтиль — то, что "
        "помечено «природа: гармоничный» в данных) точкой напряжения или "
        "конфликта — используй ТОЛЬКО готовую пометку «природа: ...» "
        "каждого аспекта, не переопределяй её по памяти о типе аспекта "
        "(реальная, найденная тестированием ошибка: трин и полусекстиль, "
        "оба гармоничные, были неверно названы «точками напряжения»). "
        "Если реальных напряжённых или неоднозначных аспектов в данных "
        "нет вообще — так и есть, не придумывай их.\n\n"
        "Важно для точности: каждый раз, когда упоминаешь чью-то точку "
        "или чей-то дом, используй ТОЛЬКО то, что указано в блоке ДАННЫЕ "
        "и ОСМЫСЛЕННЫЕ ЗАМЕТКИ выше — не путай, чья это точка и чей это "
        "дом (дом партнёра, в который попадает точка, это не дом самого "
        "владельца точки в его собственной карте); если тот же аспект "
        "упоминается в другом разделе, он должен описываться одинаково "
        "и там, и там.\n\n"
        "Если материала по какому-то разделу мало — так и напиши "
        "коротко, не выдумывай недостающее. Не добавляй отдельный "
        "мини-«Итог» внутри каждого раздела — итоговый вывод пишется "
        "только один раз, в последнем разделе «Итог» из списка выше: "
        "2-3 предложения общего вывода о паре в целом.\n\n"
        "Как называть аспект между планетами: используй готовую фразу из "
        "осмысленных заметок дословно («трин Солнца и Луны», «квадрат "
        "Меркурия и Венеры») — никогда не изобретай прилагательные вроде "
        "«Лунный», «Венерианский» вместо названия аспекта, это "
        "неинформативно и часто грамматически неверно.\n\n"
        "Не упоминай числовое значение орбиса (например «орбис 1.4°») в "
        "тексте ответа — используй его только чтобы понять силу аспекта, "
        "а описывай эту силу словами, если нужно, а не цифрой.\n\n"
        "Избегай расплывчатых формулировок вроде «структурные качества», "
        "«социальные качества», «генетические качества» — называй "
        "конкретное качество, потребность или тему отношений, реально "
        "следующую из конкретного аспекта/дома, а не абстрактную "
        "категорию.\n\n"
        "Форматирование: заголовки разделов — markdown (## Заголовок), "
        "при необходимости можно **выделять жирным** ключевые термины; "
        "юникод-значки планет/знаков/аспектов (☉ ♋ ☌ △ и т.п.) уже "
        "используются в данных выше — можешь использовать их и в своём "
        "тексте рядом с названием для краткости, но не обязательно "
        "везде. Пиши только на русском языке: никаких иероглифов, "
        "английских слов или заголовков-переводов (не дублируй русский "
        "заголовок раздела его английским эквивалентом) и никаких других "
        "языков вообще — если почувствуешь, что вот-вот повторишь то же "
        "слово ещё раз или собираешься переключиться на другой язык, "
        "перефразируй по-русски вместо этого."
    )
