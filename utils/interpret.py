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

    Four more chart_kind values, all sharing this exact same mechanism:
    "direction" (astro.get_direction_profiles — every natal point shifted
    by the same solar arc), "lunar_return"/"solar_return" (astro.get_
    lunar_return_profiles/get_solar_return_profiles — a real independent
    return chart's own points, read both on their own terms and via
    aspects to natal, like synastry's two-sided reading but for one
    person's own two charts), and "profection" (astro.get_profection_
    profiles — a short, two-item list: a synthetic calendar/ruler summary
    fact plus the year's ruling planet's own natal profile, reused
    verbatim from get_planet_profiles).

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
            elif chart_kind == "progression":
                aspects_header = "   Прогрессивные аспекты этой точки к натальным точкам:"
            elif chart_kind == "synastry":
                aspects_header = "   Синастрические аспекты этой точки к точкам партнёра:"
            elif chart_kind == "direction":
                aspects_header = "   Аспекты этой направленной точки к натальным точкам:"
            elif chart_kind in ("lunar_return", "solar_return"):
                aspects_header = "   Аспекты этой точки карты возвращения к натальным точкам:"
            elif chart_kind == "profection":
                aspects_header = "   Натальные аспекты управителя года:"
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
    elif chart_kind == "progression":
        intro = (
            "Ниже — вторичные прогрессии («день за год») значимых планет "
            "одного человека: для каждой указан её ПРОГРЕССИВНЫЙ знак и "
            "НАТАЛЬНЫЙ дом (дом карты рождения, в котором сейчас находится "
            "прогрессивная планета), её прогрессивные аспекты к натальным "
            "точкам (со знаком/домом натальной точки), и найденные "
            "справочные материалы (если материалы не найдены — рассуждай "
            "по общим принципам вторичных прогрессий). ВАЖНО: это "
            "СИМВОЛИЧЕСКАЯ техника — прогрессивная дата в исходных данных "
            "НЕ реальная дата события, а расчётная точка метода «день за "
            "год»; описывай прогрессии как медленно раскрывающийся этап "
            "внутреннего развития человека за годы/десятилетия, а не как "
            "что-то происходящее в конкретный день."
        )
        house_rule = (
            "— как натальный дом, в котором сейчас находится эта "
            "прогрессивная планета, окрашивает, В КАКОЙ ОБЛАСТИ ЖИЗНИ "
            "разворачивается это медленное изменение (12-й дом — скрыто, "
            "внутренне, не сразу заметно даже самому человеку; угловые "
            "дома 1/4/7/10 — проявляется во внешних решениях и событиях "
            "этого жизненного этапа);\n"
        )
        aspect_rule = (
            "— как КАЖДЫЙ прогрессивный аспект окрашивает эту точку: "
            "используй ГОТОВУЮ пометку «природа: ...» у каждого аспекта "
            "выше, НЕ переопределяй её сам по типу аспекта — «гармоничный» "
            "означает плавное, поддерживающее раскрытие качеств точки; "
            "«напряжённый» — внутреннее противоречие или давление, с "
            "которым человек постепенно учится справляться на этом этапе "
            "жизни (не разовый кризис, а фон целого периода); "
            "«неоднозначный» — не конфликт, а постепенная адаптация; "
            "«нейтральный (зависит от планет)» (соединение) — зависит от "
            "природы обеих планет. Учитывай СКОРОСТЬ прогрессирующей "
            "планеты: аспект от Луны формируется и проходит за пару лет, "
            "аспект от личной планеты (Солнце/Меркурий/Венера/Марс) — за "
            "несколько лет, а точный прогрессивный аспект от внешней "
            "планеты (Юпитер/Сатурн/Уран/Нептун/Плутон) — редкое, "
            "растянутое на десятилетия событие, почти не меняющееся всю "
            "оставшуюся жизнь после того, как сложилось; не описывай его "
            "как проходящий эпизод;\n"
        )
        movement_rule = (
            "— направление движения здесь про темп самого прогрессивного "
            "процесса, а не про дни или недели: «сходящийся, усиливается» "
            "значит эта тема ещё продолжает нарастать в последующие годы; "
            "«расходящийся, ослабевает» — эта тема уже прошла свой пик "
            "жизни и постепенно отходит на второй план.\n"
        )
        closing = (
            "Пиши только на русском языке, без английских слов и без "
            "иероглифов. Не придумывай имя человеку, чья это карта, — "
            "данные не содержат имени; называй его «этот человек» / "
            "«он» / «она». Слова «натальный»/«прогрессивный» — термины, "
            "а не чьи-то имена. Никогда не путай прогрессивную (расчётную, "
            "символическую) дату из данных с реальной календарной датой "
            "события.\n\n"
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
    elif chart_kind == "direction":
        intro = (
            "Ниже — направленные (солнечная дуга) положения значимых точек "
            "одного человека: ВСЕ они смещены от натальных на ОДИН И ТОТ ЖЕ "
            "угол (в отличие от прогрессий, где разные точки движутся с "
            "разной скоростью), для каждой указан её направленный знак и "
            "НАТАЛЬНЫЙ дом (дом натальной карты, через который сейчас "
            "проходит эта направленная точка), её аспекты к натальным "
            "точкам (со знаком/домом натальной точки), и найденные "
            "справочные материалы (если материалы не найдены — рассуждай "
            "по общим принципам солнечных дирекций). ВАЖНО: дирекции — это "
            "точная, вычислимая техника (используется в том числе для "
            "ректификации) — направленный Асцендент и MC читаются так же "
            "значимо, как направленные планеты, не менее важно."
        )
        house_rule = (
            "— как натальный дом, через который сейчас проходит эта "
            "направленная точка, окрашивает, В КАКОЙ ОБЛАСТИ ЖИЗНИ "
            "проявляется её влияние (12-й дом — скрыто, внутренне; "
            "угловые дома 1/4/7/10 — заметно для окружающих и во внешних "
            "обстоятельствах);\n"
        )
        aspect_rule = (
            "— как КАЖДЫЙ аспект к натальной точке окрашивает влияние "
            "направленной точки: используй ГОТОВУЮ пометку «природа: ...» "
            "у каждого аспекта выше, НЕ переопределяй её сам по типу "
            "аспекта — «гармоничный» означает плавную, поддерживающую "
            "активацию натальной точки; «напряжённый» — реальное трение "
            "или кризис, требующий сознательного усилия именно в этот "
            "период; «неоднозначный» — не конфликт, а необходимость "
            "подстроиться; «нейтральный (зависит от планет)» (соединение) "
            "— итог зависит от природы обеих планет;\n"
        )
        movement_rule = (
            "— направление здесь про то, ещё ли точная дирекция "
            "нарастает или уже прошла точный градус: «сходящийся, "
            "усиливается» значит эта дирекция ещё готовится вступить в "
            "полную силу, «расходящийся, ослабевает» — она уже прошла "
            "точное соединение/аспект и её острая фаза позади (эффект всё "
            "ещё растянут на годы, дирекции не проходят за дни или "
            "недели);\n"
        )
        closing = (
            "Пиши только на русском языке, без английских слов и без "
            "иероглифов. Не придумывай имя человеку, чья это карта, — "
            "данные не содержат имени; называй его «этот человек» / "
            "«он» / «она». Слова «натальный»/«направленный» — термины, а "
            "не чьи-то имена.\n\n"
        )
    elif chart_kind in ("lunar_return", "solar_return"):
        cycle_word = "лунара (около месяца)" if chart_kind == "lunar_return" else "солара (около года)"
        chart_word = "лунарной" if chart_kind == "lunar_return" else "солярной"
        intro = (
            f"Ниже — точки {chart_word} карты (реальной независимой карты, "
            f"построенной на момент возвращения планеты к натальному "
            f"градусу): для каждой указан её СОБСТВЕННЫЙ знак, её "
            f"СОБСТВЕННЫЙ дом (в системе домов самой карты возвращения) и "
            f"НАТАЛЬНЫЙ дом (дом натальной карты, в который она попадает "
            f"при наложении), её аспекты к натальным точкам (со знаком/"
            f"домом натальной точки), и найденные справочные материалы "
            f"(если материалы не найдены — рассуждай по общим принципам "
            f"{cycle_word})."
        )
        house_rule = (
            f"— и СОБСТВЕННЫЙ дом карты возвращения (в какой сфере ЭТОГО "
            f"{cycle_word.split(' ')[0]} это проявляется само по себе), и "
            f"НАТАЛЬНЫЙ дом, в который она попадает (какую натальную тему "
            f"это активирует) — оба вместе, не подменяя один другим;\n"
        )
        aspect_rule = (
            "— как КАЖДЫЙ аспект к натальной точке окрашивает эту тему "
            "именно в течение ТЕКУЩЕГО цикла: используй ГОТОВУЮ пометку "
            "«природа: ...» у каждого аспекта выше, НЕ переопределяй её "
            "сам по типу аспекта — «гармоничный» создаёт лёгкую, "
            "поддерживающую активацию натальной темы в этот период; "
            "«напряжённый» — трение или сложность именно сейчас; "
            "«неоднозначный» — необходимость подстроиться; «нейтральный "
            "(зависит от планет)» (соединение) — зависит от природы обеих "
            "планет;\n"
        )
        movement_rule = ""
        closing = (
            "Пиши только на русском языке, без английских слов и без "
            "иероглифов. Не придумывай имя человеку, чья это карта, — "
            "данные не содержат имени; называй его «этот человек» / "
            "«он» / «она». Слово «натальный» — термин, а не чьё-то имя. "
            f"Не путай собственный дом {chart_word} карты с натальным "
            "домом — это два разных дома одной и той же точки.\n\n"
        )
    elif chart_kind == "profection":
        intro = (
            "Ниже — профекция текущего года одного человека: техника "
            "ЦЕЛЫХ ЗНАКОВ (классическая), а не квадрантная система домов, "
            "используемая для остальных техник в этом приложении. Первая "
            "запись — сам факт: какой дом/знак профецирован в этом году и "
            "кто его управитель («хозяин года»); вторая запись — "
            "СОБСТВЕННЫЙ натальный профиль этого управителя (его знак, "
            "натальный дом и натальные аспекты) — именно управитель года "
            "является главной темой всего профекционного года."
        )
        house_rule = (
            "— профецированный дом (а не натальный дом самого управителя) "
            "определяет, В КАКОЙ ОБЛАСТИ ЖИЗНИ разворачивается год: 12-й "
            "профецированный дом — скрыто, внутренне; угловые "
            "профецированные дома 1/4/7/10 — заметно и активно;\n"
        )
        aspect_rule = (
            "— управитель года «включает» те области жизни, которые "
            "показывают его СОБСТВЕННЫЙ натальный дом и натальные "
            "аспекты — используй ГОТОВУЮ пометку «природа: ...» у "
            "каждого его аспекта выше, НЕ переопределяй её сам по типу "
            "аспекта, ровно как для натальной карты;\n"
        )
        movement_rule = ""
        closing = (
            "Пиши только на русском языке, без английских слов и без "
            "иероглифов. Не придумывай имя человеку, чья это карта — "
            "называй его «этот человек» / «он» / «она». Не путай "
            "профецированный дом (техника целых знаков) с натальным домом "
            "управителя года (обычная квадрантная система) — это два "
            "разных, не взаимозаменяемых понятия.\n\n"
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
        "которых нет в данных выше. Реальная, найденная тестированием "
        "ошибка: когда НЕСКОЛЬКО РАЗНЫХ точек оказываются в одном и том же "
        "знаке (это нормально и часто встречается, особенно у медленных "
        "планет), заметка для одной точки иногда по ошибке называет знак "
        "ДРУГОЙ точки просто потому, что он недавно упоминался рядом — "
        "перед тем как назвать знак, сверяйся именно со строкой ЭТОЙ "
        "конкретной точки в данных выше, а не с тем, какой знак был назван "
        "в предыдущей заметке. Если один и тот же тезис или отсылка "
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
    chart_kind="transit", astro.get_progression_profiles' progressed-planet
    bundles when chart_kind="progression", or astro.get_synastry_profiles'
    two combined profile lists when chart_kind="synastry"), not per
    isolated fact. Returns "" (never raises) if there are no profiles or
    anything in here fails — callers should treat that as "no digest
    available, fall back to the plain computed-data + methodology prompt"
    rather than let a digest failure block the whole answer."""
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
# A sixth section ("Совпадающие транзиты...") was added after real user
# feedback that a generated transit answer felt thin and generic (only
# 1-2 planets discussed per section despite far more material being
# available) — the same "more named section slots reliably pulls more
# material out of the model" lever already used for synastry's 5->7
# expansion. This one specifically forces the answer to address transit_
# methodology.txt's point 5 (several simultaneous transits converging on
# one theme/sign/natal point are more significant together than each is
# alone), which a real generated answer completely ignored even though
# the underlying chart had three separate transiting points in the same
# sign at once.
TRANSIT_ANSWER_SECTIONS = [
    "Общая картина текущего периода",
    "Возможности и поддерживающие тенденции",
    "Вызовы и внутреннее напряжение",
    "Совпадающие транзиты: главная тема периода",
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
        "Не рисуй искусственно благополучную картину: если аспект помечен "
        "в данных как «природа: гармоничный» — упомяни это просто как "
        "факт, одной-двумя фразами; а если он помечен как «напряжённый» "
        "или «неоднозначный» — раскрой его внимательно и подробно "
        "(особенно в разделе «Вызовы и внутреннее напряжение»): в чём "
        "именно трение, с какой натальной темой оно связано, как оно "
        "может ощущаться в повседневной жизни прямо сейчас — не сворачивай "
        "его в одну общую фразу и не смягчай до уровня гармоничного. "
        "НИКОГДА не называй гармоничный транзит (трин, секстиль, "
        "полусекстиль, квинтиль, биквинтиль) вызовом или напряжением — "
        "используй ТОЛЬКО готовую пометку «природа: ...» каждого аспекта "
        "из осмысленных заметок, не переопределяй её по памяти о типе "
        "аспекта. Если реальных напряжённых или неоднозначных транзитов в "
        "данных нет вообще — так и есть, не придумывай их.\n\n"
        "Важно для точности: каждый раз, когда упоминаешь транзитную или "
        "натальную точку, используй ТОЛЬКО её знак и дом ровно так, как "
        "указано в блоке ДАННЫЕ и ОСМЫСЛЕННЫЕ ЗАМЕТКИ выше (натальный дом "
        "транзитной планеты — это дом натальной карты, через который она "
        "сейчас проходит, а не отдельно вычисленный дом момента) — не "
        "пересчитывай и не переформулируй их по памяти; если та же точка "
        "упоминается в другом разделе, её знак и дом должны совпадать с "
        "первым упоминанием слово в слово. Реальная, найденная тестированием "
        "ошибка: если несколько РАЗНЫХ транзитных планет сейчас в одном и "
        "том же знаке (это нормально и часто встречается), модель иногда "
        "путает, какой планете какой знак принадлежит, и приписывает знак "
        "одной планеты другой просто потому, что он недавно упоминался "
        "рядом — прежде чем написать «транзитный <планета> в <знак>», "
        "сверь по имени планеты именно ЕЁ строку в данных, а не полагайся "
        "на то, какой знак был упомянут в предыдущем предложении.\n\n"
        "Раздел «Совпадающие транзиты: главная тема периода» — посмотри, "
        "есть ли в данных НЕСКОЛЬКО разных транзитных планет одновременно "
        "аспектирующих одну и ту же натальную точку, или несколько "
        "транзитных планет в одном знаке/доме прямо сейчас — если да, "
        "объясни, почему такое совпадение делает эту тему особенно "
        "значимой именно сейчас (сильнее, чем каждый аспект по "
        "отдельности); если ничего подобного в данных нет, честно напиши "
        "об этом коротко, не выдумывай совпадение.\n\n"
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


# Section headers for the final progression answer (build_progression_
# answer_prompt below) — a fourth, deliberately different breakdown:
# secondary progressions describe a slow, decades-long unfolding of
# character, not a short current period (transit) or a static personality
# snapshot (natal) or a two-person relationship (synastry). "Сроки" here
# is framed around LIFE STAGES (years/decades) rather than TRANSIT_ANSWER_
# SECTIONS' days/weeks framing — the same section NAME pattern
# ("что нарастает, что уже проходит") still applies, just at a completely
# different timescale, which build_progression_answer_prompt's own
# instructions make explicit.
PROGRESSION_ANSWER_SECTIONS = [
    "Общая картина этого этапа жизни",
    "Что раскрывается и поддерживается",
    "Внутренние противоречия этого этапа",
    "Главная тема: что доминирует прямо сейчас",
    "Этапы: что уже сложилось, что ещё впереди",
    "Итог",
]


def build_progression_answer_prompt(
    query: str, computed_text: str, digested_notes: str, general_contexts: List[Dict[str, Any]]
) -> str:
    """Progression-chart counterpart to build_transit_answer_prompt — same
    overall mechanism (fixed named markdown sections, consumed after
    digest_facts_async(..., chart_kind="progression") already reasoned over
    each profile), with PROGRESSION_ANSWER_SECTIONS and instructions
    reframed around a slow, symbolic, multi-year/decade unfolding instead
    of transit's real, current, days-to-months timescale.

    Reuses the same hard-won guardrails (markdown headers, no CJK/English
    leakage, no invented personal name, no invented aspect-naming
    adjectives, no orb numbers in prose, no vague filler language, the
    harmonious/tense "nature" label used verbatim, not re-derived) — none
    of those failure modes are specific to transit charts, so a
    progression answer needs the exact same guardrails."""
    context_parts = [f"ДАННЫЕ ДЛЯ ЭТОГО ЗАПРОСА:\n{computed_text}"]
    if digested_notes:
        context_parts.append(f"ОСМЫСЛЕННЫЕ ЗАМЕТКИ ПО ПРОГРЕССИВНЫМ ПЛАНЕТАМ:\n{digested_notes}")
    if general_contexts:
        general_text = "\n\n---\n\n".join(c["text"] for c in general_contexts)
        context_parts.append(f"ОБЩАЯ МЕТОДОЛОГИЯ И СПРАВОЧНЫЕ МАТЕРИАЛЫ:\n{general_text}")
    context_block = "\n\n===\n\n".join(context_parts)

    sections_list = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(PROGRESSION_ANSWER_SECTIONS))

    return (
        f"{context_block}\n\n"
        f"Вопрос пользователя: {query}\n\n"
        "Напиши развёрнутую астрологическую интерпретацию ВТОРИЧНЫХ "
        "ПРОГРЕССИЙ («день за год») для этого человека, разбитую строго "
        "на следующие разделы — используй эти названия дословно как "
        "markdown-заголовки (## Название раздела), каждый раздел — "
        "несколько предложений связного текста, а не одна фраза:\n\n"
        f"{sections_list}\n\n"
        "В каждом разделе опирайся на конкретные прогрессивные аспекты и "
        "осмысленные заметки выше, относящиеся к этой теме — не "
        "перечисляй их списком, а свяжи в связное описание, объясняя, "
        "ПОЧЕМУ именно сейчас (на этом этапе жизни) раскрывается та или "
        "иная тема (через конкретный прогрессивный аспект и натальный "
        "дом/точку, которую он затрагивает), а не общие свойства планеты "
        "самой по себе. ВАЖНО: прогрессии — это про ГОДЫ и ДЕСЯТИЛЕТИЯ, "
        "не про дни или недели — никогда не описывай прогрессивный аспект "
        "как быстро проходящий эпизод, даже если он помечен как "
        "«расходящийся, ослабевает» (это тоже растянуто на годы). Не "
        "описывай натальный дом сам по себе в отрыве от того, какая "
        "прогрессивная планета сейчас в нём находится. Не повторяй одну и "
        "ту же прогрессивную планету в нескольких разделах без "
        "необходимости — за исключением раздела «Этапы», где допустимо "
        "вернуться к уже упомянутым аспектам именно ради их временной "
        "динамики.\n\n"
        "Не рисуй искусственно благополучную картину: если аспект помечен "
        "в данных как «природа: гармоничный» — упомяни это просто как "
        "факт, одной-двумя фразами; а если он помечен как «напряжённый» "
        "или «неоднозначный» — раскрой его внимательно и подробно "
        "(особенно в разделе «Внутренние противоречия этого этапа»): в "
        "чём именно внутреннее напряжение, с какой натальной темой оно "
        "связано, как человек постепенно учится с ним справляться на "
        "протяжении этого этапа — не сворачивай его в одну общую фразу и "
        "не смягчай до уровня гармоничного. НИКОГДА не называй гармоничную "
        "прогрессию (трин, секстиль, полусекстиль, квинтиль, биквинтиль) "
        "внутренним противоречием — используй ТОЛЬКО готовую пометку "
        "«природа: ...» каждого аспекта из осмысленных заметок, не "
        "переопределяй её по памяти о типе аспекта.\n\n"
        "Раздел «Главная тема: что доминирует прямо сейчас» — учитывай "
        "СКОРОСТЬ прогрессирующей планеты (см. заметки): точный "
        "прогрессивный аспект от Луны — редкий, но проходит за пару лет; "
        "от личной планеты — длится несколько лет; а точный прогрессивный "
        "аспект от внешней планеты (Юпитер/Сатурн/Уран/Нептун/Плутон) — "
        "самое значимое и редкое событие всей карты прогрессий, растянутое "
        "на десятилетия — если такой есть в данных, он должен быть "
        "главной темой этого раздела, а не второстепенным упоминанием "
        "рядом с более быстрыми личными планетами.\n\n"
        "Важно для точности: каждый раз, когда упоминаешь прогрессивную "
        "или натальную точку, используй ТОЛЬКО её знак и дом ровно так, "
        "как указано в блоке ДАННЫЕ и ОСМЫСЛЕННЫЕ ЗАМЕТКИ выше (натальный "
        "дом прогрессивной планеты — это дом натальной карты, в котором "
        "она сейчас находится) — не пересчитывай и не переформулируй их "
        "по памяти; если та же точка упоминается в другом разделе, её "
        "знак и дом должны совпадать с первым упоминанием слово в слово. "
        "Если несколько РАЗНЫХ прогрессивных планет сейчас в одном и том "
        "же знаке, сверяйся по имени планеты именно с ЕЁ строкой в "
        "данных, а не с тем, какой знак был упомянут в предыдущем "
        "предложении.\n\n"
        "НИКОГДА не путай прогрессивную (расчётную, символическую) дату "
        "из блока ДАННЫЕ с реальной календарной датой какого-либо "
        "события — это не дата, когда что-то произошло, а техническая "
        "точка метода «день за год».\n\n"
        "Если материала по какому-то разделу мало — так и напиши коротко, "
        "не выдумывай недостающее. Не добавляй отдельный мини-«Итог» "
        "внутри каждого раздела — итоговый вывод пишется только один раз, "
        "в последнем разделе «Итог» из списка выше: 2-3 предложения общего "
        "вывода.\n\n"
        "Как называть аспект между планетами: используй готовую фразу из "
        "осмысленных заметок дословно («трин Солнца и Меркурия», «квадрат "
        "Луны и Сатурна») — никогда не изобретай прилагательные вроде "
        "«Лунный», «Сатурнианский» вместо названия аспекта, это "
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
        "«натальный»/«прогрессивный» в данных — термины, а не чьё-то имя."
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


# Shared closing guardrail block reused verbatim by all four answer-prompt
# builders below (direction/lunar_return/solar_return/profection) —
# identical wording to what every earlier build_*_answer_prompt function
# already has inline (aspect-naming grammar, no orb numbers in prose, no
# vague filler language, markdown formatting, language purity). Pulled out
# once here (rather than pasted four more times) purely to keep this
# already-large file from repeating ~25 lines of byte-identical text four
# additional times — every existing build_*_answer_prompt function is left
# exactly as it was rather than retrofitted onto this, since they already
# ship correctly and touching them isn't needed for this change.
def _shared_answer_guardrails() -> str:
    return (
        "Как называть аспект между планетами: используй готовую фразу из "
        "осмысленных заметок дословно (например «трин Солнца и Луны», "
        "«квадрат Меркурия и Марса») — никогда не изобретай прилагательные "
        "вроде «Лунный», «Марсианский», «Юпитерианский» вместо названия "
        "аспекта, это неинформативно и часто грамматически неверно.\n\n"
        "Не упоминай числовое значение орбиса (например «орбис 1.4°») в "
        "тексте ответа — используй его только чтобы понять силу аспекта, а "
        "описывай эту силу словами, если нужно, а не цифрой.\n\n"
        "Избегай расплывчатых формулировок вроде «структурные качества», "
        "«социальные качества», «генетические качества» — называй "
        "конкретное качество или тему, реально следующую из данных, а не "
        "абстрактную категорию.\n\n"
        "Форматирование: заголовки разделов — markdown (## Заголовок), при "
        "необходимости можно **выделять жирным** ключевые термины; "
        "юникод-значки планет/знаков/аспектов (☉ ♋ ☌ △ и т.п.) уже "
        "используются в данных выше — можешь использовать их и в своём "
        "тексте рядом с названием для краткости, но не обязательно везде. "
        "Пиши только на русском языке: никаких иероглифов, английских слов "
        "или заголовков-переводов (не дублируй русский заголовок раздела "
        "его английским эквивалентом) и никаких других языков вообще — "
        "если почувствуешь, что вот-вот повторишь то же слово ещё раз или "
        "собираешься переключиться на другой язык, перефразируй по-русски "
        "вместо этого.\n\n"
    )


def _no_invented_name_guardrail(extra_terms: str) -> str:
    """extra_terms lists which words in the data are TERMS, not a
    person's name (e.g. "натальный/направленный") — same anti-
    fabrication guardrail every existing build_*_answer_prompt already
    has, factored out for the four new answer-prompt builders below."""
    return (
        "Не придумывай имя человеку: birth-данные не содержат имени, если "
        "оно не было явно названо в вопросе пользователя — называй "
        "человека «этот человек» / «он» / «она» (грамматический род бери "
        "из формулировки вопроса), а не выдуманным именем. "
        f"{extra_terms} — термины, а не чьё-то имя."
    )


# Section headers for the final direction (solar arc) answer — mirrors
# PROGRESSION_ANSWER_SECTIONS' shape (a slow-ish, but here PRECISE and
# calculable, unfolding), with one section swapped for directions' own
# defining trait: unlike progression's gradual "what dominates now",
# direction_methodology.txt frames this technique around exact, nameable
# timing (real use case: rectification), so "Главная тема" here is framed
# around the single TIGHTEST (most exact) direction rather than the
# slowest-moving one.
DIRECTION_ANSWER_SECTIONS = [
    "Общая картина этого направленного периода",
    "Что раскрывается и поддерживается",
    "Точное напряжение: где давление ощутимее всего",
    "Главная тема: самая точная дирекция",
    "Сроки: что ещё нарастает, что уже прошло",
    "Итог",
]


def build_direction_answer_prompt(
    query: str, computed_text: str, digested_notes: str, general_contexts: List[Dict[str, Any]]
) -> str:
    """Direction-chart counterpart to build_progression_answer_prompt —
    same overall mechanism (fixed named markdown sections, consumed after
    digest_facts_async(..., chart_kind="direction") already reasoned over
    each profile), with DIRECTION_ANSWER_SECTIONS and instructions
    reframed around solar arc directions' defining trait: EVERY point
    moves by the SAME arc (unlike progression's per-point speeds), and the
    technique is prized specifically for precise, calculable timing."""
    context_parts = [f"ДАННЫЕ ДЛЯ ЭТОГО ЗАПРОСА:\n{computed_text}"]
    if digested_notes:
        context_parts.append(f"ОСМЫСЛЕННЫЕ ЗАМЕТКИ ПО НАПРАВЛЕННЫМ ТОЧКАМ:\n{digested_notes}")
    if general_contexts:
        general_text = "\n\n---\n\n".join(c["text"] for c in general_contexts)
        context_parts.append(f"ОБЩАЯ МЕТОДОЛОГИЯ И СПРАВОЧНЫЕ МАТЕРИАЛЫ:\n{general_text}")
    context_block = "\n\n===\n\n".join(context_parts)

    sections_list = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(DIRECTION_ANSWER_SECTIONS))

    return (
        f"{context_block}\n\n"
        f"Вопрос пользователя: {query}\n\n"
        "Напиши развёрнутую астрологическую интерпретацию СОЛНЕЧНЫХ ДУГ "
        "(дирекций) для этого человека, разбитую строго на следующие "
        "разделы — используй эти названия дословно как markdown-заголовки "
        "(## Название раздела), каждый раздел — несколько предложений "
        "связного текста, а не одна фраза:\n\n"
        f"{sections_list}\n\n"
        "В каждом разделе опирайся на конкретные направленные аспекты и "
        "осмысленные заметки выше, относящиеся к этой теме — не "
        "перечисляй их списком, а свяжи в связное описание, объясняя, "
        "ПОЧЕМУ именно эта дирекция значима (через конкретный направленный "
        "аспект и натальный дом/точку, которую он затрагивает), а не общие "
        "свойства планеты самой по себе. ВАЖНО: ВСЕ направленные точки "
        "смещены на ОДИН И ТОТ ЖЕ угол (солнечную дугу) — в отличие от "
        "прогрессий, где разные точки движутся с разной скоростью; никогда "
        "не описывай их как двигающиеся с разной скоростью относительно "
        "друг друга. Дирекции растянуты на годы, а не дни — даже "
        "«расходящийся, ослабевает» аспект здесь означает, что острая фаза "
        "уже прошла, а не что влияние скоро исчезнет полностью. Не "
        "описывай натальный дом сам по себе в отрыве от того, какая "
        "направленная точка сейчас через него проходит. Не повторяй одну и "
        "ту же направленную точку в нескольких разделах без "
        "необходимости — за исключением раздела «Сроки», где допустимо "
        "вернуться к уже упомянутым аспектам именно ради их временной "
        "динамики.\n\n"
        "Не рисуй искусственно благополучную картину: если аспект помечен "
        "в данных как «природа: гармоничный» — упомяни это просто как "
        "факт, одной-двумя фразами; а если он помечен как «напряжённый» "
        "или «неоднозначный» — раскрой его внимательно и подробно "
        "(особенно в разделе «Точное напряжение»): в чём именно трение, с "
        "какой натальной темой оно связано, как оно может ощущаться в этот "
        "период — не сворачивай его в одну общую фразу и не смягчай до "
        "уровня гармоничного. НИКОГДА не называй гармоничную дирекцию "
        "(трин, секстиль, полусекстиль, квинтиль, биквинтиль) точкой "
        "напряжения — используй ТОЛЬКО готовую пометку «природа: ...» "
        "каждого аспекта из осмысленных заметок, не переопределяй её по "
        "памяти о типе аспекта.\n\n"
        "Раздел «Главная тема: самая точная дирекция» — среди всех "
        "приведённых аспектов выбери тот, у которого орбис (сила аспекта, "
        "не упоминай само число) наиболее плотный/точный, и раскрой именно "
        "его подробнее остальных как определяющую тему всего периода; если "
        "несколько аспектов сопоставимо точны — можно упомянуть несколько, "
        "но не растворяй раздел в равномерном перечислении всех "
        "аспектов.\n\n"
        "Важно для точности: каждый раз, когда упоминаешь направленную или "
        "натальную точку, используй ТОЛЬКО её знак и дом ровно так, как "
        "указано в блоке ДАННЫЕ и ОСМЫСЛЕННЫЕ ЗАМЕТКИ выше — не "
        "пересчитывай и не переформулируй их по памяти; если та же точка "
        "упоминается в другом разделе, её знак и дом должны совпадать с "
        "первым упоминанием слово в слово.\n\n"
        "Если материала по какому-то разделу мало — так и напиши коротко, "
        "не выдумывай недостающее. Не добавляй отдельный мини-«Итог» "
        "внутри каждого раздела — итоговый вывод пишется только один раз, "
        "в последнем разделе «Итог» из списка выше: 2-3 предложения общего "
        "вывода.\n\n"
        f"{_shared_answer_guardrails()}"
        f"{_no_invented_name_guardrail('Слова «натальный»/«направленный» в данных')}"
    )


# Section headers for the lunar-return answer — a monthly-cycle framing:
# unlike transit's days/weeks or progression's decades, lunar_return_
# methodology.txt frames this technique around the CURRENT ~27-29 day
# cycle specifically, and around the return's own Moon placement as the
# single most important point (hence its own dedicated section rather
# than folding it into a generic "emotional world" section the way natal
# charts do).
LUNAR_RETURN_ANSWER_SECTIONS = [
    "Общая тема этого лунарного месяца",
    "Эмоциональный фон и повседневные дела (Луна лунара)",
    "Поддерживающие тенденции месяца",
    "Напряжение и вызовы месяца",
    "Какую натальную тему активирует этот лунар",
    "Итог",
]


def build_lunar_return_answer_prompt(
    query: str, computed_text: str, digested_notes: str, general_contexts: List[Dict[str, Any]]
) -> str:
    """Lunar-return counterpart to build_transit_answer_prompt — same
    overall mechanism, LUNAR_RETURN_ANSWER_SECTIONS instead, reframed
    around a REAL independent monthly return chart (own placements read
    on their own terms, per lunar_return_methodology.txt) rather than a
    moving transit moment overlaid on the natal chart alone."""
    context_parts = [f"ДАННЫЕ ДЛЯ ЭТОГО ЗАПРОСА:\n{computed_text}"]
    if digested_notes:
        context_parts.append(f"ОСМЫСЛЕННЫЕ ЗАМЕТКИ ПО ТОЧКАМ ЛУНАРНОЙ КАРТЫ:\n{digested_notes}")
    if general_contexts:
        general_text = "\n\n---\n\n".join(c["text"] for c in general_contexts)
        context_parts.append(f"ОБЩАЯ МЕТОДОЛОГИЯ И СПРАВОЧНЫЕ МАТЕРИАЛЫ:\n{general_text}")
    context_block = "\n\n===\n\n".join(context_parts)

    sections_list = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(LUNAR_RETURN_ANSWER_SECTIONS))

    return (
        f"{context_block}\n\n"
        f"Вопрос пользователя: {query}\n\n"
        "Напиши развёрнутую астрологическую интерпретацию ЛУНАРНОГО "
        "ВОЗВРАЩЕНИЯ (лунара) для этого человека — текущего примерно "
        "МЕСЯЧНОГО цикла — разбитую строго на следующие разделы — "
        "используй эти названия дословно как markdown-заголовки "
        "(## Название раздела), каждый раздел — несколько предложений "
        "связного текста, а не одна фраза:\n\n"
        f"{sections_list}\n\n"
        "В каждом разделе опирайся на конкретные точки лунарной карты и "
        "осмысленные заметки выше, относящиеся к этой теме — не "
        "перечисляй их списком, а свяжи в связное описание. Для каждой "
        "точки лунарной карты в данных указаны ДВА дома — её СОБСТВЕННЫЙ "
        "дом в системе домов самого лунара, и НАТАЛЬНЫЙ дом, в который она "
        "попадает при наложении на натальную карту — используй оба "
        "осмысленно: собственный дом лунара говорит, в какой сфере ЭТОГО "
        "месяца что-то происходит само по себе, а натальный дом — какую "
        "долгосрочную натальную тему это временно активирует; не путай их "
        "и не подменяй один другим. Положение и аспекты Луны лунара — "
        "самое важное во всей карте, ему уделено отдельным разделом выше, "
        "не растворяй его среди прочих точек в других разделах без "
        "необходимости.\n\n"
        "Не рисуй искусственно благополучную картину: если аспект помечен "
        "в данных как «природа: гармоничный» — упомяни это просто как "
        "факт, одной-двумя фразами; а если он помечен как «напряжённый» "
        "или «неоднозначный» — раскрой его внимательно и подробно "
        "(особенно в разделе «Напряжение и вызовы месяца»). НИКОГДА не "
        "называй гармоничный аспект (трин, секстиль, полусекстиль, "
        "квинтиль, биквинтиль) точкой напряжения — используй ТОЛЬКО "
        "готовую пометку «природа: ...» каждого аспекта из осмысленных "
        "заметок, не переопределяй её по памяти о типе аспекта. Если "
        "реальных напряжённых или неоднозначных аспектов в данных нет "
        "вообще — так и есть, не придумывай их.\n\n"
        "ВАЖНО: это цикл примерно в МЕСЯЦ, не год и не день — не описывай "
        "его как мимолётное настроение одного дня и не растягивай его "
        "смысл на годы вперёд.\n\n"
        "Важно для точности: каждый раз, когда упоминаешь точку лунарной "
        "или натальной карты, используй ТОЛЬКО её знак и дома ровно так, "
        "как указано в блоке ДАННЫЕ и ОСМЫСЛЕННЫЕ ЗАМЕТКИ выше — не "
        "пересчитывай и не переформулируй их по памяти; если та же точка "
        "упоминается в другом разделе, её знак и дома должны совпадать с "
        "первым упоминанием слово в слово.\n\n"
        "Если материала по какому-то разделу мало — так и напиши коротко, "
        "не выдумывай недостающее. Не добавляй отдельный мини-«Итог» "
        "внутри каждого раздела — итоговый вывод пишется только один раз, "
        "в последнем разделе «Итог» из списка выше: 2-3 предложения общего "
        "вывода.\n\n"
        f"{_shared_answer_guardrails()}"
        f"{_no_invented_name_guardrail('Слова «натальный»/«лунарный» в данных')}"
    )


# Section headers for the solar-return answer — an annual-cycle framing,
# mirroring lunar return's structure but at the year-scale, and with the
# return's own Ascendant (not its Sun placement) singled out as the most
# important point, per solar_return_methodology.txt.
SOLAR_RETURN_ANSWER_SECTIONS = [
    "Общая тема этого солярного года",
    "Асцендент солара: главный ракурс года",
    "Поддерживающие тенденции года",
    "Напряжение и вызовы года",
    "Какую натальную тему активирует этот солар",
    "Итог",
]


def build_solar_return_answer_prompt(
    query: str, computed_text: str, digested_notes: str, general_contexts: List[Dict[str, Any]]
) -> str:
    """Solar-return counterpart to build_lunar_return_answer_prompt — same
    mechanism, SOLAR_RETURN_ANSWER_SECTIONS instead, reframed around the
    annual (not monthly) cycle and around the return's own Ascendant (not
    Sun placement) as the single most important point, per solar_return_
    methodology.txt."""
    context_parts = [f"ДАННЫЕ ДЛЯ ЭТОГО ЗАПРОСА:\n{computed_text}"]
    if digested_notes:
        context_parts.append(f"ОСМЫСЛЕННЫЕ ЗАМЕТКИ ПО ТОЧКАМ СОЛЯРНОЙ КАРТЫ:\n{digested_notes}")
    if general_contexts:
        general_text = "\n\n---\n\n".join(c["text"] for c in general_contexts)
        context_parts.append(f"ОБЩАЯ МЕТОДОЛОГИЯ И СПРАВОЧНЫЕ МАТЕРИАЛЫ:\n{general_text}")
    context_block = "\n\n===\n\n".join(context_parts)

    sections_list = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(SOLAR_RETURN_ANSWER_SECTIONS))

    return (
        f"{context_block}\n\n"
        f"Вопрос пользователя: {query}\n\n"
        "Напиши развёрнутую астрологическую интерпретацию СОЛНЕЧНОГО "
        "ВОЗВРАЩЕНИЯ (солара) для этого человека — текущего примерно "
        "ГОДОВОГО цикла — разбитую строго на следующие разделы — "
        "используй эти названия дословно как markdown-заголовки "
        "(## Название раздела), каждый раздел — несколько предложений "
        "связного текста, а не одна фраза:\n\n"
        f"{sections_list}\n\n"
        "В каждом разделе опирайся на конкретные точки солярной карты и "
        "осмысленные заметки выше, относящиеся к этой теме — не "
        "перечисляй их списком, а свяжи в связное описание. Для каждой "
        "точки солярной карты в данных указаны ДВА дома — её СОБСТВЕННЫЙ "
        "дом в системе домов самого солара, и НАТАЛЬНЫЙ дом, в который она "
        "попадает при наложении на натальную карту — используй оба "
        "осмысленно, не путай их и не подменяй один другим. Асцендент "
        "солара и его знак — самое важное во всей карте (важнее даже "
        "положения Солнца по дому), ему уделён отдельный раздел выше, не "
        "растворяй его среди прочих точек без необходимости.\n\n"
        "Не рисуй искусственно благополучную картину: если аспект помечен "
        "в данных как «природа: гармоничный» — упомяни это просто как "
        "факт, одной-двумя фразами; а если он помечен как «напряжённый» "
        "или «неоднозначный» — раскрой его внимательно и подробно "
        "(особенно в разделе «Напряжение и вызовы года»). НИКОГДА не "
        "называй гармоничный аспект (трин, секстиль, полусекстиль, "
        "квинтиль, биквинтиль) точкой напряжения — используй ТОЛЬКО "
        "готовую пометку «природа: ...» каждого аспекта из осмысленных "
        "заметок, не переопределяй её по памяти о типе аспекта. Если "
        "реальных напряжённых или неоднозначных аспектов в данных нет "
        "вообще — так и есть, не придумывай их.\n\n"
        "ВАЖНО: это цикл примерно в ГОД, не месяц и не день — не растягивай "
        "его смысл на десятилетия вперёд и не сжимай до одного дня.\n\n"
        "Важно для точности: каждый раз, когда упоминаешь точку солярной "
        "или натальной карты, используй ТОЛЬКО её знак и дома ровно так, "
        "как указано в блоке ДАННЫЕ и ОСМЫСЛЕННЫЕ ЗАМЕТКИ выше — не "
        "пересчитывай и не переформулируй их по памяти; если та же точка "
        "упоминается в другом разделе, её знак и дома должны совпадать с "
        "первым упоминанием слово в слово.\n\n"
        "Если материала по какому-то разделу мало — так и напиши коротко, "
        "не выдумывай недостающее. Не добавляй отдельный мини-«Итог» "
        "внутри каждого раздела — итоговый вывод пишется только один раз, "
        "в последнем разделе «Итог» из списка выше: 2-3 предложения общего "
        "вывода.\n\n"
        f"{_shared_answer_guardrails()}"
        f"{_no_invented_name_guardrail('Слова «натальный»/«солярный» в данных')}"
    )


# Section headers for the profection answer — deliberately SHORT (four,
# not six or seven) since astro.get_profection_profiles only ever returns
# two profiles (the calendar/ruler summary fact plus the ruler's own
# natal profile): unlike every other technique here, there simply isn't
# enough distinct material to responsibly fill a six-section breakdown
# without inventing content — the same "more section slots forces more
# material out of the model" lever that helped natal/synastry would
# actively backfire here, pressuring the model to pad thin material
# instead of the intended effect.
PROFECTION_ANSWER_SECTIONS = [
    "Тема года: профецированный дом",
    "Управитель года и его натальная роль",
    "Как это может проявиться в течение года",
    "Итог",
]


def build_profection_answer_prompt(
    query: str, computed_text: str, digested_notes: str, general_contexts: List[Dict[str, Any]]
) -> str:
    """Profection counterpart to the other build_*_answer_prompt
    functions — same fixed-named-markdown-sections mechanism, but over
    the deliberately small PROFECTION_ANSWER_SECTIONS list (see its own
    comment for why), consumed after digest_facts_async(..., chart_kind=
    "profection") reasoned over the profection's two profiles (the
    calendar/ruler summary fact, and the year's ruler's own natal
    profile — see astro.get_profection_profiles)."""
    context_parts = [f"ДАННЫЕ ДЛЯ ЭТОГО ЗАПРОСА:\n{computed_text}"]
    if digested_notes:
        context_parts.append(f"ОСМЫСЛЕННЫЕ ЗАМЕТКИ ПО ПРОФЕКЦИИ:\n{digested_notes}")
    if general_contexts:
        general_text = "\n\n---\n\n".join(c["text"] for c in general_contexts)
        context_parts.append(f"ОБЩАЯ МЕТОДОЛОГИЯ И СПРАВОЧНЫЕ МАТЕРИАЛЫ:\n{general_text}")
    context_block = "\n\n===\n\n".join(context_parts)

    sections_list = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(PROFECTION_ANSWER_SECTIONS))

    return (
        f"{context_block}\n\n"
        f"Вопрос пользователя: {query}\n\n"
        "Напиши развёрнутую астрологическую интерпретацию ПРОФЕКЦИИ "
        "текущего года (классическая техника целых знаков) для этого "
        "человека, разбитую строго на следующие разделы — используй эти "
        "названия дословно как markdown-заголовки (## Название раздела), "
        "каждый раздел — несколько предложений связного текста, а не одна "
        "фраза:\n\n"
        f"{sections_list}\n\n"
        "ВАЖНО: техника целых знаков ОТЛИЧАЕТСЯ от обычной квадрантной "
        "системы домов — профецированный дом (техника целых знаков от "
        "натального Асцендента) и НАТАЛЬНЫЙ дом самого управителя года "
        "(обычная квадрантная система) — это два РАЗНЫХ понятия, не путай "
        "их и не смешивай в одно.\n\n"
        "В разделе «Тема года: профецированный дом» опирайся на "
        "профецированный дом и знак из данных — какая сфера жизни в фокусе "
        "в этом году просто по факту профекции, ещё до учёта личности "
        "самого управителя.\n\n"
        "В разделе «Управитель года и его натальная роль» опирайся на "
        "осмысленную заметку о натальном профиле управителя выше (его "
        "натальный знак, натальный дом, натальные аспекты) — раскрой, что "
        "за планета «управляет» этим годом и что она сама по себе значит "
        "в натальной карте этого человека; используй готовую пометку "
        "«природа: ...» для каждого её натального аспекта, не "
        "переопределяй её сам по типу аспекта.\n\n"
        "В разделе «Как это может проявиться в течение года» соедини "
        "профецированную сферу (первый раздел) с натальной ролью "
        "управителя (второй раздел) в одно связное объяснение — что "
        "именно из натальных склонностей и тем управителя года, скорее "
        "всего, будет активно и заметно именно в сфере профецированного "
        "дома в течение всего года (не только в какой-то один день).\n\n"
        "Если натальных аспектов управителя в данных нет вообще — так и "
        "есть, не придумывай их; если материала для какого-то раздела "
        "объективно мало, так и напиши коротко. Не добавляй отдельный "
        "мини-«Итог» внутри каждого раздела — итоговый вывод пишется "
        "только один раз, в последнем разделе «Итог»: 2-3 предложения "
        "общего вывода о годе в целом.\n\n"
        f"{_shared_answer_guardrails()}"
        f"{_no_invented_name_guardrail('Слова «натальный»/«профецированный» в данных')}"
    )


def build_help_answer_prompt(
    query: str, overview_text: str, general_contexts: List[Dict[str, Any]]
) -> str:
    """astro_help_assistant's own follow-up prompt (routes/chat.py) —
    deliberately NOT rag_utils.build_prompt's generic reasoning-mode
    template, which every other _INTERPRETED_TOOL_NAMES entry uses fine
    but this one tool cannot: that template opens by asserting its context
    IS relevant to the question ("Context below includes... relevant to
    the question", see rag_utils.build_prompt) and then has the model
    "list the specific facts from the context that matter for this
    question" — true by construction for every other astro_* tool (their
    computed_chunk is always this exact person's real, just-computed
    chart), but only SOMETIMES true here, by design: astro_help_
    assistant's whole point (per its own TOOL_REGISTRY description and
    astro_help_methodology.txt) is to also gracefully handle a question
    that turns out to have nothing to do with astrology at all.

    A real failure was observed using the generic template: asked "На
    каком материке расположен Кейптаун?" through the help-mode toggle
    (ChatRequest.force_help), the model treated the handed-in technique
    overview as "the relevant facts" per the prompt's own framing and
    answered with a technique rundown instead of the actual geography
    question — the prompt itself never gave the model permission to
    decide the context DIDN'T apply. This prompt makes that relevance
    check an explicit first step instead of an assumed premise, and
    splits the two outcomes (irrelevant -> answer normally, ignore the
    material entirely; relevant -> use the material) as concretely as
    build_profection_answer_prompt etc. split their own section
    instructions, rather than trusting a vaguer "use if relevant" aside to
    override the template's opening assumption.

    overview_text is astro_help_overview()'s fixed cheat-sheet (already
    wrapped with its own intro line by routes/chat.py's computed_chunk
    special-case for this tool); general_contexts is whatever
    rag_utils.retrieve_context returned (always includes astro_help_
    methodology.txt's chunks via topic_hint, per _TOOL_TOPIC)."""
    context_parts = [overview_text]
    if general_contexts:
        context_parts.append(
            "\n\n---\n\n".join(c["text"] for c in general_contexts)
        )
    context_block = "\n\n===\n\n".join(context_parts)

    return (
        f'Пользователь написал: "{query}"\n\n'
        "Ниже — справочный материал этого приложения о его "
        "астрологических методиках (что каждая делает, когда какую "
        "выбирать, как оформить запрос). Он МОЖЕТ быть релевантен вопросу "
        "пользователя, а может быть совершенно НЕ релевантен — реши это "
        "первым делом, прежде чем отвечать:\n\n"
        f"{context_block}\n\n"
        "Шаг 1 (не показывай его в ответе, просто пройди про себя): "
        "вопрос пользователя ДЕЙСТВИТЕЛЬНО о выборе или объяснении "
        "методик этого приложения, либо о том, как оформить к нему "
        "запрос?\n\n"
        "Если НЕТ — вопрос не про астрологию и не про методики этого "
        "приложения (общие знания, бытовой вопрос, что угодно другое): "
        "полностью проигнорируй весь материал выше и ответь на реальный "
        "вопрос пользователя обычным образом, кратко и по существу, ни "
        "словом не упоминая методики или астрологию.\n\n"
        "Если ДА — используй материал выше: подскажи, какая методика "
        "(или несколько) подходит, объясни разницу между техниками если "
        "об этом спросили, и/или подскажи, как правильно оформить запрос "
        "— в дружелюбном тоне для человека без астрологического опыта, не "
        "перечисляя сразу все методики без разбора.\n\n"
        "Ответь на языке вопроса пользователя. Не упоминай эти шаги и не "
        "объясняй ход рассуждения — сразу дай итоговый ответ."
    )
