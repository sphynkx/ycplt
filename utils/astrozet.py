"""AstroZet (.zbs) file format — parsing and writing.

AstroZet is a third-party Windows astrology program. Its .zbs format is a
semicolon-delimited, one-record-per-line birth-data interchange format —
NOT this app's own storage shape. Per the user's explicit design choice: we
store birth profiles however is convenient for us (see db/connection.py's
birth_profiles table and db/repository.py), and use THIS module only at the
boundary — importing a .zbs file into that table, or exporting rows back out
to .zbs text for use in AstroZet itself.

Line shape (9 semicolon-separated fields, each line terminated by a trailing
';'), confirmed against a real user-provided example:

    Name; DD.MM.YYYY; HH:MM:SS; UTC_offset; Place; Lat; Lon; Sex; Comment;

    Иван Петров; 15.08.1985; 12:00:00; +4; Винница, Винницкая обл., Украина; 49n14; 28e29; M; Далее комментарий в свободной форме|Значок пайпа обозначает перевод строки|PHOTO: ClosePeople\\plysyi.jpg|строка начинающаяся с "PHOTO: " и далее относительный путь к фото;

Field notes:
  - Date is DD.MM.YYYY (not ISO) — converted to/from this app's own
    'YYYY-MM-DD' (the shape utils/astro.py's _build_subject() expects).
  - Time is HH:MM:SS, but this app only keeps HH:MM internally (seconds
    are discarded on import; export always re-adds ':00').
  - UTC_offset (e.g. '+4') is kept verbatim, unused for anything but
    faithful re-export — this app resolves the real IANA timezone from
    lat/lon via astro._resolve_timezone(), never from a bare UTC offset.
  - Lat/Lon use degrees+hemisphere-letter+minutes, no separator, e.g.
    '49n14' = 49°14' N, '28e29' = 28°29' E. No seconds component in the
    format itself (whole minutes only).
  - Comment uses '|' as a display newline. A segment starting with
    'PHOTO: ' followed by a relative path marks an embedded photo
    reference, and may appear anywhere among the '|'-separated segments
    (not necessarily last) — this parser scans all segments for it rather
    than assuming a fixed position.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

_PHOTO_PREFIX = "PHOTO: "

_LATLON_RE = re.compile(r"^(\d+)\s*([nsewNSEW])\s*(\d+(?:\.\d+)?)$")
_DATE_RE = re.compile(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$")
_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$")


class ZbsParseError(ValueError):
    """Raised for a single malformed .zbs line. Carries the offending line
    number and raw text so callers (see routes/profiles.py's import
    endpoint) can report a per-line error without discarding the rest of
    an otherwise-valid file."""

    def __init__(self, line_number: int, raw_line: str, reason: str):
        self.line_number = line_number
        self.raw_line = raw_line
        self.reason = reason
        super().__init__(f"line {line_number}: {reason} ({raw_line!r})")


# ---------- parsing: .zbs text -> list of profile dicts ----------
def parse_zbs(text: str) -> Tuple[List[Dict[str, Any]], List[ZbsParseError]]:
    """Parses every non-blank line of a .zbs file. Returns (profiles, errors)
    rather than raising on the first bad line — a real interchange file with
    one malformed record shouldn't block importing the rest (see
    routes/profiles.py's import endpoint, which surfaces both halves to the
    caller)."""
    profiles: List[Dict[str, Any]] = []
    errors: List[ZbsParseError] = []
    for i, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            profiles.append(_parse_zbs_line(line))
        except ZbsParseError as e:
            e.line_number = i
            errors.append(e)
        except ValueError as e:
            errors.append(ZbsParseError(i, line, str(e)))
    return profiles, errors


def _parse_zbs_line(line: str) -> Dict[str, Any]:
    # Each record ends with a trailing ';' — splitting on ';' leaves one
    # empty trailing piece to drop.
    parts = [p.strip() for p in line.split(";")]
    if parts and parts[-1] == "":
        parts = parts[:-1]
    if len(parts) < 8:
        raise ValueError(
            f"expected at least 8 semicolon-separated fields (Name; Date; Time; "
            f"UTC_offset; Place; Lat; Lon; Sex[; Comment]), got {len(parts)}"
        )

    name, date_raw, time_raw, utc_offset, place, lat_raw, lon_raw, sex = parts[:8]
    comment_raw = parts[8] if len(parts) > 8 else ""

    if not name:
        raise ValueError("Name field is empty")

    comment, photo_path = _split_comment(comment_raw)

    return {
        "name": name,
        "date": _parse_date(date_raw),
        "time": _parse_time(time_raw),
        "utc_offset": utc_offset,
        "place": place,
        "lat": _parse_latlon(lat_raw, "lat"),
        "lon": _parse_latlon(lon_raw, "lon"),
        "sex": sex,
        "comment": comment,
        "photo_path": photo_path,
    }


def _parse_date(s: str) -> str:
    m = _DATE_RE.match(s.strip())
    if not m:
        raise ValueError(f"malformed date, expected DD.MM.YYYY: {s!r}")
    day, month, year = (int(g) for g in m.groups())
    return f"{year:04d}-{month:02d}-{day:02d}"


def _parse_time(s: str) -> str:
    m = _TIME_RE.match(s.strip())
    if not m:
        raise ValueError(f"malformed time, expected HH:MM or HH:MM:SS: {s!r}")
    hh, mm, _ss = m.groups()
    return f"{int(hh):02d}:{mm}"


def _parse_latlon(s: str, kind: str) -> float:
    m = _LATLON_RE.match(s.strip())
    if not m:
        raise ValueError(
            f"malformed {kind}, expected e.g. '49n14' (degrees+hemisphere+minutes): {s!r}"
        )
    degrees, hemi, minutes = m.groups()
    hemi = hemi.lower()
    expected = ("n", "s") if kind == "lat" else ("e", "w")
    if hemi not in expected:
        raise ValueError(f"{kind} hemisphere must be one of {expected}, got {hemi!r}: {s!r}")
    value = int(degrees) + float(minutes) / 60.0
    if hemi in ("s", "w"):
        value = -value
    return value


def _split_comment(raw: str) -> Tuple[str, Optional[str]]:
    """'|'-separated segments -> (plain comment text with real newlines,
    photo path or None). The PHOTO segment is excluded from the returned
    comment text and surfaced separately."""
    if not raw:
        return "", None
    photo_path = None
    text_segments = []
    for seg in raw.split("|"):
        seg = seg.strip()
        if not seg:
            continue
        if seg.startswith(_PHOTO_PREFIX):
            photo_path = seg[len(_PHOTO_PREFIX):].strip()
        else:
            text_segments.append(seg)
    return "\n".join(text_segments), photo_path


# ---------- writing: profile dicts -> .zbs text ----------
def export_zbs(profiles: List[Dict[str, Any]]) -> str:
    """Joins one .zbs line per profile, each terminated by ';' and a
    newline — the same shape a real AstroZet .zbs file uses."""
    return "".join(_format_zbs_line(p) + "\n" for p in profiles)


def _format_zbs_line(profile: Dict[str, Any]) -> str:
    fields = [
        profile["name"],
        _format_date(profile["date"]),
        _format_time(profile.get("time") or "12:00"),
        profile.get("utc_offset") or "+0",
        profile.get("place") or "",
        _format_latlon(profile["lat"], "lat"),
        _format_latlon(profile["lon"], "lon"),
        profile.get("sex") or "",
        _join_comment(profile.get("comment") or "", profile.get("photo_path")),
    ]
    return "; ".join(fields) + ";"


def _format_date(iso_date: str) -> str:
    year, month, day = iso_date.split("-")
    return f"{int(day):02d}.{int(month):02d}.{int(year):04d}"


def _format_time(hhmm: str) -> str:
    parts = hhmm.split(":")
    hh = int(parts[0])
    mm = int(parts[1]) if len(parts) > 1 else 0
    return f"{hh:02d}:{mm:02d}:00"


def _format_latlon(value: float, kind: str) -> str:
    if kind == "lat":
        hemi = "n" if value >= 0 else "s"
    else:
        hemi = "e" if value >= 0 else "w"
    value = abs(value)
    degrees = int(value)
    minutes = round((value - degrees) * 60)
    if minutes >= 60:
        degrees += 1
        minutes -= 60
    return f"{degrees}{hemi}{minutes:02d}"


def _join_comment(comment: str, photo_path: Optional[str]) -> str:
    segments = [line for line in comment.split("\n") if line.strip()] if comment else []
    if photo_path:
        segments.append(f"{_PHOTO_PREFIX}{photo_path}")
    return "|".join(segments)


# ---------- using an uploaded .zbs file as a chat request's birth data ----------
# A real, user-suggested use case beyond plain import/export: AstroZet users
# commonly keep one .zbs file PER PERSON that holds both that person's own
# birth-data record AND, on separate lines in the same file, their life
# events (comments hold free-text explanations) — exactly the input
# utils/rectification_events.py's astro_rectification_events tool already
# wants (a birth time to refine plus a list of life events to test candidate
# times against). This lets that same file be attached directly to a chat
# message instead of retyping all of it as prose.
#
# Heuristic for telling a person's own birth record apart from an event
# record within one file (confirmed with the user — AstroZet's own files
# don't mark this explicitly): the FIRST record in the file is the subject;
# every record after it is either a life event (rectification) or a second
# person (synastry), depending entirely on which tool utils/tool_router.py
# picks for the message's own typed instruction. zbs_profiles_to_spec_text
# doesn't try to guess which — it emits data for BOTH interpretations at
# once (see its own docstring) and lets whichever technique's own
# extraction code (astro._parse_spec / astro._extract_two_person_fields /
# rectification_events._try_parse_semicolon_event) pick out only the keys
# or lines it understands. This is the same "hand every candidate source to
# extraction, let each field's own resolver find what it needs" approach
# astro._extract_fields' own docstring already describes for combining
# req.query + the router's own transcription + conversation history — the
# .zbs-derived text is simply one more candidate source, added the same way.
def zbs_profiles_to_spec_text(profiles: List[Dict[str, Any]]) -> str:
    """Converts already-parsed .zbs profiles (from parse_zbs) into plain
    text this app's OWN extraction machinery already reliably parses, so an
    uploaded file's structured data flows through the exact same code paths
    as typed key=value input — no changes needed to astro.py's or
    rectification_events.py's own extraction logic. Returns "" for an empty
    list.

    Emits, in order:
      1. "name=...;date=...;time=...;lat=...;lon=..." for the FIRST
         profile — astro._parse_spec's fast path, feeding every
         single-subject technique (natal, transit, direction, progression,
         return, profection) and rectification's own subject/window
         extraction (rectification_events._extract_birth_window_fields
         calls astro._parse_spec directly on the same kind of text).
      2. The same data again as "_a"/"_b"-suffixed keys for the first TWO
         profiles — astro._extract_two_person_fields' own suffix
         convention, feeding synastry specifically. Harmless, unused
         extra keys for every other technique.
      3. One "description; date; time; place; lat; lon; comment" line per
         REMAINING profile (i.e. profiles[1:]) — the exact shape
         rectification_events._try_parse_semicolon_event already parses,
         picked up ONLY by astro_rectification_events; every other
         technique simply never looks at these lines."""
    if not profiles:
        return ""

    # Every line ends with ';' — astro._parse_spec splits its ENTIRE input
    # on ';' as one long string (it has no idea about newlines at all), so
    # a line missing its own trailing ';' would silently bleed its last
    # field into the next line's first key=value pair (a real bug caught
    # by this module's own stub verification: "lon=28.48...\nname_a=..."
    # with no ';' between them parsed as one field named "lon" whose value
    # was "28.48...\nname_a=Иван Петров").
    lines: List[str] = [_spec_line(profiles[0]) + ";"]

    if len(profiles) >= 2:
        lines.append(_spec_line(profiles[0], suffix="a") + ";" + _spec_line(profiles[1], suffix="b") + ";")

    for ev in profiles[1:]:
        desc = (ev.get("name") or "").strip() or "событие"
        parts = [
            desc,
            ev["date"],
            ev.get("time") or "12:00",
            ev.get("place") or "",
            str(ev.get("lat", "")),
            str(ev.get("lon", "")),
        ]
        if ev.get("comment"):
            parts.append(ev["comment"].replace("\n", " "))
        lines.append("; ".join(parts) + ";")

    return "\n".join(lines)


def _spec_line(profile: Dict[str, Any], suffix: str = "") -> str:
    sfx = f"_{suffix}" if suffix else ""
    parts = []
    if profile.get("name"):
        parts.append(f"name{sfx}={profile['name']}")
    parts.append(f"date{sfx}={profile['date']}")
    parts.append(f"time{sfx}={profile.get('time') or '12:00'}")
    parts.append(f"lat{sfx}={profile['lat']}")
    parts.append(f"lon{sfx}={profile['lon']}")
    return ";".join(parts)
