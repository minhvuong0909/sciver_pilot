"""Tách số trong claim và trong ô bằng chứng — regex + decimal.Decimal (SPEC 7-M5)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from sciverx.schemas import TimeSpec

_SCALE = {"k": Decimal(1_000), "thousand": Decimal(1_000), "m": Decimal(1_000_000),
          "million": Decimal(1_000_000), "b": Decimal(1_000_000_000), "billion": Decimal(1_000_000_000)}

NUM_RE = re.compile(
    r"(?<![\w.])(?P<sign>[-+−])?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s*(?P<scale>[kKmMbB]\b|thousand|million|billion))?"
    r"\s*(?P<unit>%|percentage points?|pp\b|points?\b|×|x\b|times\b|-fold\b|fold\b|ms\b|s\b|seconds?\b|hours?\b|"
    r"minutes?\b|days?\b)?", re.I)

_QUALIFIERS = [
    ("at_least", r"(at least|no less than|≥|>=|a minimum of)\s*$"),
    ("at_most", r"(at most|up to|no more than|≤|<=|a maximum of)\s*$"),
    ("more_than", r"(more than|over|above|exceeds?|exceeding|greater than|>)\s*$"),
    ("less_than", r"(less than|under|below|fewer than|<)\s*$"),
    ("approximately", r"(about|around|approximately|roughly|nearly|almost|~|≈|close to)\s*$"),
]
_YEAR = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")
_QUARTER = re.compile(r"\bQ([1-4])\s*(19[5-9]\d|20[0-4]\d)\b", re.I)
_RANGE = re.compile(r"\bfrom\s+(19[5-9]\d|20[0-4]\d)\s+to\s+(19[5-9]\d|20[0-4]\d)\b", re.I)


@dataclass
class Quantity:
    text: str
    value: Decimal
    unit: str | None
    scale: str | None
    qualifier: str
    precision_decimals: int
    start: int
    end: int


def _unit(u: str | None) -> str | None:
    if not u:
        return None
    u = u.lower()
    if u == "%":
        return "%"
    if u.startswith("percentage point") or u == "pp" or u.startswith("point"):
        return "pp"
    if u in ("×", "x", "times", "-fold", "fold"):
        return "x"
    if u in ("ms",):
        return "ms"
    if u in ("s", "second", "seconds"):
        return "s"
    return u.rstrip("s") if u.endswith("s") and len(u) > 3 else u


def to_decimal(text: str) -> Decimal | None:
    t = text.replace(",", "").replace("−", "-").strip()
    try:
        return Decimal(t)
    except InvalidOperation:
        return None


def qualifier_before(text: str, start: int) -> str:
    before = text[max(0, start - 25):start].lower()
    for name, pat in _QUALIFIERS:
        if re.search(pat, before):
            return name
    return "exact"


def quantities(text: str, skip_years: bool = True) -> list[Quantity]:
    out = []
    for m in NUM_RE.finditer(text):
        num = m.group("num")
        if skip_years and _YEAR.fullmatch(num) and not m.group("unit"):
            continue
        val = to_decimal(num)
        if val is None:
            continue
        if m.group("sign") in ("-", "−"):
            val = -val
        scale = (m.group("scale") or "").lower() or None
        if scale:
            val = val * _SCALE[scale]
        decimals = len(num.split(".")[1]) if "." in num else 0
        end = m.end() if m.group("unit") or scale else m.end("num")
        out.append(Quantity(text=text[m.start():end].strip(), value=val, unit=_unit(m.group("unit")),
                            scale=scale, qualifier=qualifier_before(text, m.start()),
                            precision_decimals=decimals, start=m.start(), end=end))
    return out


def time_spec(text: str) -> TimeSpec | None:
    if m := _RANGE.search(text):
        return TimeSpec(kind="range", start=m.group(1), end=m.group(2), text=m.group(0))
    if m := _QUARTER.search(text):
        q = f"{m.group(2)}-Q{m.group(1)}"
        return TimeSpec(kind="quarter", start=q, end=q, text=m.group(0))
    if m := _YEAR.search(text):
        return TimeSpec(kind="year", start=m.group(1), end=m.group(1), text=m.group(0))
    return None


def cell_number(text: str) -> Decimal | None:
    """Số chính của một ô (bỏ "± sai số", ký hiệu chú thích, %)."""
    t = text.replace("−", "-")
    m = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", t)
    return to_decimal(m.group(0)) if m else None
