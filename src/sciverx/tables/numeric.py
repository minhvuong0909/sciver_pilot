"""Trích số trong bảng và precision/recall theo multiset — SPEC M1.5."""

from __future__ import annotations

import re
from collections import Counter
from decimal import Decimal, InvalidOperation

from sciverx.schemas import TableRecord

# Số có dấu, phần nghìn đã bỏ ở bước chuẩn hóa; giữ dạng thập phân.
# "_" và "^" (ký hiệu chỉ số "78.9_{1.2}", "10^{-3}") không coi là dính chữ.
_NUM = re.compile(r"(?<![A-Za-z0-9.])[-+]?\d+(?:\.\d+)?(?![A-Za-z0-9]|\.\d)")


def canonical_number(text: str) -> str | None:
    try:
        d = Decimal(text)
    except InvalidOperation:
        return None
    # 93.20 và 93.2 coi là cùng số; giữ dấu âm.
    return format(d.normalize(), "f") if d != 0 else "0"


def numbers_in_text(text: str) -> list[str]:
    out = []
    for m in _NUM.finditer(text):
        val = canonical_number(m.group(0))
        if val is not None:
            out.append(val)
    return out


def table_numbers(record: TableRecord, use_norm: bool = True) -> Counter:
    bag: Counter = Counter()
    for c in record.cells:
        text = c.text_norm if (use_norm and c.text_norm is not None) else c.text_raw
        bag.update(numbers_in_text(text))
    return bag


def numeric_precision_recall(pred: TableRecord, gold: TableRecord) -> tuple[float, float]:
    p, g = table_numbers(pred), table_numbers(gold)
    hit = sum((p & g).values())
    prec = hit / sum(p.values()) if p else (1.0 if not g else 0.0)
    rec = hit / sum(g.values()) if g else 1.0
    return prec, rec
