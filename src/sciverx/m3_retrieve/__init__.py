"""M3 — Hybrid Search."""

from __future__ import annotations

from sciverx.schemas import EvidenceUnit

_GROUP = {"paragraph": "section", "table_full": "table", "table_rows": "table",
          "chart_caption": "chart"}


def parent_key(unit: EvidenceUnit) -> str:
    """Khóa cha duy nhất trong một bài: section "1" và bảng "1" không được trùng."""
    return f"{_GROUP[unit.kind]}:{unit.parent_id}"
