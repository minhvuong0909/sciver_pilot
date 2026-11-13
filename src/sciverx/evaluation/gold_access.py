"""Module DUY NHẤT đọc GoldInfo ở runtime đánh giá (SPEC 5.5).

Chuyển GoldInfo thành tập bằng chứng đúng ở mức cha, cùng khóa với M3:
bảng "table:{id}", biểu đồ "chart:{id}", section "section:{id}".
"""

from __future__ import annotations

from pathlib import Path

from sciverx.schemas import GoldInfo
from sciverx.utils.io import read_jsonl


def load_gold(path: str | Path) -> dict[str, GoldInfo]:
    return {g.claim_id: g for g in read_jsonl(path, GoldInfo)}


def gold_visual_parents(gold: GoldInfo) -> list[str]:
    out = []
    for v in gold.visual_items:
        key = f"{'table' if v['item_type'] == 'table' else 'chart'}:{v['item_id']}"
        if key not in out:
            out.append(key)
    return out


def gold_section_parents(gold: GoldInfo) -> list[str]:
    out = []
    for s in gold.section_ids:
        key = f"section:{s}"
        if key not in out:
            out.append(key)
    return out


def section_matches(retrieved: str, gold: str) -> bool:
    """Section truy xuất khớp section chuẩn nếu trùng hoặc là tiểu mục của nó
    (độ chi tiết theo D5: nhãn ghi tới mục/tiểu mục, đơn vị của kho là section lá)."""
    if not (retrieved.startswith("section:") and gold.startswith("section:")):
        return False
    r, g = retrieved.split(":", 1)[1], gold.split(":", 1)[1]
    return r == g or r.startswith(g + ".")
