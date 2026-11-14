"""Ngân sách đầu vào cho verifier, chung cho BM25, dense và hybrid (SPEC 7-M3).

Lấy cha theo thứ hạng cho tới khi chạm max_parents hoặc max_evidence_tokens
(đếm bằng tokenizer của Qwen). Một cha vượt ngân sách riêng nó → chỉ đưa các
phần có điểm cao nhất (bảng: nhóm hàng kèm caption và tiêu đề), ghi truncated.
Ảnh chỉ dùng cho biểu đồ khi allow_mixed, tối đa max_images.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from sciverx.m3_retrieve.retrieve import PaperIndex, units_of_parent
from sciverx.schemas import RetrievalResult, TableRecord
from sciverx.tables.linearize import linearize_table

Counter_ = Callable[[str], int]


@dataclass
class SelectedParent:
    parent_id: str
    kind: str  # table | section | chart
    rank: int
    text: str
    truncated: bool = False
    rows: list[int] | None = None   # hàng bảng được đưa vào khi truncated
    image_path: str | None = None


@dataclass
class Selection:
    parents: list[SelectedParent] = field(default_factory=list)
    budget: dict = field(default_factory=dict)


def _section_text(index: PaperIndex, parent: str) -> str:
    return "\n".join(u.text for _, u in units_of_parent(index, parent))


def _table_rows_by_priority(index: PaperIndex, parent: str, best_unit_id: str) -> list[list[int]]:
    """Nhóm hàng của bảng: nhóm của đơn vị tốt nhất trước, rồi theo thứ tự hàng."""
    groups = []
    for _, u in units_of_parent(index, parent):
        if u.row_range is None:
            continue
        rows = list(range(u.row_range[0], u.row_range[1] + 1))
        if u.unit_id == best_unit_id:
            groups.insert(0, rows)
        elif rows not in groups:
            groups.append(rows)
    return groups


def select_evidence(result: RetrievalResult, index: PaperIndex, tables: dict[str, TableRecord],
                    budget_cfg: dict, count: Counter_, chart_images: dict[str, str] | None = None,
                    allow_mixed: bool = False) -> Selection:
    max_tokens = budget_cfg.get("max_evidence_tokens", 6000)
    max_parents = budget_cfg.get("max_parents", 3)
    max_images = budget_cfg.get("max_images", 2)
    sel = Selection()
    used = images = 0
    dropped: list[str] = []
    truncated_any = False
    for cand in result.candidates:
        if len(sel.parents) >= max_parents:
            break
        group, pid = cand.parent_id.split(":", 1)
        sp = SelectedParent(cand.parent_id, group, cand.final_rank, "")
        if group == "table":
            rec = tables[pid]
            sp.text = linearize_table(rec)
            if count(sp.text) > max_tokens:   # một bảng vượt ngân sách riêng nó
                chosen: list[int] = []
                for rows in _table_rows_by_priority(index, cand.parent_id, cand.best_unit_id):
                    trial = sorted(set(chosen + rows))
                    if count(linearize_table(rec, rows=trial)) + used > max_tokens:
                        break
                    chosen = trial
                sp.rows, sp.truncated = chosen, True
                sp.text = linearize_table(rec, rows=chosen) if chosen else rec.caption
                truncated_any = True
        elif group == "section":
            sp.text = _section_text(index, cand.parent_id)
            if count(sp.text) > max_tokens:
                best = next(u.text for _, u in units_of_parent(index, cand.parent_id)
                            if u.unit_id == cand.best_unit_id)
                sp.text, sp.truncated = best, True
                truncated_any = True
        else:  # chart
            if not allow_mixed:
                continue
            sp.text = _section_text(index, cand.parent_id)
            img = (chart_images or {}).get(pid)
            if img and images < max_images:
                sp.image_path = img
                images += 1
        n = count(sp.text)
        if used + n > max_tokens:
            dropped.append(cand.parent_id)
            break
        used += n
        sel.parents.append(sp)
    # cha còn trong top max_parents nhưng không được đưa vào vì hết ngân sách
    taken = {p.parent_id for p in sel.parents}
    for cand in result.candidates[:max_parents]:
        if cand.parent_id not in taken and cand.parent_id not in dropped:
            if not cand.parent_id.startswith("chart:") or allow_mixed:
                dropped.append(cand.parent_id)
    sel.budget = {"tokens_used": used, "images_used": images, "dropped_parents": dropped,
                  "truncated": truncated_any, "max_evidence_tokens": max_tokens,
                  "max_parents": max_parents, "max_images": max_images}
    return sel
