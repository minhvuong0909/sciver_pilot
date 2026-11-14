"""Tập so sánh ghép cặp — SPEC 7-M1 "Chính sách tập so sánh ghép cặp".

- PAIRED_ORACLE: claim mà MỌI bảng bằng chứng có G (status ok) và có E (bất kể
  trạng thái trích). RQ1 so G↔E chỉ trên tập này.
- PAIRED_RETR: claim mà MỌI bảng của bài có G ok. RQ2/RQ3 so G1↔E1 chỉ trên tập này.
Module đánh giá được đọc GoldInfo (vị trí bằng chứng) để dựng tập này.
"""

from __future__ import annotations

from sciverx.schemas import GoldInfo, TableRecord


def paired_retr(golds: dict[str, GoldInfo], paper_of: dict[str, str], g_tables: list[TableRecord]) -> list[str]:
    bad_papers = {t.paper_id for t in g_tables if t.status != "ok"}
    papers_with_tables = {t.paper_id for t in g_tables}
    return sorted(c for c in golds if paper_of[c] in papers_with_tables and paper_of[c] not in bad_papers)


def paired_oracle(golds: dict[str, GoldInfo], paper_of: dict[str, str], g_tables: list[TableRecord],
                  e_tables: list[TableRecord]) -> list[str]:
    g_ok = {(t.paper_id, t.table_id) for t in g_tables if t.status == "ok"}
    e_any = {(t.paper_id, t.table_id) for t in e_tables}
    out = []
    for cid, g in golds.items():
        keys = [(paper_of[cid], v["item_id"]) for v in g.visual_items if v["item_type"] == "table"]
        if keys and all(k in g_ok and k in e_any for k in keys):
            out.append(cid)
    return sorted(out)
