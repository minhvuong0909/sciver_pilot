"""Ánh xạ ô E↔G cho cùng (paper, table) — SPEC M1.5.

Bước 1 (`header_key`): ô thân bảng ghép theo (tiêu đề hàng, tiêu đề cột) đã
chuẩn hóa. Một ô E phủ nhiều vị trí (ô gộp) có thể ghép với nhiều ô G.
Bước 2 (`content`): ô còn lại (kể cả ô tiêu đề) ghép theo độ giống nội dung,
kèm ràng buộc thứ tự hàng: không cho cặp mới "bắt chéo" cặp đã có.
Ô không ghép được nằm trong unmatched_e / unmatched_g.
"""

from __future__ import annotations

import re

from sciverx.schemas import CellMapping, SourceCell, TableRecord
from sciverx.tables.grid import Grid, build_grid, merged_column_headers
from sciverx.tables.teds import normalized_levenshtein

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", text.strip().lower())


def _text(c: SourceCell) -> str:
    return _norm(c.text_norm if c.text_norm is not None else c.text_raw)


def _keys(grid: Grid, cell: SourceCell, col_headers: list[str], n_header: int) -> set[tuple[str, str]]:
    """Các khóa (tiêu đề hàng, tiêu đề cột) mà ô phủ. Ô tiêu đề và cột tiêu đề hàng không có khóa."""
    if cell.is_header or cell.row < n_header:
        return set()
    keys = set()
    for r in range(cell.row, cell.row + cell.rowspan):
        rh_cell = grid.cell_at(r, 0)
        rh = _text(rh_cell) if rh_cell is not None else ""
        for c in range(cell.col, cell.col + cell.colspan):
            if c == 0:
                continue
            keys.add((rh, _norm(col_headers[c])))
    return keys


def map_cells(e: TableRecord, g: TableRecord, min_content_sim: float = 0.8) -> CellMapping:
    ge, gg = build_grid(e.cells), build_grid(g.cells)
    he = merged_column_headers(ge, e.n_header_rows)
    hg = merged_column_headers(gg, g.n_header_rows)

    g_by_key: dict[tuple[str, str], list[str]] = {}
    for c in g.cells:
        for k in _keys(gg, c, hg, g.n_header_rows):
            g_by_key.setdefault(k, []).append(c.cell_id)
    # Khóa trùng trong G (hai hàng cùng tiêu đề) là mơ hồ → không dùng ở bước 1.
    unique_g = {k: v[0] for k, v in g_by_key.items() if len(v) == 1}

    pairs: list[dict] = []
    used_g: set[str] = set()
    used_e: set[str] = set()
    for c in e.cells:
        targets = sorted({unique_g[k] for k in _keys(ge, c, he, e.n_header_rows) if k in unique_g})
        if targets:
            pairs.append({"e_cell_id": c.cell_id, "g_cell_ids": targets, "method": "header_key",
                          "score": 1.0})
            used_e.add(c.cell_id)
            used_g.update(targets)

    g_cells = {c.cell_id: c for c in g.cells}
    e_cells = {c.cell_id: c for c in e.cells}
    anchors = [(e_cells[p["e_cell_id"]].row, g_cells[gid].row)
               for p in pairs for gid in p["g_cell_ids"]]

    def order_ok(er: int, gr: int) -> bool:
        return all((ar < er and ag <= gr) or (ar > er and ag >= gr) or ar == er
                   for ar, ag in anchors)

    for c in sorted(e.cells, key=lambda x: (x.row, x.col)):
        if c.cell_id in used_e:
            continue
        best = None
        for gc in g.cells:
            if gc.cell_id in used_g or gc.is_header != c.is_header:
                continue
            if not order_ok(c.row, gc.row):
                continue
            sim = 1.0 - normalized_levenshtein(_text(c), _text(gc))
            if sim < min_content_sim:
                continue
            dist = abs(gc.row - c.row) + abs(gc.col - c.col)
            cand = (sim, -dist, gc.cell_id)
            if best is None or cand > best:
                best = cand
        if best is not None:
            gid = best[2]
            pairs.append({"e_cell_id": c.cell_id, "g_cell_ids": [gid], "method": "content",
                          "score": round(best[0], 4)})
            used_e.add(c.cell_id)
            used_g.add(gid)
            anchors.append((c.row, g_cells[gid].row))

    pairs.sort(key=lambda p: (e_cells[p["e_cell_id"]].row, e_cells[p["e_cell_id"]].col))
    return CellMapping(
        paper_id=g.paper_id, table_id=g.table_id, e_condition=e.condition, g_condition=g.condition,
        pairs=pairs,
        unmatched_e=[c.cell_id for c in e.cells if c.cell_id not in used_e],
        unmatched_g=[c.cell_id for c in g.cells if c.cell_id not in used_g])


def map_e_cells_to_g(mapping: CellMapping, e_cell_ids: list[str]) -> tuple[set[str], list[str]]:
    """Đưa tập ô chọn trên E về G (dùng chấm M4 trên E1, SPEC mục 10).
    Trả (tập ô G, danh sách ô E không ánh xạ được)."""
    lookup = {p["e_cell_id"]: p["g_cell_ids"] for p in mapping.pairs}
    out: set[str] = set()
    missing = []
    for cid in e_cell_ids:
        if cid in lookup:
            out.update(lookup[cid])
        else:
            missing.append(cid)
    return out, missing
