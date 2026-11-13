"""Bảng → văn bản cho kho tìm kiếm, M4 và verifier — SPEC M1.4, M1.6, 8.2.

Dạng: một dòng tiêu đề đã gộp nhiều tầng, rồi mỗi hàng thân một dòng, ô
ngăn bằng " | ". Ô gộp lặp lại ở mọi vị trí nó phủ. Chữ đậm mặc định KHÔNG
đưa vào (include_bold=False) để G và E công bằng. Chú thích bảng in ở cuối.
"""

from __future__ import annotations

from sciverx.schemas import TableRecord
from sciverx.tables.grid import Grid, build_grid, merged_column_headers

SEP = " | "
TEXT_LABEL = "Table content (as text):"


def _cell_text(grid: Grid, r: int, c: int, normalized: bool, include_bold: bool,
               with_ids: bool) -> str:
    cell = grid.cell_at(r, c)
    if cell is None:
        return ""
    text = cell.text_norm if (normalized and cell.text_norm is not None) else cell.text_raw
    if include_bold and cell.bold and text:
        text = f"**{text}**"
    if with_ids:
        text = f"[{cell.cell_id}] {text}"
    return text


def header_line(record: TableRecord, normalized: bool = True) -> str:
    grid = build_grid(record.cells)
    return SEP.join(merged_column_headers(grid, record.n_header_rows, normalized))


def body_lines(record: TableRecord, rows: range | list[int] | None = None, normalized: bool = True,
               include_bold: bool = False, with_ids: bool = False) -> list[str]:
    """Các dòng thân bảng. `rows` là chỉ số hàng trong lưới (mặc định mọi hàng thân)."""
    grid = build_grid(record.cells)
    if rows is None:
        rows = range(record.n_header_rows, grid.n_rows)
    return [SEP.join(_cell_text(grid, r, c, normalized, include_bold, with_ids)
                     for c in range(grid.n_cols)) for r in rows]


def footnote_lines(record: TableRecord) -> list[str]:
    return [f"[{mark}] {text}" for mark, text in sorted(record.footnotes.items())]


def linearize_table(record: TableRecord, rows: range | list[int] | None = None,
                    normalized: bool = True, include_bold: bool = False, with_ids: bool = False,
                    with_caption: bool = True) -> str:
    parts: list[str] = []
    if with_caption and record.caption:
        parts.append(record.caption.strip())
    head = header_line(record, normalized)
    if record.n_header_rows and head.strip(" |"):
        parts.append(head)
    parts.extend(body_lines(record, rows, normalized, include_bold, with_ids))
    notes = footnote_lines(record)
    if notes:
        parts.append("Notes: " + " ".join(notes))
    return "\n".join(parts)


def verifier_table_block(record: TableRecord, include_bold: bool = False) -> str:
    """Khối bảng văn bản cho verifier (SPEC 8.2): nhãn cố định + bảng + chú thích."""
    return f"{TEXT_LABEL}\n{linearize_table(record, include_bold=include_bold)}"
