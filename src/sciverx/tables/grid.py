"""Suy lưới từ danh sách ô có span. `cells` là nguồn sự thật; lưới chỉ là dẫn xuất.

Ô gộp được lặp lại ở mọi vị trí nó phủ (SPEC M1.4, dòng "Ô gộp"), và lưới lưu
ánh xạ vị trí → cell_id để truy ngược về ô gốc.
"""

from __future__ import annotations

from dataclasses import dataclass

from sciverx.schemas import SourceCell


@dataclass
class Grid:
    n_rows: int
    n_cols: int
    ids: list[list[str | None]]  # ids[r][c] = cell_id phủ vị trí (r, c), None nếu trống
    by_id: dict[str, SourceCell]

    def cell_at(self, r: int, c: int) -> SourceCell | None:
        cid = self.ids[r][c]
        return self.by_id[cid] if cid is not None else None

    def text_rows(self, normalized: bool = True) -> list[list[str]]:
        """Ma trận chữ, ô gộp lặp lại; vị trí trống là chuỗi rỗng."""
        out = []
        for r in range(self.n_rows):
            row = []
            for c in range(self.n_cols):
                cell = self.cell_at(r, c)
                if cell is None:
                    row.append("")
                elif normalized and cell.text_norm is not None:
                    row.append(cell.text_norm)
                else:
                    row.append(cell.text_raw)
            out.append(row)
        return out


def build_grid(cells: list[SourceCell]) -> Grid:
    if not cells:
        return Grid(0, 0, [], {})
    n_rows = max(c.row + c.rowspan for c in cells)
    n_cols = max(c.col + c.colspan for c in cells)
    ids: list[list[str | None]] = [[None] * n_cols for _ in range(n_rows)]
    by_id: dict[str, SourceCell] = {}
    for cell in cells:
        if cell.cell_id in by_id:
            raise ValueError(f"cell_id trùng: {cell.cell_id}")
        by_id[cell.cell_id] = cell
        for r in range(cell.row, cell.row + cell.rowspan):
            for c in range(cell.col, cell.col + cell.colspan):
                if ids[r][c] is not None:
                    raise ValueError(f"Ô {cell.cell_id} chồng lên {ids[r][c]} tại ({r},{c})")
                ids[r][c] = cell.cell_id
    return Grid(n_rows, n_cols, ids, by_id)


def merged_column_headers(grid: Grid, n_header_rows: int, normalized: bool = True,
                          sep: str = " / ") -> list[str]:
    """Tiêu đề cột gộp nhiều tầng: "Cấp 1 / Cấp 2". Bỏ tầng trùng do colspan lặp lại."""
    rows = grid.text_rows(normalized)[:n_header_rows]
    headers = []
    for c in range(grid.n_cols):
        parts: list[str] = []
        for row in rows:
            text = row[c].strip()
            if text and (not parts or parts[-1] != text):
                parts.append(text)
        headers.append(sep.join(parts))
    return headers


def row_header(grid: Grid, r: int, normalized: bool = True) -> str:
    """Tiêu đề hàng = chữ ở cột đầu tiên của hàng (giả định thường gặp ở bảng khoa học)."""
    if grid.n_cols == 0:
        return ""
    return grid.text_rows(normalized)[r][0].strip()
