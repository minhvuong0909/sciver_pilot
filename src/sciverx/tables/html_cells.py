"""HTML bảng → danh sách SourceCell giữ nguyên rowspan/colspan — SPEC M1.2 bước 3.

- `is_header`: ô `<th>` hoặc nằm trong `<thead>`.
- `bold`: có `<b>`/`<strong>` hoặc class `ltx_font_bold` ở ô hay phần tử con.
- Ký hiệu chú thích `* † ‡ § ¶` ở cuối ô (hoặc trong `<sup>`) → `footnote_marks`.
- Bỏ `<annotation>`/`<annotation-xml>` của MathML khi lấy chữ.
"""

from __future__ import annotations

import copy
import re

from bs4 import BeautifulSoup, Tag

from sciverx.schemas import SourceCell

FOOTNOTE_CHARS = "*†‡§¶"
_TRAILING_MARKS = re.compile(rf"\s*([{re.escape(FOOTNOTE_CHARS)}]+)\s*$")
_WS = re.compile(r"\s+")


_REF_ARTIFACT = re.compile(r"\s*###\w+###")  # dấu vết tiền xử lý của SciVer: "[7 ###reference_b7###]"


def _span(tag: Tag, attr: str) -> int:
    """Đọc thuộc tính rowspan/colspan, hoặc class ltx_rowspan_N / ltx_colspan_N (dạng <span>)."""
    value = tag.get(attr)
    if value is None:
        for cls in tag.get("class") or []:
            if cls.startswith(f"ltx_{attr}_"):
                value = cls.rsplit("_", 1)[1]
                break
    try:
        return max(1, int(str(value or "1").strip() or 1))
    except ValueError:
        return 1


def _is_bold(tag: Tag) -> bool:
    if tag.find(["b", "strong"]) is not None:
        return True
    for el in [tag, *tag.find_all(True)]:
        if "ltx_font_bold" in (el.get("class") or []):
            return True
        style = str(el.get("style") or "").replace(" ", "").lower()
        if "font-weight:bold" in style:
            return True
    return False


def cell_text(tag: Tag) -> str:
    """Chữ hiển thị của ô, bỏ phần chú thích MathML, gộp khoảng trắng."""
    tag = copy.copy(tag)  # bs4 sao chép sâu; không sửa cây gốc
    for ann in tag.find_all(["annotation", "annotation-xml"]):
        ann.decompose()
    # Nối phần tử inline không chèn khoảng trắng ("92.4<sup>*</sup>" → "92.4*");
    # chỉ <br> và khối (p, div) mới tách từ.
    for br in tag.find_all("br"):
        br.replace_with(" ")
    for block in tag.find_all(lambda e: e.name in ("p", "div", "li", "tr", "td", "th")
                              or {"ltx_tr", "ltx_td", "ltx_p"} & set(e.get("class") or [])):
        block.insert_after(" ")
    return _WS.sub(" ", _REF_ARTIFACT.sub("", tag.get_text(""))).strip()


def split_footnote_marks(text: str) -> tuple[str, list[str]]:
    """Tách ký hiệu chú thích ở cuối ô: "93.2*†" → ("93.2", ["*", "†"])."""
    m = _TRAILING_MARKS.search(text)
    if not m or not m.group(1) or m.start() == 0:
        return text, []
    return text[: m.start()].rstrip(), list(m.group(1))


# LaTeXML viết bảng theo hai cách: thẻ <table>/<tr>/<td>, hoặc <span> với class
# ltx_tabular / ltx_tr / ltx_td (ltx_th cho ô tiêu đề, ltx_thead cho phần đầu).
def _classes(el: Tag) -> list[str]:
    return el.get("class") or []


def _is_table(el: Tag) -> bool:
    return el.name == "table" or "ltx_tabular" in _classes(el)


def _is_row(el: Tag) -> bool:
    return el.name == "tr" or "ltx_tr" in _classes(el)


def _is_cell(el: Tag) -> bool:
    return el.name in ("td", "th") or "ltx_td" in _classes(el)


def _owner_table(el: Tag) -> Tag | None:
    return el.find_parent(_is_table)


def find_tables(root: Tag) -> list[Tag]:
    """Các bảng ngoài cùng (không nằm trong ô của bảng khác), theo thứ tự xuất hiện."""
    found = [root] if _is_table(root) else []
    found += root.find_all(_is_table)
    return [t for t in found if _owner_table(t) is None or _owner_table(t) not in found]


def _rows(table: Tag) -> list[tuple[Tag, bool]]:
    """Các hàng của `table` (bỏ hàng của bảng lồng), kèm cờ nằm trong phần đầu."""
    out = []
    for tr in table.find_all(_is_row):
        if _owner_table(tr) is not table:
            continue
        in_head = tr.find_parent("thead") is not None or tr.find_parent(
            lambda e: "ltx_thead" in _classes(e)) is not None
        out.append((tr, in_head))
    return out


def _row_cells(tr: Tag) -> list[Tag]:
    return [td for td in tr.find_all(_is_cell) if td.find_parent(_is_row) is tr]


def _parse_one(table: Tag, row_offset: int) -> tuple[list[SourceCell], int, int]:
    """Trả (cells, n_header_rows, n_rows) của một bảng; hàng đánh số từ row_offset."""
    occupied: set[tuple[int, int]] = set()
    cells: list[SourceCell] = []
    thead_rows = 0
    row_all_header: list[bool] = []
    rows = _rows(table)
    for i, (tr, in_head) in enumerate(rows):
        r = row_offset + i
        thead_rows += in_head
        col = 0
        all_header = True
        any_cell = False
        for td in _row_cells(tr):
            while (r, col) in occupied:
                col += 1
            rs, cs = _span(td, "rowspan"), _span(td, "colspan")
            for dr in range(rs):
                for dc in range(cs):
                    occupied.add((r + dr, col + dc))
            raw = cell_text(td)
            _, marks = split_footnote_marks(raw)
            is_header = td.name == "th" or "ltx_th" in _classes(td) or in_head
            all_header &= is_header
            any_cell = True
            cells.append(SourceCell(cell_id=f"r{r}c{col}", row=r, col=col, rowspan=rs, colspan=cs,
                                    is_header=is_header, text_raw=raw, bold=_is_bold(td),
                                    footnote_marks=marks))
            col += cs
        row_all_header.append(all_header and any_cell)
    if thead_rows:
        n_header = thead_rows
    else:
        n_header = 0
        for flag in row_all_header:
            if not flag:
                break
            n_header += 1
    # rowspan có thể kéo quá hàng cuối; số hàng thật của phần này tính theo ô
    n_rows = max([len(rows)] + [c.row - row_offset + c.rowspan for c in cells])
    return cells, n_header, n_rows


_NUMERIC = re.compile(r"^[-+−–]?\d[\d,]*(\.\d+)?\s*%?")


def guess_header_rows(cells: list[SourceCell]) -> int:
    """Khi HTML không khai báo hàng tiêu đề (LaTeXML chỉ đánh ltx_th_row cho cột
    đầu): tiêu đề = các hàng đầu tới trước hàng đầu tiên có ô số ở cột ≥ 1.
    Dùng CÙNG luật cho bảng chuẩn và bảng trích (như bản pilot)."""
    rows: dict[int, list[SourceCell]] = {}
    for c in cells:
        rows.setdefault(c.row, []).append(c)
    if not rows:
        return 0
    for i, r in enumerate(sorted(rows)):
        if any(c.col >= 1 and _NUMERIC.match(c.text_raw.strip()) for c in rows[r]):
            return i
    return 1 if len(rows) > 1 else 0


def parse_table_html(html: str | Tag) -> tuple[list[SourceCell], int, int]:
    """Trả (cells, n_header_rows, n_parts). Nhiều bảng con ngoài cùng (vd hai
    panel đặt cạnh nhau) được xếp chồng theo chiều dọc; n_header_rows lấy theo
    phần đầu tiên, ô tiêu đề của các phần sau vẫn giữ is_header."""
    root = html if isinstance(html, Tag) else BeautifulSoup(html, "lxml")
    tables = find_tables(root)
    if not tables:
        raise ValueError("Không tìm thấy bảng (<table> hoặc ltx_tabular)")
    all_cells: list[SourceCell] = []
    n_header = 0
    offset = 0
    for k, t in enumerate(tables):
        cells, nh, n_rows = _parse_one(t, offset)
        if k == 0:
            n_header = nh
        all_cells += cells
        offset += n_rows
        if k == 0:
            first_rows = n_rows
    if n_header == 0:
        n_header = guess_header_rows([c for c in all_cells if c.row < first_rows])
    return all_cells, n_header, len(tables)


def parse_html_table(html: str | Tag) -> tuple[list[SourceCell], int]:
    """Trả (cells, n_header_rows). Vị trí cột tính theo lưới có span (ô bị
    rowspan từ hàng trên phủ thì nhảy qua)."""
    cells, n_header, _ = parse_table_html(html)
    return cells, n_header
