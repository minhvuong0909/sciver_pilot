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


def _span(tag: Tag, attr: str) -> int:
    try:
        return max(1, int(str(tag.get(attr, "1")).strip() or 1))
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
    for block in tag.find_all(["p", "div", "li"]):
        block.insert_after(" ")
    return _WS.sub(" ", tag.get_text("")).strip()


def split_footnote_marks(text: str) -> tuple[str, list[str]]:
    """Tách ký hiệu chú thích ở cuối ô: "93.2*†" → ("93.2", ["*", "†"])."""
    m = _TRAILING_MARKS.search(text)
    if not m or not m.group(1) or m.start() == 0:
        return text, []
    return text[: m.start()].rstrip(), list(m.group(1))


def _rows(table: Tag) -> list[tuple[Tag, bool]]:
    """Các hàng <tr> theo thứ tự, kèm cờ nằm trong <thead>. Bỏ bảng lồng."""
    out = []
    for tr in table.find_all("tr"):
        if tr.find_parent("table") is not table:
            continue
        out.append((tr, tr.find_parent("thead") is not None))
    return out


def parse_html_table(html: str | Tag) -> tuple[list[SourceCell], int]:
    """Trả (cells, n_header_rows). Vị trí cột tính theo lưới có span (ô bị
    rowspan từ hàng trên phủ thì nhảy qua)."""
    if isinstance(html, Tag):
        table = html if html.name == "table" else html.find("table")
    else:
        table = BeautifulSoup(html, "lxml").find("table")
    if table is None:
        raise ValueError("Không tìm thấy thẻ <table>")

    occupied: set[tuple[int, int]] = set()
    cells: list[SourceCell] = []
    thead_rows = 0
    row_all_header: list[bool] = []
    for r, (tr, in_thead) in enumerate(_rows(table)):
        thead_rows += in_thead
        col = 0
        all_header = True
        any_cell = False
        for td in tr.find_all(["td", "th"], recursive=False):
            while (r, col) in occupied:
                col += 1
            rs, cs = _span(td, "rowspan"), _span(td, "colspan")
            for dr in range(rs):
                for dc in range(cs):
                    occupied.add((r + dr, col + dc))
            raw = cell_text(td)
            _, marks = split_footnote_marks(raw)
            is_header = td.name == "th" or in_thead
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
    return cells, n_header
