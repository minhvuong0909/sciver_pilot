"""Biểu diễn bảng dùng chung cho bảng chuẩn (G) và bảng trích (E).

Một bảng = dict {"grid": list[list[str]], "n_header_rows": int, "source": str}
  - grid đã "trải" rowspan/colspan (ô gộp được lặp lại ở mọi vị trí nó phủ),
    để G và E được so sánh trên cùng một dạng.

Hàm chính:
  html_to_grid(html)        HTML <table> (LaTeXML của arXiv hoặc HTML của Docling) -> grid
  normalize_table(tab)      bộ chuẩn hóa cho bảng khoa học, trả về (bảng mới, log)
  linearize(tab, caption)   bảng -> văn bản đưa vào prompt (dạng "|" giống PoT)
  numeric_tokens(tab)       các số trong bảng (để đo độ khớp số giữa G và E)
"""
import copy
import re

from bs4 import BeautifulSoup

# ----------------------------------------------------------------------------- HTML -> grid

def _cell_text(cell):
    cell = copy.copy(cell)
    # LaTeXML: <math alttext="\pm"> ... <annotation> chứa mã TeX -> bỏ annotation để không bị lặp
    for ann in cell.find_all(["annotation", "annotation-xml"]):
        ann.decompose()
    for br in cell.find_all("br"):
        br.replace_with(" ")
    txt = cell.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", txt).strip()


def html_to_grid(html):
    """Trả về (grid, n_header_rows). Bảng lồng nhau: chỉ lấy bảng ngoài cùng lớn nhất."""
    soup = BeautifulSoup(html, "lxml")
    tables = soup.find_all("table")
    if not tables:
        return [], 0
    table = max(tables, key=lambda t: len(t.find_all(["td", "th"])))
    rows = [tr for tr in table.find_all("tr") if tr.find_parent("table") is table]

    grid, pending = [], {}  # pending[(r, c)] = text cho rowspan
    n_header = 0
    for r, tr in enumerate(rows):
        row, c = [], 0
        cells = [td for td in tr.find_all(["td", "th"]) if td.find_parent("tr") is tr]
        is_header = bool(cells) and (all(td.name == "th" for td in cells)
                                     or tr.find_parent("thead") is not None)
        for td in cells:
            while (r, c) in pending:
                row.append(pending.pop((r, c)))
                c += 1
            txt = _cell_text(td)
            rs = max(int(td.get("rowspan", 1) or 1), 1)
            cs = max(int(td.get("colspan", 1) or 1), 1)
            for dc in range(cs):
                row.append(txt)
                for dr in range(1, rs):
                    pending[(r + dr, c + dc)] = txt
            c += cs
        while (r, c) in pending:
            row.append(pending.pop((r, c)))
            c += 1
        grid.append(row)
        if is_header and r == n_header:
            n_header += 1
    width = max((len(x) for x in grid), default=0)
    grid = [x + [""] * (width - len(x)) for x in grid]
    grid = [x for x in grid if any(v.strip() for v in x)]
    return grid, n_header


NUM_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)*(?:\.\d+)?")


def _is_numeric_cell(s):
    s = s.replace("±", " ").replace("%", "")
    return bool(NUM_RE.fullmatch(s.strip().split(" ")[0])) if s.strip() else False


def guess_header_rows(grid, declared=0):
    """Nếu HTML không khai báo <th>/<thead>: header = các dòng đầu đến trước dòng đầu tiên
    có >= 1 ô số ở các cột sau cột 0. Dùng CÙNG luật cho G và E."""
    if declared:
        return declared
    for i, row in enumerate(grid):
        if any(_is_numeric_cell(v) for v in row[1:]):
            return max(i, 1) if i > 0 else 1
    return 1 if grid else 0


def make_table(html, source):
    grid, nh = html_to_grid(html)
    return {"grid": grid, "n_header_rows": guess_header_rows(grid, nh), "source": source}


# ----------------------------------------------------------------------------- chuẩn hóa

MINUS_CHARS = "−‒–—"  # − ‒ – —
FOOTNOTE_RE = re.compile(r"(?<=[\d%)])\s*[\*†‡§¶]+$")
THOUSANDS_RE = re.compile(r"^[-+]?\d{1,3}(,\d{3})+(\.\d+)?$")
MISSING = {"-", "–", "—", "−", "n/a", "na", "--", "—/—", "/", "×", "✗"}


def normalize_cell(s):
    """Trả về (giá trị mới, [tên luật đã áp dụng])."""
    rules, orig = [], s
    v = re.sub(r"\s+", " ", s).strip()
    if v.lower() in MISSING:
        return "N/A", ["missing->N/A"]
    # dấu trừ unicode đứng trước số
    v2 = re.sub(f"[{MINUS_CHARS}](?=\\d)", "-", v)
    if v2 != v:
        rules.append("unicode_minus"); v = v2
    # ký hiệu chú thích sau số: 85.3* , 12.1†
    v2 = FOOTNOTE_RE.sub("", v)
    if v2 != v:
        rules.append("footnote_mark"); v = v2
    # dấu phân cách hàng nghìn 1,234,567 -> 1234567
    if THOUSANDS_RE.match(v):
        v = v.replace(",", ""); rules.append("thousands_sep")
    # ± : chuẩn khoảng trắng "85.3 ± 0.2" / "85.3±0.2" / "85.3 +- 0.2" -> "85.3 ± 0.2"
    v2 = re.sub(r"\s*(?:±|\+/-|\+-)\s*", " ± ", v)
    if v2 != v:
        rules.append("plus_minus"); v = v2
    # % dính/không dính: "85.3 %" -> "85.3%"
    v2 = re.sub(r"(\d)\s+%", r"\1%", v)
    if v2 != v:
        rules.append("percent_space"); v = v2
    # mũi tên chiều tốt hơn trong tiêu đề
    v2 = v.replace("↑", " (higher is better)").replace("↓", " (lower is better)")
    if v2 != v:
        rules.append("arrow"); v = re.sub(r"\s+", " ", v2).strip()
    return v, rules


def flatten_header(grid, n_header):
    """Gộp các dòng tiêu đề nhiều tầng thành 1 dòng: 'Cấp1 / Cấp2'."""
    if n_header <= 1:
        return grid, n_header
    width = len(grid[0])
    flat = []
    for j in range(width):
        parts = []
        for i in range(n_header):
            p = grid[i][j].strip()
            if p and (not parts or parts[-1] != p):
                parts.append(p)
        flat.append(" / ".join(parts))
    return [flat] + grid[n_header:], 1


def normalize_table(tab):
    """Áp dụng giống hệt cho G và E. Trả về (bảng chuẩn hóa, log)."""
    grid = [list(r) for r in tab["grid"]]
    log = []
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            new, rules = normalize_cell(cell)
            if rules:
                log.append({"row": i, "col": j, "before": cell, "after": new, "rules": rules})
                row[j] = new
    grid, nh = flatten_header(grid, tab.get("n_header_rows", 1))
    if nh != tab.get("n_header_rows", 1):
        log.append({"row": -1, "col": -1, "before": f"{tab.get('n_header_rows')} header rows",
                    "after": "1 header row", "rules": ["flatten_header"]})
    return {"grid": grid, "n_header_rows": nh, "source": tab["source"] + "+norm"}, log


# ----------------------------------------------------------------------------- linearize

def linearize(tab, caption="", name="Table"):
    """Định dạng giống PoT: các ô nối bằng ' | ', mỗi dòng xuống dòng."""
    lines = [f"{name}: {caption}".strip()]
    if not tab or not tab.get("grid"):
        lines.append("[EMPTY TABLE - extraction failed]")
        return "\n".join(lines)
    for r in tab["grid"]:
        lines.append(" | ".join(c if c else " " for c in r))
    return "\n".join(lines)


def numeric_tokens(tab):
    out = []
    for r in tab.get("grid", []):
        for c in r:
            for m in NUM_RE.findall(c.replace(",", "")):
                try:
                    out.append(round(float(m), 6))
                except ValueError:
                    pass
    return out
