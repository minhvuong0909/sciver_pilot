"""E0 từ PDF arXiv đúng phiên bản bài (quyết định của nhóm 2026-11-13: dataset chỉ có
ảnh cho một phần bảng, nên chạy Docling trên PDF để mọi bảng đều có bảng trích).

1. Tải https://arxiv.org/pdf/{paperid} (paperid đã có vN), cache; nghỉ ≥ 3 giây giữa
   các request; User-Agent mô tả mục đích nghiên cứu (SPEC M1.2).
2. Tìm trang chứa caption "Table N" (hoặc "TABLE IV") bằng chữ của PDF, chỉ cho
   Docling xử lý các trang đó để chạy nhanh trên CPU.
3. Ghép bảng Docling với table_id theo số trong caption; caption thiếu thì ghép theo
   trang khi chỉ còn đúng một bảng và một ID chưa ghép trên trang đó.
"""

from __future__ import annotations

import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

USER_AGENT = "sciverx-research/0.1 (academic table-verification study; polite crawler)"
_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50}
# Nhãn in trên caption: "3", "3.1", "C.1", "A2", "S4" hoặc số La Mã "IV".
_LABEL = r"([A-Z]\.?\d+(?:\.\d+)?|\d+(?:\.\d+)?|[IVXL]+)"
_CAPTION = re.compile(rf"\b(?:Table|TABLE)\s+{_LABEL}\s*[:.|]")
_CAPTION_START = re.compile(rf"^\s*(?:Table|TABLE)\s+{_LABEL}(?=[\s:.|]|$)")


def roman_to_int(s: str) -> int | None:
    if s.isdigit():
        return int(s)
    total, prev = 0, 0
    for ch in reversed(s.upper()):
        v = _ROMAN.get(ch)
        if v is None:
            return None
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    return total


def caption_number(text: str) -> str | None:
    """Nhãn bảng trong caption, chuẩn hóa: số La Mã → số; "C.1" giữ nguyên."""
    m = _CAPTION_START.search(text or "") or _CAPTION.search(text or "")
    if not m:
        return None
    label = m.group(1)
    if re.fullmatch(r"[IVXL]+", label):
        n = roman_to_int(label)
        return str(n) if n else None
    return label


def table_labels(tables: list[dict]) -> dict[str, str]:
    """table_id của dataset → nhãn in trên caption ("7" → "C.1"); không đọc được thì dùng chính ID."""
    return {t["table_id"]: caption_number(t.get("caption", "")) or t["table_id"] for t in tables}


# ------------------------------------------------------------------ tải PDF

_last_request = [0.0]


def download_pdf(paper_id: str, dest_dir: Path, sleep_s: float = 3.0) -> Path | None:  # pragma: no cover - mạng
    dest = dest_dir / f"{paper_id}.pdf"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    wait = sleep_s - (time.time() - _last_request[0])
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(f"https://arxiv.org/pdf/{paper_id}", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            data = r.read()
    except Exception:
        return None
    finally:
        _last_request[0] = time.time()
    if not data.startswith(b"%PDF"):
        return None
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return dest


# ------------------------------------------------------------------ trang có bảng

def caption_pages(page_texts: list[str], labels: list[str]) -> dict[str, list[int]]:
    """nhãn bảng → các trang (đánh số từ 1) có dòng caption của bảng đó."""
    wanted = set(labels)
    out: dict[str, list[int]] = {}
    for p, text in enumerate(page_texts, 1):
        for line in text.splitlines():
            n = caption_number(line) if _CAPTION_START.search(line) else None
            if n in wanted and p not in out.setdefault(n, []):
                out[n].append(p)
    return {k: v for k, v in out.items() if v}


def pdf_page_texts(pdf: Path) -> list[str]:  # pragma: no cover - cần pypdfium2
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(str(pdf))
    try:
        return [doc[i].get_textpage().get_text_range() for i in range(len(doc))]
    finally:
        doc.close()


def page_groups(pages: list[int]) -> list[tuple[int, int]]:
    """Gom trang thành các khoảng liên tiếp để gọi Docling ít lần."""
    out: list[tuple[int, int]] = []
    for p in sorted(set(pages)):
        if out and p == out[-1][1] + 1:
            out[-1] = (out[-1][0], p)
        else:
            out.append((p, p))
    return out


# ------------------------------------------------------------------ Docling trên PDF

@dataclass
class FoundTable:
    page: int
    caption: str
    html: str
    n_cells: int


class PdfTableFinder(Protocol):
    name: str

    def find(self, pdf: Path, page_range: tuple[int, int]) -> list[FoundTable]: ...


class DoclingPdfFinder:  # pragma: no cover - cần docling
    def __init__(self, mode: str = "ACCURATE", do_ocr: bool = False, do_cell_matching: bool = False):
        import importlib.metadata as md

        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
        from docling.document_converter import DocumentConverter, PdfFormatOption

        opts = PdfPipelineOptions(do_table_structure=True, do_ocr=do_ocr)
        opts.table_structure_options.mode = getattr(TableFormerMode, mode)
        opts.table_structure_options.do_cell_matching = do_cell_matching
        self._conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
        self.name = f"docling-pdf@{md.version('docling')}"

    def find(self, pdf: Path, page_range: tuple[int, int]) -> list[FoundTable]:
        doc = self._conv.convert(str(pdf), page_range=page_range).document
        out = []
        for t in doc.tables:
            page = t.prov[0].page_no if t.prov else page_range[0]
            out.append(FoundTable(page=page, caption=t.caption_text(doc) or "",
                                  html=t.export_to_html(doc=doc), n_cells=t.data.num_rows * t.data.num_cols))
        return out


def match_tables(found: list[FoundTable], pages_of: dict[str, list[int]]) -> dict[str, tuple[FoundTable, str]]:
    """table_id → (bảng Docling, cách ghép). Ưu tiên số trong caption; sau đó theo trang."""
    matched: dict[str, tuple[FoundTable, str]] = {}
    used: set[int] = set()
    for i, f in enumerate(found):
        n = caption_number(f.caption)
        if n in pages_of and n not in matched:
            matched[n] = (f, "caption")
            used.add(i)
    for tid, pages in pages_of.items():
        if tid in matched:
            continue
        # chỉ bảng KHÔNG có số trong caption (bảng có số khác là của ID khác)
        cands = [i for i, f in enumerate(found)
                 if i not in used and f.page in pages and caption_number(f.caption) is None]
        others = [t for t, ps in pages_of.items() if t not in matched and set(ps) & set(pages)]
        if len(cands) == 1 and others == [tid]:
            matched[tid] = (found[cands[0]], "page")
            used.add(cands[0])
    return matched
