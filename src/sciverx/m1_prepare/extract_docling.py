"""M1.3 — bảng trích E0 bằng Docling cho CÙNG danh sách bảng như G0 (SPEC 7-M1.3).

Theo quyết định của nhóm (2026-11-13): chỉ dùng ảnh có sẵn trong dataset.
Bảng không có ảnh → status `no_table_found`, cells rỗng, vẫn nằm trong kho để
G và E có cùng danh sách parent_id. Lấy ảnh/PDF cho các bảng này để sau.

Docling: TableFormer ACCURATE, bật OCR, do_cell_matching=False, lấy bảng lớn
nhất trong ảnh. Docling không cho biết chữ đậm → bold=None. Cache theo hash
ảnh + phiên bản Docling + cấu hình. Chạy trong môi trường `docling`.
[CẦN KIỂM] API Docling (ImageFormatOption, export_to_html) trên máy có Docling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from sciverx.m1_prepare import sources_path, tables_path, work_dir
from sciverx.schemas import TableRecord
from sciverx.tables.grid import build_grid
from sciverx.tables.html_cells import parse_table_html
from sciverx.utils.hashing import sha256_bytes, sha256_obj
from sciverx.utils.io import read_jsonl, write_jsonl


class TableExtractor(Protocol):
    name: str  # vd "docling@2.15.0"

    def extract_html(self, image_path: Path) -> str | None:
        """HTML của bảng lớn nhất trong ảnh, hoặc None nếu không thấy bảng."""


class DoclingExtractor:  # pragma: no cover - cần môi trường docling
    def __init__(self, mode: str = "ACCURATE", do_ocr: bool = True, do_cell_matching: bool = False):
        import importlib.metadata as md

        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
        from docling.document_converter import DocumentConverter, ImageFormatOption

        opts = PdfPipelineOptions(do_table_structure=True, do_ocr=do_ocr)
        opts.table_structure_options.mode = getattr(TableFormerMode, mode)
        opts.table_structure_options.do_cell_matching = do_cell_matching
        self._conv = DocumentConverter(
            format_options={InputFormat.IMAGE: ImageFormatOption(pipeline_options=opts)})
        self.name = f"docling@{md.version('docling')}"

    def extract_html(self, image_path: Path) -> str | None:
        doc = self._conv.convert(str(image_path)).document
        if not doc.tables:
            return None
        biggest = max(doc.tables, key=lambda t: t.data.num_rows * t.data.num_cols)
        return biggest.export_to_html(doc=doc)


def _cached_html(extractor: TableExtractor, image: Path, cache_dir: Path, cfg_key: str) -> str | None:
    key = sha256_obj({"img": sha256_bytes(image.read_bytes()), "ext": extractor.name, "cfg": cfg_key})
    hit = cache_dir / f"{key}.json"
    if hit.exists():
        return json.loads(hit.read_text(encoding="utf-8"))["html"]
    html = extractor.extract_html(image)
    cache_dir.mkdir(parents=True, exist_ok=True)
    hit.write_text(json.dumps({"html": html}), encoding="utf-8")
    return html


def extract_table(paper_id: str, table_id: str, caption: str, image: Path | None,
                  extractor: TableExtractor, cache_dir: Path, cfg_key: str = "") -> TableRecord:
    base = dict(table_record_id=TableRecord.make_id(paper_id, table_id, "E0"), paper_id=paper_id,
                table_id=table_id, condition="E0", extractor=extractor.name, caption=caption)
    if image is None or not image.exists():
        return TableRecord(**base, cells=[], n_header_rows=0, status="no_table_found",
                           quality={"image_missing": True})
    try:
        html = _cached_html(extractor, image, cache_dir, cfg_key)
        if not html:
            return TableRecord(**base, cells=[], n_header_rows=0, status="no_table_found")
        cells, n_header, _ = parse_table_html(html)
        build_grid(cells)
    except Exception as exc:  # Docling hoặc HTML lỗi: ghi lại, không dừng cả lượt
        return TableRecord(**base, cells=[], n_header_rows=0, status="error",
                           quality={"error": str(exc)[:200]})
    cells = [c.model_copy(update={"bold": None}) for c in cells]
    return TableRecord(**base, cells=cells, n_header_rows=n_header, status="ok")


def extract_all(sources: list[dict], extractor: TableExtractor, data_root: str | Path,
                cache_dir: Path, cfg_key: str = "") -> list[TableRecord]:
    root = Path(data_root)
    out = []
    for src in sources:
        for t in src["tables"]:
            img = root / t["image_path"] if t.get("image_path") else None
            out.append(extract_table(src["paper_id"], t["table_id"], t["caption"], img, extractor,
                                     cache_dir, cfg_key))
    return out


def _record_from_html(paper_id: str, table_id: str, caption: str, name: str, html: str | None,
                      quality: dict) -> TableRecord:
    base = dict(table_record_id=TableRecord.make_id(paper_id, table_id, "E0"), paper_id=paper_id,
                table_id=table_id, condition="E0", extractor=name, caption=caption)
    if not html:
        return TableRecord(**base, cells=[], n_header_rows=0, status="no_table_found", quality=quality)
    try:
        cells, n_header, _ = parse_table_html(html)
        build_grid(cells)
    except Exception as exc:
        return TableRecord(**base, cells=[], n_header_rows=0, status="error",
                           quality={**quality, "error": str(exc)[:200]})
    cells = [c.model_copy(update={"bold": None}) for c in cells]
    return TableRecord(**base, cells=cells, n_header_rows=n_header, status="ok", quality=quality)


def extract_paper_from_pdf(src: dict, pdf: Path | None, finder, page_texts, cache_dir: Path,
                           cfg_key: str = "") -> list[TableRecord]:
    """E0 cho mọi bảng của một bài từ PDF. `page_texts(pdf) -> list[str]` (chữ từng trang)."""
    from sciverx.m1_prepare.pdf_tables import (FoundTable, caption_pages, match_tables, page_groups,
                                               table_labels)
    tids = [t["table_id"] for t in src["tables"]]
    caps = {t["table_id"]: t["caption"] for t in src["tables"]}
    labels = table_labels(src["tables"])   # ID của dataset ≠ nhãn in trong PDF ("7" → "C.1")
    if pdf is None or not pdf.exists():
        return [_record_from_html(src["paper_id"], t, caps[t], finder.name, None,
                                  {"source": "pdf", "reason": "pdf_missing"}) for t in tids]
    key = sha256_obj({"pdf": sha256_bytes(pdf.read_bytes()), "ext": finder.name, "cfg": cfg_key,
                      "labels": labels})
    hit = cache_dir / f"{key}.json"
    if hit.exists():
        cached = json.loads(hit.read_text(encoding="utf-8"))
        pages_of = cached["pages_of"]
        found = [FoundTable(**f) for f in cached["found"]]
    else:
        try:
            pages_of = caption_pages(page_texts(pdf), list(labels.values()))
            found = []
            for rng in page_groups([p for ps in pages_of.values() for p in ps]):
                found += finder.find(pdf, rng)
        except Exception as exc:  # một PDF lỗi không được làm dừng cả lượt chạy
            err = f"{type(exc).__name__}: {str(exc)[:160]}"
            return [TableRecord(table_record_id=TableRecord.make_id(src["paper_id"], t, "E0"),
                                paper_id=src["paper_id"], table_id=t, condition="E0", extractor=finder.name,
                                caption=caps[t], cells=[], n_header_rows=0, status="error",
                                quality={"source": "pdf", "error": err}) for t in tids]
        cache_dir.mkdir(parents=True, exist_ok=True)
        hit.write_text(json.dumps({"pages_of": pages_of, "found": [vars(f) for f in found]}), encoding="utf-8")
    matched = match_tables(found, pages_of)
    out = []
    for t in tids:
        lab = labels[t]
        if lab in matched:
            f, how = matched[lab]
            q = {"source": "pdf", "page": f.page, "match": how, "label": lab}
            out.append(_record_from_html(src["paper_id"], t, caps[t], finder.name, f.html, q))
        else:
            reason = "caption_not_found" if lab not in pages_of else "no_docling_table"
            out.append(_record_from_html(src["paper_id"], t, caps[t], finder.name, None,
                                         {"source": "pdf", "reason": reason}))
    return out


def run(cfg: dict, args) -> int:  # pragma: no cover - cần docling
    dcfg = cfg.get("m1", {}).get("docling", {})
    sources = read_jsonl(sources_path(cfg))
    if cfg.get("limit"):
        sources = sources[: cfg["limit"]]
    if dcfg.get("source", "pdf") == "pdf":
        from sciverx.m1_prepare.pdf_tables import DoclingPdfFinder, download_pdf, pdf_page_texts
        finder = DoclingPdfFinder(dcfg.get("mode", "ACCURATE"), dcfg.get("do_ocr_pdf", False),
                                  dcfg.get("do_cell_matching", False))
        pdf_dir = Path(cfg["paths"]["data_root"]).parent / "arxiv_pdf"
        recs = []
        for i, src in enumerate(sources, 1):
            pdf = download_pdf(src["paper_id"], pdf_dir, cfg["m1"]["gold"].get("arxiv_sleep_s", 3))
            recs += extract_paper_from_pdf(src, pdf, finder, pdf_page_texts,
                                           work_dir(cfg) / "cache" / "docling_pdf", sha256_obj(dcfg))
            if i % 10 == 0:
                print(f"[extract] {i}/{len(sources)} bài", flush=True)
        name = finder.name
    else:
        extractor = DoclingExtractor(dcfg.get("mode", "ACCURATE"), dcfg.get("do_ocr", True),
                                     dcfg.get("do_cell_matching", False))
        recs = extract_all(sources, extractor, cfg["paths"]["data_root"],
                           work_dir(cfg) / "cache" / "docling", sha256_obj(dcfg))
        name = extractor.name
    write_jsonl(tables_path(cfg, "E0"), recs)
    by: dict[str, int] = {}
    for r in recs:
        by[r.status] = by.get(r.status, 0) + 1
    print(f"[extract] {len(recs)} bảng E0 ({name}): {by}")
    return 0
