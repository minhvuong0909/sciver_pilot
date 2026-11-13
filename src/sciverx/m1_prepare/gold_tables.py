"""M1.2 — bảng chuẩn G0 cho MỌI bảng trong danh sách nguồn (SPEC 7-M1.2).

Dữ liệu thật đã có `tables[id].table_html` (LaTeXML của arXiv đúng phiên bản
bài) nên luôn dùng nhánh `extractor=gold_json`; không cần tải HTML arXiv.
- Không có bảng HTML (ảnh hoặc rỗng) → status `gold_unavailable`.
- HTML lỗi (ô chồng nhau...) → status `error`.
Xuất `annotation/gold_audit.csv` để người kiểm tay: mọi bảng lỗi, bảng nhiều
phần, bảng phải đoán hàng tiêu đề, và mẫu ngẫu nhiên phân tầng 100 bảng
(seed 2026) → [DỪNG & HỎI].
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

from sciverx.m1_prepare import sources_path, tables_path, work_dir
from sciverx.schemas import TableRecord
from sciverx.tables.grid import build_grid
from sciverx.tables.html_cells import parse_table_html
from sciverx.utils.io import read_jsonl, write_jsonl

AUDIT_FIELDS = ["table_record_id", "paper_id", "table_id", "status", "reason", "n_cells", "n_parts",
                "n_header_rows", "caption", "image_path", "checked_ok", "notes"]


def build_gold_table(paper_id: str, table_id: str, entry: dict) -> tuple[TableRecord, dict]:
    """Trả (TableRecord G0, thông tin cho audit)."""
    caption = entry.get("capture") or entry.get("caption") or ""
    rid = TableRecord.make_id(paper_id, table_id, "G0")
    info = {"n_parts": 0, "reason": ""}
    try:
        cells, n_header, n_parts = parse_table_html(entry.get("table_html") or "")
        build_grid(cells)  # kiểm ô chồng nhau
        status, info["n_parts"] = "ok", n_parts
    except ValueError as exc:
        cells, n_header = [], 0
        missing = "Không tìm thấy bảng" in str(exc)
        status = "gold_unavailable" if missing else "error"
        info["reason"] = str(exc)[:200]
    rec = TableRecord(table_record_id=rid, paper_id=paper_id, table_id=table_id, condition="G0",
                      extractor="gold_json", source_version=paper_id.rsplit("v", 1)[-1]
                      if "v" in paper_id else None,
                      caption=caption, cells=cells, n_header_rows=n_header, status=status)
    return rec, info


def audit_rows(records: list[TableRecord], infos: list[dict], images: dict[str, str | None],
               n_random: int = 100, seed: int = 2026) -> list[dict]:
    """Bảng cần kiểm: lỗi, nhiều phần, và mẫu ngẫu nhiên phân tầng theo kích thước."""
    rows, chosen = [], set()

    def add(rec, info, reason):
        if rec.table_record_id in chosen:
            return
        chosen.add(rec.table_record_id)
        rows.append({"table_record_id": rec.table_record_id, "paper_id": rec.paper_id,
                     "table_id": rec.table_id, "status": rec.status, "reason": reason,
                     "n_cells": len(rec.cells), "n_parts": info["n_parts"],
                     "n_header_rows": rec.n_header_rows, "caption": rec.caption[:200],
                     "image_path": images.get(rec.table_record_id) or "", "checked_ok": "",
                     "notes": info["reason"]})

    for rec, info in zip(records, infos):
        if rec.status != "ok":
            add(rec, info, "status")
        elif info["n_parts"] > 1:
            add(rec, info, "multi_part")
    ok = [(r, i) for r, i in zip(records, infos) if r.status == "ok"
          and r.table_record_id not in chosen]
    strata: dict[str, list] = {"small": [], "medium": [], "large": []}
    for r, i in ok:
        n = len(r.cells)
        strata["small" if n < 30 else "medium" if n < 100 else "large"].append((r, i))
    rng = random.Random(seed)
    total = sum(len(v) for v in strata.values())
    for name, items in strata.items():
        k = round(n_random * len(items) / total) if total else 0
        for r, i in rng.sample(items, min(k, len(items))):
            add(r, i, f"random_{name}")
    return rows


def write_audit_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
        w.writeheader()
        w.writerows(rows)


def run(cfg: dict, args) -> int:
    from sciverx.data.loader import load_paper
    root = cfg["paths"]["data_root"]
    records, infos, images = [], [], {}
    for src in read_jsonl(sources_path(cfg)):
        paper = load_paper(root, f"papers/{src['paper_id']}.json")
        for t in src["tables"]:
            rec, info = build_gold_table(src["paper_id"], t["table_id"],
                                         paper["tables"][t["table_id"]])
            records.append(rec)
            infos.append(info)
            images[rec.table_record_id] = t["image_path"]
    write_jsonl(tables_path(cfg, "G0"), records)
    gcfg = cfg.get("m1", {}).get("gold", {})
    rows = audit_rows(records, infos, images, gcfg.get("audit_random_n", 100),
                      cfg.get("seed_global", 2026))
    out = work_dir(cfg) / "annotation" / "gold_audit.csv"
    write_audit_csv(out, rows)
    by_status: dict[str, int] = {}
    for r in records:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    print(f"[gold] {len(records)} bảng G0: {by_status}; {len(rows)} bảng cần kiểm tay → {out}")
    print("[gold] [DỪNG & HỎI] cần người kiểm gold_audit.csv trước khi dùng G0 cho thực nghiệm")
    return 0
