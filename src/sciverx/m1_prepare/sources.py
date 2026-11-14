"""M1.1 — danh sách nguồn chung của mỗi bài (SPEC 7-M1.1).

Với mỗi bài trong tập đã lọc: mọi section, mọi bảng, mọi biểu đồ có trong
paper JSON, kèm ảnh của từng item nếu dataset có. Danh sách này dùng chung
cho mọi điều kiện (G và E). Chỉ đọc ClaimInput (paper_id) và paper JSON;
không đọc trường cấp claim nào.
"""

from __future__ import annotations

from pathlib import Path

from sciverx.data.loader import table_image_rel
from sciverx.m1_prepare import sources_path, work_dir
from sciverx.schemas import ClaimInput
from sciverx.utils.hashing import sha256_obj
from sciverx.utils.io import read_jsonl, write_jsonl


def paper_sources(paper_id: str, paper: dict, data_root: str | Path) -> dict:
    root = Path(data_root)
    tables = []
    for tid, t in paper.get("tables", {}).items():
        rel = table_image_rel(paper_id, tid)
        tables.append({"table_id": str(tid), "caption": t.get("capture") or t.get("caption") or "",
                       "image_path": rel if (root / rel).exists() else None})
    charts = []
    for cid, c in paper.get("image_paths", {}).items():
        rel = f"images/{c['figure_path']}" if c.get("figure_path") else None
        charts.append({"chart_id": str(cid), "caption": c.get("caption", ""),
                       "image_path": rel if rel and (root / rel).exists() else None})
    sections = [{"section_id": str(s["section_id"]), "section_name": s.get("section_name", ""),
                 "n_chars": len(s.get("text") or "")} for s in paper.get("sections", [])]
    return {"paper_id": paper_id, "sections": sections, "tables": tables, "charts": charts}


def build_sources(paper_ids: list[str], load, data_root: str | Path) -> list[dict]:
    """`load(paper_id) -> paper dict`."""
    return [paper_sources(pid, load(pid), data_root) for pid in sorted(set(paper_ids))]


def run(cfg: dict, args) -> int:
    from sciverx.data.loader import load_paper
    root = cfg["paths"]["data_root"]
    claims = read_jsonl(work_dir(cfg) / "claims_input.jsonl", ClaimInput)
    srcs = build_sources([c.paper_id for c in claims],
                         lambda pid: load_paper(root, f"papers/{pid}.json"), root)
    write_jsonl(sources_path(cfg), srcs)
    h = sha256_obj(srcs)
    sources_path(cfg).with_suffix(".sha256").write_text(h + "\n", encoding="utf-8")
    n_t = sum(len(s["tables"]) for s in srcs)
    n_img = sum(t["image_path"] is not None for s in srcs for t in s["tables"])
    print(f"[sources] {len(srcs)} bài, {n_t} bảng ({n_img} có ảnh), "
          f"{sum(len(s['charts']) for s in srcs)} biểu đồ, hash={h[:12]}")
    return 0
