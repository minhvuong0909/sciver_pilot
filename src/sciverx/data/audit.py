"""Audit dữ liệu D1–D8 — SPEC 5.2. Viết một lần, dùng lại nguyên vẹn cho test ở GĐ B4.

--structure-only (ngoại lệ trước B4, SPEC 5.4): chỉ in tên trường và kiểu dữ
liệu, không in giá trị, không đếm theo nhãn hay claim_type.

Kết quả ghi vào {work}/{split}/audit.json. Trả mã 4 khi D2 thiếu trường
mong đợi (cần [DỪNG & HỎI]).
"""

from __future__ import annotations

import json
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Callable

from sciverx.data.loader import (CLAIM_LEVEL_FIELDS, PAPER_ANNOTATION_FIELDS, SPLIT_FILES,
                                 evidence_items, load_paper, load_split, table_image_rel)
from sciverx.data.select import table_status

EXPECTED_QUERY_COMMON = {"claim", "claim_type", "label", "paper_path", "section"}
EXPECTED_QUERY_SINGLE = {"type", "item", "image_path"}
EXPECTED_QUERY_PAIR = {"item1_type", "item1", "item1_path", "item2_type", "item2", "item2_path"}
EXPECTED_PAPER = {"sections", "tables", "image_paths"}

ImageChecker = Callable[[str], bool]


def local_image_checker(data_root: str | Path) -> ImageChecker:
    root = Path(data_root)
    return lambda rel: (root / rel).exists()


def remote_image_checker(base: str = "https://huggingface.co/datasets/chengyewang/SciVer/resolve/main/"
                         ) -> ImageChecker:  # pragma: no cover - cần mạng
    cache: dict[str, bool] = {}

    def check(rel: str) -> bool:
        if rel not in cache:
            req = urllib.request.Request(base + rel, method="HEAD")
            try:
                cache[rel] = urllib.request.urlopen(req, timeout=30).status == 200
            except Exception:
                cache[rel] = False
        return cache[rel]
    return check


def _type_name(v) -> str:
    return "null" if v is None else type(v).__name__


def structure(rows: list[dict]) -> dict:
    """Tên trường → tập kiểu dữ liệu. Không chứa giá trị."""
    types: dict[str, set[str]] = {}
    for r in rows:
        for k, v in r.items():
            types.setdefault(k, set()).add(_type_name(v))
    return {k: sorted(v) for k, v in sorted(types.items())}


def check_fields(rows: list[dict]) -> dict:
    seen = set().union(*(r.keys() for r in rows)) if rows else set()
    missing = set(EXPECTED_QUERY_COMMON - seen)
    singles = [r for r in rows if r.get("claim_type") in ("direct", "analytical")]
    pairs = [r for r in rows if r.get("claim_type") in ("parallel", "sequential")]
    if singles:
        missing |= EXPECTED_QUERY_SINGLE - set().union(*(r.keys() for r in singles))
    if pairs:
        missing |= EXPECTED_QUERY_PAIR - set().union(*(r.keys() for r in pairs))
    expected = EXPECTED_QUERY_COMMON | EXPECTED_QUERY_SINGLE | EXPECTED_QUERY_PAIR
    return {"missing_expected": sorted(missing), "extra": sorted(seen - expected),
            "extra_label_leaking": sorted((seen - expected) & {"origin_statement",
                                                                "perturbed_statement",
                                                                "perturbed_explanation"})}


def audit(data_root: str | Path, split: str, image_exists: ImageChecker | None = None,
          structure_only: bool = False) -> dict:
    rows = load_split(data_root, split)
    report: dict = {"split": split, "D1": {"file": SPLIT_FILES[split], "n_claims": len(rows)}}
    report["D2"] = {"query_structure": structure(rows), **check_fields(rows)}
    if structure_only:
        papers = [load_paper(data_root, p) for p in sorted({r["paper_path"] for r in rows})[:20]]
        report["paper_structure"] = structure(papers)
        return report

    paper_ids = sorted({r["paperid"] for r in rows})
    papers = {r["paperid"]: load_paper(data_root, r["paper_path"]) for r in rows}
    image_exists = image_exists or local_image_checker(data_root)

    # D3: nội dung tables[id]
    entry_keys: Counter = Counter()
    n_tables = n_html = 0
    for p in papers.values():
        for t in p.get("tables", {}).values():
            entry_keys.update(t.keys())
            n_tables += 1
            n_html += bool(t.get("table_html"))
    report["D3"] = {"table_entry_keys": dict(entry_keys), "n_tables": n_tables,
                    "with_table_html": n_html}

    # D4: kho có đủ bảng/biểu đồ và ảnh không
    with_img = 0
    per_paper = []
    for pid in paper_ids:
        tids = list(papers[pid].get("tables", {}))
        imgs = sum(image_exists(table_image_rel(pid, t)) for t in tids)
        with_img += imgs
        per_paper.append((len(tids), imgs, len(papers[pid].get("image_paths", {}))))
    report["D4"] = {
        "n_papers": len(paper_ids), "n_tables": n_tables, "tables_with_image": with_img,
        "papers_all_tables_have_image": sum(1 for n, i, _ in per_paper if n == i),
        "mean_tables_per_paper": round(n_tables / max(1, len(paper_ids)), 2),
        "mean_charts_per_paper": round(sum(c for *_, c in per_paper) / max(1, len(paper_ids)), 2),
        "paper_top_level_keys": sorted(set().union(*(p.keys() for p in papers.values()))),
        "annotation_fields_present": sorted(set().union(*(p.keys() for p in papers.values()))
                                            & PAPER_ANNOTATION_FIELDS),
        "note": "So với HTML arXiv chưa làm; tables_with_image đếm theo image_exists.",
    }

    # D5: sections
    n_sec = empty = 0
    depth: Counter = Counter()
    for p in papers.values():
        for s in p["sections"]:
            n_sec += 1
            empty += not s.get("text", "").strip()
    for r in rows:
        for s in r.get("section") or []:
            depth[str(s).count(".") + 1] += 1
    report["D5"] = {"n_sections": n_sec, "empty_text_sections": empty,
                    "annotated_section_depth": dict(sorted(depth.items()))}

    # D6: dạng ID item, phiên bản arXiv
    item_ids: Counter = Counter()
    for r in rows:
        for _, iid, _ in evidence_items(r):
            item_ids["digit" if str(iid).isdigit() else "other"] += 1
    report["D6"] = {"item_id_format": dict(item_ids),
                    "paperid_has_version": sum("v" in pid for pid in paper_ids),
                    "n_papers": len(paper_ids)}

    # D7, D8: claim thỏa tiêu chí; bảng dùng chung
    status = [table_status(r) for r in rows]
    sel = [r for r, s in zip(rows, status) if s == "all_table"]
    report["D7"] = {
        "by_table_status": dict(Counter(status)),
        "selected": len(sel),
        "selected_by_type_label": {f"{k[0]}/{k[1]}": v for k, v in
                                   sorted(Counter((r["claim_type"], bool(r["label"])) for r in sel).items())},
    }
    use: Counter = Counter()
    for r in sel:
        for t, iid, _ in evidence_items(r):
            if t == "table":
                use[(r["paperid"], iid)] += 1
    report["D8"] = {"unique_evidence_tables": len(use),
                    "tables_shared_by_multiple_claims": sum(1 for v in use.values() if v > 1),
                    "claims_on_shared_tables": sum(v for v in use.values() if v > 1),
                    "n_papers_selected": len({r["paperid"] for r in sel})}
    report["claim_level_fields_never_in_corpus"] = sorted(CLAIM_LEVEL_FIELDS)
    return report


def run(cfg: dict, args) -> int:
    split = cfg["split"]
    structure_only = bool(getattr(args, "structure_only", False))
    if split == "test" and not structure_only and not cfg.get("allow_test_audit", False):
        print("[audit] test chỉ được audit --structure-only trước GĐ B4 (SPEC 5.4)")
        return 2
    checker = remote_image_checker() if cfg.get("audit", {}).get("remote_images") else None
    report = audit(cfg["paths"]["data_root"], split, checker, structure_only)
    out = Path(cfg["paths"]["work"]) / split / ("audit_structure.json" if structure_only else "audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2)[:4000])
    if report["D2"]["missing_expected"]:
        print("[audit] D2 thiếu trường mong đợi → [DỪNG & HỎI]")
        return 4
    return 0
