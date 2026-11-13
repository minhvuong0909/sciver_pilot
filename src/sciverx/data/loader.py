"""Đọc dataset SciVer và paper JSON — chuyển từ common.py của bản pilot.

Cấu trúc thật đã kiểm trên validation (xem docs/VERIFICATION_LOG.md):
- split: valset.json / testset.json, mỗi file là list query.
- query: claim, claim_type, label, paperid, paper_path, section, request_id,
  origin_statement, perturbed_statement, perturbed_explanation và
    direct/analytical: type, item, image_path
    parallel/sequential: item{1,2}_type, item{1,2}, item{1,2}_path
- paper: title, abstract, sections[{section_id, parent_section_id, section_name, text}],
  tables[id]{table_html, capture}, image_paths[id]{figure_path, caption, url}, url, ...

Một số trường của query lộ nhãn (origin_statement / perturbed_statement /
perturbed_explanation) và một số trường của paper là sản phẩm của quá trình
gán nhãn (extract_*_info, *_result). Hai nhóm này không bao giờ vào kho tài
liệu hay model.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

SPLIT_FILES = {"validation": "valset.json", "test": "testset.json"}
PAIR_TYPES = ("parallel", "sequential")
SINGLE_TYPES = ("direct", "analytical")

# Trường cấp claim: chỉ data/select.py đọc (SPEC 5.5).
CLAIM_LEVEL_FIELDS = {
    "label", "claim_type", "section", "type", "item", "image_path",
    "item1_type", "item1", "item1_path", "item2_type", "item2", "item2_path",
    "origin_statement", "perturbed_statement", "perturbed_explanation", "request_id",
}
# Trường cấp paper sinh ra khi gán nhãn (liên kết câu ↔ item): không đưa vào kho M1.
PAPER_ANNOTATION_FIELDS = {"extract_table_info", "extract_figure_info", "table_result",
                           "image_result", "validation", "Chart"}


def split_path(data_root: str | Path, split: str) -> Path:
    return Path(data_root) / SPLIT_FILES[split]


def load_split(data_root: str | Path, split: str) -> list[dict]:
    rows = json.loads(split_path(data_root, split).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{SPLIT_FILES[split]} phải là list, nhận {type(rows).__name__}")
    return rows


def dataset_rel(path: str) -> str:
    """'./SciVer/papers/x.json' → 'papers/x.json' (đường dẫn trong dataset có tiền tố repo)."""
    p = path.replace("\\", "/")
    for prefix in ("./SciVer/", "SciVer/", "./"):
        if p.startswith(prefix):
            return p[len(prefix):]
    return p


def resolve(data_root: str | Path, path: str) -> Path:
    return Path(data_root) / dataset_rel(path)


@lru_cache(maxsize=256)
def _load_json_cached(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_paper(data_root: str | Path, paper_path: str) -> dict:
    return _load_json_cached(str(resolve(data_root, paper_path)))


def evidence_items(query: dict) -> list[tuple[str, str, str]]:
    """[(item_type, item_id, image_path)] theo thứ tự như code gốc."""
    if query["claim_type"] in SINGLE_TYPES:
        return [(query["type"], str(query["item"]), query["image_path"])]
    if query["claim_type"] in PAIR_TYPES:
        return [(query["item1_type"], str(query["item1"]), query["item1_path"]),
                (query["item2_type"], str(query["item2"]), query["item2_path"])]
    raise ValueError(f"claim_type lạ: {query['claim_type']!r}")


def table_image_rel(paper_id: str, table_id: str) -> str:
    """Tên ảnh bảng trong dataset: images/{paperid}-Table{N}-1.png."""
    return f"images/{paper_id}-Table{table_id}-1.png"


def prepare_context(paper: dict, section_ids: list[str]) -> str:
    """Giống code gốc SciVer: nối toàn bộ section cấp 1 chứa bằng chứng."""
    top: list[str] = []
    for s in section_ids:
        k = str(s).split(".")[0]
        if k not in top:
            top.append(k)
    ctx = ""
    for sec in paper["sections"]:
        if str(sec["section_id"]).split(".")[0] in top:
            ctx += sec["section_name"] + ":\n" + sec["text"] + "\n"
    return ctx


def label_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "entailed", "yes", "1", "supported"):
        return True
    if s in ("false", "refuted", "no", "0"):
        return False
    raise ValueError(f"Nhãn không rõ: {v!r}")
