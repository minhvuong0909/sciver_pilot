"""M1.7 — chỉ mục theo từng bài và từng điều kiện (G1, E1) (SPEC 7-M1.7).

BM25 dựng lại khi truy xuất (rẻ, tất định); phần tốn kém là embedding nên
được lưu: {work}/{split}/index/{cond}/vectors.npy + units.json, cache theo
hash nội dung + model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sciverx.m1_prepare import units_path, work_dir
from sciverx.m3_retrieve.retrieve import PaperIndex, pool_hash
from sciverx.schemas import EvidenceUnit
from sciverx.utils.hashing import sha256_obj
from sciverx.utils.io import read_jsonl


def index_dir(cfg: dict, cond: str) -> Path:
    return work_dir(cfg) / "index" / cond


def save_vectors(out: Path, units: list[EvidenceUnit], vectors: np.ndarray, model: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "vectors.npy", vectors.astype(np.float32))
    meta = {"model": model, "unit_ids": [u.unit_id for u in units],
            "content_hash": sha256_obj({"model": model, "pool": pool_hash(units)})}
    (out / "units.json").write_text(json.dumps(meta), encoding="utf-8")


def load_vectors(out: Path, units: list[EvidenceUnit], model: str) -> np.ndarray | None:
    """Trả ma trận nếu cache khớp đúng nội dung và model, ngược lại None."""
    meta_p = out / "units.json"
    if not meta_p.exists():
        return None
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    if meta["content_hash"] != sha256_obj({"model": model, "pool": pool_hash(units)}):
        return None
    return np.load(out / "vectors.npy")


def build_indexes(units: list[EvidenceUnit], condition: str, vectors: np.ndarray | None
                  ) -> dict[str, PaperIndex]:
    """Tách ma trận chung thành kho của từng bài."""
    by: dict[str, list[int]] = {}
    for i, u in enumerate(units):
        by.setdefault(u.paper_id, []).append(i)
    out = {}
    for pid, ids in by.items():
        us = [units[i] for i in ids]
        idx = PaperIndex.build(pid, condition, us)
        idx.vectors = vectors[ids] if vectors is not None else None
        out[pid] = idx
    return out


def load_indexes(cfg: dict, cond: str, need_dense: bool) -> dict[str, PaperIndex]:
    units = read_jsonl(units_path(cfg, cond), EvidenceUnit)
    vecs = None
    if need_dense:
        model = cfg["m1"]["units"]["embed_model"]
        vecs = load_vectors(index_dir(cfg, cond), units, model)
        if vecs is None:
            raise RuntimeError(f"Chưa có vectors cho {cond} (chạy stage index trước)")
    return build_indexes(units, cond, vecs)


def run(cfg: dict, args) -> int:  # pragma: no cover - cần sentence-transformers
    from sciverx.m3_retrieve.rankers import SentenceTransformerEmbedder
    model = cfg["m1"]["units"]["embed_model"]
    emb = SentenceTransformerEmbedder(model)
    for cond in ("G1", "E1"):
        p = units_path(cfg, cond)
        if not p.exists():
            continue
        units = read_jsonl(p, EvidenceUnit)
        if load_vectors(index_dir(cfg, cond), units, model) is not None:
            print(f"[index] {cond}: cache khớp, bỏ qua")
            continue
        save_vectors(index_dir(cfg, cond), units, emb.encode([u.text for u in units]), model)
        print(f"[index] {cond}: {len(units)} vectors")
    return 0
