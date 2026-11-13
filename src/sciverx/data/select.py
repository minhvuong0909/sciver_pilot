"""Chọn claim và tách ClaimInput / GoldInfo — SPEC 5.3.

Đây là module DUY NHẤT đọc annotation để lọc. Đầu ra:
  {work}/{split}/claims_input.jsonl   ClaimInput (không có gì cấp claim ngoài text)
  {work}/{split}/gold.jsonl           GoldInfo (file riêng)
  {work}/{split}/ids.txt (+ ids.sha256)
Tiêu chí mặc định: mọi bằng chứng thị giác là bảng. allow_mixed: có ít nhất
một bảng, phần còn lại là biểu đồ. claim_id = "{split}-{vị trí trong file}"
vì request_id của dataset bị trùng.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from sciverx.data.loader import dataset_rel, evidence_items, label_bool, load_split
from sciverx.schemas import ClaimInput, GoldInfo
from sciverx.utils.hashing import ids_hash
from sciverx.utils.io import write_jsonl


def table_status(query: dict) -> str:
    types = [t for t, _, _ in evidence_items(query)]
    n_tab = sum(t == "table" for t in types)
    if n_tab == len(types):
        return "all_table"
    return "mixed" if n_tab else "no_table"


def select_claims(rows: list[dict], split: str, allow_mixed: bool = False
                  ) -> tuple[list[ClaimInput], list[GoldInfo]]:
    ok = {"all_table", "mixed"} if allow_mixed else {"all_table"}
    claims, golds = [], []
    for i, q in enumerate(rows):
        if table_status(q) not in ok:
            continue
        cid = f"{split}-{i}"
        claims.append(ClaimInput(claim_id=cid, split=split, paper_id=q["paperid"], text=q["claim"]))
        golds.append(GoldInfo(
            claim_id=cid, label=label_bool(q["label"]), claim_type=q["claim_type"],
            section_ids=[str(s) for s in q.get("section") or []],
            visual_items=[{"item_id": iid, "item_type": t, "image_path": dataset_rel(p)}
                          for t, iid, p in evidence_items(q)]))
    return claims, golds


def write_selection(out_dir: str | Path, claims: list[ClaimInput], golds: list[GoldInfo]) -> dict:
    out = Path(out_dir)
    write_jsonl(out / "claims_input.jsonl", claims)
    write_jsonl(out / "gold.jsonl", golds)
    ids = [c.claim_id for c in claims]
    (out / "ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    h = ids_hash(ids)
    (out / "ids.sha256").write_text(h + "\n", encoding="utf-8")
    return {"n_claims": len(ids), "ids_hash": h,
            "n_papers": len({c.paper_id for c in claims})}


def distribution(golds: list[GoldInfo]) -> dict[str, int]:
    return {f"{ct}/{'T' if lab else 'F'}": n
            for (ct, lab), n in sorted(Counter((g.claim_type, g.label) for g in golds).items())}


def run(cfg: dict, args) -> int:
    split = cfg["split"]
    rows = load_split(cfg["paths"]["data_root"], split)
    claims, golds = select_claims(rows, split, cfg.get("selection", {}).get("allow_mixed", False))
    if cfg.get("limit"):
        claims, golds = claims[: cfg["limit"]], golds[: cfg["limit"]]
    stats = write_selection(Path(cfg["paths"]["work"]) / split, claims, golds)
    print(f"[select] {split}: {stats['n_claims']}/{len(rows)} claim, {stats['n_papers']} bài, "
          f"ids_hash={stats['ids_hash'][:12]}")
    if split != "test":  # không in phân bố nhãn của test
        print("[select] phân bố (claim_type/nhãn):", distribution(golds))
    return 0
