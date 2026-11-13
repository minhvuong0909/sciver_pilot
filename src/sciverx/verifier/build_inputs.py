"""Dựng đầu vào cho verifier — SPEC 8.2.

| Thiết lập | Bằng chứng | Prompt |
| S-oracle  | item từ OracleEvidenceLocator, context = prepare_context như code gốc | sciver_p* (IMG) / sciver_p*_text (G*, E*), hoặc unified_v1 (cầu nối X1u) |
| S-retr    | cha theo ngân sách M3 | unified_v1 |
| S-align4  | bằng chứng sau M4 (văn bản do M6 dựng) | unified_v1 |
| S-align45 | như trên + mô tả numeric_checks của M5 | unified_v1_numeric |

Module chỉ nhận ClaimInput và OracleEvidenceLocator (không có label); đưa
GoldInfo vào sẽ bị từ chối.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from sciverx.data.loader import prepare_context
from sciverx.schemas import ClaimInput, OracleEvidenceLocator, TableRecord, VerifierRequest
from sciverx.tables.linearize import linearize_table
from sciverx.verifier.prompts import CLAIM_TYPE_TO_ID, TEXT_TABLE_LABEL, render
from sciverx.verifier.request_hash import make_request


@dataclass
class BuiltInput:
    request: VerifierRequest
    images: list[Path] = field(default_factory=list)
    meta: list[dict] = field(default_factory=list)


def _caption(paper: dict, item_type: str, item_id: str) -> str:
    if item_type == "table":
        t = paper["tables"][item_id]
        return t.get("capture") or t.get("caption") or ""
    return paper.get("image_paths", {}).get(item_id, {}).get("caption", "")


def _require_locator(loc) -> OracleEvidenceLocator:
    if not isinstance(loc, OracleEvidenceLocator) or hasattr(loc, "label"):
        raise TypeError("build_inputs chỉ nhận OracleEvidenceLocator (không có label)")
    return loc


def build_oracle(claim: ClaimInput, loc: OracleEvidenceLocator, paper: dict, table_condition: str,
                 tables: dict[str, TableRecord], data_root: str | Path, unified: bool = False) -> BuiltInput:
    """S-oracle. table_condition ∈ IMG | G0 | G1 | E0 | E1. Biểu đồ luôn là ảnh."""
    loc = _require_locator(loc)
    root = Path(data_root)
    items = loc.visual_items
    ctx = prepare_context(paper, loc.section_ids)
    caps = [_caption(paper, v["item_type"], v["item_id"]) for v in items]
    images, blocks, meta = [], [], []
    for k, (v, cap) in enumerate(zip(items, caps), 1):
        if v["item_type"] == "table" and table_condition != "IMG":
            rec = tables[v["item_id"]]
            text = linearize_table(rec)
            if not rec.cells:
                text = (rec.caption or cap) + "\n[EMPTY TABLE - extraction failed]"
            blocks.append(text if len(items) == 1 else f"item{k} {text}")
            meta.append({"item": v["item_id"], "as": "text", "status": rec.status})
        else:
            images.append(root / v["image_path"])
            meta.append({"item": v["item_id"], "as": "image"})
    table_block = (TEXT_TABLE_LABEL + "\n" + "\n\n".join(blocks) + "\n") if blocks else ""
    if unified:
        pid = "unified_v1"
        evidence = "\n".join(x for x in [table_block.strip(), "Captions: " + " | ".join(caps),
                                         "Context:\n" + ctx] if x)
        text = render(pid, claim=claim.text, evidence=evidence)
    else:
        pid = CLAIM_TYPE_TO_ID[loc.claim_type] + ("_text" if blocks else "")
        fields = ({"caption": caps[0]} if len(items) == 1 else {"caption1": caps[0], "caption2": caps[1]})
        text = table_block + render(pid, claim=claim.text, context=ctx, **fields)
    return BuiltInput(make_request(claim.claim_id, pid, text, images), images, meta)


def evidence_from_selection(parents: list[dict]) -> str:
    """Văn bản bằng chứng S-retr từ các cha đã chọn theo ngân sách (theo thứ hạng)."""
    out = []
    for p in parents:
        kind, pid = p["parent_id"].split(":", 1)
        label = {"table": "Table", "section": "Section", "chart": "Figure"}[kind]
        head = f"[{label} {pid}]" + (" (truncated)" if p.get("truncated") else "")
        body = p["text"]
        if kind == "table":
            body = TEXT_TABLE_LABEL + "\n" + body
        out.append(f"{head}\n{body}")
    return "\n\n".join(out)


def build_retrieved(claim: ClaimInput, parents: list[dict], data_root: str | Path) -> BuiltInput:
    """S-retr: chỉ nhận claim và các cha M3 đã chọn."""
    images = [Path(data_root) / p["image_path"] for p in parents if p.get("image_path")]
    text = render("unified_v1", claim=claim.text, evidence=evidence_from_selection(parents))
    meta = [{"parent": p["parent_id"], "truncated": p.get("truncated", False)} for p in parents]
    return BuiltInput(make_request(claim.claim_id, "unified_v1", text, images), images, meta)


def build_aligned(claim: ClaimInput, evidence_text: str, numeric_text: str | None = None,
                  images: list[Path] | None = None) -> BuiltInput:
    """S-align4 (numeric_text None) / S-align45. Văn bản bằng chứng do M6 dựng từ package."""
    images = images or []
    if numeric_text is None:
        pid, text = "unified_v1", render("unified_v1", claim=claim.text, evidence=evidence_text)
    else:
        pid = "unified_v1_numeric"
        text = render(pid, claim=claim.text, evidence=evidence_text,
                      numeric_checks=numeric_text or "(none)")
    return BuiltInput(make_request(claim.claim_id, pid, text, images), images)
