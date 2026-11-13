"""M1.6 — đơn vị tìm kiếm ≤ max_wordpieces (SPEC 7-M1.6).

Hai kho G1 và E1 dùng CÙNG danh sách nguồn, cùng đơn vị đoạn văn, cùng quy
tắc chia; chỉ phần biểu diễn bảng khác nhau.
- paragraph: tách section text theo "\\n\\n"; đoạn quá dài thì gom câu tới giới
  hạn. Văn bản "{section_name}: {đoạn}", lưu char_start/char_end trong section.
- table_full: caption + tiêu đề + mọi hàng, chỉ khi vừa giới hạn.
- table_rows: nhóm hàng liên tiếp, mỗi nhóm lặp lại caption và tiêu đề gộp.
  Hàng đơn lẻ quá dài → chia theo cột, mỗi phần kèm cột tiêu đề hàng.
- Bảng không có nội dung (no_table_found, gold_unavailable...) → một đơn vị
  table_rows chỉ có caption để hai kho có cùng parent_id.
- chart_caption: chỉ khi allow_mixed.
Không cắt âm thầm: mọi lần phải chia nhỏ hơn mức tự nhiên đều ghi log.
"""

from __future__ import annotations

import re
from typing import Callable

from sciverx.schemas import EvidenceUnit, TableRecord
from sciverx.tables.grid import build_grid, merged_column_headers
from sciverx.tables.linearize import SEP

Counter_ = Callable[[str], int]
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def hf_wordpiece_counter(model_name: str) -> Counter_:  # pragma: no cover - cần transformers
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    return lambda text: len(tok(text, add_special_tokens=True)["input_ids"])


def whitespace_counter(text: str) -> int:
    """Bộ đếm thay thế cho test: số từ + 2 token đặc biệt."""
    return len(text.split()) + 2


# ------------------------------------------------------------------ đoạn văn

def _pieces(text: str, start: int, limit: int, count: Counter_, prefix: str,
            log: list[dict], where: str) -> list[tuple[int, int]]:
    """Chia [start, start+len(text)) thành các khoảng vừa giới hạn: theo câu, rồi theo từ."""
    if count(prefix + text) <= limit:
        return [(start, start + len(text))]
    sents, pos = [], 0
    for m in _SENT.finditer(text):
        sents.append((pos, m.start()))
        pos = m.end()
    sents.append((pos, len(text)))
    out: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = None
    for s, e in sents:
        if count(prefix + text[s:e]) > limit:  # một câu quá dài → chia theo từ
            if cur:
                out.append(cur)
                cur = None
            log.append({"where": where, "action": "split_sentence_by_words"})
            words = [(m.start() + s, m.end() + s) for m in re.finditer(r"\S+", text[s:e])]
            ws = words[0][0]
            last_end = ws
            for a, b in words:
                if count(prefix + text[ws:b]) > limit and last_end > ws:
                    out.append((ws, last_end))
                    ws = a
                last_end = b
            out.append((ws, last_end))
            continue
        if cur and count(prefix + text[cur[0]:e]) <= limit:
            cur = (cur[0], e)
        else:
            if cur:
                out.append(cur)
            cur = (s, e)
    if cur:
        out.append(cur)
    return [(start + a, start + b) for a, b in out]


def paragraph_units(paper_id: str, paper: dict, limit: int, count: Counter_,
                    log: list[dict]) -> list[EvidenceUnit]:
    units = []
    for sec in paper.get("sections", []):
        text = sec.get("text") or ""
        sid = str(sec["section_id"])
        prefix = f"{sec.get('section_name', '')}: "
        k = 0
        pos = 0
        for block in text.split("\n\n"):
            b_start = text.index(block, pos) if block else pos
            pos = b_start + len(block)
            lead = len(block) - len(block.lstrip())
            block_s = block.strip()
            if not block_s:
                continue
            for a, b in _pieces(block_s, b_start + lead, limit, count, prefix, log, f"{paper_id}::{sid}"):
                body = text[a:b]
                units.append(EvidenceUnit(
                    unit_id=f"{paper_id}::paragraph::{sid}::{k}", paper_id=paper_id, kind="paragraph",
                    parent_id=sid, text=prefix + body, n_wordpieces=count(prefix + body),
                    char_start=a, char_end=b))
                k += 1
    return units


# ------------------------------------------------------------------ bảng

def _row_line(grid, r: int, cols: list[int]) -> str:
    rows = grid.text_rows()
    return SEP.join(rows[r][c] for c in cols)


def table_units(rec: TableRecord, limit: int, count: Counter_, log: list[dict]) -> list[EvidenceUnit]:
    cond = rec.condition
    base = dict(paper_id=rec.paper_id, parent_id=rec.table_id, table_condition=cond)
    uid = f"{rec.paper_id}::{{kind}}::{rec.table_id}::{{part}}"
    caption = rec.caption.strip()
    if rec.status != "ok" or not rec.cells:
        text = caption or f"Table {rec.table_id}"
        return [EvidenceUnit(unit_id=uid.format(kind="table_rows", part=f"{cond}-caption"),
                             kind="table_rows", text=text, n_wordpieces=count(text), **base)]
    grid = build_grid(rec.cells)
    nh = rec.n_header_rows
    all_cols = list(range(grid.n_cols))
    headers = merged_column_headers(grid, nh)
    head_line = SEP.join(headers) if nh and any(headers) else ""
    body = list(range(nh, grid.n_rows))

    def text_for(rows: list[int], cols: list[int]) -> str:
        head = SEP.join(headers[c] for c in cols) if head_line else ""
        lines = [x for x in [caption, head] if x] + [_row_line(grid, r, cols) for r in rows]
        return "\n".join(lines)

    notes = " ".join(f"[{m}] {t}" for m, t in sorted(rec.footnotes.items()))
    full = text_for(body, all_cols) + (f"\nNotes: {notes}" if notes else "")
    if count(full) <= limit:
        return [EvidenceUnit(unit_id=uid.format(kind="table_full", part=f"{cond}-0"),
                             kind="table_full", text=full, n_wordpieces=count(full),
                             row_range=(nh, grid.n_rows - 1) if body else None, **base)]
    units: list[EvidenceUnit] = []
    group: list[int] = []

    def flush():
        if group:
            t = text_for(group, all_cols)
            units.append(EvidenceUnit(unit_id=uid.format(kind="table_rows", part=f"{cond}-{len(units)}"),
                                      kind="table_rows", text=t, n_wordpieces=count(t),
                                      row_range=(group[0], group[-1]), **base))

    for r in body:
        if count(text_for(group + [r], all_cols)) <= limit:
            group.append(r)
            continue
        flush()
        group = []
        if count(text_for([r], all_cols)) <= limit:
            group = [r]
            continue
        # một hàng vượt giới hạn: chia theo cột, mỗi phần giữ cột 0 (tiêu đề hàng)
        log.append({"where": rec.table_record_id, "action": "split_row_by_columns", "row": r})
        cols: list[int] = [0]
        for c in all_cols[1:]:
            if count(text_for([r], cols + [c])) <= limit or cols == [0]:
                cols.append(c)
            else:
                t = text_for([r], cols)
                units.append(EvidenceUnit(unit_id=uid.format(kind="table_rows", part=f"{cond}-{len(units)}"),
                                          kind="table_rows", text=t, n_wordpieces=count(t),
                                          row_range=(r, r), **base))
                cols = [0, c]
        t = text_for([r], cols)
        units.append(EvidenceUnit(unit_id=uid.format(kind="table_rows", part=f"{cond}-{len(units)}"),
                                  kind="table_rows", text=t, n_wordpieces=count(t),
                                  row_range=(r, r), **base))
    flush()
    if not units:  # bảng chỉ có tiêu đề
        t = text_for([], all_cols)
        units.append(EvidenceUnit(unit_id=uid.format(kind="table_rows", part=f"{cond}-0"),
                                  kind="table_rows", text=t, n_wordpieces=count(t), **base))
    return units


def chart_units(paper_id: str, paper: dict, count: Counter_) -> list[EvidenceUnit]:
    out = []
    for cid, c in paper.get("image_paths", {}).items():
        text = (c.get("caption") or "").strip()
        if text:
            out.append(EvidenceUnit(unit_id=f"{paper_id}::chart_caption::{cid}::0", paper_id=paper_id,
                                    kind="chart_caption", parent_id=str(cid), text=text,
                                    n_wordpieces=count(text)))
    return out


def build_units(paper_id: str, paper: dict, tables: list[TableRecord], limit: int, count: Counter_,
                allow_mixed: bool = False) -> tuple[list[EvidenceUnit], list[dict]]:
    log: list[dict] = []
    units = paragraph_units(paper_id, paper, limit, count, log)
    for rec in tables:
        units += table_units(rec, limit, count, log)
    if allow_mixed:
        units += chart_units(paper_id, paper, count)
    over = [u.unit_id for u in units if u.n_wordpieces > limit]
    if over:
        raise ValueError(f"Đơn vị vượt {limit} word pieces: {over[:3]}")
    return units, log


def run(cfg: dict, args) -> int:  # pragma: no cover - cần tokenizer của model embedding
    from sciverx.data.loader import load_paper
    from sciverx.m1_prepare import sources_path, tables_path, units_path, work_dir
    from sciverx.utils.io import read_jsonl, write_jsonl
    ucfg = cfg["m1"]["units"]
    count = hf_wordpiece_counter(ucfg["embed_model"])
    limit = ucfg["max_wordpieces"]
    allow_mixed = cfg.get("selection", {}).get("allow_mixed", False)
    root = cfg["paths"]["data_root"]
    srcs = read_jsonl(sources_path(cfg))
    all_log: list[dict] = []
    for cond in ("G1", "E1"):
        path = tables_path(cfg, cond)
        if not path.exists():
            print(f"[units] bỏ qua {cond}: chưa có bảng")
            continue
        by_paper: dict[str, list[TableRecord]] = {}
        for r in read_jsonl(path, TableRecord):
            by_paper.setdefault(r.paper_id, []).append(r)
        units: list[EvidenceUnit] = []
        for s in srcs:
            paper = load_paper(root, f"papers/{s['paper_id']}.json")
            u, log = build_units(s["paper_id"], paper, by_paper.get(s["paper_id"], []), limit, count,
                                 allow_mixed)
            units += u
            all_log += [{**e, "condition": cond} for e in log]
        write_jsonl(units_path(cfg, cond), units)
        print(f"[units] {cond}: {len(units)} đơn vị, max {max(u.n_wordpieces for u in units)} wp")
    write_jsonl(work_dir(cfg) / "units_log.jsonl", all_log)
    return 0
