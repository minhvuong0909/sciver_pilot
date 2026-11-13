"""Dựng tests/fixtures/packages_mock.jsonl: 10 EvidencePackage viết tay (SPEC GĐ A0).

Dữ liệu là GIẢ, chỉ để M4–M6 và verifier phát triển song song trước khi có
dữ liệu SciVer thật. Không dùng để báo cáo. Chạy lại:
    python tests/fixtures/build_packages_mock.py
"""

from __future__ import annotations

from pathlib import Path

from sciverx.schemas import (AlignedEvidence, CellRef, EvidencePackage, NumericCheck, Operand,
                             RetrievalCandidate, RetrievalResult, SpanRef, TimeSpec)
from sciverx.utils.io import write_jsonl

OUT = Path(__file__).with_name("packages_mock.jsonl")


def cell(paper, table, cid, rh, ch, text, cond="G1"):
    return CellRef(table_record_id=f"{paper}::{table}::{cond}", condition=cond, cell_id=cid,
                   row_header=rh, col_header=ch, text_raw=text, text_norm=text)


def span(paper, sec, start, text):
    return SpanRef(unit_id=f"{paper}::paragraph::{sec}::0", section_id=sec, char_start=start,
                   char_end=start + len(text), text=text)


def retrieval(cid, parents, cond="G1"):
    cands = [RetrievalCandidate(parent_id=p, best_unit_id=f"u-{p}", kind=k, final_rank=i + 1,
                                rrf_score=round(1 / (61 + i) + 1 / (62 + i), 6))
             for i, (p, k) in enumerate(parents)]
    return RetrievalResult(claim_id=cid, method="hybrid", table_condition=cond,
                           params={"tokenizer_version": "v1", "rrf_constant": 60, "k_units": 50,
                                   "embed_model": "sentence-transformers/all-mpnet-base-v2"},
                           n_units_in_pool=40, n_parents_in_pool=12, candidates=cands,
                           pool_hash="mock")


def claim_op(text, value, unit=None):
    return Operand(source="claim", value_text=text, value=value, unit=unit)


def ev_op(ref, value, unit=None):
    return Operand(source="evidence_cell" if isinstance(ref, CellRef) else "evidence_span",
                   ref=ref, value_text=ref.text_raw if isinstance(ref, CellRef) else ref.text,
                   value=value, unit=unit)


def pkg(i, text, status, evidence, checks=(), cond="G1", setting="S-align45", missing=(),
        cands=(), errors=(), parents=None):
    cid = f"validation-mock{i:02d}"
    paper = f"mockpaper{i:02d}"
    parents = parents or [("S4.T2", "table_rows"), ("S4", "paragraph"), ("S5.T3", "table_full")]
    return EvidencePackage(
        claim_id=cid, claim_text=text, paper_id=paper, evidence_setting=setting,
        table_condition=cond, retrieval=retrieval(cid, parents, cond), evidence=list(evidence),
        alignment_status=status, alignment_candidates=list(cands), numeric_checks=list(checks),
        missing_information=list(missing),
        budget={"tokens_used": 1800 + 100 * i, "images_used": 0, "dropped_parents": [],
                "truncated": False},
        errors=list(errors),
        provenance={"model": "mock/scripted", "prompt_id": "align_v1", "prompt_hash": "mock",
                    "config_hash": "mock", "run_id": "mock", "versions": {}})


def build() -> list[EvidencePackage]:
    P = [f"mockpaper{i:02d}" for i in range(11)]
    out = []

    # 1. value_equals khớp một ô
    c = cell(P[1], "S4.T2", "r3c2", "BERT-large", "F1", "93.2")
    out.append(pkg(1, "BERT-large reaches 93.2 F1 on SQuAD.", "matched",
                   [AlignedEvidence(evidence_id="S4.T2", type="table", cells=[c], from_candidate_rank=1)],
                   [NumericCheck(check_id="c1", relation="value_equals",
                                 claim_operands=[claim_op("93.2", "93.2")],
                                 evidence_operands=[ev_op(c, "93.2")],
                                 comparison_set_status="not_applicable", unit="F1",
                                 qualifier="exact", precision_decimals=1, decided_by="rule")]))

    # 2. percentage_point_diff: 80 → 90 là +10 điểm %
    a = cell(P[2], "S4.T2", "r1c1", "Baseline", "Acc (%)", "80.0")
    b = cell(P[2], "S4.T2", "r2c1", "Ours", "Acc (%)", "90.0")
    out.append(pkg(2, "Our method improves accuracy by 10 points over the baseline.", "matched",
                   [AlignedEvidence(evidence_id="S4.T2", type="table", cells=[a, b], from_candidate_rank=1)],
                   [NumericCheck(check_id="c1", relation="percentage_point_diff",
                                 claim_operands=[claim_op("10 points", "10", "pp")],
                                 evidence_operands=[ev_op(a, "80.0", "%"), ev_op(b, "90.0", "%")],
                                 comparison_set_status="not_applicable", unit="pp",
                                 direction="higher_better", qualifier="exact", decided_by="rule")]))

    # 3. relative_change: 80 → 90 là +12.5% tương đối
    out.append(pkg(3, "Accuracy increases by about 12.5% relative to the baseline.", "matched",
                   [AlignedEvidence(evidence_id="S4.T2", type="table", cells=[a, b], from_candidate_rank=1)],
                   [NumericCheck(check_id="c1", relation="relative_change",
                                 claim_operands=[claim_op("about 12.5%", "12.5", "%")],
                                 evidence_operands=[ev_op(a, "80.0", "%"), ev_op(b, "90.0", "%")],
                                 comparison_set_status="not_applicable", unit="%",
                                 qualifier="approximately", tolerance_rule="approximately: làm tròn 1 chữ số",
                                 decided_by="rule")]))

    # 4. extremum với tập so sánh đầy đủ
    rows = [cell(P[4], "S5.T3", f"r{r}c3", m, "BLEU", v)
            for r, (m, v) in enumerate([("LSTM", "27.1"), ("Transformer", "28.4"), ("Ours", "29.3")], 1)]
    out.append(pkg(4, "Our model achieves the highest BLEU among all compared systems.", "matched",
                   [AlignedEvidence(evidence_id="S5.T3", type="table", cells=rows, from_candidate_rank=1)],
                   [NumericCheck(check_id="c1", relation="extremum", claim_operands=[],
                                 evidence_operands=[ev_op(rows[2], "29.3")], comparison_set=rows,
                                 comparison_set_status="complete", unit="BLEU",
                                 direction="higher_better", qualifier="none", decided_by="rule")]))

    # 5. ratio_multiple với thời gian (tỷ số đảo chiều)
    t1 = cell(P[5], "S4.T2", "r1c2", "Baseline", "Latency (ms)", "120")
    t2 = cell(P[5], "S4.T2", "r2c2", "Ours", "Latency (ms)", "40")
    out.append(pkg(5, "Our system is 3x faster than the baseline.", "matched",
                   [AlignedEvidence(evidence_id="S4.T2", type="table", cells=[t1, t2], from_candidate_rank=1)],
                   [NumericCheck(check_id="c1", relation="ratio_multiple",
                                 claim_operands=[claim_op("3x", "3", "x")],
                                 evidence_operands=[ev_op(t1, "120", "ms"), ev_op(t2, "40", "ms")],
                                 comparison_set_status="not_applicable", unit="ms",
                                 direction="lower_better", quantity_kind="time", qualifier="exact",
                                 decided_by="rule")]))

    # 6. bằng chứng văn bản + bảng, partial (thiếu năm)
    s = span(P[6], "S3", 120, "The dataset was collected in 2019 and contains 12,000 dialogues.")
    c6 = cell(P[6], "S3.T1", "r1c1", "Dialogues", "Count", "12,000")
    out.append(pkg(6, "The corpus, collected in 2019, has at least 12k dialogues.", "partial",
                   [AlignedEvidence(evidence_id="S3.T1", type="table", cells=[c6], from_candidate_rank=1),
                    AlignedEvidence(evidence_id="S3", type="paragraph", spans=[s], from_candidate_rank=2)],
                   [NumericCheck(check_id="c1", relation="comparison",
                                 claim_operands=[claim_op("12k", "12000")],
                                 evidence_operands=[ev_op(c6, "12000")],
                                 comparison_set_status="not_applicable", unit="dialogues",
                                 time=TimeSpec(kind="year", start="2019", end="2019", text="in 2019"),
                                 quantity_kind="count", qualifier="at_least", decided_by="rule")],
                   missing=["time: bảng không ghi năm, chỉ đoạn văn ghi 2019"]))

    # 7. ambiguous: hai bảng có cùng chỉ tiêu
    x = cell(P[7], "S4.T2", "r2c2", "Ours", "EM", "71.4")
    y = cell(P[7], "S5.T3", "r2c2", "Ours", "EM", "68.9")
    out.append(pkg(7, "Our model obtains an EM of 71.4.", "ambiguous", [],
                   cands=[AlignedEvidence(evidence_id="S4.T2", type="table", cells=[x], from_candidate_rank=1),
                          AlignedEvidence(evidence_id="S5.T3", type="table", cells=[y], from_candidate_rank=3)],
                   missing=["không rõ claim nói tới dev hay test set"]))

    # 8. not_found
    out.append(pkg(8, "The model was trained on 64 TPUs for two weeks.", "not_found", [],
                   missing=["không ứng viên nào nêu phần cứng huấn luyện"], setting="S-align4"))

    # 9. parse_error (khác not_found)
    out.append(pkg(9, "Recall improves on all five datasets.", "parse_error", [],
                   errors=["align_v1: JSON không hợp lệ sau 2 lần thử lại"], setting="S-align4"))

    # 10. bảng E1, universal với tập so sánh partial
    u = [cell(P[10], "S4.T2", f"r{r}c4", d, "Δ Recall", v, cond="E1")
         for r, (d, v) in enumerate([("D1", "+1.2"), ("D2", "+0.4"), ("D3", "-0.3")], 1)]
    out.append(pkg(10, "Our approach improves recall on every dataset.", "matched",
                   [AlignedEvidence(evidence_id="S4.T2", type="table", cells=u, from_candidate_rank=1)],
                   [NumericCheck(check_id="c1", relation="universal", claim_operands=[],
                                 evidence_operands=[ev_op(c, v) for c, v in zip(u, ["1.2", "0.4", "-0.3"])],
                                 comparison_set=u, comparison_set_status="partial", unit="Recall",
                                 direction="higher_better", qualifier="none", decided_by="rule")],
                   cond="E1", missing=["bảng E1 thiếu 2 hàng so với G1; tập so sánh chưa đủ"]))
    return out


if __name__ == "__main__":
    n = write_jsonl(OUT, build())
    print(f"Đã ghi {n} package vào {OUT}")
