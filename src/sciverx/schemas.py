"""Schema dữ liệu dùng chung (Pydantic v2) — SPEC mục 6.

Mọi bản ghi kế thừa `Record`: có `schema_version` và cấm trường lạ, để một
trường như `label` không thể lọt vào đối tượng runtime mà không bị phát hiện.
`ClaimInput` và `GoldInfo` luôn nằm ở hai file khác nhau.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from sciverx import SCHEMA_VERSION

Split = Literal["validation", "test"]
TableCondition = Literal["G0", "G1", "E0", "E1"]
VerifierTableCondition = Literal["IMG", "G0", "G1", "E0", "E1"]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = SCHEMA_VERSION


# ---------------------------------------------------------------- claim & gold

class ClaimInput(Record):
    claim_id: str
    split: Split
    paper_id: str
    text: str


class GoldInfo(Record):
    """File gold.jsonl; chỉ data/select.py ghi, chỉ evaluation/ đọc đầy đủ."""
    claim_id: str
    label: bool
    claim_type: str
    section_ids: list[str]
    visual_items: list[dict]  # [{item_id, item_type, image_path}]


class OracleEvidenceLocator(Record):
    """Dẫn xuất từ GoldInfo, KHÔNG có label. Dùng cho S-oracle."""
    claim_id: str
    claim_type: str
    section_ids: list[str]
    visual_items: list[dict]

    @classmethod
    def from_gold(cls, gold: GoldInfo) -> "OracleEvidenceLocator":
        return cls(claim_id=gold.claim_id, claim_type=gold.claim_type,
                   section_ids=list(gold.section_ids),
                   visual_items=[dict(v) for v in gold.visual_items])


# ---------------------------------------------------------------- bảng

class SourceCell(Record):
    """Ô theo cấu trúc gốc (còn span). cell_id = "r{row}c{col}" của góc trên trái."""
    cell_id: str
    row: int
    col: int
    rowspan: int = Field(default=1, ge=1)
    colspan: int = Field(default=1, ge=1)
    is_header: bool
    text_raw: str
    text_norm: str | None = None
    bold: bool | None = None  # None = không xác định được (vd Docling)
    footnote_marks: list[str] = []


class TableRecord(Record):
    table_record_id: str  # f"{paper_id}::{table_id}::{condition}"
    paper_id: str
    table_id: str
    condition: TableCondition
    extractor: str  # gold_json | gold_arxiv_html@v{N} | docling@{version}
    source_version: str | None = None
    caption: str
    cells: list[SourceCell]
    n_header_rows: int
    status: Literal["ok", "no_table_found", "error", "gold_unmatched", "gold_unavailable"]
    normalization_log_id: str | None = None
    footnotes: dict[str, str] = {}
    quality: dict | None = None  # chỉ E*: TEDS, TEDS-Struct, num_prec/rec

    @staticmethod
    def make_id(paper_id: str, table_id: str, condition: str) -> str:
        return f"{paper_id}::{table_id}::{condition}"


class CellMapping(Record):
    paper_id: str
    table_id: str
    e_condition: str
    g_condition: str
    pairs: list[dict]  # {e_cell_id, g_cell_ids: [..], method, score}
    unmatched_e: list[str]
    unmatched_g: list[str]


# ---------------------------------------------------------------- truy xuất

class EvidenceUnit(Record):
    unit_id: str  # f"{paper_id}::{kind}::{parent_id}::{part}"
    paper_id: str
    kind: Literal["paragraph", "table_full", "table_rows", "chart_caption"]
    parent_id: str
    table_condition: TableCondition | None = None
    text: str
    n_wordpieces: int
    char_start: int | None = None
    char_end: int | None = None
    row_range: tuple[int, int] | None = None


class RetrievalCandidate(Record):
    parent_id: str
    best_unit_id: str
    kind: str
    bm25_rank: int | None = None
    bm25_score: float | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    rrf_score: float | None = None
    final_rank: int


class RetrievalResult(Record):
    claim_id: str
    method: Literal["bm25", "dense", "hybrid"]
    table_condition: str
    params: dict  # tokenizer_version, embed_model, rrf_constant, k_units
    n_units_in_pool: int
    n_parents_in_pool: int
    candidates: list[RetrievalCandidate]  # mức cha, đã loại trùng
    pool_hash: str


# ---------------------------------------------------------------- căn chỉnh & số liệu

class CellRef(Record):
    table_record_id: str
    condition: str
    cell_id: str
    row_header: str  # lấy từ TableRecord, không lấy từ model
    col_header: str
    text_raw: str
    text_norm: str | None = None


class SpanRef(Record):
    unit_id: str
    section_id: str
    char_start: int
    char_end: int
    text: str


class Operand(Record):
    source: Literal["evidence_cell", "evidence_span", "claim"]
    ref: CellRef | SpanRef | None = None  # None khi source == "claim"
    value_text: str
    value: str | None = None  # Decimal dạng chuỗi
    unit: str | None = None
    scale: str | None = None


class TimeSpec(Record):
    kind: Literal["year", "quarter", "range", "point", "relative", "none"]
    start: str | None = None
    end: str | None = None
    text: str


Relation = Literal["value_equals", "percent_value", "relative_change", "percentage_point_diff",
                   "comparison", "extremum", "universal", "ratio_multiple", "average", "sum",
                   "temporal"]


class NumericCheck(Record):
    check_id: str
    relation: Relation
    claim_operands: list[Operand]  # giá trị NÊU TRONG CLAIM (cần kiểm)
    evidence_operands: list[Operand]  # giá trị LẤY TỪ BẰNG CHỨNG
    comparison_set: list[CellRef] = []
    comparison_set_status: Literal["complete", "partial", "unknown", "not_applicable"]
    unit: str | None = None
    time: TimeSpec | None = None
    direction: Literal["higher_better", "lower_better", "unknown"] | None = None
    quantity_kind: Literal["speed", "time", "throughput", "score", "count", "other",
                           "unknown"] | None = None
    qualifier: Literal["exact", "approximately", "at_least", "at_most", "more_than",
                       "less_than", "none"]
    precision_decimals: int | None = None
    tolerance_rule: str | None = None
    decided_by: Literal["rule", "llm"]
    computed: dict | None = None  # chỉ khi m5.calculator = true


class AlignedEvidence(Record):
    evidence_id: str
    type: Literal["table", "paragraph", "chart"]
    cells: list[CellRef] = []
    spans: list[SpanRef] = []
    from_candidate_rank: int


AlignmentStatus = Literal["matched", "partial", "ambiguous", "not_found", "parse_error", "skipped"]


class EvidencePackage(Record):
    claim_id: str
    claim_text: str
    paper_id: str
    evidence_setting: Literal["S-retr", "S-align4", "S-align45"]
    table_condition: TableCondition
    retrieval: RetrievalResult
    evidence: list[AlignedEvidence]
    alignment_status: AlignmentStatus
    alignment_candidates: list[AlignedEvidence] = []
    numeric_checks: list[NumericCheck] = []
    missing_information: list[str]
    budget: dict  # tokens_used, images_used, dropped_parents, truncated
    errors: list[str]
    provenance: dict  # model, revision, prompt_id, prompt_hash, config_hash, run_id, versions


# ---------------------------------------------------------------- verifier

class VerifierRequest(Record):
    claim_id: str
    prompt_id: str
    text: str
    image_sha256: list[str]
    request_hash: str


class VerifierOutput(Record):
    claim_id: str
    run_id: str
    system: Literal["B0", "P0", "P1"]
    evidence_setting: Literal["S-oracle", "S-retr", "S-align4", "S-align45"]
    table_condition: VerifierTableCondition
    prompt_id: str
    request_hash: str
    sample_name: str
    temperature: float
    seed: int
    response: str
    finish_reason: str
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    n_images: int
    pred_old: bool | None = None
    pred_new: bool | None = None
    error: str | None = None


class RunManifest(Record):
    """outputs/runs/<run_id>/manifest.json."""
    run_id: str
    created_at: str
    git_commit: str | None
    config: dict
    config_hash: str
    claim_ids: list[str]
    claim_ids_hash: str
    versions: dict[str, str | None]
    model: str | None = None
    model_revision: str | None = None
    gpu: dict | None = None
    seeds: list[int] = []
    duration_s: float | None = None
    error_counts: dict[str, int] = {}
