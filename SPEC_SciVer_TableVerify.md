# SPEC — Pipeline kiểm chứng claim có bảng trên SciVer (mục 4.3)

> **Dành cho agent thực thi.** Đọc toàn bộ file trước khi viết dòng code đầu tiên.
> **[DỪNG & HỎI]** = bắt buộc dừng, báo cáo cho người phụ trách và chờ trả lời.
> **[CẦN KIỂM]** = giả định chưa xác minh trên dữ liệu thật; kiểm và ghi kết quả vào `docs/VERIFICATION_LOG.md`.

Phiên bản: **1.1** · Ngày: 2026-11-12 · Chủ dự án: nhóm WorkplaceFactBench
Mã nguồn khởi đầu: `sciver_pilot.zip` (đã kiểm thử trên dữ liệu giả). Tái sử dụng, nhưng sửa theo đặc tả này ở những chỗ khác nhau, ví dụ TEDS, chuẩn hóa và schema.

### Thay đổi so với 1.0
| # | Thay đổi |
|---|---|
| 1 | Bỏ mốc pilot 40 claim: mọi chạy chính thức dùng **toàn bộ** claim đã lọc. Tách **Mức A** (triển khai và bàn giao module, trên validation) và **Mức B** (thực nghiệm đầy đủ và báo cáo test). |
| 2 | Bảo vệ test: GĐ1 chỉ audit validation; test được audit bằng cùng mã ở GĐ B4. |
| 3 | Quyền đọc dữ liệu được quy định theo **chức năng**; tách `ClaimInput` và `GoldInfo` thành hai file; nhãn kết luận không bao giờ vào model. |
| 4 | Kho G và E dùng **cùng danh sách nguồn** (mọi bảng của bài); có chính sách cho bảng không dựng được bảng chuẩn; so sánh ghép cặp chỉ trên tập đủ cặp. |
| 5 | Luật chuẩn hóa có điều kiện áp dụng; thông tin có nghĩa được giữ trong đầu vào của verifier, không chỉ nằm trong log; bảng chuẩn ghép theo **phiên bản** bài. |
| 6 | Schema ô: `cell_id`, `table_record_id`, cấu trúc span, ánh xạ ô E↔G; toán hạng có nguồn; `numeric_checks` là danh sách; tập so sánh; thời gian và đơn vị có cấu trúc. |
| 7 | Chỉ số truy xuất: Hit@k, Recall@k, VisualComplete@k, MRR. |
| 8 | Bootstrap theo cụm bài báo; phân tích nguyên nhân cho chọn nhiều, có mức chắc chắn. |
| 9 | Tách đóng góp của M4 và M5 (ba điều kiện); S-oracle không còn được gọi là "mức trần". |
| 10 | Danh sách prompt được phép; ngân sách token và ảnh; `parse_error` tách khỏi `not_found`. |
| 11 | B0 chạy trên toàn validation và trên tập đã lọc; ±3 điểm chỉ là tín hiệu; cấu hình GPU phải thử thực tế. |
| 12 | Kiểm thử chống rò rỉ bằng hash request và mock model; TEDS tính trên cấu trúc có span, kiểm bằng cặp bảng có kết quả biết trước. |

---

## 0. Tóm tắt

1. **Bài toán:** cho claim khoa học và bài báo nguồn, tìm bằng chứng (đoạn văn, bảng), căn chỉnh tới đúng ô hoặc đoạn, trích thông tin số, đóng gói cho bộ kiểm chứng kết luận Entailed/Refuted.
2. **Dữ liệu:** SciVer (ACL 2025), **toàn bộ** claim thỏa tiêu chí ở validation (1.000 claim gốc) và test (2.000 claim gốc). Tiêu chí: bằng chứng thị giác là bảng, không cần biểu đồ.
3. **Model:** `Qwen/Qwen2.5-VL-7B-Instruct` qua vLLM, chưa fine-tune.
4. **Module:** M1 chuẩn bị và lập chỉ mục · M2 trích atomic claim (để sau) · M3 Hybrid Search · M4 căn chỉnh · M5 số liệu · M6 đóng gói.
5. **Ba trục thực nghiệm:** hệ thống (B0/P0/P1) × thiết lập bằng chứng (S-oracle/S-retr/S-align4/S-align45) × dạng bảng (IMG/G0/G1/E0/E1).
6. **Validation = tập phát triển.** **Test = khóa**, chỉ chạy một lần với cấu hình đã đóng băng.
7. Quyền đọc dữ liệu chuẩn (gold) quy định theo chức năng (mục 5.5). Ở runtime S-retr/S-align, module chỉ nhận claim, ID bài và kho tài liệu.
8. Truy xuất và kết luận được đánh giá **riêng**. `alignment_status = matched` **không** có nghĩa claim đúng.
9. Mọi lượt chạy tái lập được: cấu hình, seed, phiên bản, danh sách ID và dữ liệu trung gian đều được lưu.
10. **Mức A** kết thúc bằng bàn giao module chạy được trên toàn validation. **Mức B** là thực nghiệm đầy đủ. Agent không chuyển sang Mức B khi chưa được duyệt.

---

## 1. Mục tiêu và phạm vi

### 1.1 Câu hỏi nghiên cứu
- **RQ1 (dạng bảng):** Cùng claim, cùng model, cùng vị trí bằng chứng chuẩn: đưa bảng dưới dạng ảnh (IMG), bảng chuẩn văn bản (G0/G1) và bảng trích tự động (E0/E1) làm kết luận thay đổi thế nào? Lỗi trích xuất nào **liên quan** đến việc đổi kết luận?
- **RQ2 (truy xuất):** Cùng kho, cùng model, cùng ngân sách đầu vào: BM25, vector và hybrid tìm bằng chứng tốt đến đâu, và kết luận thay đổi ra sao?
- **RQ3 (căn chỉnh và số liệu):** So với chỉ đưa kết quả truy xuất, thêm M4 thay đổi gì, và thêm M4+M5 thay đổi gì?

### 1.2 Hai mức phạm vi
| Mức | Nội dung | Kết thúc khi |
|---|---|---|
| **A — Triển khai và bàn giao** | GĐ A0–A6: kiểm dữ liệu validation → M1 → B0 → M3 → M4 → M5 → M6, chạy trên **toàn validation đã lọc**, kèm chỉ số từng module | Bàn giao theo mục 14.1. **[DỪNG & HỎI]** trước khi sang Mức B |
| **B — Thực nghiệm đầy đủ** | GĐ B1–B4: các thí nghiệm X1–X4 trên validation, gán nhãn ô, đóng băng cấu hình, chạy test, thống kê và báo cáo | Bàn giao theo mục 14.2 |

Không có pilot theo mẫu nhỏ. Được phép chạy **smoke test** trên ≤ 20 claim validation để gỡ lỗi code. Kết quả smoke test không được báo cáo như kết quả thực nghiệm.

### 1.3 Ngoài phạm vi (chỉ làm khi có yêu cầu)
M2; PoT verifier; embedding thứ hai; reranker; kho nhiều bài; Qwen2.5-VL-72B; P1 (fine-tune); biểu đồ làm bằng chứng cần đọc số.

### 1.4 Nguồn tham chiếu
- Paper SciVer: https://aclanthology.org/2025.acl-long.420 (arXiv 2506.15569)
- Code gốc: https://github.com/QDRhhhh/SciVer (`utils/input_processing.py`, `utils/constant.py`, `utils/vllm_input_preparation.py`, `model_inference/vllm_inference.py`, `acc_evaluation.py`)
- Dữ liệu: https://huggingface.co/datasets/chengyewang/SciVer
- TEDS: Zhong et al. 2020 (PubTabNet) · Docling: https://github.com/docling-project/docling
- `sentence-transformers/all-mpnet-base-v2`: model card ghi mặc định cắt đầu vào dài hơn 384 word pieces
- `rank_bm25`: không tự tiền xử lý, nhận danh sách token do người dùng cung cấp

---

## 2. Quy tắc bắt buộc cho agent

1. **Test:** trước GĐ B4 không chạy model trên test, không đọc giá trị nhãn, không thống kê phân bố test. Ngoại lệ duy nhất xem mục 5.4.
2. **Quyền đọc dữ liệu chuẩn** theo bảng mục 5.5. Kiểm bằng test tự động (mục 12.4).
3. **Nhãn kết luận** (`label`) không bao giờ xuất hiện trong bất kỳ đầu vào nào của model, ở mọi thiết lập.
4. **Không bịa:** dữ liệu thiếu thì ghi `missing_information` hoặc trạng thái lỗi.
5. **Không cắt âm thầm:** mọi việc cắt bớt (token, hàng, ứng viên, ảnh) phải ghi log kèm số lượng.
6. **Prompt:** chỉ dùng các biến thể trong danh sách mục 8.3.
7. **Mọi khác biệt với code gốc** ghi vào `docs/DEVIATIONS.md`.
8. Dữ liệu thật khác giả định → **[DỪNG & HỎI]**, kèm bản in cấu trúc thật.
9. Không xóa dữ liệu trung gian; không ghi đè lượt chạy (mỗi lượt có `run_id` riêng).
10. Commit nhỏ, thông điệp rõ; không commit dữ liệu, model hay khóa API.
11. Khi viết báo cáo: chưa có thí nghiệm can thiệp thì dùng "liên quan đến", không dùng "gây ra".

---

## 3. Môi trường và phần cứng

### 3.1 Phần cứng
| Thành phần | Yêu cầu |
|---|---|
| Verifier, M4 (vLLM, Qwen2.5-VL-7B) | GPU NVIDIA; khuyến nghị A100 40/80 GB hoặc L40S 48 GB. GPU 24 GB **chưa chắc** đủ: phải thử trên các mẫu có prompt dài nhất và nhiều ảnh nhất, rồi ghi `max_model_len` và mức dùng bộ nhớ thực tế vào `VERIFICATION_LOG.md` |
| Docling | CPU đủ; GPU nhanh hơn |
| Embedding | CPU được; GPU nhanh hơn |
| Đĩa | ≥ 60 GB |

### 3.2 Hai môi trường Python (tránh xung đột phiên bản torch)
| Tên | Python | Dùng cho | Gói chính |
|---|---|---|---|
| `sciver` | 3.10 | M3–M6, verifier, đánh giá | `vllm==0.9.0.1` (như repo SciVer), `sentence-transformers`, `rank_bm25`, `pydantic>=2`, `numpy`, `pandas`, `scipy`, `pillow`, `beautifulsoup4`, `lxml`, `apted`, `requests`, `pyyaml`, `pytest` |
| `docling` | 3.11 | Trích bảng từ ảnh | `docling`, `beautifulsoup4`, `lxml`, `pydantic>=2`, `pyyaml` |

Ghim phiên bản thật bằng `pip freeze` vào `requirements/*.lock.txt`. Ghi phiên bản Docling, vLLM, torch, CUDA, transformers và sentence-transformers vào manifest của mỗi lượt chạy.

### 3.3 Tải tài nguyên
```bash
git lfs install
git clone https://huggingface.co/datasets/chengyewang/SciVer data/raw/SciVer
git clone https://github.com/QDRhhhh/SciVer third_party/SciVer
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"
```
Ghi commit hash của dataset và repo gốc vào `VERIFICATION_LOG.md`.

---

## 4. Cấu trúc repo

```
sciver-tableverify/
├── pyproject.toml                  # package "sciverx", CLI: python -m sciverx <stage>
├── README.md
├── requirements/{sciver,docling}.txt + *.lock.txt
├── configs/ base.yaml · experiments/*.yaml · frozen/*.yaml
├── src/sciverx/
│   ├── schemas.py                  # mục 6
│   ├── data/          loader.py, resolver.py, audit.py, select.py (chỉ module này đọc annotation để lọc)
│   ├── tables/        html_cells.py, grid.py, normalize.py, linearize.py, teds.py, cell_mapping.py, numeric.py
│   ├── m1_prepare/    sources.py, gold_tables.py, extract_docling.py, units.py, index_bm25.py, index_dense.py
│   ├── m3_retrieve/   tokenizer.py, bm25.py, dense.py, rrf.py, budget.py, retrieve.py
│   ├── m4_align/      prompts.py, align.py, validators.py
│   ├── m5_numeric/    parse.py, relations.py, tolerance.py, calculator.py
│   ├── m6_package/    package.py
│   ├── verifier/      prompts.py, build_inputs.py, budget.py, run_vllm.py, scoring.py, request_hash.py
│   ├── evaluation/    gold_access.py (chỉ module này đọc GoldInfo ở runtime đánh giá), retrieval_metrics.py,
│   │                  align_metrics.py, numeric_metrics.py, verdict_metrics.py, stats.py, report.py
│   ├── annotation/    export_tasks.py, import_labels.py, agreement.py
│   └── utils/         io.py, logging.py, manifest.py, hashing.py, seed.py
├── scripts/  ·  tests/ (fixtures/, mocks/)
├── docs/ VERIFICATION_LOG.md, DEVIATIONS.md, DATA_CARD.md, EXPERIMENTS.md, ANNOTATION_GUIDE.md, reports/
├── data/ (gitignore) raw/, interim/, processed/
└── outputs/ (gitignore) runs/<run_id>/
```

Ánh xạ từ `sciver_pilot.zip`: `common.py` → `data/`; `tables.py` → `tables/` (**sửa**: giữ cấu trúc span và luật chuẩn hóa mới); `teds.py` → **viết lại** theo mục 7-M1.5; `prompts.py` → `verifier/prompts.py`; `s03`, `s04` → `m1_prepare/`; `s06` → `verifier/`; `s07` → `evaluation/`.

---

## 5. Dữ liệu

### 5.1 Cấu trúc giả định theo code gốc — **[CẦN KIỂM]**
- **Query:** `claim`, `claim_type ∈ {direct, parallel, sequential, analytical}`, `label`, `paper_path`, `section` (danh sách ID section chứa bằng chứng văn bản). Ngoài ra:
  - direct/analytical: `type`, `item`, `image_path`
  - parallel/sequential: `item1_type`, `item1`, `item1_path`, `item2_type`, `item2`, `item2_path`
- **Paper JSON:** `sections[{section_id, section_name, text}]`, `tables[id]{capture, …}`, `image_paths[id]{caption, …}`.

### 5.2 Audit dữ liệu (GĐ A1 — **chỉ validation**)
Viết `data/audit.py` một lần, dùng lại nguyên vẹn cho test ở GĐ B4.

| # | Nội dung | Vì sao |
|---|---|---|
| D1 | Tên file split; số claim validation | Kế hoạch |
| D2 | Tên trường thật có khớp mục 5.1 không | Khác → **[DỪNG & HỎI]** |
| D3 | `tables[id]` chứa gì (HTML/text/ô hay chỉ caption) | Nguồn bảng chuẩn |
| D4 | Paper JSON có **mọi** bảng và biểu đồ của bài, kèm ảnh, hay chỉ các item được tuyển chọn? Đếm số bảng/bài trong JSON và so với HTML arXiv của bài | Quyết định có làm S-retr/S-align được không → **[DỪNG & HỎI]** |
| D5 | `sections` là toàn văn hay phần trích; độ chi tiết của `section` (mục / tiểu mục / đoạn) | Mức chấm truy xuất văn bản |
| D6 | Dạng ID item (LaTeXML như `S4.T2`?); có arXiv ID và **phiên bản** (vN) cho mỗi bài không | Ghép bảng chuẩn đúng phiên bản |
| D7 | Số claim validation thỏa tiêu chí, theo (claim_type, label) | Kích thước thực nghiệm |
| D8 | Số bảng duy nhất trên validation; số claim dùng chung bảng; số bài | Lịch trích bảng, bootstrap theo cụm |

### 5.3 Tiêu chí chọn claim (cố định trước khi chạy)
- **Mặc định:** claim có ≥ 1 bằng chứng thị giác và **mọi bằng chứng thị giác là bảng**; giữ nguyên bằng chứng văn bản.
- **Tùy chọn `allow_mixed=true`:** giữ claim có cả bảng và biểu đồ. Biểu đồ vào kho dưới dạng caption, vào verifier dưới dạng ảnh. Báo kết quả riêng.
- Chọn **toàn bộ** claim thỏa tiêu chí trong split. ID ổn định: `"{split}-{qid}"`.
- `data/select.py` là module **duy nhất** đọc annotation để lọc. Đầu ra:
  - `data/processed/{split}/claims_input.jsonl` (`ClaimInput`)
  - `data/processed/{split}/gold.jsonl` (`GoldInfo`, file riêng)
  - `ids.txt` + hash
- Báo phân bố thực tế. Sau khi lọc, nhãn có thể không còn 50/50.

### 5.4 Vai trò các split
- **validation:** phát triển, chỉnh tham số, phân tích lỗi.
- **test:** GĐ B4 chạy `audit.py` và `select.py` (mã đã đóng băng), rồi các thí nghiệm đã đăng ký trước, với `configs/frozen/` (lưu hash). Mỗi cấu hình chạy đúng một lần.
- **Ngoại lệ trước B4:** chỉ được chạy `audit.py --structure-only` trên test để kiểm **tên trường và kiểu dữ liệu**. Không in giá trị nhãn, không đếm theo nhãn hay claim_type. Chạy ngoại lệ này cần **[DỪNG & HỎI]** để chỉ định người chạy, và người đó chỉ chuyển cho nhóm phát triển kết luận "khớp / không khớp schema".
- P1 (nếu được duyệt): chỉ huấn luyện trên validation (tách train/dev nội bộ, seed cố định) hoặc dữ liệu ngoài.

### 5.5 Quyền đọc dữ liệu theo chức năng
| Thành phần | Được đọc | Không được đọc |
|---|---|---|
| Chuẩn bị tập đánh giá (`data/select.py`) | Annotation gốc, để lọc và tạo `ClaimInput` + `GoldInfo` | — |
| M1 (kho tài liệu) | Toàn bộ nội dung khả dụng của bài: sections, mọi bảng, mọi biểu đồ, kèm đường dẫn ảnh **của từng item trong kho** | Bất kỳ trường nào cấp claim (`section`, `item*`, `image_path`, `item*_path`, `label`, `claim_type`) |
| Runtime S-retr / S-align (M3–M6, verifier) | `ClaimInput` (claim_id, text, paper_id) + kho M1 | `GoldInfo` |
| Runtime S-oracle (verifier) | `ClaimInput` + **vị trí bằng chứng** từ `GoldInfo` (section, item, đường dẫn ảnh); `claim_type` chỉ để chọn prompt gốc như code SciVer | `label` |
| Đánh giá (`evaluation/`) | `GoldInfo` đầy đủ | — |

Đọc đường dẫn ảnh của một item **trong kho** để tải ảnh là hợp lệ. Điều bị cấm là dùng đường dẫn hoặc ID được chỉ định trong annotation của claim để chọn trước bằng chứng.

---

## 6. Schema dữ liệu (Pydantic v2, `schemas.py`)

Mọi file trung gian là JSONL, có `schema_version: "1.1"`. `ClaimInput` và `GoldInfo` nằm ở **hai file khác nhau**. Không model runtime nào nhận `GoldInfo`, trừ `verifier/build_inputs.py` ở S-oracle; module này chỉ nhận đối tượng `OracleEvidenceLocator` không có trường `label`.

```python
class ClaimInput(BaseModel):
    claim_id: str; split: Literal["validation", "test"]; paper_id: str; text: str

class GoldInfo(BaseModel):                       # file gold.jsonl, chỉ data/select.py ghi
    claim_id: str; label: bool; claim_type: str
    section_ids: list[str]
    visual_items: list[dict]                     # [{item_id, item_type, image_path}]

class OracleEvidenceLocator(BaseModel):          # dẫn xuất từ GoldInfo, KHÔNG có label
    claim_id: str; claim_type: str; section_ids: list[str]; visual_items: list[dict]

class SourceCell(BaseModel):                     # ô theo cấu trúc gốc (còn span)
    cell_id: str                                 # ổn định trong một TableRecord: "r{row}c{col}" của góc trên trái
    row: int; col: int; rowspan: int = 1; colspan: int = 1
    is_header: bool
    text_raw: str
    text_norm: str | None = None
    bold: bool | None = None                     # None = không xác định được (vd Docling)
    footnote_marks: list[str] = []

class TableRecord(BaseModel):
    table_record_id: str                         # f"{paper_id}::{table_id}::{condition}"
    paper_id: str; table_id: str
    condition: Literal["G0", "G1", "E0", "E1"]
    extractor: str                               # gold_json | gold_arxiv_html@v{N} | docling@{version}
    source_version: str | None                   # phiên bản arXiv dùng cho bảng chuẩn
    caption: str
    cells: list[SourceCell]                      # nguồn sự thật; grid suy ra từ cells
    n_header_rows: int
    status: Literal["ok", "no_table_found", "error", "gold_unmatched", "gold_unavailable"]
    normalization_log_id: str | None
    footnotes: dict[str, str] = {}               # ký hiệu → nội dung chú thích (nếu có)
    quality: dict | None = None                  # chỉ E*: TEDS, TEDS-Struct, num_prec/rec

class CellMapping(BaseModel):                    # ánh xạ ô E ↔ G cho cùng (paper, table)
    paper_id: str; table_id: str; e_condition: str; g_condition: str
    pairs: list[dict]                            # {e_cell_id, g_cell_ids: [..], method, score}
    unmatched_e: list[str]; unmatched_g: list[str]

class EvidenceUnit(BaseModel):
    unit_id: str                                 # f"{paper_id}::{kind}::{parent_id}::{part}"
    paper_id: str
    kind: Literal["paragraph", "table_full", "table_rows", "chart_caption"]
    parent_id: str                               # section_id / table_id / chart_id
    table_condition: Literal["G1", "E1", "G0", "E0"] | None
    text: str; n_wordpieces: int
    char_start: int | None; char_end: int | None # với paragraph: vị trí trong section text
    row_range: tuple[int, int] | None

class RetrievalCandidate(BaseModel):
    parent_id: str; best_unit_id: str; kind: str
    bm25_rank: int | None; bm25_score: float | None
    dense_rank: int | None; dense_score: float | None
    rrf_score: float | None; final_rank: int

class RetrievalResult(BaseModel):
    claim_id: str; method: Literal["bm25", "dense", "hybrid"]; table_condition: str
    params: dict                                 # tokenizer_version, embed_model, rrf_constant, k_units
    n_units_in_pool: int; n_parents_in_pool: int
    candidates: list[RetrievalCandidate]         # mức cha, đã loại trùng
    pool_hash: str

class CellRef(BaseModel):
    table_record_id: str; condition: str; cell_id: str
    row_header: str; col_header: str             # lấy từ TableRecord, không lấy từ model
    text_raw: str; text_norm: str | None

class SpanRef(BaseModel):
    unit_id: str; section_id: str; char_start: int; char_end: int; text: str

class Operand(BaseModel):
    source: Literal["evidence_cell", "evidence_span", "claim"]
    ref: CellRef | SpanRef | None                # None khi source == "claim"
    value_text: str
    value: str | None                            # Decimal dạng chuỗi
    unit: str | None; scale: str | None

class TimeSpec(BaseModel):
    kind: Literal["year", "quarter", "range", "point", "relative", "none"]
    start: str | None; end: str | None; text: str

class NumericCheck(BaseModel):
    check_id: str
    relation: Literal["value_equals", "percent_value", "relative_change", "percentage_point_diff",
                      "comparison", "extremum", "universal", "ratio_multiple", "average", "sum", "temporal"]
    claim_operands: list[Operand]                # giá trị NÊU TRONG CLAIM (cần kiểm)
    evidence_operands: list[Operand]             # giá trị LẤY TỪ BẰNG CHỨNG
    comparison_set: list[CellRef] = []           # cho extremum / universal / comparison
    comparison_set_status: Literal["complete", "partial", "unknown", "not_applicable"]
    unit: str | None; time: TimeSpec | None
    direction: Literal["higher_better", "lower_better", "unknown"] | None
    quantity_kind: Literal["speed", "time", "throughput", "score", "count", "other", "unknown"] | None
    qualifier: Literal["exact", "approximately", "at_least", "at_most", "more_than", "less_than", "none"]
    precision_decimals: int | None
    tolerance_rule: str | None
    decided_by: Literal["rule", "llm"]
    computed: dict | None = None                 # chỉ khi m5.calculator = true: {value, formula, config_id}

class AlignedEvidence(BaseModel):
    evidence_id: str; type: Literal["table", "paragraph", "chart"]
    cells: list[CellRef] = []; spans: list[SpanRef] = []
    from_candidate_rank: int

class EvidencePackage(BaseModel):
    schema_version: str = "1.1"
    claim_id: str; claim_text: str; paper_id: str
    evidence_setting: Literal["S-retr", "S-align4", "S-align45"]
    table_condition: Literal["G0", "G1", "E0", "E1"]
    retrieval: RetrievalResult
    evidence: list[AlignedEvidence]
    alignment_status: Literal["matched", "partial", "ambiguous", "not_found", "parse_error", "skipped"]
    alignment_candidates: list[AlignedEvidence] = []
    numeric_checks: list[NumericCheck] = []
    missing_information: list[str]
    budget: dict                                 # tokens_used, images_used, dropped_parents, truncated
    errors: list[str]
    provenance: dict                             # model, revision, prompt_id, prompt_hash, config_hash, run_id, versions

class VerifierRequest(BaseModel):                # để hash và kiểm tra rò rỉ
    claim_id: str; prompt_id: str; text: str; image_sha256: list[str]; request_hash: str

class VerifierOutput(BaseModel):
    claim_id: str; run_id: str
    system: Literal["B0", "P0", "P1"]
    evidence_setting: Literal["S-oracle", "S-retr", "S-align4", "S-align45"]
    table_condition: Literal["IMG", "G0", "G1", "E0", "E1"]
    prompt_id: str; request_hash: str
    sample_name: str; temperature: float; seed: int
    response: str; finish_reason: str
    prompt_tokens: int | None; output_tokens: int | None; n_images: int
    pred_old: bool | None; pred_new: bool | None
    error: str | None
```

`RunManifest` (`outputs/runs/<run_id>/manifest.json`): run_id, thời điểm, git commit, toàn bộ cấu hình và hash, danh sách claim_id và hash, phiên bản thư viện, model và revision, GPU và bộ nhớ, seed, thời gian chạy, số lỗi theo loại.

---

## 7. Đặc tả module

### M1 — Chuẩn bị và lập chỉ mục

**M1.1 Kho nguồn của bài (`sources.py`).** Với mỗi bài có trong tập đã lọc, liệt kê **mọi** section, bảng và biểu đồ khả dụng (theo kết quả D4), kèm ảnh của từng item. Đây là danh sách nguồn **chung** cho mọi điều kiện. Ghi `sources_manifest.jsonl` (paper_id, danh sách table_id, chart_id, section_id) và hash.

**M1.2 Bảng chuẩn G0** — cho **mọi bảng trong danh sách nguồn**, không chỉ bảng làm bằng chứng.
1. `tables[id]` có HTML → `extractor=gold_json`.
2. Nếu không → HTML arXiv **đúng phiên bản**: dùng phiên bản ghi trong dataset nếu có; nếu không có, dùng phiên bản mới nhất có ngày ≤ thời điểm thu thập dữ liệu SciVer (theo paper; **[CẦN KIỂM]**). URL dạng `https://arxiv.org/html/<id>v<N>`. Cache lại; nghỉ ≥ 3 giây giữa các request; User-Agent mô tả mục đích nghiên cứu. Ghép theo id LaTeXML, sau đó theo caption. Lưu `source_version` và `match_score`.
3. HTML → `cells` **giữ nguyên rowspan/colspan**, `is_header` (thẻ `<th>`/`<thead>`), `bold` (thẻ `<b>`/`<strong>` hoặc class `ltx_font_bold`), ký hiệu chú thích. Bỏ `<annotation>` của MathML khi lấy chữ.
4. Không ghép được → `status=gold_unmatched`; bài không có HTML → `gold_unavailable`.
5. **Audit:** xuất `annotation/gold_audit.csv` gồm mọi bảng có `match_score < 0.8`, mọi bảng lỗi, và một mẫu ngẫu nhiên phân tầng 100 bảng (seed 2026). **[DỪNG & HỎI]** để người kiểm tay. Báo **tỷ lệ đúng ước lượng** kèm khoảng tin cậy. Ghi rõ: audit ngẫu nhiên chỉ ước lượng chất lượng, không chứng minh mọi bảng còn lại đúng.

**M1.3 Bảng trích E0** (môi trường `docling`) — cho **cùng danh sách bảng** như M1.2. Dùng TableFormer chế độ ACCURATE, bật OCR, `do_cell_matching=False`; lấy bảng lớn nhất. Đọc cấu trúc span từ đầu ra của Docling; nếu Docling không cho biết chữ đậm thì `bold=None`. Không tìm thấy bảng → `no_table_found`, giữ trong kho với nội dung rỗng. Cache theo hash ảnh + phiên bản Docling + cấu hình.

**M1.4 Chuẩn hóa → G1, E1** — **cùng một hàm** cho G và E. Luôn giữ `text_raw`, ghi `text_norm`. Thông tin có nghĩa phải nằm **trong `text_norm` hoặc trong khối bảng đưa vào verifier**, không chỉ trong log.

| Nội dung | Luật |
|---|---|
| Dấu trừ Unicode `−‒–—` đứng ngay trước chữ số | → `-` |
| Ô chỉ gồm `–`, `—`, `-`, `n/a`, `N/A` | → `N/A` |
| Ký hiệu chú thích `* † ‡ § ¶` sau số | **Giữ** dưới dạng tách rời: `93.2 [*]`; đưa nội dung chú thích (nếu tìm thấy trong caption hoặc chú thích bảng) vào `footnotes` và in ở cuối khối bảng |
| Dấu phân cách nghìn | Chỉ đổi khi khớp đầy đủ mẫu `^\d{1,3}(,\d{3})+(\.\d+)?$` **và** cùng cột không có số dùng dấu phẩy thập phân; ngược lại giữ nguyên và ghi `ambiguous_separator` |
| `±`, `+/-`, `+-` | Chuẩn thành `x ± y`, **giữ cả hai giá trị** |
| `↑` / `↓` | Chỉ trong **ô tiêu đề**: giữ ký hiệu và thêm chú giải, ví dụ `Acc ↑ (higher is better)`. Ô không phải tiêu đề: giữ nguyên |
| Khoảng trắng trước `%` | `85.3 %` → `85.3%` |
| Tiêu đề nhiều tầng | Khi linearize, gộp thành `Cấp 1 / Cấp 2`; **không** sửa `cells`; lưu ánh xạ cột → tiêu đề gộp |
| Ô gộp | Giữ trong `cells`; khi linearize, lặp lại nội dung ở mọi vị trí bị phủ; lưu ánh xạ vị trí → `cell_id` |
| Chữ đậm | Mặc định **không** đưa vào văn bản (để G và E công bằng vì `bold` của E thường là `None`). Gắn cờ `format_dependent=true` cho claim chứa các từ chỉ thứ hạng như "best", "highest", "lowest", "outperform", "state-of-the-art", "bold" (danh sách cố định, lưu trong repo). Biến thể `G1b` (có `**x**`) là thí nghiệm phụ, chỉ chạy khi được yêu cầu |

Log: mỗi thay đổi ghi `{table_record_id, cell_id, before, after, rules}` vào `normalize_log.jsonl`.

**M1.5 Chất lượng trích** (E so với G cùng bảng):
- **TEDS và TEDS-Struct** theo định nghĩa PubTabNet, tính trên cây HTML **còn rowspan/colspan** (dựng lại từ `cells`, không dùng grid đã trải). Nút `td` khác nhau về span thì chi phí đổi tên là 1. Nội dung so bằng khoảng cách Levenshtein chuẩn hóa.
- Precision/recall của multiset các số.
- **Ánh xạ ô E↔G** (`tables/cell_mapping.py`): ghép theo (tiêu đề hàng, tiêu đề cột) chuẩn hóa, rồi theo độ giống nội dung kèm ràng buộc thứ tự. Cho phép một–nhiều và không ghép được. Lưu `CellMapping`. Báo tỷ lệ ô G không ghép được.
- Ghi `table_quality.csv`.

**M1.6 Đơn vị tìm kiếm** — hai bộ kho, **G1** và **E1**, dùng **cùng danh sách nguồn**, cùng đơn vị đoạn văn và cùng quy tắc chia. Chỉ phần biểu diễn bảng là khác nhau.
- `paragraph`: tách `sections[].text` theo đoạn (`\n\n`; không có thì gom câu tới giới hạn). Văn bản: `"{section_name}: {đoạn}"`. Lưu `char_start` và `char_end`.
- `table_full`: caption + tiêu đề + mọi hàng, **chỉ khi** ≤ giới hạn.
- `table_rows`: nhóm hàng liên tiếp, **mỗi nhóm lặp lại caption và dòng tiêu đề đã gộp**, ≤ giới hạn. Hàng đơn lẻ vượt giới hạn → chia theo cột, mỗi phần kèm tiêu đề; ghi log.
- `chart_caption`: chỉ khi `allow_mixed`.
- Giới hạn `max_wordpieces = 384`, đo bằng **tokenizer của model embedding**. Không đơn vị nào vượt giới hạn; không cắt âm thầm.
- Bảng `no_table_found` hoặc `gold_unmatched`: vẫn tạo đơn vị caption-only (`table_rows` với 0 hàng) để hai kho có **cùng danh sách parent_id**. Ghi cờ trạng thái.

**M1.7 Chỉ mục** theo từng bài và từng điều kiện (G1, E1): BM25 dùng tokenizer của M3; dense là ma trận embedding đã chuẩn hóa L2 (`.npy`). Cache theo hash nội dung + model. Lưu `pool_hash`.

**Chính sách tập so sánh ghép cặp (bắt buộc):**
- `PAIRED_ORACLE`: claim mà **mọi bảng bằng chứng** có cả G (status ok, đạt audit nếu nằm trong mẫu audit) và E (bất kể trạng thái trích). RQ1 so G↔E **chỉ** trên tập này.
- `PAIRED_RETR`: claim mà **mọi bảng của bài** có G ok. RQ2/RQ3 so G1↔E1 **chỉ** trên tập này.
- Nhánh E (và IMG) được báo **thêm** trên tập rộng hơn, ghi rõ là tập khác và **không** so ghép cặp với G.
- Ghi kích thước từng tập vào `DATA_CARD.md`.

### M2 — Trích atomic claims (ngoài phạm vi Mức A/B)
Chỉ định nghĩa giao diện `extract_claims(draft_text) -> list[ClaimInput]` và prompt nháp. Nguyên tắc: một thông tin kiểm chứng được cho mỗi claim; giữ thực thể, chỉ tiêu, tập dữ liệu, điều kiện; giải đại từ; giữ vị trí câu gốc; không tách tới mức mất nghĩa.

### M3 — Hybrid Search
- **Tokenizer** (dùng chung cho query và tài liệu, vì `rank_bm25` nhận token do mình đưa vào): chuyển chữ thường; regex khởi đầu `[a-z0-9]+(?:[.\-@/][a-z0-9]+)*%?`. Phải giữ nguyên `93.2`, `gpt-4o`, `f1@5`, `top-1`, `12.5%`. Bỏ stopword theo danh sách cố định trong repo; không stemming. Ghi `tokenizer_version`.
- **BM25:** `BM25Okapi` (k1=1.5, b=0.75; ghi lại).
- **Dense:** `all-mpnet-base-v2`, cosine trên vector đã chuẩn hóa.
- **RRF:** `score(u) = Σ_m 1 / (rrf_constant + rank_m(u))`, `rrf_constant = 60` là cấu hình khởi đầu, chỉnh trên validation. Không nhầm với top-k.
- **Gộp theo cha:** điểm cha = điểm cao nhất của các đơn vị con; loại trùng trước khi xếp hạng. Năm nhóm hàng của một bảng chỉ chiếm **một** vị trí.
- **Phạm vi chính:** kho của cùng bài. Kho nhiều bài là mở rộng.
- **Ngân sách đầu vào cho verifier** (`budget.py`), chung cho BM25, dense và hybrid:
  - `max_evidence_tokens` (đếm bằng tokenizer của Qwen, mặc định 6000), `max_parents` (mặc định 3), `max_images` = 2.
  - Lấy cha theo thứ hạng cho tới khi chạm một trong các giới hạn. Một bảng cha vượt `max_evidence_tokens` riêng nó → đưa các nhóm hàng có điểm cao nhất kèm tiêu đề, ghi `truncated`.
  - Ghi `tokens_used`, `images_used`, `dropped_parents`. Báo phân bố token thực tế theo phương pháp, vì cùng `k` không có nghĩa là cùng số token.

### M4 — Căn chỉnh claim với bằng chứng
- **Đầu vào:** `ClaimInput.text` + các cha đã chọn theo ngân sách M3. Với bảng dài: tiêu đề + các nhóm hàng đã truy xuất + chỉ số hàng thật. Áp dụng `max_align_tokens` (mặc định 12000); vượt thì cắt theo thứ hạng và ghi log.
- Gọi Qwen2.5-VL-7B (văn bản), `temperature=0`, prompt `align_v1` yêu cầu JSON: `evidence_id`, `cell_id` (với bảng, dùng ID in sẵn trong bảng đưa vào), đoạn văn (chỉ số câu hoặc offset), và lý do ngắn (thực thể, chỉ tiêu, tập dữ liệu, điều kiện).
- Kiểm định bằng Pydantic. Sai định dạng → thử lại tối đa 2 lần → **`parse_error`** (khác `not_found`).
- **Kiểm bằng code:** evidence_id thuộc ứng viên; `cell_id` tồn tại trong `TableRecord`; `row_header`/`col_header` lấy từ bảng thật.
- Model báo không chắc chắn hoặc chọn nhiều phương án ngang nhau → **`ambiguous`**, lưu `alignment_candidates`.
- Cho phép nhiều bằng chứng. Lưu `prompt_id` và hash prompt.

### M5 — Xử lý số liệu theo claim
- Tách **`claim_operands`** (số nêu trong claim) khỏi **`evidence_operands`** (số lấy từ bằng chứng). Không bao giờ dùng giá trị trong claim làm dữ kiện xác nhận chính nó.
- Một claim có thể có **nhiều** `NumericCheck`.
- Parse bằng regex + `decimal.Decimal`; xác định đơn vị, thang đo, `qualifier`, `precision_decimals`, `TimeSpec`.
- Phân biệt bắt buộc: `percentage_point_diff` khác `relative_change` (80% → 90%: +10 điểm %, +12,5% tương đối).
- `extremum` / `universal` / `comparison`: điền `comparison_set` và `comparison_set_status`. Chỉ ghi `complete` khi đã bao đủ mọi hàng/cột thuộc phạm vi claim nêu (ví dụ "mọi dataset trong bảng").
- `ratio_multiple`: xác định `quantity_kind` (tốc độ/throughput hay thời gian; tỷ số đảo chiều). Không rõ → `unknown` + `missing_information`.
- **Dung sai:** chỉ ghi `tolerance_rule` dựa trên câu chữ (`qualifier`), độ chính xác của nguồn và quy ước làm tròn. Không áp ngưỡng cứng chung cho mọi claim.
- Luật Python trước; LLM chỉ khi luật không quyết được (`decided_by`).
- `computed` chỉ có khi `m5.calculator=true`, là cấu hình thực nghiệm riêng.

### M6 — Đóng gói
Ghép thành `EvidencePackage` và kiểm định. Đưa `numeric_checks` vào verifier dưới dạng văn bản mô tả (toán hạng, nguồn, phép tính), **không** đưa kết luận. Báo: % hợp lệ schema, % truy được về nguồn (mọi `CellRef` và `SpanRef` tồn tại), phân bố `alignment_status`.

---

## 8. Bộ kiểm chứng

### 8.1 Model và sinh
`Qwen/Qwen2.5-VL-7B-Instruct`, vLLM 0.9.0.1, `limit_mm_per_prompt={"image": 2}`, `max_model_len=32768` (giảm nếu thử thực tế không đủ bộ nhớ, ghi lại), `max_tokens=10240` như paper.

Bốn lượt sinh: `t1_s1`, `t1_s2`, `t1_s3` (temperature 1.0, seed 1/2/3, truyền vào `SamplingParams`) và `t0` (temperature 0).

### 8.2 Dựng đầu vào
| Thiết lập | Bằng chứng | Ngữ cảnh văn bản | Prompt |
|---|---|---|---|
| S-oracle | Item từ `OracleEvidenceLocator` (như code gốc) | `prepare_context(paper, section_ids)` như code gốc | 4 prompt gốc theo `claim_type`; thêm `unified_v1` làm cầu nối |
| S-retr | Các cha theo ngân sách M3 (chỉ truy xuất) | Đoạn văn được truy xuất | `unified_v1` |
| S-align4 | Bằng chứng sau M4 (ô và đoạn đã chọn, kèm bảng cha theo ngân sách) | Đoạn văn đã căn chỉnh | `unified_v1` |
| S-align45 | Như S-align4 + mô tả `numeric_checks` từ M5 | Như trên | `unified_v1_numeric` |

- Dạng bảng: IMG → ảnh, tiền xử lý giống code gốc (RGB, phóng ảnh có cạnh < 224, JPEG). G\*/E\* → linearize dạng `" | "`, mỗi hàng một dòng, chèn dưới nhãn "Table content (as text):", kèm chú thích bảng (`footnotes`).
- Ở S-oracle với IMG chỉ dùng ảnh bằng chứng, nên tối đa 2 ảnh. Ở các thiết lập truy xuất, bảng là văn bản nên không dùng ảnh; nếu bật `allow_mixed` thì áp dụng `max_images=2` theo thứ hạng và ghi các ảnh bị bỏ.
- Trước khi gửi, mỗi request được ghi thành `VerifierRequest` kèm `request_hash` = SHA-256 của (prompt_id, văn bản, danh sách sha256 ảnh).

### 8.3 Danh sách prompt được phép (đầy đủ)
| prompt_id | Nội dung | Dùng ở |
|---|---|---|
| `sciver_p1..p4` | Nguyên văn `utils/constant.py` | S-oracle (IMG) |
| `sciver_p1..p4_text` | Như trên + chỉnh `TEXT_MODE_EDITS` ("image" → "table") + khối bảng văn bản | S-oracle (G\*/E\*) |
| `unified_v1` | Một prompt dùng chung, không phụ thuộc `claim_type`: dựa trên prompt 1 của SciVer, đổi "the image" thành "the provided evidence (tables and/or text)", giữ yêu cầu hoài nghi và câu kết `Answer: $ANSWER`. Bản nháp do agent viết, **[DỪNG & HỎI]** để duyệt, sau đó đóng băng | S-oracle cầu nối, S-retr, S-align4 |
| `unified_v1_numeric` | `unified_v1` + khối "Numeric checks to consider:" liệt kê toán hạng có nguồn và phép tính (không có kết quả, không có kết luận) | S-align45 |
| `align_v1` | Prompt M4 | M4 |

Không prompt nào khác được dùng. Mọi prompt có hash trong `DEVIATIONS.md`.

### 8.4 Chấm điểm
- `pred_old`: `'yes' in response.lower()` (như `acc_evaluation.py`).
- `pred_new`: lần xuất hiện **cuối** của `Answer: yes|no` (cho phép `**`, ngoặc, khoảng trắng); dự phòng bằng `final answer is yes|no`; không đọc được → `None`, tính là sai và báo tỷ lệ riêng.
- Báo cả hai; kết luận chính dùng `pred_new`.

### 8.5 B0 — tái lập code gốc
- **B0-full:** chạy `third_party/SciVer` nguyên trạng trên **toàn validation**. Chấm bằng `acc_evaluation.py` gốc và bằng `pred_new`. Đối chiếu với số paper (Qwen2.5-VL-7B validation 53,5).
- **B0-filtered:** cùng đầu ra, tính lại trên tập validation đã lọc. Đây là **mốc** cho các thí nghiệm của nhóm, không so với số paper.
- Lệch > 3 điểm ở B0-full chỉ là **tín hiệu** để kiểm tra cấu hình (phiên bản, prompt, ảnh, cách chấm). Không phải tiêu chuẩn bảo đảm tái lập. Ghi lại và **[DỪNG & HỎI]**.

---

## 9. Ma trận thực nghiệm

Trục: **Hệ thống** (B0 | P0 | P1) × **Thiết lập** (S-oracle | S-retr | S-align4 | S-align45) × **Dạng bảng** (IMG | G0 | G1 | E0 | E1).

| ID | Mức | Hệ thống | Thiết lập | Bảng | Prompt | Lượt sinh | Tập claim | Trả lời |
|---|---|---|---|---|---|---|---|---|
| X0 | A | B0 | gốc | IMG | gốc | 1 (như code gốc) | full + filtered | Tái lập |
| X1 | B | P0 | S-oracle | IMG, G0, G1, E0, E1 | sciver_p* | 4 | PAIRED_ORACLE (+ IMG/E trên tập rộng, báo riêng) | RQ1 |
| X1u | B | P0 | S-oracle | IMG, G1, E1 | unified_v1 | 4 | PAIRED_ORACLE | Cầu nối prompt |
| X2 | A | — | truy xuất | kho G1, kho E1 | — | — | toàn tập lọc; so G↔E trên PAIRED_RETR | RQ2 (chỉ số truy xuất) |
| X3 | B | P0 | S-retr | G1, E1 | unified_v1 | hybrid: 4; BM25/dense: t0 | PAIRED_RETR | RQ2 (kết luận) |
| X4a | B | P0 | S-align4 | G1, E1 | unified_v1 | 4 | PAIRED_RETR | RQ3: M4 |
| X4b | B | P0 | S-align45 | G1, E1 | unified_v1_numeric | 4 | PAIRED_RETR | RQ3: M4+M5 |
| X4c | B (tùy chọn) | P0 | S-align45 + calculator | G1, E1 | unified_v1_numeric | t0 | PAIRED_RETR | RQ3 phụ |
| X6 | — | P1 | … | … | … | … | … | Chỉ khi được duyệt |

So sánh chính (ghép cặp trên cùng claim):
- **RQ1:** IMG↔G1, G0↔E0, G1↔E1, G0↔G1, E0↔E1, IMG↔E1 (trên PAIRED_ORACLE).
- **Cầu nối:** X1 ↔ X1u cùng dạng bảng.
- **RQ2:** BM25 ↔ dense ↔ hybrid (cùng kho, cùng ngân sách); kho G1 ↔ kho E1.
- **RQ3:** X3 ↔ X4a (đóng góp của M4); X4a ↔ X4b (đóng góp thêm của M5); X3 ↔ X4b (tổng). Nếu ngân sách chỉ đủ hai điều kiện, chạy X3 và X4b, rồi ghi RQ3 là **tác động kết hợp** của M4 và M5.

Diễn giải:
- S-oracle là **đối chứng có sẵn vị trí bằng chứng chuẩn**, không phải mức trần: model vẫn có thể suy luận sai hoặc bị ngữ cảnh làm nhiễu, và kết quả truy xuất có thể cao hơn ở một số claim.
- B0 và P0 nhận thông tin khác nhau nên không quy chênh lệch giữa chúng cho một thành phần cụ thể.

Ước lượng chi phí: (số claim của từng tập) × (số điều kiện) × (số lượt). Tính con số thật sau D7 và **[DỪNG & HỎI]** nếu vượt ngân sách GPU. Phương án giảm: G0/E0 và X4c chỉ chạy t0.

---

## 10. Gán nhãn bổ sung (Mức B)

- **Mục đích:** chấm M4 ở mức ô, chấm M5, và phân tích lỗi trích xuất có rơi vào ô bằng chứng hay không.
- **Tập con:** lấy mẫu phân tầng theo `claim_type` (seed 2026) trong PAIRED_RETR. Đề xuất 150 claim validation + 150 claim test. **[DỪNG & HỎI]** để chốt cỡ mẫu. Tập con test chỉ được gán nhãn sau khi cấu hình đã đóng băng.
- **Nhãn** (gán trên **G1**, theo `cell_id`): các ô cần để kiểm chứng; đoạn văn cần thiết (offset); với mỗi phép kiểm tra: `relation`, toán hạng claim, toán hạng bằng chứng, đơn vị, thời gian, `qualifier`, `direction`, tập so sánh.
- **Chấm M4 trên E1:** đưa ô được chọn trên E1 về G1 qua `CellMapping`. Báo riêng:
  - (a) tỷ lệ ô E không ánh xạ được;
  - (b) precision, recall và F1 tập ô **sau ánh xạ**;
  - (c) lỗi chọn ô khi ánh xạ thành công.
  - Không so trực tiếp tọa độ giữa G1 và E1.
- **Quy trình:** `annotation/export_tasks.py` xuất CSV (ảnh bảng, bảng G1 có in `cell_id`, claim). Người gán thứ nhất gán toàn bộ; người thứ hai gán độc lập 30% mẫu. Tính Cohen's κ cho `relation` và F1 tập ô; bất đồng được thảo luận rồi chốt. Hướng dẫn trong `ANNOTATION_GUIDE.md` (agent viết nháp, người duyệt).
- **Phân công đề xuất:** Vương gán, Viễn kiểm chéo. **[DỪNG & HỎI]** để nhóm chốt.

---

## 11. Giai đoạn triển khai

### Mức A — Triển khai và bàn giao module (chỉ validation)
| GĐ | Nội dung | Điều kiện hoàn thành |
|---|---|---|
| A0 | Repo, môi trường, CI, `schemas.py`, fixtures, mock model | `pytest` xanh; hai môi trường cài được; lock file; `tests/fixtures/packages_mock.jsonl` (10 mẫu viết tay) để M4–M6 phát triển song song |
| A1 | `audit.py`, `select.py` trên validation | `VERIFICATION_LOG.md` đủ D1–D8; ghép claim–bài–ảnh 100% hoặc lỗi được liệt kê; **[DỪNG & HỎI]** với D2/D4 |
| A2 | M1 đầy đủ: nguồn, G0, E0, G1, E1, chất lượng, ánh xạ ô, đơn vị, chỉ mục | Mọi bảng trong danh sách nguồn có TableRecord; audit bảng chuẩn đã duyệt; PAIRED_ORACLE/PAIRED_RETR có kích thước trong `DATA_CARD.md`; 0 đơn vị > 384 |
| A3 | X0 (B0-full, B0-filtered) | Kết quả và giải trình nếu lệch |
| A4 | M3 + X2 | Hit@k, Recall@k, VisualComplete@k, MRR cho 3 phương pháp × 2 kho; tham số chọn trên validation được ghi lại; phân bố ngân sách |
| A5 | M4 | Mọi đầu ra hợp lệ schema; báo tỷ lệ matched/partial/ambiguous/not_found/parse_error; độ chính xác chọn item trên toàn validation |
| A6 | M5 + M6 | `packages.jsonl` 100% hợp lệ; 100% `CellRef`/`SpanRef` truy được về nguồn (hoặc lỗi được liệt kê); bàn giao theo mục 14.1 → **[DỪNG & HỎI]** |

### Mức B — Thực nghiệm đầy đủ
| GĐ | Nội dung | Điều kiện hoàn thành |
|---|---|---|
| B1 | X1, X1u, X3, X4a, X4b (validation); duyệt `unified_v1` | Kết quả có manifest; phân tích lỗi validation |
| B2 | Gán nhãn tập con validation; chấm M4 mức ô, M5 | Chỉ số đồng thuận; chỉ số M4/M5 |
| B3 | Đóng băng cấu hình; đăng ký trước các so sánh sẽ báo trên test trong `EXPERIMENTS.md` | `configs/frozen/` + hash → **[DỪNG & HỎI]** |
| B4 | Test: `audit.py` → `select.py` → M1 → X0, X1, X1u, X2, X3, X4a, X4b; gán nhãn tập con test; thống kê; báo cáo | Bàn giao theo mục 14.2 |

Sau mỗi giai đoạn: `docs/reports/<GĐ>.md` gồm việc đã làm, số liệu, vấn đề và quyết định cần người chốt.

---

## 12. Đánh giá

### 12.1 Chỉ số truy xuất (mức cha, đã loại trùng)
Gọi G là tập bằng chứng đúng của claim (bảng theo `item`, đoạn văn theo `section_ids` ở độ chi tiết D5), R_k là top-k cha.
| Chỉ số | Định nghĩa |
|---|---|
| Hit@k | 1 nếu \|G ∩ R_k\| ≥ 1 |
| Recall@k | \|G ∩ R_k\| / \|G\| |
| VisualComplete@k | 1 nếu mọi bằng chứng **thị giác** (bảng) trong G nằm trong R_k. Không chứng minh đã tìm đủ đoạn văn |
| TextRecall@k | Recall@k chỉ tính trên đoạn văn, ở độ chi tiết D5 |
| MRR | 1 / thứ hạng của bằng chứng đúng đầu tiên |

Báo với k ∈ {1, 3, 5, 10}, tách theo `claim_type` và theo kho G1/E1. Báo thêm số cha trong kho của mỗi bài.

### 12.2 Chỉ số theo thành phần
| Thành phần | Chỉ số |
|---|---|
| M1 | % ghép nguồn đúng; % `no_table_found`; TEDS và TEDS-Struct (có span); precision/recall số; % ô G không ánh xạ được sang E; tỷ lệ bảng chuẩn đúng (ước lượng từ audit, kèm khoảng tin cậy); phân bố luật chuẩn hóa |
| M4 | Độ chính xác chọn item (toàn split); F1 tập ô sau ánh xạ (tập con); tỷ lệ không ánh xạ được; % ambiguous / not_found / parse_error |
| M5 | Độ chính xác từng trường: relation, claim_operands, evidence_operands, unit, time, qualifier, direction, comparison_set_status (tập con) |
| M6 | % hợp lệ schema; % truy được về nguồn |
| Toàn chuỗi | Accuracy, macro-F1; theo `claim_type`, theo nhãn, theo `format_dependent`; % không đọc được câu trả lời; % bị cắt (`length`); token, ảnh và thời gian mỗi claim |

### 12.3 Ổn định, thống kê và phân tích nguyên nhân
- **Theo điều kiện:** accuracy trung bình ± độ lệch chuẩn qua 3 seed; % claim nhất quán ở cả 3 seed; accuracy theo đa số; accuracy ở t0.
- **So sánh cặp:** McNemar chính xác (trên t0 và trên đa số). Khoảng tin cậy 95% của hiệu accuracy bằng **bootstrap theo cụm bài báo** (lấy mẫu lại bài, giữ mọi claim của bài; 1.000 lần; seed 2026). Báo thêm số bài và số claim trung bình mỗi bài.
- **Phân tích đổi kết luận:** với mỗi so sánh chính, xuất `flips.csv`. Người phân tích một mẫu ngẫu nhiên ≥ 100 trường hợp (hoặc tất cả nếu ít hơn), với ba cột:
  - `observed_errors` (chọn nhiều): mất dấu âm, sai cột, lệch hàng, mất tiêu đề, sai số, thiếu bằng chứng, chọn sai ô, sai phép tính, sai đơn vị, khác;
  - `suspected_causes` (chọn nhiều): đọc ảnh, trích bảng, chuẩn hóa, truy xuất, căn chỉnh, số liệu, suy luận, nhiễu seed, **chưa xác định**;
  - `certainty`: có căn cứ rõ / khả dĩ / chưa xác định.
- Trên tập con có nhãn ô: báo tỷ lệ đổi kết luận theo việc lỗi trích xuất **có hay không** rơi vào ô bằng chứng (qua `CellMapping`). Diễn giải là **liên quan**, không phải nhân quả.

### 12.4 Kiểm thử tự động bắt buộc (pytest, mock model)
| Test | Nội dung |
|---|---|
| `test_no_leakage_requests` | Với S-retr/S-align: xáo trộn ngẫu nhiên mọi trường của `GoldInfo` → `pool_hash`, kết quả truy xuất (tất định), và **`request_hash`** gửi tới M4/verifier **không đổi**. Dùng mock model, không so văn bản sinh ra |
| `test_no_label_anywhere` | Ở mọi thiết lập kể cả S-oracle, chuỗi nhãn và giá trị `label` không xuất hiện trong bất kỳ request nào; `build_inputs` không nhận được đối tượng có trường `label` |
| `test_pool_parity` | Kho G1 và E1 của mỗi bài có **cùng tập parent_id** và cùng đơn vị đoạn văn |
| `test_tokenizer` | Giữ đúng `93.2`, `gpt-4o`, `f1@5`, `top-1`, `12.5%` |
| `test_units_limit` | Không đơn vị nào > 384 word pieces |
| `test_parent_dedup` | Năm đơn vị cùng bảng chỉ chiếm một vị trí ở mức cha |
| `test_rrf` | Ví dụ tính tay |
| `test_retrieval_metrics` | Hit/Recall/VisualComplete/MRR trên ví dụ có nhiều bằng chứng |
| `test_html_cells_spans` | Giữ đúng rowspan/colspan, tiêu đề, chữ đậm |
| `test_normalize_rules` | Từng luật ở M1.4, gồm các ca mơ hồ phải giữ nguyên; idempotent |
| `test_teds_reference` | TEDS(x,x)=1 và **≥ 3 cặp bảng** có thay đổi cấu trúc (mất hàng, gộp ô, lệch cột) với giá trị tham chiếu tính tay hoặc đối chiếu với cài đặt PubTabNet |
| `test_cell_mapping` | Ca một–nhiều, ca không ghép được, ca lệch hàng |
| `test_m5_relations` | 80→90: +10 điểm %, +12,5% tương đối; "at least 12%" khác "about 12%"; "2× faster" với thời gian khác với throughput; claim_operands tách khỏi evidence_operands |
| `test_budget` | Cùng ngân sách token giữa các phương pháp; ghi đúng `dropped_parents` |
| `test_scoring` | Các chuỗi mẫu cho `pred_old` và `pred_new` |
| `test_package_schema` | Mọi package hợp lệ; mọi ref tồn tại; `parse_error` khác `not_found` |

---

## 13. Cấu hình và tái lập

```yaml
paths: {data_root: data/raw/SciVer, work: data/processed, outputs: outputs/runs}
split: validation
seed_global: 2026
selection: {allow_mixed: false}
m1:
  gold: {use_arxiv_html: true, version_policy: dataset_or_latest_before_collection, arxiv_sleep_s: 3,
         audit_random_n: 100, audit_score_threshold: 0.8}
  docling: {mode: ACCURATE, do_ocr: true, do_cell_matching: false}
  normalize: {format_dependent_terms: configs/format_terms.txt, include_bold: false}
  units: {max_wordpieces: 384, embed_model: sentence-transformers/all-mpnet-base-v2}
m3: {methods: [bm25, dense, hybrid], bm25: {k1: 1.5, b: 0.75}, rrf_constant: 60, k_units: 50,
     tokenizer_version: v1, budget: {max_evidence_tokens: 6000, max_parents: 3, max_images: 2}}
m4: {model: Qwen/Qwen2.5-VL-7B-Instruct, temperature: 0, max_retries: 2, prompt_id: align_v1, max_align_tokens: 12000}
m5: {calculator: false}
verifier:
  model: Qwen/Qwen2.5-VL-7B-Instruct
  max_model_len: 32768
  max_tokens: 10240
  samples: [{name: t1_s1, temperature: 1.0, seed: 1}, {name: t1_s2, temperature: 1.0, seed: 2},
            {name: t1_s3, temperature: 1.0, seed: 3}, {name: t0, temperature: 0.0, seed: 0}]
eval: {bootstrap_n: 1000, bootstrap_seed: 2026, bootstrap_unit: paper}
```
- `run_id = {date}_{experiment}_{split}_{confighash[:8]}`. Không ghi đè. Có `manifest.json`. Chạy tiếp được khi bị ngắt, nếu hash cấu hình trùng.
- CLI: `python -m sciverx <stage> --config configs/experiments/<x>.yaml`. Các stage: `audit`, `select`, `sources`, `gold`, `extract`, `normalize`, `quality`, `cellmap`, `units`, `index`, `retrieve`, `align`, `numeric`, `package`, `verify`, `evaluate`, `report`, `annotate-export`, `annotate-import`.
- Các ngưỡng 6000 / 12000 token là giá trị khởi đầu: chỉnh trên validation sau khi xem phân bố token thật, rồi đóng băng ở B3.

---

## 14. Bàn giao

### 14.1 Cuối Mức A
1. Source code, `README.md`, lock file; `pytest` xanh.
2. `VERIFICATION_LOG.md` (D1–D8, hash dataset và repo, cấu hình GPU thực tế).
3. `DATA_CARD.md` (tiêu chí lọc, số claim, PAIRED_ORACLE, PAIRED_RETR, bảng bị loại và lý do).
4. `DEVIATIONS.md`:
   - seed được truyền vào vLLM;
   - cách chấm `pred_new`;
   - chỉnh sửa "image" → "table" và khối bảng văn bản;
   - `unified_v1` và các prompt khác kèm hash;
   - `max_model_len`;
   - xử lý chữ đậm và cờ `format_dependent`;
   - luật chuẩn hóa;
   - chính sách phiên bản arXiv.
5. Dữ liệu trung gian validation: sources, TableRecord, CellMapping, normalize_log, table_quality, units, chỉ mục, retrieval, alignment, packages.
6. Kết quả X0 và X2; chỉ số M1, M3, M4 (mức item), M6.
7. `docs/reports/A*.md`.

### 14.2 Cuối Mức B
Mọi mục ở 14.1 cho cả test, cộng thêm:
- kết quả X1, X1u, X3, X4a, X4b (và X4c nếu chạy) có manifest;
- `EXPERIMENTS.md` gồm các so sánh đăng ký trước và kết quả;
- `ANNOTATION_GUIDE.md`, nhãn, chỉ số đồng thuận;
- `flips.csv` đã phân tích;
- `outputs/report/` gồm bảng chỉ số (CSV + markdown), khoảng tin cậy bootstrap theo cụm, McNemar.

---

## 15. Điểm cần người chốt ([DỪNG & HỎI])

| # | Khi nào | Nội dung |
|---|---|---|
| H0 | Trước A1 | Có chạy `audit.py --structure-only` trên test không; nếu có, ai chạy |
| H1 | A1 | D2 (tên trường), D4 (kho có đủ bảng và ảnh không): quyết định khả năng làm S-retr/S-align |
| H2 | A2 | Duyệt audit bảng chuẩn; chấp nhận kích thước PAIRED_ORACLE và PAIRED_RETR |
| H3 | A3 | B0-full lệch số paper > 3 điểm |
| H4 | A6 | Duyệt bàn giao Mức A; cho phép sang Mức B |
| H5 | B1 | Duyệt nội dung `unified_v1` và `unified_v1_numeric` |
| H6 | B1–B4 | Chi phí GPU vượt ngân sách |
| H7 | B2 | Cỡ tập con gán nhãn; người gán và người kiểm chéo |
| H8 | B3 | Duyệt cấu hình đóng băng và danh sách so sánh đăng ký trước |
| H9 | B4 | Cho phép chạy test |
| H10 | Bất kỳ | Bật mở rộng (M2, PoT, G1b, embedding 2, reranker, nhiều bài, 72B, P1) |

---

## 16. Phụ lục — Chi tiết đã xác minh từ code gốc SciVer

- Mỗi claim nhận 1 ảnh (direct/analytical) hoặc 2 ảnh (parallel/sequential) đúng các bằng chứng, cộng các section cấp 1 chứa bằng chứng (`prepare_context`: lấy phần trước dấu chấm đầu tiên của ID section rồi nối toàn bộ section cấp 1 tương ứng).
- 4 prompt CoT theo `claim_type`; prompt 2 và prompt 4 giống hệt nhau.
- `temperature=1.0`, `max_tokens=10240`, `n=1`. `GENERATION_SEED=215` được khai báo nhưng không truyền vào `SamplingParams`.
- Ảnh: chuyển RGB; với Qwen2.5-VL, phóng ảnh có cạnh < 224 (giữ tỉ lệ); nén lại JPEG. `AutoProcessor(min_pixels=256*28*28, max_pixels=1280*28*28)` chỉ dùng để dựng chat template.
- `acc_evaluation.py`: dự đoán Entailed khi `'yes' in response.lower()`; đọc `label` từ file split.
- Số trong paper (Qwen2.5-VL-7B, CoT): validation 53,5; test 57,6. Qwen2.5-VL-72B: test 69,4.
