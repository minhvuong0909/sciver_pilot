# SciVer-TableVerify

**Kiểm chứng claim khoa học có bảng biểu: truy xuất, căn chỉnh và xử lý số liệu trên benchmark SciVer**

![status](https://img.shields.io/badge/status-in%20development-orange) ![python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue) ![model](https://img.shields.io/badge/verifier-Qwen2.5--VL--7B-purple) ![spec](https://img.shields.io/badge/spec-v1.1-lightgrey)

Dự án thuộc hướng **Numeric & Table-Aware Verification** của nhóm WorkplaceFactBench. Hệ thống nhận một claim và bài báo nguồn, rồi thực hiện bốn bước:

1. Tìm bằng chứng (đoạn văn và bảng) trong bài.
2. Căn chỉnh claim tới đúng ô hoặc đúng đoạn.
3. Trích các phép kiểm tra số liệu.
4. Đóng gói kết quả cho bộ kiểm chứng kết luận **Entailed** hoặc **Refuted**.

Thực nghiệm chạy trên **toàn bộ** claim có bằng chứng bảng của SciVer (ACL 2025). Model kiểm chứng giống paper gốc là `Qwen2.5-VL-7B-Instruct`.

> Đặc tả kỹ thuật đầy đủ nằm ở [`SPEC_SciVer_TableVerify.md`](SPEC_SciVer_TableVerify.md). Khi README và SPEC khác nhau, **SPEC là chuẩn**.

---

## Mục lục
- [Câu hỏi nghiên cứu](#câu-hỏi-nghiên-cứu)
- [Kiến trúc](#kiến-trúc)
- [Thiết kế thực nghiệm](#thiết-kế-thực-nghiệm)
- [Cài đặt](#cài-đặt)
- [Dữ liệu](#dữ-liệu)
- [Chạy pipeline](#chạy-pipeline)
- [Kết quả đầu ra](#kết-quả-đầu-ra)
- [Đánh giá](#đánh-giá)
- [Tái lập và nguyên tắc dữ liệu](#tái-lập-và-nguyên-tắc-dữ-liệu)
- [Kiểm thử](#kiểm-thử)
- [Cấu trúc repo](#cấu-trúc-repo)
- [Lộ trình](#lộ-trình)
- [Khác biệt so với code gốc SciVer](#khác-biệt-so-với-code-gốc-sciver)
- [Nhóm thực hiện](#nhóm-thực-hiện)
- [Trích dẫn và giấy phép](#trích-dẫn-và-giấy-phép)

---

## Câu hỏi nghiên cứu

| | Câu hỏi |
|---|---|
| **RQ1 — Dạng bảng** | Cùng claim, cùng model, cùng vị trí bằng chứng: đưa bảng dưới dạng **ảnh**, **bảng chuẩn dạng văn bản** hoặc **bảng trích tự động bằng Docling** làm kết luận thay đổi thế nào? Lỗi trích xuất nào liên quan đến việc đổi kết luận? |
| **RQ2 — Truy xuất** | Cùng kho và cùng ngân sách đầu vào: **BM25**, **vector** và **hybrid (RRF)** tìm bằng chứng tốt đến đâu, và kết luận thay đổi ra sao? |
| **RQ3 — Căn chỉnh và số liệu** | So với chỉ đưa kết quả truy xuất, thêm bước **căn chỉnh (M4)**, rồi thêm **xử lý số liệu (M5)**, thay đổi kết luận thế nào? |

---

## Kiến trúc

```
                        ┌──────────────────────── Tài liệu nguồn (bài báo SciVer) ───────────────────────┐
                        │ sections · mọi bảng (ảnh) · mọi biểu đồ                                          │
                        └───────────────────────────────────────┬─────────────────────────────────────────┘
                                                                ▼
 M1  Chuẩn bị & lập chỉ mục ── bảng chuẩn G0 (JSON / HTML arXiv đúng phiên bản)  ─┐
                              bảng trích  E0 (Docling)                              ├─► chuẩn hóa → G1 / E1
                              TEDS · ánh xạ ô E↔G · đơn vị tìm kiếm ≤ 384 token    ─┘   chỉ mục BM25 + vector
                                                                ▼
 Claim ──► (M2 trích atomic claim — mở rộng) ──► M3  Hybrid Search: BM25 + vector + RRF, gộp theo bảng/đoạn cha
                                                                ▼
                                                 M4  Căn chỉnh: chọn đúng bảng, ô, đoạn  (matched / ambiguous / …)
                                                                ▼
                                                 M5  Số liệu: toán hạng có nguồn, phép tính, đơn vị, thời gian
                                                                ▼
                                                 M6  Đóng gói EvidencePackage (JSON, kiểm định Pydantic)
                                                                ▼
                                                 Verifier  Qwen2.5-VL-7B (vLLM) ──► Entailed / Refuted
```

| Module | Nhiệm vụ | Công cụ chính |
|---|---|---|
| **M1** Chuẩn bị & lập chỉ mục | Ghép nguồn; bảng chuẩn và bảng trích cho **mọi** bảng của bài; chuẩn hóa; TEDS; ánh xạ ô; chia đơn vị; lập chỉ mục | Docling, BeautifulSoup/lxml, apted, Sentence Transformers, rank_bm25 |
| **M2** Trích atomic claims | Tách bản thảo thành các claim kiểm chứng được *(mở rộng, chưa làm)* | Qwen2.5, Pydantic |
| **M3** Hybrid Search | Truy xuất BM25 và vector, hợp nhất bằng RRF, gộp theo cha, áp ngân sách đầu vào | rank_bm25, `all-mpnet-base-v2`, NumPy |
| **M4** Căn chỉnh | Chọn đúng nguồn, ô và đoạn; kiểm ID bằng code; ghi trạng thái mơ hồ | Qwen2.5-VL-7B (văn bản), Pydantic |
| **M5** Số liệu | Phép kiểm tra số có cấu trúc: điểm % hay % tương đối, cực trị, "mọi/tất cả", bội số, dung sai theo câu chữ | Python `re`, `decimal` |
| **M6** Đóng gói | JSON thống nhất, truy được về nguồn, ghi thông tin còn thiếu | Pydantic, JSONL |

> `alignment_status = matched` chỉ có nghĩa là **đã tìm được bằng chứng phù hợp để kiểm tra**. Nó không có nghĩa claim đúng.

---

## Thiết kế thực nghiệm

Ba trục độc lập:

| Trục | Giá trị |
|---|---|
| **Hệ thống** | `B0` code SciVer gốc · `P0` pipeline của nhóm, chưa fine-tune · `P1` có fine-tune *(chỉ khi được duyệt)* |
| **Thiết lập bằng chứng** | `S-oracle` cho sẵn vị trí bằng chứng chuẩn · `S-retr` chỉ truy xuất · `S-align4` + M4 · `S-align45` + M4 + M5 |
| **Dạng bảng** | `IMG` ảnh · `G0` / `G1` bảng chuẩn chưa / đã chuẩn hóa · `E0` / `E1` bảng Docling chưa / đã chuẩn hóa |

| ID | Hệ thống | Thiết lập | Bảng | Trả lời |
|---|---|---|---|---|
| X0 | B0 | gốc | IMG | Tái lập paper (toàn validation) và mốc trên tập đã lọc |
| X1 / X1u | P0 | S-oracle | IMG, G0, G1, E0, E1 | RQ1; cầu nối prompt gốc ↔ prompt thống nhất |
| X2 | — | truy xuất | kho G1, kho E1 | RQ2 (chỉ số truy xuất) |
| X3 | P0 | S-retr | G1, E1 | RQ2 (kết luận) |
| X4a / X4b | P0 | S-align4 / S-align45 | G1, E1 | RQ3: đóng góp của M4, rồi M4+M5 |

So sánh G↔E chỉ thực hiện trên các tập **ghép cặp** (`PAIRED_ORACLE`, `PAIRED_RETR`). Đây là các claim có đủ cả bảng chuẩn lẫn bảng trích, nên kết quả không bị lệch vì so hai tập claim khác nhau.

---

## Cài đặt

### Yêu cầu
- Linux, NVIDIA GPU. Khuyến nghị A100 40/80 GB hoặc L40S 48 GB. GPU 24 GB cần thử trên các mẫu dài nhất trước khi dùng.
- Conda, `git-lfs`, khoảng 60 GB đĩa trống.

### Hai môi trường riêng
Dùng hai môi trường vì vLLM và Docling có thể cần các phiên bản torch khác nhau.

```bash
# 1) Môi trường chính: vLLM, truy xuất, căn chỉnh, đánh giá
conda create -y -n sciver python=3.10 && conda activate sciver
pip install -r requirements/sciver.txt          # gồm vllm==0.9.0.1 (giống repo SciVer)
pip install -e .

# 2) Môi trường Docling: trích bảng từ ảnh (chạy được trên CPU)
conda create -y -n docling python=3.11 && conda activate docling
pip install -r requirements/docling.txt
pip install -e .
```

### Tải dữ liệu và model
```bash
git lfs install
git clone https://huggingface.co/datasets/chengyewang/SciVer data/raw/SciVer
git clone https://github.com/QDRhhhh/SciVer third_party/SciVer
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"
```

Kiểm tra nhanh:
```bash
python -c "import vllm, torch; print(vllm.__version__, torch.cuda.is_available())"
pytest -q
```

---

## Dữ liệu

| | Mô tả |
|---|---|
| Nguồn | SciVer: claim khoa học trên bài arXiv ngành CS, gán nhãn bởi chuyên gia |
| Split | `validation` (1.000 claim): phát triển · `test` (2.000 claim): khóa, chỉ chạy một lần ở cuối |
| Tiêu chí chọn | Claim có bằng chứng thị giác **chỉ là bảng**; giữ nguyên bằng chứng văn bản. Tùy chọn `allow_mixed` để gồm cả biểu đồ |
| Quy mô | **Toàn bộ** claim thỏa tiêu chí. Số lượng thực tế được ghi trong [`docs/DATA_CARD.md`](docs/DATA_CARD.md) |

Bảng chuẩn được lấy từ JSON của dataset nếu có. Nếu không có, lấy từ **HTML arXiv đúng phiên bản** của bài. Chất lượng bảng chuẩn được kiểm tay trên mẫu ngẫu nhiên và mọi bảng có độ khớp thấp.

---

## Chạy pipeline

Mọi bước dùng một CLI chung với file cấu hình YAML:

```bash
python -m sciverx <stage> --config configs/experiments/<experiment>.yaml
```

### Mức A — Triển khai module trên validation
```bash
conda activate sciver
python -m sciverx audit     --config configs/base.yaml          # kiểm dữ liệu D1–D8 (chỉ validation)
python -m sciverx select    --config configs/base.yaml          # ClaimInput + GoldInfo (hai file riêng)
python -m sciverx sources   --config configs/base.yaml          # danh sách mọi bảng, biểu đồ, section của mỗi bài
python -m sciverx gold      --config configs/base.yaml          # bảng chuẩn G0 + file audit cần kiểm tay

conda activate docling
python -m sciverx extract   --config configs/base.yaml          # bảng trích E0 bằng Docling

conda activate sciver
python -m sciverx normalize --config configs/base.yaml          # G1, E1 + normalize_log
python -m sciverx quality   --config configs/base.yaml          # TEDS, TEDS-Struct, độ khớp số
python -m sciverx cellmap   --config configs/base.yaml          # ánh xạ ô E↔G
python -m sciverx units     --config configs/base.yaml          # đơn vị tìm kiếm ≤ 384 word pieces
python -m sciverx index     --config configs/base.yaml          # chỉ mục BM25 + vector cho kho G1 và E1

python -m sciverx verify    --config configs/experiments/x0_b0.yaml          # B0
python -m sciverx retrieve  --config configs/experiments/x2_retrieval.yaml   # BM25 / dense / hybrid
python -m sciverx align     --config configs/experiments/x4_align.yaml       # M4
python -m sciverx numeric   --config configs/experiments/x4_align.yaml       # M5
python -m sciverx package   --config configs/experiments/x4_align.yaml       # M6
python -m sciverx evaluate  --config configs/experiments/x2_retrieval.yaml
```

### Mức B — Thực nghiệm đầy đủ
```bash
python -m sciverx verify   --config configs/experiments/x1_oracle.yaml       # RQ1
python -m sciverx verify   --config configs/experiments/x3_retr.yaml         # RQ2
python -m sciverx verify   --config configs/experiments/x4a_align4.yaml      # RQ3 – M4
python -m sciverx verify   --config configs/experiments/x4b_align45.yaml     # RQ3 – M4+M5
python -m sciverx evaluate --config configs/experiments/x1_oracle.yaml
python -m sciverx report   --config configs/base.yaml

# Test: chỉ sau khi cấu hình đã đóng băng trong configs/frozen/ và đã được duyệt
python -m sciverx <stage> --config configs/frozen/<experiment>.yaml --split test
```

Thử nhanh khi phát triển: thêm `--limit 20` để chạy smoke test. Kết quả smoke test không dùng để báo cáo.

---

## Kết quả đầu ra

Mỗi lượt chạy được lưu ở `outputs/runs/<run_id>/` với `run_id = {ngày}_{thí nghiệm}_{split}_{hash cấu hình}`. Lượt chạy cũ không bao giờ bị ghi đè.

| File | Nội dung |
|---|---|
| `manifest.json` | Cấu hình đầy đủ, git commit, phiên bản thư viện và model, GPU, seed, danh sách ID, thời gian, số lỗi |
| `tables/*.jsonl` | `TableRecord` cho G0/G1/E0/E1 (giữ cấu trúc ô gộp) |
| `table_quality.csv`, `cell_mapping.jsonl` | TEDS, độ khớp số, ánh xạ ô E↔G |
| `retrieval.jsonl` | Ứng viên theo thứ hạng: điểm BM25, vector và RRF |
| `packages.jsonl` | `EvidencePackage`: bằng chứng, ô, đoạn, phép kiểm tra số, trạng thái, nguồn |
| `verifier_requests.jsonl`, `verifier_outputs.jsonl` | Request (kèm hash) và câu trả lời của model, chấm theo cách cũ và cách mới |
| `eval/` | Chỉ số, độ ổn định, McNemar, bootstrap theo cụm, `flips.csv` |

---

## Đánh giá

| Thành phần | Chỉ số |
|---|---|
| Trích bảng (M1) | TEDS, TEDS-Struct (giữ ô gộp), precision/recall số, % không tìm thấy bảng, % ô không ánh xạ được |
| Truy xuất (M3) | Hit@k, Recall@k, VisualComplete@k, TextRecall@k, MRR, với k ∈ {1, 3, 5, 10}, tính ở mức bảng/đoạn cha |
| Căn chỉnh (M4) | Độ chính xác chọn bảng; F1 tập ô **sau ánh xạ E↔G**; tỷ lệ ambiguous / not_found / parse_error |
| Số liệu (M5) | Độ chính xác từng trường (quan hệ, toán hạng, đơn vị, thời gian, qualifier, tập so sánh) |
| Đóng gói (M6) | % hợp lệ schema, % truy được về nguồn |
| Toàn chuỗi | Accuracy, macro-F1, tỷ lệ không đọc được câu trả lời, độ ổn định qua 3 seed, token và thời gian mỗi claim |

So sánh cặp dùng **McNemar chính xác** và **khoảng tin cậy 95% bằng bootstrap theo cụm bài báo**. Phân tích các claim đổi kết luận ghi lỗi quan sát được, nguyên nhân nghi ngờ (cho chọn nhiều) và mức chắc chắn. Khi diễn giải chỉ nói "liên quan", không nói "gây ra".

---

## Tái lập và nguyên tắc dữ liệu

- **Test bị khóa:** mọi lựa chọn được chỉnh trên validation. Test chỉ chạy một lần, với cấu hình đóng băng đã có hash và các so sánh đã đăng ký trước.
- **Không lộ vị trí bằng chứng:** ở các thiết lập truy xuất, module chỉ nhận `ClaimInput` (claim, ID bài) và kho tài liệu. `GoldInfo` nằm ở file riêng và chỉ module đánh giá được đọc.
- **Nhãn kết luận không bao giờ vào model**, ở mọi thiết lập.
- **Kho công bằng:** kho bảng chuẩn và kho bảng trích có cùng danh sách nguồn và cùng quy tắc chia đơn vị.
- **Không cắt âm thầm:** mọi việc cắt bớt token, hàng, ứng viên hay ảnh đều được ghi log.
- **Sinh có kiểm soát:** 3 seed ở temperature 1.0 (giống paper) và 1 lượt temperature 0. Seed được truyền thật vào vLLM.

---

## Kiểm thử

```bash
pytest -q                     # toàn bộ, dùng mock model, không cần GPU
pytest -q tests/test_leakage* # chống lộ vị trí bằng chứng: so hash request khi xáo trộn GoldInfo
```

Bộ test gồm:
- chống rò rỉ và kiểm tra kho G/E có cùng danh sách nguồn;
- tokenizer giữ nguyên các dạng như `93.2`, `gpt-4o`, `f1@5`;
- giới hạn 384 word pieces mỗi đơn vị;
- RRF và gộp theo cha;
- định nghĩa các chỉ số truy xuất;
- ô gộp, luật chuẩn hóa, TEDS với các cặp bảng có kết quả biết trước, ánh xạ ô;
- các quan hệ số của M5, ngân sách đầu vào, chấm điểm và schema.

---

## Cấu trúc repo

```
sciver-tableverify/
├── configs/            base.yaml · experiments/ · frozen/
├── src/sciverx/
│   ├── schemas.py      Pydantic: ClaimInput, GoldInfo, TableRecord, EvidencePackage, …
│   ├── data/           audit, select, loader
│   ├── tables/         ô gộp, chuẩn hóa, linearize, TEDS, ánh xạ ô
│   ├── m1_prepare/     nguồn, bảng chuẩn, Docling, đơn vị, chỉ mục
│   ├── m3_retrieve/    tokenizer, BM25, dense, RRF, ngân sách
│   ├── m4_align/       căn chỉnh và kiểm định
│   ├── m5_numeric/     phân tích số, quan hệ, dung sai
│   ├── m6_package/     đóng gói
│   ├── verifier/       prompt, dựng đầu vào, vLLM, chấm điểm
│   ├── evaluation/     chỉ số, thống kê, báo cáo
│   └── annotation/     xuất/nhập nhãn, độ đồng thuận
├── tests/              fixtures, mocks
├── docs/               VERIFICATION_LOG · DEVIATIONS · DATA_CARD · EXPERIMENTS · ANNOTATION_GUIDE · reports/
├── requirements/       sciver.txt · docling.txt · *.lock.txt
├── data/               (không commit) raw · interim · processed
└── outputs/            (không commit) runs/<run_id>/
```

---

## Lộ trình

**Mức A — Triển khai và bàn giao module (validation)**
- [ ] A0 Repo, môi trường, schema, test và mock
- [ ] A1 Kiểm dữ liệu D1–D8, chọn claim
- [ ] A2 M1: bảng chuẩn, Docling, chuẩn hóa, TEDS, ánh xạ ô, đơn vị, chỉ mục
- [ ] A3 X0: tái lập B0
- [ ] A4 M3 + X2: truy xuất
- [ ] A5 M4: căn chỉnh
- [ ] A6 M5 + M6: số liệu, đóng gói, bàn giao

**Mức B — Thực nghiệm đầy đủ**
- [ ] B1 X1, X1u, X3, X4a, X4b trên validation
- [ ] B2 Gán nhãn ô trên tập con validation; chấm M4/M5
- [ ] B3 Đóng băng cấu hình và đăng ký trước các so sánh
- [ ] B4 Chạy test, thống kê, báo cáo

**Mở rộng:** M2 · PoT verifier · embedding thứ hai · reranker · kho nhiều bài · Qwen2.5-VL-72B · fine-tune (P1)

---

## Khác biệt so với code gốc SciVer

Danh sách đầy đủ kèm hash prompt nằm ở [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md). Các điểm chính:

| Điểm | Code gốc | Dự án này |
|---|---|---|
| Seed | Khai báo `GENERATION_SEED=215` nhưng không truyền vào vLLM | Truyền thật; 3 seed (temperature 1.0) + 1 lượt temperature 0 |
| Chấm điểm | Entailed nếu có chữ "yes" ở bất kỳ đâu | Báo cả cách cũ và cách mới (đọc dòng `Answer:` cuối cùng) |
| Đầu vào bảng | Chỉ dạng ảnh | Thêm bảng chuẩn và bảng trích dạng văn bản (chỉ đổi "image" → "table") |
| Prompt | 4 prompt theo loại claim | Giữ nguyên ở S-oracle; thêm prompt thống nhất cho các thiết lập không biết loại claim |
| Bằng chứng | Cho sẵn theo nhãn | Thêm các thiết lập tự truy xuất và căn chỉnh |

---

## Nhóm thực hiện

Nhóm **WorkplaceFactBench**. Phân công đề xuất, chờ nhóm chốt:

| Thành viên | Phần phụ trách |
|---|---|
| Vương | M1, M3: chuẩn bị nguồn, biểu diễn bảng, truy xuất; gán nhãn ô |
| Viễn | M4–M6: căn chỉnh, xử lý số liệu, đóng gói; kiểm chéo nhãn |
| Khoa | Bộ kiểm chứng, tích hợp đầu ra |

---

## Trích dẫn và giấy phép

Dự án dùng dữ liệu và code của **SciVer**. Khi công bố kết quả, hãy trích dẫn paper gốc theo BibTeX trên ACL Anthology: <https://aclanthology.org/2025.acl-long.420>. Code gốc ở <https://github.com/QDRhhhh/SciVer>, dữ liệu ở <https://huggingface.co/datasets/chengyewang/SciVer>.

Các thành phần bên thứ ba khác: [Docling](https://github.com/docling-project/docling), [vLLM](https://github.com/vllm-project/vllm), [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), [Sentence Transformers](https://www.sbert.net/), [rank_bm25](https://github.com/dorianbrown/rank_bm25). Mỗi thành phần tuân theo giấy phép riêng của nó. Trước khi phân phối lại dữ liệu hoặc trọng số model, hãy kiểm tra giấy phép tương ứng.

Giấy phép của mã nguồn dự án: *chưa chọn* (nhóm cần quyết định trước khi công khai repo).
