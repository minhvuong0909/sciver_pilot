# DEVIATIONS

Mọi khác biệt so với code gốc SciVer và so với SPEC (SPEC 2.7, 14.1-4).

## So với code gốc SciVer

| Điểm | Code gốc | Dự án này | Vì sao |
|---|---|---|---|
| Seed | `GENERATION_SEED=215` khai báo nhưng không truyền vào vLLM | Seed truyền thật vào `SamplingParams`; 3 lượt temperature 1.0 (seed 1/2/3) + 1 lượt temperature 0 | Tái lập được, đo độ ổn định (SPEC 8.1) |
| Chấm điểm | Entailed nếu `'yes' in response.lower()` | Báo cả `pred_old` (cách cũ) và `pred_new`: lần cuối của `Answer: yes|no` (cho phép `**`, ngoặc, khoảng trắng), dự phòng `final answer is`, không đọc được → `None`, tính là sai | Cách cũ tính "eyes", "yes" trong lập luận là Entailed (SPEC 8.4) |
| Đầu vào bảng | Chỉ ảnh | Thêm bảng văn bản G0/G1/E0/E1: khối `Table content (as text):` + chú thích bảng; prompt 1 và 3 đổi "image" → "table" theo `TEXT_MODE_EDITS` | RQ1 |
| Prompt | 4 prompt theo `claim_type` | Giữ nguyên ở S-oracle; thêm `unified_v1` / `unified_v1_numeric` cho các thiết lập không biết loại claim | SPEC 8.3 |
| Bằng chứng | Cho sẵn theo nhãn | Thêm S-retr, S-align4, S-align45 | RQ2, RQ3 |
| `max_model_len` | Mặc định của vLLM | 32768 (giảm nếu GPU không đủ, ghi vào VERIFICATION_LOG) | SPEC 8.1 — **chưa thử trên GPU**. Kiểm trước bằng `python -m sciverx count-tokens --config …` |
| `pred_old` với đầu ra rỗng | Mẫu rỗng sai ở cả hai nhãn (`if item["response"]`) | `pred_old("")` = `None` (tính sai), giống gốc | Trước đây pilot tính mẫu rỗng là đúng với nhãn Refuted |
| Trạng thái đầu ra | Không có | `VerifierOutput.status` ∈ `ok` / `parse_error` / `empty` / `error`; báo tỷ lệ trong `condition_summary.status_pct` | Tách lỗi kỹ thuật, đầu ra rỗng và lỗi định dạng |
| Retry | Không có (một lần gọi cả tập) | Lô lỗi → gửi lại từng request, tối đa `verifier.max_retries` (mặc định 1); số lần gọi ghi ở `attempts`, tổng retry ở manifest | Quy tắc cố định, áp dụng đều cho mọi cấu hình |
| Cách gửi cho vLLM | `llm.generate` cả tập trong một lần, template dựng bằng `AutoProcessor` | `llm.chat` theo lô `verifier.chunk_size` (mặc định 64), mỗi request một `SamplingParams` có seed riêng | Nhanh như gửi cả tập nhưng ghi ra đĩa sau mỗi lô. [CẦN KIỂM trên GPU] so chuỗi prompt và vài đầu ra với cách gửi từng câu |
| Chạy tiếp từ chỗ dừng | Không có (ngắt giữa chừng là mất cả điều kiện) | Chạy lại cùng lệnh (cùng `run_id`) bỏ qua (claim, lượt) đã có đầu ra không lỗi và cùng `request_hash`; mẫu `status=error` được chạy lại | Tránh tốn giờ GPU khi máy bị ngắt |

## Prompt và hash

Hash = 16 ký tự đầu SHA-256 của nội dung template (prompt `_text` gồm cả `TEXT_MODE_EDITS`). Prompt 4 của SciVer giống hệt prompt 2 nên trùng hash. Đã so nguyên văn với `utils/constant.py` của repo gốc (commit `0a97282`).

| prompt_id | hash |
|---|---|
| `sciver_p1` | `c71f88b0664f3c53` |
| `sciver_p1_text` | `e149e8e061671a81` |
| `sciver_p2` | `4f15a22567ce6907` |
| `sciver_p2_text` | `9257b106a22a16f3` |
| `sciver_p3` | `a3ebd796e47ceae9` |
| `sciver_p3_text` | `05c1fa8e6b46c282` |
| `sciver_p4` | `4f15a22567ce6907` |
| `sciver_p4_text` | `9257b106a22a16f3` |
| `unified_v1` | `4e9d96fb8eb7e230` — **bản nháp, chờ duyệt (H5)** |
| `unified_v1_numeric` | `3eea2c1965b7f479` — **bản nháp, chờ duyệt (H5)** |
| `align_v1` | `cc50b9137c12613e` |

## Bảng và chuẩn hóa

- **Bảng chuẩn**: dataset đã có `tables[id].table_html` (LaTeXML đúng phiên bản bài) cho mọi bảng, nên luôn dùng `extractor=gold_json`; nhánh tải HTML arXiv của SPEC M1.2 bước 2 không cần dùng. `source_version` lấy từ hậu tố `vN` của `paperid`.
- **Bảng nhiều phần** (nhiều tabular ngoài cùng trong một figure): xếp chồng theo chiều dọc, đưa hết vào danh sách kiểm tay.
- **Hàng tiêu đề không khai báo**: đoán = các hàng trước hàng đầu tiên có ô số ở cột ≥ 1 (luật của bản pilot), dùng chung cho G và E.
- **TEDS**: viết lại so với bản pilot — cây dựng từ `cells` còn rowspan/colspan và có `thead`/`tbody`, thay vì lưới đã trải.
- **Luật chuẩn hóa** (thứ tự áp): tách ký hiệu chú thích → dấu trừ Unicode → khoảng trắng trước `%` → `±` → dấu nghìn → gắn lại `x [*]`; ô `–`, `—`, `-`, `n/a`, `na`, `NA` → `N/A`; mũi tên chỉ chú giải trong ô tiêu đề. Khác bản pilot: giữ ký hiệu chú thích và mũi tên thay vì xóa.
- **Chữ đậm**: mặc định không đưa vào văn bản (`include_bold=false`) vì Docling không cho biết chữ đậm (`bold=None`). Claim chứa từ trong `configs/format_terms.txt` được gắn cờ `format_dependent`.
- **Dấu vết tiền xử lý** `###reference_xx###` trong chữ ô bị bỏ. Từ 26/11/2026 cũng bị bỏ trong **đơn vị đoạn văn** của kho tìm kiếm (khoảng 52% đơn vị còn dính, theo đánh giá 25/11), vì nó làm nhiễu BM25/embedding và làm prompt S-retr dài thêm. `char_start`/`char_end` vẫn trỏ vào section text gốc. **Context của S-oracle không đổi** (vẫn nguyên văn như code SciVer) để giữ baseline. Phải chạy lại `units` và `index`.

## Truy xuất

- Khóa cha có tiền tố (`section:1`, `table:1`) vì ID section và ID bảng trùng nhau trong một bài.
- Section truy xuất được tính khớp section chuẩn khi trùng hoặc là tiểu mục của nó (nhãn ghi tới cấp 1–3, kho chia theo section lá).
- Ô bảng quá dài (cả đoạn văn trong một ô) được chia theo từ, mỗi phần lặp lại caption và tiêu đề.
- **Tùy chọn cải thiện truy xuất (X2b, mặc định tắt):** `m3.rrf_weights` (RRF có trọng số; SPEC dùng RRF 1:1), `m3.rerank` (cross-encoder chấm lại `top_n` đơn vị đầu), `m3.visual_slots` (giữ ≥ n bảng/biểu đồ trong top-k). Lý do: trên 1.000 claim validation, hybrid 1:1 kém BM25 (Recall@5 75,8 so với 79,8; VisualComplete@5 66,2 so với 75,1) vì dense yếu (Recall@5 64,0). Chỉ chọn biến thể trên phần chỉnh của validation, xác nhận trên phần xác nhận (hyper_parameter.md mục 8).
- **`max_parents` 3 → 5 (25/11/2026, tạm thời):** với 3 cha, chỉ 69% claim (291 claim G1, BM25) có đủ bảng chuẩn trong bằng chứng gửi cho model; parallel/sequential 44%/47% vì claim cần 2 bảng + đoạn văn. Với 5 cha: 80% (parallel 69%, sequential 65%), token bằng chứng TB 1.862 → 3.038, 18 claim chạm trần 6.000 token. Con số đo trên chính 291 claim dùng để quyết định, nên phải xác nhận trên phần xác nhận; chốt cuối cùng theo accuracy của verifier khi có GPU (chạy S-retr với 3 và 5). Biến thể `bm25-mp3` trong X2b giữ cấu hình cũ để so.
- **`visual_slots.k` mặc định = `max_parents`:** bản đầu đặt k = 5 khi model chỉ nhận 3 cha, nên bảng được giữ chỗ ở hạng 4–5 bị cắt (top-5 tăng 81% → 90% nhưng đủ bảng khi đến model vẫn 69%). `evaluate` nay báo thêm "đủ bảng khi đến model" (`delivered/…`, đọc `selection/*.jsonl`).
- **Bỏ rerank ms-marco khỏi X2b:** 66% đủ bảng khi đến model, kém BM25 (69%); model huấn luyện trên văn bản web, không đọc tốt bảng toàn số. Code `m3.rerank` vẫn giữ cho model khác.

## E0: Docling trên PDF arXiv

SPEC M1.3 chạy Docling trên **ảnh bảng**. Dataset chỉ có ảnh cho 379/1.327 bảng của các bài đã chọn (và gần như trùng với bảng làm bằng chứng), nên theo quyết định của nhóm (2026-11-13) E0 được lấy từ **PDF arXiv đúng phiên bản** của bài (`m1.docling.source: pdf`):

- Chỉ đưa vào Docling các trang có dòng caption "Table N" (tìm bằng lớp chữ của PDF) để chạy được trên CPU.
- **OCR tắt** khi đọc PDF (`do_ocr_pdf: false`) vì PDF arXiv đã có lớp chữ; OCR vẫn bật khi chạy trên ảnh.
- **Cell matching bật** khi đọc PDF (`do_cell_matching_pdf: true`), khác SPEC (`do_cell_matching=False`, đặt cho ảnh). Với thiết lập của SPEC, Docling để trống chữ ở hàng tiêu đề của 238/1.220 bảng; thử trên 27 bảng của 3 bài, bật lên làm TEDS tăng 0,571 → 0,777 và không còn bảng mất tiêu đề. Nhánh ảnh vẫn tắt như SPEC.
- Bảng Docling được ghép với `table_id` của dataset qua **nhãn in trên caption** lấy từ caption của dataset (ID của dataset đánh số liên tục, trong khi bảng phụ lục in là "Table C.1"; số La Mã được đổi sang số), hoặc theo trang khi chỉ còn đúng một bảng và một ID chưa ghép. Bảng không ghép được nằm trong kho với `status=no_table_found` và lý do trong `quality.reason`.
- Lớp chữ PDF nối độ lệch chuẩn viết dạng chỉ số dưới vào số chính (`83.20.3`); Docling không tách được, lỗi này được phản ánh trong TEDS và độ khớp số.
- Nhánh ảnh (`source: image`) vẫn giữ để so sánh trên các bảng có ảnh.

## Chỉ số dưới / trên trong ô

LaTeXML viết độ lệch chuẩn bằng `<sub>` (hoặc MathML `msub`). Chữ của ô ghi thành `78.9_{1.2}` (chỉ số trên không phải ký hiệu chú thích: `10^{-3}`) để không dính thành `78.91.2`; ký hiệu chú thích trong `<sup>` vẫn nối liền (`92.4*`).

## Môi trường

SPEC tách hai môi trường (`sciver` với vLLM, `docling`) để tránh xung đột torch. Máy phát triển (Windows, không GPU) không cài vLLM nên dùng **một** `.venv` Python 3.12 với torch 2.15 CPU, sentence-transformers 6.3, Docling 2.150. Trên máy GPU vẫn dùng hai môi trường theo `requirements/`.

## Trường dữ liệu bị loại

`origin_statement`, `perturbed_statement`, `perturbed_explanation` (lộ nhãn) và các trường gán nhãn của paper (`extract_table_info`, `extract_figure_info`, `table_result`, `image_result`, `Chart`, `validation`) không vào kho tài liệu hay model.
