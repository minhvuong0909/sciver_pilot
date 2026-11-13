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
| `max_model_len` | Mặc định của vLLM | 32768 (giảm nếu GPU không đủ, ghi vào VERIFICATION_LOG) | SPEC 8.1 — **chưa thử trên GPU** |

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
- **Dấu vết tiền xử lý** `###reference_xx###` trong chữ ô bị bỏ.

## Truy xuất

- Khóa cha có tiền tố (`section:1`, `table:1`) vì ID section và ID bảng trùng nhau trong một bài.
- Section truy xuất được tính khớp section chuẩn khi trùng hoặc là tiểu mục của nó (nhãn ghi tới cấp 1–3, kho chia theo section lá).
- Ô bảng quá dài (cả đoạn văn trong một ô) được chia theo từ, mỗi phần lặp lại caption và tiêu đề.

## Ảnh và E0

Theo quyết định của nhóm (2026-11-13): chỉ dùng ảnh có trong dataset. Bảng không có ảnh nằm trong kho E với `status=no_table_found`, `quality.image_missing=true`. Việc lấy ảnh/PDF cho các bảng này để sau.

## Trường dữ liệu bị loại

`origin_statement`, `perturbed_statement`, `perturbed_explanation` (lộ nhãn) và các trường gán nhãn của paper (`extract_table_info`, `extract_figure_info`, `table_result`, `image_result`, `Chart`, `validation`) không vào kho tài liệu hay model.
