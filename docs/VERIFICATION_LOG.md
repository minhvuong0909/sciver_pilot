# VERIFICATION_LOG

Ghi lại các giả định [CẦN KIỂM] của SPEC đã được kiểm trên dữ liệu thật và kết quả. Chỉ dùng **validation**; `testset.json` chưa được tải (SPEC 2.1, 5.4).

## Nguồn

| Thành phần | Phiên bản |
|---|---|
| Dataset `chengyewang/SciVer` (Hugging Face) | commit `5695ac384d247ae5df2a7b73568dc63d189296b0` (2025-06-19) |
| Code gốc `QDRhhhh/SciVer` (GitHub) | commit `0a97282dec3f500e0b10d12ea4d3b3eaf19993e3` (2025-07-24) |
| Tải về bằng | `scripts/download_sciver_val.py` (chỉ `valset.json`, paper JSON và ảnh của validation) |

## Audit D1–D8 (validation, `python -m sciverx audit`)

| # | Kết quả |
|---|---|
| D1 | `valset.json`, 1.000 claim (`testset.json` 2.000 claim, chưa mở) |
| D2 | Đủ mọi trường SPEC 5.1. Trường thêm: `paperid`, `request_id`, `origin_statement`, `perturbed_statement`, `perturbed_explanation`. **Ba trường cuối lộ nhãn**: `claim == origin_statement` ở 505 claim, `claim == perturbed_statement` ở 495 claim; `perturbed_explanation` giải thích vì sao claim sai. Ba trường này bị bỏ hoàn toàn ở `select.py`, không vào `ClaimInput` lẫn `GoldInfo`. `request_id` **không duy nhất** → `claim_id = "{split}-{vị trí trong file}"`. |
| D3 | `tables[id]` = `{table_html, capture}`. **Mọi bảng (2.590/2.590) có `table_html`** là LaTeXML của arXiv đúng phiên bản bài → bảng chuẩn G0 lấy thẳng từ dataset (`extractor=gold_json`), không cần tải HTML arXiv. |
| D4 | 405 bài, 2.590 bảng (6,4 bảng/bài), 10,5 biểu đồ/bài. Paper JSON có **mọi** bảng của bài. **Ảnh**: dataset chỉ có ảnh cho một phần bảng (mẫu 15 bài: 29/85 bảng có ảnh `images/{paperid}-Table{N}-1.png`), nhiều khả năng chỉ bảng từng làm bằng chứng. Paper JSON còn có các trường sinh ra khi gán nhãn (`extract_table_info`, `extract_figure_info`, `table_result`, `image_result`, `Chart`, `validation`) — liên kết câu ↔ item, **không** đưa vào kho M1. Chưa so số bảng với trang HTML arXiv. |
| D5 | 7.217 section, 614 section rỗng (tiêu đề cha). Nhãn `section` ghi tới cấp 1 (210), cấp 2 (783), cấp 3 (150) → chấm truy xuất văn bản ở mức section: section lá khớp section chuẩn khi trùng hoặc là tiểu mục của nó. |
| D6 | ID bảng là số (`"1"`, `"4"`); ID biểu đồ có thể là hình con (`"3(b)"`). Mọi `paperid` có phiên bản (`2411.03788v1`). |
| D7 | Theo trạng thái bằng chứng: `all_table` 291, `mixed` (bảng + biểu đồ) 194, `no_table` 515. Chọn mặc định 291 claim: analytical 117 (T59/F58), direct 108 (T51/F57), parallel 32 (T10/F22), sequential 34 (T20/F14). |
| D8 | 315 bảng bằng chứng duy nhất trên 213 bài; 41 bảng dùng chung bởi nhiều claim (83 claim). |

## Bảng HTML thật khác giả định của bản pilot

Kiểm trên toàn bộ 2.590 bảng của 405 bài:

- 142 bảng viết bằng `<span class="ltx_tabular">` (`ltx_tr`, `ltx_td`, `ltx_th`), span ghi bằng class `ltx_colspan_N` / `ltx_rowspan_N` → parser đọc cả hai dạng.
- 68 bảng có nhiều bảng con ngoài cùng (panel cạnh nhau) → xếp chồng theo chiều dọc, đưa vào danh sách kiểm tay.
- ~1.300 bảng không khai báo hàng tiêu đề (LaTeXML chỉ đánh `ltx_th_row` cho cột đầu) → đoán hàng tiêu đề bằng luật của bản pilot, dùng chung cho G và E.
- Chữ trong ô còn dấu vết tiền xử lý `###reference_b7###` → bỏ đi.
- Sau khi sửa: 2.555 bảng đọc được, 33 không có bảng HTML (ảnh hoặc rỗng), 2 bảng lỗi span.

## M1 trên 291 claim đã chọn

| Bước | Kết quả |
|---|---|
| sources | 213 bài, 1.327 bảng, 2.235 biểu đồ |
| gold (G0) | 1.307 `ok`, 20 `gold_unavailable`; 154 bảng vào `annotation/gold_audit.csv` (lỗi, nhiều phần, mẫu ngẫu nhiên phân tầng seed 2026) — **[DỪNG & HỎI] H2: chờ người kiểm tay** |
| normalize (G1) | 2.866 ô đổi: `na_token` 1.113, `plus_minus` 894, `thousands_separator` 635, `footnote_mark` 142, `percent_space` 49, `unicode_minus` 32, `ambiguous_separator` 3 |
| units (G1) | 10.167 đơn vị (7.426 paragraph, 903 table_full, 1.838 table_rows), dài nhất 384 word pieces (tokenizer `all-mpnet-base-v2`); chia ép: 87 hàng theo cột, 39 ô theo từ, 4 câu theo từ — đều có trong `units_log.jsonl` |
| PAIRED_RETR | 271/291 claim |
| E0 (Docling) | **chưa chạy**: máy phát triển không có môi trường Docling |

## Chạy thử truy xuất (smoke, không dùng để báo cáo)

BM25 trên kho G1, 291 claim, `k_units = 50`: Hit@1 0,83 · Hit@3 0,96 · Recall@3 0,75 · VisualComplete@3 0,70 · TextRecall@3 0,78 · MRR 0,90. VisualComplete@3 của parallel/sequential chỉ ~0,47 (cần hai bảng). Dense và hybrid chưa chạy (cần torch + sentence-transformers).

## Chưa kiểm

- Cấu hình GPU thật cho vLLM (`max_model_len`, bộ nhớ), API Docling (`ImageFormatOption`, `export_to_html`).
- B0 (code SciVer gốc) chưa chạy.
- So số bảng trong paper JSON với trang HTML arXiv.
