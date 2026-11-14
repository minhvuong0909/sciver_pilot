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
| ảnh trong dataset | đủ 1.256/1.256 ảnh bằng chứng của validation (315/315 bảng bằng chứng của 291 claim có ảnh); nhưng chỉ 379/1.327 bảng của 213 bài có ảnh |
| E0 (Docling 2.150, PDF arXiv) | 1.327 bảng: **1.244 `ok`** (1.180 ghép theo nhãn caption, 64 theo trang), 83 `no_table_found` (51 Docling không thấy bảng, 32 không thấy caption trong PDF); 0 bảng mất chữ ở hàng tiêu đề sau khi bật cell matching (trước đó 238/1.220) |
| normalize (G1, E1) | 7.212 ô đổi: `plus_minus` 3.004, `na_token` 2.162, `thousands_separator` 1.246, `header_arrow_legend` 410, `footnote_mark` 225, `percent_space` 137, `unicode_minus` 32, `ambiguous_separator` 3 |
| quality (E1 so với G1, 1.225 bảng) | TEDS trung bình 0,76 (trung vị 0,87); TEDS-Struct 0,87 (0,976); precision số 0,82 (1,0); recall số 0,95 (1,0) |
| cellmap (E1↔G1) | 1.225 bảng; ô G không ánh xạ được 35.231/97.516 (36%) |
| units | G1: 10.174 đơn vị; E1: 9.986 đơn vị; dài nhất 384 word pieces (tokenizer `all-mpnet-base-v2`); chia ép G1/E1: hàng theo cột 87/71, ô theo từ 39/10, câu theo từ 4/4 — đều có trong `units_log.jsonl` |
| index | embedding `all-mpnet-base-v2` (CPU): G1 10.174 vector, E1 9.986 vector |
| PAIRED_RETR / PAIRED_ORACLE | 271 / 287 trên 291 claim |

## Chạy thử truy xuất (smoke, không dùng để báo cáo)

BM25 trên kho G1, 291 claim, `k_units = 50`: Hit@1 0,83 · Hit@3 0,96 · Recall@3 0,75 · VisualComplete@3 0,70 · TextRecall@3 0,78 · MRR 0,90. VisualComplete@3 của parallel/sequential chỉ ~0,47 (cần hai bảng). Dense và hybrid chưa chạy (cần torch + sentence-transformers).

## Môi trường đã chạy

Windows 11, không GPU, `.venv` Python 3.12: torch 2.15.0+cpu, sentence-transformers 6.3.0, docling 2.150.0, transformers (tokenizer). Docling trên PDF: ~25–60 giây/bài trên CPU. API `DocumentConverter` + `PdfFormatOption`, `TableItem.caption_text`, `export_to_html`, `prov[0].page_no` đã chạy thật.

## Chưa kiểm

- Cấu hình GPU thật cho vLLM (`max_model_len`, bộ nhớ); M4 và verifier chưa chạy với Qwen (phần đánh giá giao cho người khác).
- Nhánh Docling trên ảnh (`source: image`) chưa chạy thật.
- B0 (code SciVer gốc) chưa chạy.
- So số bảng trong paper JSON với trang HTML arXiv.
