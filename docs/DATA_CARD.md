# DATA_CARD

## Tiêu chí lọc (SPEC 5.3)

- **Mặc định:** claim có ≥ 1 bằng chứng thị giác và **mọi** bằng chứng thị giác là bảng; giữ nguyên bằng chứng văn bản.
- **`allow_mixed=true`:** thêm claim có cả bảng và biểu đồ (báo riêng).
- Chọn **toàn bộ** claim thỏa tiêu chí. `claim_id = "{split}-{vị trí trong file}"` (vì `request_id` của dataset bị trùng).
- Trường lộ nhãn `origin_statement`, `perturbed_statement`, `perturbed_explanation` bị bỏ hoàn toàn.

## Validation

| | Số lượng |
|---|---|
| Claim gốc | 1.000 |
| Claim chỉ có bảng (mặc định) | **291** trên 213 bài |
| Claim bảng + biểu đồ (`allow_mixed`) | 194 |
| Phân bố (claim_type / nhãn) | analytical 59T/58F · direct 51T/57F · parallel 10T/22F · sequential 20T/14F |
| Bảng trong kho (mọi bảng của 213 bài) | 1.327 |
| Bảng chuẩn G0 `ok` | 1.307 |
| Bảng loại khỏi G0 | 20 `gold_unavailable` (bảng là ảnh hoặc HTML rỗng) |
| **PAIRED_RETR** (mọi bảng của bài có G ok) | **271** |
| Bảng trích E0 `ok` (Docling trên PDF arXiv) | 1.244 (83 `no_table_found`) |
| **PAIRED_ORACLE** (bảng bằng chứng có G ok và có E) | **287** |

Sau khi lọc, nhãn không còn 50/50: 140 Entailed / 151 Refuted.

## Ảnh bảng và E0

Dataset có đủ ảnh cho 315 bảng làm bằng chứng của 291 claim (dùng cho điều kiện IMG), nhưng chỉ 379/1.327 bảng của 213 bài có ảnh. Vì vậy (quyết định của nhóm, 2026-11-13) E0 được trích bằng Docling từ **PDF arXiv đúng phiên bản** của bài, cho mọi bảng trong kho. Bảng Docling không tìm thấy vẫn nằm trong kho E với `status=no_table_found` để G và E có cùng danh sách nguồn.

## Test

Chưa mở. Sẽ chạy `audit.py` và `select.py` (mã đã đóng băng) ở GĐ B4.
