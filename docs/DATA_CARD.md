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
| **PAIRED_ORACLE** (bảng bằng chứng có G ok và có E) | chưa tính — cần E0 từ Docling |

Sau khi lọc, nhãn không còn 50/50: 140 Entailed / 151 Refuted.

## Ảnh bảng

Theo quyết định của nhóm (2026-11-13): chỉ dùng ảnh có sẵn trong dataset. Dataset chỉ có ảnh cho một phần bảng; bảng không có ảnh được giữ trong kho E dưới dạng `no_table_found` (`image_missing`) để G và E có cùng danh sách nguồn. Lấy ảnh/PDF cho các bảng này để làm sau.

## Test

Chưa mở. Sẽ chạy `audit.py` và `select.py` (mã đã đóng băng) ở GĐ B4.
