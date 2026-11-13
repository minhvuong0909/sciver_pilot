---
name: git-conventions
description: Quy tắc đặt tên nhánh, viết commit message và mô tả pull request cho repo sciver_pilot. Dùng mỗi khi tạo nhánh, commit, đổi tên nhánh, mở hoặc cập nhật pull request, hoặc khi được hỏi về quy ước git của dự án.
---

# Quy ước Git của sciver_pilot

Áp dụng cho mọi người và mọi agent làm việc trên repo này.

## 1. Luồng làm việc

- `main` được bảo vệ: không push thẳng, không force-push, không xóa.
- Mọi thay đổi đi qua nhánh riêng và pull request (PR) vào `main`.
- Chỉ chủ repo (`@minhvuong0909`) duyệt và merge PR. Người khác không tự merge, kể cả khi PR đã được duyệt.
- Merge kiểu **squash**; tiêu đề PR trở thành commit trên `main`, nên tiêu đề PR phải theo đúng mục 3.
- Xóa nhánh sau khi merge.

## 2. Tên nhánh

Định dạng:

```
<type>/<scope>-<mo-ta-ngan>
```

- Chỉ dùng chữ thường không dấu, số và dấu gạch nối (kebab-case). Không dấu cách, không gạch dưới, không tiếng Việt có dấu.
- `<type>` lấy từ bảng ở mục 3.
- `<scope>` lấy từ danh sách ở mục 4; bỏ được nếu thay đổi không thuộc module nào.
- Mô tả 2–5 từ, tổng độ dài tên nhánh ≤ 50 ký tự.
- Một nhánh làm một việc. Việc mới thì tạo nhánh mới từ `main` mới nhất.

Ví dụ đúng:

```
feat/m3-bm25-tokenizer
fix/tables-merged-cell-span
docs/spec-budget-rules
test/m5-numeric-tolerance
chore/add-gitignore
exp/m3-rrf-constant-sweep
```

Ví dụ sai: `Feature_M3`, `fix bug`, `vuong-test`, `sua-loi-bảng`, `dev2`.

## 3. Commit message

Theo Conventional Commits:

```
<type>(<scope>): <tiêu đề>

<thân bài>

<footer>
```

### Tiêu đề

- Tối đa 72 ký tự, không dấu chấm cuối câu.
- Viết ở thể mệnh lệnh, mô tả việc commit làm: `thêm`, `sửa`, `bỏ`, `đổi` — không viết `đã thêm`, `đang sửa`.
- Viết bằng tiếng Việt; giữ nguyên tên kỹ thuật (`BM25`, `TEDS`, `parse_error`, tên hàm, tên file).
- `<scope>` bỏ được nếu thay đổi trải trên nhiều module.

### Bảng `type`

| Type | Dùng khi |
|---|---|
| `feat` | Thêm chức năng mới |
| `fix` | Sửa lỗi |
| `docs` | Chỉ đổi tài liệu (README, SPEC, `docs/`) |
| `test` | Thêm hoặc sửa test, fixture, mock |
| `refactor` | Đổi cấu trúc code, không đổi hành vi |
| `perf` | Tăng tốc hoặc giảm bộ nhớ, không đổi kết quả |
| `exp` | Cấu hình hoặc script thí nghiệm, kết quả chạy được ghi lại |
| `build` | Phụ thuộc, `requirements/`, file lock, đóng gói |
| `ci` | Workflow CI, hook, cấu hình repo |
| `chore` | Việc lặt vặt không thuộc các loại trên |
| `revert` | Hoàn tác một commit trước |

### Thân bài

Bắt buộc khi thay đổi không hiển nhiên từ tiêu đề. Cách tiêu đề một dòng trống, xuống dòng ở khoảng 72 ký tự. Trả lời:

1. **Vì sao** cần thay đổi này (lỗi gì, yêu cầu nào trong SPEC).
2. **Thay đổi gì** ở mức hành vi, không kể lại từng dòng diff.
3. **Ảnh hưởng** tới kết quả: chỉ số, cache, định dạng dữ liệu, cấu hình đã đóng băng.

Thay đổi làm lệch khỏi SPEC hoặc khỏi paper gốc phải nêu rõ trong thân bài và ghi vào `docs/DEVIATIONS.md`.

### Footer

- `BREAKING CHANGE: <mô tả>` khi đổi schema, định dạng file đầu ra, hoặc làm cache cũ mất hiệu lực. Thêm `!` sau scope: `feat(schemas)!: ...`.
- `Refs: #12` hoặc `Closes: #12` khi có issue liên quan.
- `Co-Authored-By: ...` khi có đồng tác giả.

### Ví dụ

```
feat(m3): thêm tokenizer giữ nguyên số thập phân và tên model

BM25 của rank_bm25 nhận token do mình đưa vào, nên tokenizer mặc định
tách `93.2` thành `93` và `2`, làm mất khớp với ô số trong bảng.

Tokenizer mới dùng regex ở SPEC mục M3, giữ nguyên `93.2`, `gpt-4o`,
`f1@5`, `top-1`, `12.5%`. Ghi `tokenizer_version: v1` vào params.

BREAKING CHANGE: chỉ mục BM25 dựng bằng tokenizer cũ phải dựng lại.
Refs: #7
```

```
fix(tables): tính đúng span khi ô gộp nằm ở hàng tiêu đề cuối
```

```
docs(spec): làm rõ parse_error tách khỏi not_found
```

Ví dụ sai: `update`, `fix bug`, `WIP`, `sửa lại code m3 cho chạy được.`, `Feat: Add Tokenizer`.

### Quy tắc khác

- Mỗi commit một thay đổi logic, tự nó chạy được và qua test.
- Không commit `data/`, `outputs/`, file `.npy`, cache, khóa API hay file `.env`.
- Không dùng `--no-verify` để bỏ qua hook.
- Không sửa lịch sử (`rebase`, `amend`, force-push) trên nhánh người khác đang dùng.

## 4. Danh sách `scope`

| Scope | Phạm vi |
|---|---|
| `schemas` | `src/sciverx/schemas.py` |
| `data` | `src/sciverx/data/` — audit, select, loader |
| `tables` | `src/sciverx/tables/` — ô gộp, chuẩn hóa, linearize, TEDS |
| `m1` | `m1_prepare/` — nguồn, bảng chuẩn, Docling, đơn vị, chỉ mục |
| `m3` | `m3_retrieve/` — tokenizer, BM25, dense, RRF, ngân sách |
| `m4` | `m4_align/` — căn chỉnh và kiểm định |
| `m5` | `m5_numeric/` — phân tích số, quan hệ, dung sai |
| `m6` | `m6_package/` — đóng gói |
| `verifier` | `verifier/` — prompt, dựng đầu vào, vLLM, chấm điểm |
| `eval` | `evaluation/` — chỉ số, thống kê, báo cáo |
| `annotation` | `annotation/` — xuất/nhập nhãn, độ đồng thuận |
| `utils` | `utils/` — JSONL, hash, run_id, manifest, seed, logging |
| `configs` | `configs/` |
| `spec` | `SPEC_SciVer_TableVerify.md` |
| `readme` | `README.md` |
| `deps` | `requirements/` |
| `repo` | `.github/`, `.claude/`, `.gitignore`, cấu hình repo |

Module mới thì thêm scope vào bảng này trong cùng PR.

## 5. Pull request

**Tiêu đề:** đúng định dạng tiêu đề commit ở mục 3.

**Mô tả** theo mẫu:

```markdown
## Mục đích
Vì sao cần PR này; dẫn mục SPEC hoặc issue liên quan.

## Thay đổi
- Gạch đầu dòng theo hành vi, không theo file.

## Kiểm thử
- Lệnh đã chạy và kết quả (vd. `pytest tests/test_tokenizer.py` — 12 passed).
- Nếu chưa chạy được phần nào thì ghi rõ.

## Ảnh hưởng
- Chỉ số, cache, schema, cấu hình đóng băng bị đổi; hoặc "Không".
- Lệch khỏi SPEC/paper: mô tả và dẫn dòng trong `docs/DEVIATIONS.md`; hoặc "Không".
```

- PR nhỏ, một mục đích; nên dưới khoảng 400 dòng thay đổi.
- PR chưa xong thì mở ở dạng draft.
- Không tự merge. Chờ `@minhvuong0909` duyệt và merge.

## 6. Kiểm tra trước khi commit hoặc mở PR

1. Đang đứng trên nhánh riêng, không phải `main`.
2. Tên nhánh đúng mục 2.
3. `git status` không có file dữ liệu, đầu ra, cache hay bí mật.
4. Test liên quan đã chạy và qua.
5. Tiêu đề commit đúng `<type>(<scope>): <tiêu đề>`, ≤ 72 ký tự.
6. Thân bài nêu được vì sao và ảnh hưởng, nếu thay đổi không hiển nhiên.
