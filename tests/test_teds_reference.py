"""SPEC 12.4 test_teds_reference: TEDS(x,x)=1 và ≥ 3 cặp bảng có thay đổi cấu trúc
với giá trị tham chiếu tính tay.

Bảng gốc A (không tiêu đề): table > tbody > 2 tr > 2 td = 8 nút.
"""

import pytest

from sciverx.schemas import TableRecord
from sciverx.tables.html_cells import parse_html_table
from sciverx.tables.numeric import numbers_in_text, numeric_precision_recall
from sciverx.tables.teds import build_tree, normalized_levenshtein, teds, tree_size


def rec(html, cond="G1", n_header=0):
    cells, _ = parse_html_table(html)  # số hàng tiêu đề cố định để giữ giá trị tính tay
    return TableRecord(table_record_id=f"p::T::{cond}", paper_id="p", table_id="T", condition=cond,
                       extractor="test", caption="", cells=cells, n_header_rows=n_header,
                       status="ok")


A = rec("<table><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr></table>")


def test_tree_size_and_identity():
    assert tree_size(build_tree(A.cells, A.n_header_rows)) == 8
    assert teds(A, A) == 1.0
    assert teds(A, A, structure_only=True) == 1.0


@pytest.mark.parametrize("html,expected,expected_struct", [
    # mất hàng: xóa 1 tr + 2 td = 3 → 1 - 3/8
    ("<table><tr><td>a</td><td>b</td></tr></table>", 0.625, 0.625),
    # gộp ô: td(a) → td(ab, colspan=2) khác span = 1, xóa td(b) = 1 → 1 - 2/8
    ("<table><tr><td colspan='2'>ab</td></tr><tr><td>c</td><td>d</td></tr></table>", 0.75, 0.75),
    # thêm cột (lệch cột): chèn 2 td, cây 10 nút → 1 - 2/10
    ("<table><tr><td></td><td>a</td><td>b</td></tr><tr><td></td><td>c</td><td>d</td></tr></table>",
     0.8, 0.8),
    # chỉ đổi nội dung một ô: Levenshtein("b","x")/1 = 1 → 1 - 1/8; cấu trúc không đổi
    ("<table><tr><td>a</td><td>x</td></tr><tr><td>c</td><td>d</td></tr></table>", 0.875, 1.0),
])
def test_teds_reference_pairs(html, expected, expected_struct):
    b = rec(html, cond="E1")
    assert teds(b, A) == pytest.approx(expected)
    assert teds(b, A, structure_only=True) == pytest.approx(expected_struct)
    assert teds(A, b) == pytest.approx(expected)  # đối xứng


def test_partial_content_cost():
    a = rec("<table><tr><td>93.2</td></tr></table>")
    b = rec("<table><tr><td>93.4</td></tr></table>", cond="E1")
    # table, tbody, tr, td = 4 nút; Levenshtein 1/4 → 1 - 0.25/4
    assert teds(b, a) == pytest.approx(1 - 0.25 / 4)


def test_header_section_is_structure():
    a = rec("<table><thead><tr><th>h</th></tr></thead><tr><td>1</td></tr></table>", n_header=1)
    b = rec("<table><tr><td>h</td></tr><tr><td>1</td></tr></table>", cond="E1")
    # a: table, thead, tr, td, tbody, tr, td = 7; b: table, tbody, 2 tr, 2 td = 6
    # b → a: xóa tbody (2 tr lên làm con table), chèn thead bọc tr đầu, chèn tbody
    # bọc tr sau = 3 → 1 - 3/7
    assert teds(b, a) == pytest.approx(1 - 3 / 7)


def test_empty_tables():
    empty = A.model_copy(update={"cells": []})
    assert teds(empty, A) == 0.0
    assert teds(empty, empty) == 1.0


def test_normalized_levenshtein():
    assert normalized_levenshtein("kitten", "sitting") == pytest.approx(3 / 7)
    assert normalized_levenshtein("", "") == 0.0


def test_numbers_and_multiset_pr():
    assert numbers_in_text("93.20 [*] vs -1.0 and gpt-4o, F1, top-1, 85.3%") == [
        "93.2", "-1", "1", "85.3"]
    g = rec("<table><tr><td>1.0</td><td>2</td><td>2</td></tr></table>")
    e = rec("<table><tr><td>1</td><td>2</td><td>3</td></tr></table>", cond="E1")
    p, r = numeric_precision_recall(e, g)
    assert (p, r) == (pytest.approx(2 / 3), pytest.approx(2 / 3))


def test_numbers_in_script_notation():
    assert numbers_in_text("78.9_{1.2}") == ["78.9", "1.2"]
    assert numbers_in_text("10^{-3}") == ["10", "-3"]
