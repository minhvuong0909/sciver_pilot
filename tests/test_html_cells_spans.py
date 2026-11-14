"""SPEC 12.4 test_html_cells_spans: giữ đúng rowspan/colspan, tiêu đề, chữ đậm."""

import pytest

from sciverx.tables.grid import build_grid, merged_column_headers, row_header
from sciverx.tables.html_cells import parse_html_table, split_footnote_marks

# Tiêu đề 2 tầng (colspan) + ô "Model" rowspan 2 + một ô gộp dọc trong thân bảng.
HTML = """
<table>
 <thead>
  <tr><th rowspan="2">Model</th><th colspan="2">SQuAD</th><th>Speed</th></tr>
  <tr><th>EM</th><th>F1</th><th>(ms) &#8595;</th></tr>
 </thead>
 <tbody>
  <tr><td>BERT</td><td>84.1</td><td>90.9</td><td rowspan="2">120</td></tr>
  <tr><td>Ours</td><td><b>86.0</b></td><td><span class="ltx_font_bold">92.4</span>*</td></tr>
 </tbody>
</table>
"""


def _by_id(cells):
    return {c.cell_id: c for c in cells}


def test_spans_positions_and_header_rows():
    cells, n_header = parse_html_table(HTML)
    assert n_header == 2
    by = _by_id(cells)
    assert (by["r0c0"].rowspan, by["r0c0"].colspan) == (2, 1)
    assert (by["r0c1"].colspan, by["r0c1"].text_raw) == (2, "SQuAD")
    assert by["r0c3"].text_raw == "Speed"
    # hàng 1 bắt đầu ở cột 1 vì cột 0 bị rowspan phủ
    assert [c.cell_id for c in cells if c.row == 1] == ["r1c1", "r1c2", "r1c3"]
    assert by["r2c3"].rowspan == 2
    # hàng 3 không có ô ở cột 3
    assert [c.cell_id for c in cells if c.row == 3] == ["r3c0", "r3c1", "r3c2"]
    assert all(c.is_header for c in cells if c.row < 2)
    assert not any(c.is_header for c in cells if c.row >= 2)


def test_bold_and_footnotes():
    by = _by_id(parse_html_table(HTML)[0])
    assert by["r3c1"].bold is True          # <b>
    assert by["r3c2"].bold is True          # class ltx_font_bold
    assert by["r2c1"].bold is False
    assert by["r3c2"].footnote_marks == ["*"]
    assert by["r3c2"].text_raw == "92.4*"   # text_raw giữ nguyên, chuẩn hóa ở M1.4


def test_grid_repeats_merged_cells_and_merges_headers():
    cells, n_header = parse_html_table(HTML)
    grid = build_grid(cells)
    assert (grid.n_rows, grid.n_cols) == (4, 4)
    assert grid.ids[3][3] == "r2c3"          # ô 120 phủ xuống hàng dưới
    assert grid.text_rows(False)[3][3] == "120"
    assert merged_column_headers(grid, n_header, normalized=False) == [
        "Model", "SQuAD / EM", "SQuAD / F1", "Speed / (ms) ↓"]
    assert row_header(grid, 3, normalized=False) == "Ours"


def test_th_without_thead_counts_leading_header_rows():
    html = "<table><tr><th>A</th><th>B</th></tr><tr><th>x</th><td>1</td></tr></table>"
    cells, n_header = parse_html_table(html)
    assert n_header == 1
    assert _by_id(cells)["r1c0"].is_header


def test_mathml_annotation_is_dropped():
    html = ("<table><tr><td><math><semantics><mrow><mn>93.2</mn><mo>±</mo><mn>0.1</mn></mrow>"
            "<annotation encoding='application/x-tex'>93.2\\pm 0.1</annotation></semantics></math>"
            "</td></tr></table>")
    cells, _ = parse_html_table(html)
    assert cells[0].text_raw == "93.2±0.1"  # khoảng trắng quanh ± do M1.4 chuẩn hóa


def test_br_and_blocks_separate_words():
    cells, _ = parse_html_table("<table><tr><td>Top-1<br/>Acc</td><td><p>a</p><p>b</p></td></tr></table>")
    assert [c.text_raw for c in cells] == ["Top-1 Acc", "a b"]


def test_nested_table_rows_ignored():
    html = "<table><tr><td>a<table><tr><td>inner</td></tr></table></td><td>b</td></tr></table>"
    cells, _ = parse_html_table(html)
    assert [c.cell_id for c in cells] == ["r0c0", "r0c1"]


def test_split_footnote_marks():
    assert split_footnote_marks("93.2*†") == ("93.2", ["*", "†"])
    assert split_footnote_marks("93.2") == ("93.2", [])
    assert split_footnote_marks("*") == ("*", [])


def test_overlapping_cells_rejected():
    from sciverx.schemas import SourceCell
    a = SourceCell(cell_id="r0c0", row=0, col=0, colspan=2, is_header=False, text_raw="a")
    b = SourceCell(cell_id="r0c1", row=0, col=1, is_header=False, text_raw="b")
    with pytest.raises(ValueError):
        build_grid([a, b])


# ---- Dạng LaTeXML thật trong SciVer: bảng viết bằng <span class="ltx_tabular">
SPAN_TABLE = """<figure class="ltx_table"><figcaption>Table 1: x</figcaption><span class="ltx_tabular">
 <span class="ltx_thead"><span class="ltx_tr">
  <span class="ltx_td ltx_th"><span class="ltx_tabular"><span class="ltx_tr"><span class="ltx_td">Generative</span></span>
     <span class="ltx_tr"><span class="ltx_td">model</span></span></span></span>
  <span class="ltx_td ltx_th ltx_colspan ltx_colspan_2">AwA</span></span>
 <span class="ltx_tr"><span class="ltx_td ltx_th"></span><span class="ltx_td ltx_th">S</span><span class="ltx_td ltx_th">U</span></span></span>
 <span class="ltx_tbody"><span class="ltx_tr"><span class="ltx_td ltx_rowspan ltx_rowspan_2">VAE</span>
   <span class="ltx_td">72.6</span><span class="ltx_td">cVAE [7 ###reference_b7###]</span></span>
  <span class="ltx_tr"><span class="ltx_td">1</span><span class="ltx_td">2</span></span></span></span></figure>"""


def test_latexml_span_tabular_with_class_spans_and_nested_header():
    from sciverx.tables.html_cells import parse_table_html
    cells, n_header, parts = parse_table_html(SPAN_TABLE)
    by = _by_id(cells)
    assert (n_header, parts) == (2, 1)
    assert by["r0c0"].text_raw == "Generative model"   # bảng lồng trong ô tiêu đề
    assert by["r0c1"].colspan == 2 and by["r2c0"].rowspan == 2
    assert by["r2c2"].text_raw == "cVAE [7]"            # bỏ dấu vết ###reference_b7###
    assert [c.cell_id for c in cells if c.row == 3] == ["r3c1", "r3c2"]
    grid = build_grid(cells)
    assert merged_column_headers(grid, n_header, normalized=False) == ["Generative model", "AwA / S",
                                                                       "AwA / U"]


def test_multi_part_tables_are_stacked():
    from sciverx.tables.html_cells import parse_table_html
    html = ("<figure><table><thead><tr><th>A</th><th>x</th></tr></thead><tr><td>a</td><td>1</td></tr></table>"
            "<table><thead><tr><th>B</th><th>y</th></tr></thead><tr><td>b</td><td>2</td></tr></table></figure>")
    cells, n_header, parts = parse_table_html(html)
    assert (n_header, parts) == (1, 2)
    assert [c.cell_id for c in cells] == ["r0c0", "r0c1", "r1c0", "r1c1", "r2c0", "r2c1", "r3c0", "r3c1"]
    assert _by_id(cells)["r2c0"].is_header


def test_guess_header_rows_when_not_declared():
    # LaTeXML chỉ đánh ltx_th_row cho cột đầu, không có thead
    html = ("<table><tr><th class='ltx_th_row'>Model</th><td>RTE</td><td>MRPC</td></tr>"
            "<tr><th class='ltx_th_row'>BERT</th><td>66.4</td><td>84.8</td></tr></table>")
    cells, n_header = parse_html_table(html)
    assert n_header == 1
    numeric_first = "<table><tr><td>a</td><td>1</td></tr><tr><td>b</td><td>2</td></tr></table>"
    assert parse_html_table(numeric_first)[1] == 0


def test_sub_and_sup_scripts_are_kept_apart():
    html = ("<table><tr><td>78.9<sub class='ltx_sub'>1.2</sub></td><td>92.4<sup>*</sup></td>"
            "<td>10<sup>-3</sup></td><td><math><msub><mn>51.9</mn><mn>0.2</mn></msub>"
            "<annotation>51.9_{0.2}</annotation></math></td></tr></table>")
    cells, _ = parse_html_table(html)
    assert [c.text_raw for c in cells] == ["78.9_{1.2}", "92.4*", "10^{-3}", "51.9_{0.2}"]
    assert cells[1].footnote_marks == ["*"]
