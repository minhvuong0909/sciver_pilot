"""SPEC 12.4 test_cell_mapping: ca một–nhiều, ca không ghép được, ca lệch hàng."""

from sciverx.schemas import TableRecord
from sciverx.tables.cell_mapping import map_cells, map_e_cells_to_g
from sciverx.tables.html_cells import parse_html_table

HEAD = "<thead><tr><th>Model</th><th>EM</th><th>F1</th></tr></thead>"


def rec(body, cond, head=HEAD):
    cells, n_header = parse_html_table(f"<table>{head}<tbody>{body}</tbody></table>")
    return TableRecord(table_record_id=f"p::T::{cond}", paper_id="p", table_id="T", condition=cond,
                       extractor="test", caption="", cells=cells, n_header_rows=n_header,
                       status="ok")


G = rec("<tr><td>BERT</td><td>84.1</td><td>90.9</td></tr>"
        "<tr><td>Ours</td><td>86.0</td><td>92.4</td></tr>", "G1")


def pairs(m):
    return {p["e_cell_id"]: (p["g_cell_ids"], p["method"]) for p in m.pairs}


def test_identical_tables_map_one_to_one():
    m = map_cells(rec("<tr><td>BERT</td><td>84.1</td><td>90.9</td></tr>"
                      "<tr><td>Ours</td><td>86.0</td><td>92.4</td></tr>", "E1"), G)
    assert m.unmatched_e == [] and m.unmatched_g == []
    p = pairs(m)
    assert p["r1c1"] == (["r1c1"], "header_key")
    assert p["r0c0"] == (["r0c0"], "content")  # ô tiêu đề ghép theo nội dung
    assert (m.e_condition, m.g_condition) == ("E1", "G1")


def test_one_to_many_when_extraction_merges_cells():
    # Docling gộp hai ô EM, F1 của hàng Ours thành một ô colspan=2
    e = rec("<tr><td>BERT</td><td>84.1</td><td>90.9</td></tr>"
            "<tr><td>Ours</td><td colspan='2'>86.0 92.4</td></tr>", "E1")
    m = map_cells(e, G)
    assert pairs(m)["r2c1"] == (["r2c1", "r2c2"], "header_key")
    assert m.unmatched_g == []


def test_unmatched_cells_reported():
    # E mất hàng Ours và có một hàng rác
    e = rec("<tr><td>BERT</td><td>84.1</td><td>90.9</td></tr>"
            "<tr><td>@@@</td><td>zz</td><td>qq</td></tr>", "E1")
    m = map_cells(e, G)
    assert set(m.unmatched_e) == {"r2c0", "r2c1", "r2c2"}
    assert set(m.unmatched_g) == {"r2c0", "r2c1", "r2c2"}


def test_row_shift_matched_by_header_not_position():
    # E có thêm một hàng trống ở đầu thân bảng → mọi hàng lệch xuống 1
    e = rec("<tr><td></td><td></td><td></td></tr>"
            "<tr><td>BERT</td><td>84.1</td><td>90.9</td></tr>"
            "<tr><td>Ours</td><td>86.0</td><td>92.4</td></tr>", "E1")
    m = map_cells(e, G)
    p = pairs(m)
    assert p["r2c2"] == (["r1c2"], "header_key")
    assert p["r3c1"] == (["r2c1"], "header_key")
    assert {"r1c0", "r1c1", "r1c2"} <= set(m.unmatched_e)


def test_content_match_respects_row_order():
    # Tiêu đề cột E bị OCR sai nên bước 1 thất bại; bước 2 ghép theo nội dung, giữ thứ tự hàng
    e = rec("<tr><td>BERT</td><td>84.1</td><td>90.9</td></tr>"
            "<tr><td>Ours</td><td>86.0</td><td>92.4</td></tr>", "E1",
            head="<thead><tr><th>Mdl</th><th>E M</th><th>Fl</th></tr></thead>")
    m = map_cells(e, G)
    p = pairs(m)
    assert p["r1c1"][0] == ["r1c1"] and p["r2c2"][0] == ["r2c2"]
    assert p["r1c1"][1] == "content"


def test_map_e_cells_to_g():
    e = rec("<tr><td>BERT</td><td>84.1</td><td>90.9</td></tr>"
            "<tr><td>Ours</td><td colspan='2'>86.0 92.4</td></tr>", "E1")
    m = map_cells(e, G)
    g_ids, missing = map_e_cells_to_g(m, ["r2c1", "r9c9"])
    assert g_ids == {"r2c1", "r2c2"} and missing == ["r9c9"]
