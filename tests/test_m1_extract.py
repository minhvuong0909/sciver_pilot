"""Docling được thay bằng bộ trích giả: test kiểm phần bao quanh (trạng thái, cache,
bold=None, cùng danh sách bảng với G0), không kiểm chất lượng của Docling."""

from pathlib import Path

from sciverx.m1_prepare.extract_docling import extract_all, extract_table


class FakeExtractor:
    name = "fake@1"

    def __init__(self, html="<table><tr><th>M</th><th>F1</th></tr><tr><td>A</td><td><b>9</b></td></tr></table>"):
        self.html = html
        self.calls = 0

    def extract_html(self, image_path: Path):
        self.calls += 1
        return self.html


def test_missing_image_is_no_table_found(tmp_path):
    rec = extract_table("p", "1", "cap", None, FakeExtractor(), tmp_path)
    assert rec.status == "no_table_found" and rec.cells == [] and rec.quality == {"image_missing": True}
    assert rec.table_record_id == "p::1::E0" and rec.extractor == "fake@1"


def test_extraction_sets_bold_unknown_and_uses_cache(tmp_path):
    img = tmp_path / "t.png"
    img.write_bytes(b"png")
    ext = FakeExtractor()
    a = extract_table("p", "1", "cap", img, ext, tmp_path / "cache")
    b = extract_table("p", "1", "cap", img, ext, tmp_path / "cache")
    assert a.status == "ok" and a.n_header_rows == 1
    assert all(c.bold is None for c in a.cells)
    assert ext.calls == 1 and a == b


def test_no_table_and_error(tmp_path):
    img = tmp_path / "t.png"
    img.write_bytes(b"png")
    assert extract_table("p", "1", "", img, FakeExtractor(html=None), tmp_path / "c1").status == "no_table_found"

    class Boom(FakeExtractor):
        def extract_html(self, image_path):
            raise RuntimeError("ocr failed")
    rec = extract_table("p", "1", "", img, Boom(), tmp_path / "c2")
    assert rec.status == "error" and "ocr failed" in rec.quality["error"]


def test_extract_all_keeps_same_table_list(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "p-Table1-1.png").write_bytes(b"x")
    sources = [{"paper_id": "p", "tables": [
        {"table_id": "1", "caption": "T1", "image_path": "images/p-Table1-1.png"},
        {"table_id": "2", "caption": "T2", "image_path": None}]}]
    recs = extract_all(sources, FakeExtractor(), tmp_path, tmp_path / "cache")
    assert [(r.table_id, r.status) for r in recs] == [("1", "ok"), ("2", "no_table_found")]
