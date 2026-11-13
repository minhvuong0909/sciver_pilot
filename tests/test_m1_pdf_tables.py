"""E0 từ PDF: tìm trang có caption, ghép bảng Docling với table_id. Docling và PDF được giả lập."""

from pathlib import Path

from sciverx.m1_prepare.extract_docling import extract_paper_from_pdf
from sciverx.m1_prepare.pdf_tables import (FoundTable, caption_number, caption_pages, match_tables,
                                           page_groups, roman_to_int)

HTML = "<table><tr><th>M</th><th>F1</th></tr><tr><td>A</td><td>{v}</td></tr></table>"


def test_caption_number_arabic_and_roman():
    assert caption_number("Table 3: Results on SQuAD") == "3"
    assert caption_number("TABLE IV. Ablation") == "4"
    assert caption_number("As shown in Table 2, ...") is None or caption_number("As shown in Table 2, ...") == "2"
    assert caption_number("Figure 2: curve") is None
    assert roman_to_int("XII") == 12 and roman_to_int("IX") == 9


def test_caption_pages_only_counts_caption_lines():
    pages = ["Intro. As shown in Table 2, ours wins.\nTable 1: Setup", "Text\nTable 2: Main results\n...",
             "TABLE III: Ablation"]
    assert caption_pages(pages, ["1", "2", "3", "4"]) == {"1": [1], "2": [2], "3": [3]}


def test_page_groups():
    assert page_groups([5, 2, 3, 3, 9]) == [(2, 3), (5, 5), (9, 9)]


def test_match_by_caption_then_by_page():
    found = [FoundTable(2, "Table 2: Main", HTML.format(v=1), 4), FoundTable(3, "", HTML.format(v=2), 4),
             FoundTable(3, "Table 9: other paper part", HTML.format(v=3), 4)]
    m = match_tables(found, {"2": [2], "3": [3], "5": [7]})
    assert m["2"][1] == "caption" and m["3"][1] == "page" and "5" not in m


def test_page_match_refused_when_ambiguous():
    found = [FoundTable(3, "", "a", 1), FoundTable(3, "", "b", 1)]
    assert match_tables(found, {"3": [3]}) == {}


class FakeFinder:
    name = "docling-pdf@fake"

    def __init__(self):
        self.calls = []

    def find(self, pdf, page_range):
        self.calls.append(page_range)
        return [FoundTable(p, f"Table {p}: x", HTML.format(v=p), 4)
                for p in range(page_range[0], page_range[1] + 1)]


def test_extract_paper_from_pdf_statuses_and_cache(tmp_path):
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF fake")
    src = {"paper_id": "p", "tables": [{"table_id": t, "caption": f"Table {t}"} for t in ("1", "2", "3")]}
    texts = lambda p: ["Table 1: a", "Table 2: b", "no caption here"]
    finder = FakeFinder()
    recs = extract_paper_from_pdf(src, pdf, finder, texts, tmp_path / "cache")
    assert [(r.table_id, r.status) for r in recs] == [("1", "ok"), ("2", "ok"), ("3", "no_table_found")]
    assert recs[0].quality == {"source": "pdf", "page": 1, "match": "caption", "label": "1"}
    assert recs[2].quality["reason"] == "caption_not_found"
    assert all(c.bold is None for c in recs[0].cells) and recs[0].extractor == "docling-pdf@fake"
    assert finder.calls == [(1, 2)]
    again = extract_paper_from_pdf(src, pdf, finder, texts, tmp_path / "cache")
    assert again == recs and finder.calls == [(1, 2)]   # lần hai đọc cache


def test_missing_pdf_keeps_every_table(tmp_path):
    src = {"paper_id": "p", "tables": [{"table_id": "1", "caption": "T1"}]}
    recs = extract_paper_from_pdf(src, None, FakeFinder(), lambda p: [], tmp_path)
    assert recs[0].status == "no_table_found" and recs[0].quality["reason"] == "pdf_missing"


def test_appendix_labels_map_dataset_ids_to_printed_labels(tmp_path):
    from sciverx.m1_prepare.pdf_tables import table_labels
    tables = [{"table_id": "6", "caption": "Table 6: Main"}, {"table_id": "7", "caption": "Table C.1: Stats"},
              {"table_id": "8", "caption": "TABLE IV. Ablation"}, {"table_id": "9", "caption": ""}]
    assert table_labels(tables) == {"6": "6", "7": "C.1", "8": "4", "9": "9"}
    assert caption_pages(["x\nTable C.1: Stats", "Table 6: Main\nsee Table C.1 for"], ["C.1", "6"]) == {
        "C.1": [1], "6": [2]}

    class Finder(FakeFinder):
        def find(self, pdf, page_range):
            return [FoundTable(1, "Table C.1: Stats", HTML.format(v=1), 4)]
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF")
    src = {"paper_id": "p", "tables": tables[1:2]}
    rec = extract_paper_from_pdf(src, pdf, Finder(), lambda p: ["Table C.1: Stats"], tmp_path / "c")[0]
    assert rec.table_id == "7" and rec.status == "ok" and rec.quality["label"] == "C.1"
