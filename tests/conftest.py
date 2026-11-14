"""Bộ dữ liệu SciVer thu nhỏ, cấu trúc chép theo dữ liệu validation thật (tên trường,
đường dẫn "./SciVer/...", tables[id]{table_html, capture}). Nội dung là giả."""

import json
from pathlib import Path

import pytest

TABLE_HTML = """<figure class="ltx_table" id="S4.T{n}"><figcaption>Table {n}: Results.</figcaption>
<table class="ltx_tabular"><thead><tr><th>Model</th><th>F1</th></tr></thead>
<tbody><tr><td>BERT</td><td>90.9</td></tr><tr><td>Ours</td><td>92.4</td></tr></tbody></table></figure>"""


def _paper(pid, n_tables=2, n_charts=1):
    return {
        "title": f"Paper {pid}", "abstract": "Abstract.", "url": f"http://arxiv.org/html/{pid}",
        "sections": [
            {"section_id": "1", "parent_section_id": None, "section_name": "Introduction", "text": "Intro text."},
            {"section_id": "4", "parent_section_id": None, "section_name": "Experiments", "text": ""},
            {"section_id": "4.1", "parent_section_id": "4", "section_name": "Results",
             "text": "Our model reaches 92.4 F1.\n\nBERT reaches 90.9 F1."},
        ],
        "appendix": [], "references": [],
        "tables": {str(n): {"table_html": TABLE_HTML.format(n=n), "capture": f"Table {n}: Results."}
                   for n in range(1, n_tables + 1)},
        "image_paths": {str(n_tables + k): {"figure_path": f"{pid}_figure_{k}.png",
                                            "caption": f"Figure {k}: Curve.", "url": "x"}
                        for k in range(1, n_charts + 1)},
        "validation": True, "Chart": [], "extract_table_info": {"1": {"4.1": ["Our model reaches 92.4 F1."]}},
        "extract_figure_info": {}, "table_result": {"1": {"4.1": {"related_sentences_ids": [0]}}},
        "image_result": {},
    }


def _q(pid, ctype, label, **kw):
    claim = f"{ctype} claim about {pid}"
    base = {"paperid": pid, "paper_path": f"./SciVer/papers/{pid}.json", "claim_type": ctype,
            "section": ["4.1"], "request_id": 7, "claim": claim, "label": label,
            "origin_statement": claim if label else "orig", "perturbed_statement": "pert" if label else claim,
            "perturbed_explanation": "LEAK: this statement is incorrect because ..."}
    base.update(kw)
    return base


def _img(pid, n):
    return f"./SciVer/images/{pid}-Table{n}-1.png"


@pytest.fixture
def mini_sciver(tmp_path) -> Path:
    root = tmp_path / "SciVer"
    (root / "papers").mkdir(parents=True)
    (root / "images").mkdir()
    for pid in ("2409.00001v1", "2409.00002v2"):
        (root / "papers" / f"{pid}.json").write_text(json.dumps(_paper(pid)), encoding="utf-8")
    a, b = "2409.00001v1", "2409.00002v2"
    rows = [
        _q(a, "direct", True, type="table", item="1", image_path=_img(a, 1)),
        _q(a, "direct", False, type="chart", item="3", image_path=f"./SciVer/images/{a}_figure_1.png"),
        _q(b, "parallel", False, item1_type="table", item1="1", item1_path=_img(b, 1),
           item2_type="table", item2="2", item2_path=_img(b, 2)),
        _q(b, "sequential", True, item1_type="chart", item1="3",
           item1_path=f"./SciVer/images/{b}_figure_1.png", item2_type="table", item2="2",
           item2_path=_img(b, 2)),
        _q(b, "analytical", True, type="table", item="2", image_path=_img(b, 2)),
    ]
    (root / "valset.json").write_text(json.dumps(rows), encoding="utf-8")
    # chỉ bảng làm bằng chứng mới có ảnh, như dữ liệu thật
    for p in (_img(a, 1), _img(b, 1), _img(b, 2)):
        (root / p.replace("./SciVer/", "")).write_bytes(b"\x89PNG fake")
    return root
