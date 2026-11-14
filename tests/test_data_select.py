import argparse

from sciverx.data import loader
from sciverx.data.select import run, select_claims, table_status
from sciverx.schemas import ClaimInput, GoldInfo
from sciverx.utils.io import read_jsonl


def test_table_status(mini_sciver):
    rows = loader.load_split(mini_sciver, "validation")
    assert [table_status(q) for q in rows] == ["all_table", "no_table", "all_table", "mixed",
                                               "all_table"]


def test_select_default_table_only(mini_sciver):
    rows = loader.load_split(mini_sciver, "validation")
    claims, golds = select_claims(rows, "validation")
    assert [c.claim_id for c in claims] == ["validation-0", "validation-2", "validation-4"]
    g = golds[1]
    assert g.claim_type == "parallel" and g.label is False
    assert g.visual_items == [
        {"item_id": "1", "item_type": "table", "image_path": "images/2409.00002v2-Table1-1.png"},
        {"item_id": "2", "item_type": "table", "image_path": "images/2409.00002v2-Table2-1.png"}]
    assert g.section_ids == ["4.1"]


def test_select_allow_mixed(mini_sciver):
    rows = loader.load_split(mini_sciver, "validation")
    claims, _ = select_claims(rows, "validation", allow_mixed=True)
    assert [c.claim_id for c in claims] == ["validation-0", "validation-2", "validation-3",
                                            "validation-4"]


def test_run_writes_separate_files_without_leaks(mini_sciver, tmp_path):
    cfg = {"split": "validation", "paths": {"data_root": str(mini_sciver), "work": str(tmp_path / "w")},
           "selection": {"allow_mixed": False}}
    assert run(cfg, argparse.Namespace()) == 0
    out = tmp_path / "w" / "validation"
    claims = read_jsonl(out / "claims_input.jsonl", ClaimInput)
    golds = read_jsonl(out / "gold.jsonl", GoldInfo)
    assert [c.claim_id for c in claims] == [g.claim_id for g in golds]
    text = (out / "claims_input.jsonl").read_text(encoding="utf-8")
    for leak in ("LEAK", "perturbed", "origin_statement", "label", "claim_type", "Table1-1.png", '"section'):
        assert leak not in text
    assert (out / "ids.txt").read_text().split() == [c.claim_id for c in claims]
    assert len((out / "ids.sha256").read_text().strip()) == 64


def test_run_limit(mini_sciver, tmp_path):
    cfg = {"split": "validation", "limit": 1,
           "paths": {"data_root": str(mini_sciver), "work": str(tmp_path / "w")}}
    run(cfg, argparse.Namespace())
    assert len(read_jsonl(tmp_path / "w" / "validation" / "gold.jsonl")) == 1


def test_loader_helpers(mini_sciver):
    rows = loader.load_split(mini_sciver, "validation")
    paper = loader.load_paper(mini_sciver, rows[0]["paper_path"])
    assert set(paper["tables"]) == {"1", "2"}
    assert loader.prepare_context(paper, ["4.1"]) == (
        "Experiments:\n\nResults:\nOur model reaches 92.4 F1.\n\nBERT reaches 90.9 F1.\n")
    assert loader.dataset_rel("./SciVer/papers/x.json") == "papers/x.json"
    assert loader.table_image_rel("p", "3") == "images/p-Table3-1.png"
    assert loader.label_bool("Refuted") is False
