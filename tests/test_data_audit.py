import argparse
import json

from sciverx.data.audit import audit, run


def test_audit_counts_on_mini_dataset(mini_sciver):
    r = audit(mini_sciver, "validation")
    assert r["D1"]["n_claims"] == 5
    assert r["D2"]["missing_expected"] == []
    assert r["D2"]["extra_label_leaking"] == ["origin_statement", "perturbed_explanation",
                                              "perturbed_statement"]
    assert r["D3"] == {"table_entry_keys": {"table_html": 4, "capture": 4}, "n_tables": 4,
                       "with_table_html": 4}
    assert r["D4"]["tables_with_image"] == 3 and r["D4"]["papers_all_tables_have_image"] == 1
    assert "extract_table_info" in r["D4"]["annotation_fields_present"]
    assert r["D5"]["empty_text_sections"] == 2
    assert r["D6"]["paperid_has_version"] == 2
    assert r["D7"]["selected"] == 3
    assert r["D7"]["by_table_status"] == {"all_table": 3, "no_table": 1, "mixed": 1}
    assert r["D8"]["unique_evidence_tables"] == 3


def test_structure_only_has_no_values_or_label_counts(mini_sciver):
    r = audit(mini_sciver, "validation", structure_only=True)
    assert set(r) == {"split", "D1", "D2", "paper_structure"}
    dumped = json.dumps(r)
    assert "LEAK" not in dumped and "claim about" not in dumped
    assert r["D2"]["query_structure"]["label"] == ["bool"]


def test_missing_field_returns_stop_code(mini_sciver, tmp_path):
    rows = json.loads((mini_sciver / "valset.json").read_text())
    for q in rows:
        q.pop("section")
    (mini_sciver / "valset.json").write_text(json.dumps(rows))
    cfg = {"split": "validation", "paths": {"data_root": str(mini_sciver), "work": str(tmp_path)}}
    assert run(cfg, argparse.Namespace(structure_only=False)) == 4
    assert (tmp_path / "validation" / "audit.json").exists()


def test_full_audit_refused_on_test_split(mini_sciver, tmp_path):
    cfg = {"split": "test", "paths": {"data_root": str(mini_sciver), "work": str(tmp_path)}}
    assert run(cfg, argparse.Namespace(structure_only=False)) == 2
