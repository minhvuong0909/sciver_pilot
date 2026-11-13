from collections import Counter
from pathlib import Path

import pytest

from sciverx.llm import LLMRequest, ScriptedLLM
from sciverx.schemas import EvidencePackage
from sciverx.utils.io import read_jsonl

FIX = Path(__file__).parent / "fixtures" / "packages_mock.jsonl"


def test_packages_mock_are_valid_and_cover_statuses():
    pkgs = read_jsonl(FIX, EvidencePackage)
    assert len(pkgs) == 10
    assert len({p.claim_id for p in pkgs}) == 10
    statuses = Counter(p.alignment_status for p in pkgs)
    assert {"matched", "partial", "ambiguous", "not_found", "parse_error"} <= set(statuses)
    relations = {c.relation for p in pkgs for c in p.numeric_checks}
    assert {"percentage_point_diff", "relative_change", "extremum", "ratio_multiple",
            "universal"} <= relations


def test_packages_mock_file_is_up_to_date():
    import importlib.util
    spec = importlib.util.spec_from_file_location("bpm", FIX.with_name("build_packages_mock.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert [p.model_dump() for p in mod.build()] == [
        p.model_dump() for p in read_jsonl(FIX, EvidencePackage)]


def test_claim_operands_never_reference_evidence():
    for p in read_jsonl(FIX, EvidencePackage):
        for c in p.numeric_checks:
            assert all(o.source == "claim" and o.ref is None for o in c.claim_operands)
            assert all(o.source != "claim" for o in c.evidence_operands)


def test_scripted_llm_records_requests_and_cycles():
    llm = ScriptedLLM(script=["a", "b"])
    out = llm.generate([LLMRequest(prompt="p1"), LLMRequest(prompt="p2"), LLMRequest(prompt="p3")])
    assert [o.text for o in out] == ["a", "b", "b"]
    assert [r.prompt for r in llm.requests] == ["p1", "p2", "p3"]


def test_scripted_llm_responder():
    llm = ScriptedLLM(responder=lambda r: r.prompt.upper())
    assert llm.generate([LLMRequest(prompt="yes")])[0].text == "YES"
    with pytest.raises(ValueError):
        ScriptedLLM()
