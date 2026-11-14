import pytest

from sciverx.evaluation.stats import cluster_bootstrap_diff, flips, mcnemar_exact
from sciverx.evaluation.verdict_metrics import accuracy_f1, condition_summary, majority
from sciverx.schemas import GoldInfo, VerifierOutput


def test_accuracy_counts_unparsed_as_wrong():
    m = accuracy_f1([True, False, None, True], [True, False, True, False])
    assert m["accuracy"] == 0.5 and m["unparsed_pct"] == 25.0
    # lớp True: tp=1 fp=1 fn=1 → F1 0.5; lớp False: tp=1 fp=0 fn=1 → 2/3
    assert m["macro_f1"] == pytest.approx((0.5 + 2 / 3) / 2)


def test_majority():
    assert majority([True, True, False]) is True
    assert majority([True, False, None]) is None
    assert majority([None, None]) is None


def test_mcnemar_exact_hand_computed():
    a = [True] * 6 + [False] * 1 + [True] * 3
    b = [False] * 6 + [True] * 1 + [True] * 3
    r = mcnemar_exact(a, b)          # b=6, c=1, n=7: p = 2·(1+7)/128
    assert (r["b"], r["c"]) == (6, 1)
    assert r["p_value"] == pytest.approx(2 * 8 / 128)
    assert mcnemar_exact([True], [True])["p_value"] == 1.0


def test_cluster_bootstrap_is_seeded_and_keeps_clusters():
    a = {f"c{i}": i % 2 == 0 for i in range(20)}
    b = {f"c{i}": True for i in range(20)}
    cl = {f"c{i}": f"p{i // 4}" for i in range(20)}
    r1 = cluster_bootstrap_diff(a, b, cl, n_boot=200)
    r2 = cluster_bootstrap_diff(a, b, cl, n_boot=200)
    assert r1 == r2 and r1["diff"] == pytest.approx(0.5)
    assert r1["ci_low"] <= 0.5 <= r1["ci_high"] and r1["n_clusters"] == 5
    assert r1["mean_claims_per_cluster"] == 4


def test_flips_lists_changed_claims_with_analysis_columns():
    f = flips({"a": True, "b": False, "c": True}, {"a": True, "b": True, "c": False})
    assert [x["claim_id"] for x in f] == ["b", "c"]
    assert set(f[0]) >= {"observed_errors", "suspected_causes", "certainty"}


def test_condition_summary_seeds_and_t0():
    golds = {c: GoldInfo(claim_id=c, label=lab, claim_type="direct", section_ids=[], visual_items=[])
             for c, lab in (("x", True), ("y", False))}

    def out(c, s, pred, fr="stop"):
        return VerifierOutput(claim_id=c, run_id="r", system="P0", evidence_setting="S-oracle",
                              table_condition="G1", prompt_id="p", request_hash="h", sample_name=s,
                              temperature=1.0, seed=1, response="", finish_reason=fr, n_images=0,
                              pred_new=pred)
    outs = [out("x", "t1_s1", True), out("y", "t1_s1", False), out("x", "t1_s2", True),
            out("y", "t1_s2", True), out("x", "t1_s3", False, "length"), out("y", "t1_s3", False),
            out("x", "t0", True), out("y", "t0", False)]
    s = condition_summary(outs, golds)
    assert s["seed_accuracy_mean"] == pytest.approx((1 + 0.5 + 0.5) / 3)
    assert s["consistent_pct"] == 0.0 and s["t0"]["accuracy"] == 1.0
    assert s["majority"]["accuracy"] == 1.0 and s["truncated_pct"] == pytest.approx(100 / 8)
