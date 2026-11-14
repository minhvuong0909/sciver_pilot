"""SPEC 12.4 test_scoring (pred_old / pred_new) và danh sách prompt được phép."""

import pytest

from sciverx.verifier import prompts as P
from sciverx.verifier.request_hash import make_request, request_hash
from sciverx.verifier.scoring import is_correct, pred_new, pred_old


@pytest.mark.parametrize("resp,new", [
    ("... Therefore, the final answer is: Answer: yes", True),
    ("Therefore, the final answer is: Answer: **no**", False),
    ("Answer: (Yes)", True),
    ("Answer: [no]", False),
    ("Answer:no.", False),
    ("First I thought Answer: yes, but rechecking... Answer: no", False),   # lấy lần cuối
    ("The final answer is yes.", True),                                     # dự phòng
    ("I cannot decide.", None),
    ("Answer: maybe", None),
    ("ANSWER : YES", True),
])
def test_pred_new(resp, new):
    assert pred_new(resp) is new


def test_pred_old_matches_original_rule():
    assert pred_old("Answer: no, the claim says yes") is True    # cách cũ: có "yes" ở đâu cũng tính
    assert pred_old("Answer: no") is False
    assert pred_old("eyes") is True                               # quirk của code gốc, giữ nguyên


def test_unreadable_counts_as_wrong():
    assert is_correct(None, True) is False and is_correct(True, True) is True


def test_render_original_and_text_mode():
    p1 = P.render("sciver_p1", claim="C", context="X", caption="Cap")
    assert "based on the image and the caption" in p1
    p1t = P.render("sciver_p1_text", claim="C", context="X", caption="Cap")
    assert "based on the table and the caption" in p1t and "image" not in p1t
    assert "Answer: $ANSWER" in p1
    p2 = P.render("sciver_p2_text", claim="C", context="X", caption1="a", caption2="b")
    assert p2 == P.render("sciver_p2", claim="C", context="X", caption1="a", caption2="b")


def test_unified_prompts_and_whitelist():
    u = P.render("unified_v1", claim="C", evidence="E")
    assert "provided evidence (tables and/or text)" in u and "Answer: $ANSWER" in u
    un = P.render("unified_v1_numeric", claim="C", evidence="E", numeric_checks="- check")
    assert "Numeric checks to consider:\n- check" in un
    for bad in ("unified_v2", "unified_v1_text", "my_prompt"):
        with pytest.raises(KeyError):
            P.render(bad, claim="C", evidence="E")
    assert P.CLAIM_TYPE_TO_ID == {"direct": "sciver_p1", "parallel": "sciver_p2",
                                  "analytical": "sciver_p3", "sequential": "sciver_p4"}
    hashes = P.prompt_hashes()
    assert set(hashes) == {"sciver_p1", "sciver_p2", "sciver_p3", "sciver_p4", "sciver_p1_text",
                           "sciver_p2_text", "sciver_p3_text", "sciver_p4_text", "unified_v1",
                           "unified_v1_numeric"}


def test_request_hash_depends_on_all_parts(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"x")
    r = make_request("c", "unified_v1", "text", [img])
    assert r.request_hash == request_hash("unified_v1", "text", r.image_sha256)
    assert r.request_hash != request_hash("unified_v1", "text2", r.image_sha256)
    assert r.request_hash != request_hash("sciver_p1", "text", r.image_sha256)
    assert r.request_hash != request_hash("unified_v1", "text", [])
