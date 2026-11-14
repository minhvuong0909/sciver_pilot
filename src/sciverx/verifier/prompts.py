"""Prompt CoT: chép NGUYÊN VĂN từ SciVer (utils/constant.py, repo QDRhhhh/SciVer).

Với điều kiện bảng-dạng-văn-bản (G0/G1/E0/E1) chỉ thay các cụm "image" -> "table"
ở prompt 1 và 3 (claim một bằng chứng); nội dung bảng được chèn ở đầu phần văn bản.
Mọi thay đổi được liệt kê trong TEXT_MODE_EDITS để ghi vào báo cáo.
"""
from string import Template

OPEN_ENDED_COT_PROMPT1 = Template("""
Claim: $claim
Context: $context
Caption: $caption

Your task is to critically evaluate the claim based on the image and the caption. Carefully examine whether the information in the caption truly supports the claim. Be skeptical and cautious: if there is any inconsistency, missing evidence, or ambiguity, consider the claim incorrect.

Start by explaining your reasoning process clearly, focusing on identifying potential contradictions, lack of support, or misleading interpretations. If the claim is unsupported or contradicted by the caption and image, respond with 'no'. Only respond with 'yes' if the claim is fully and clearly supported.

Conclude your analysis by stating: 'Therefore, the final answer is: Answer: $$ANSWER' (without quotes), where $$ANSWER is your final answer. Think step by step before answering.
""")

OPEN_ENDED_COT_PROMPT2 = Template("""
Claim: $claim
Context: $context
item1 Caption: $caption1
item2 Caption: $caption2

You will be provided with two images, two tables, or one image and one table, along with contextual information explaining their relationship. Your task is to critically assess whether the claim is fully and clearly supported by the provided evidence.

Approach the task with skepticism. Carefully examine both the captions and the context for contradictions, missing links, or insufficient support. If there is any ambiguity, inconsistency, or lack of direct evidence, consider the claim incorrect. Only answer 'yes' if the claim is explicitly and unambiguously confirmed by the content.

Begin by explaining your reasoning step by step, focusing on identifying potential errors or unsupported assumptions. Conclude by stating your judgment using the format: 'Therefore, the final answer is: Answer: $$ANSWER' (without quotes), where $$ANSWER is your final answer.
""")

OPEN_ENDED_COT_PROMPT3 = Template("""
Claim: $claim
Context: $context
Caption: $caption

Your task is to evaluate the claim based on the image and the caption. Carefully examine whether the information in the caption truly supports the claim. Apply any relevant scientific principles, statistical logic, or domain knowledge necessary to link the evidence to the claim. Be balanced: actively look for confirming details as well as inconsistencies, missing evidence, or ambiguities.

Start by explaining your reasoning process step by step—describe what you observe in the image and caption, what background knowledge you use, and how you test whether each key part of the claim is supported. If every essential component of the claim is clearly and completely backed by the caption and image, respond with 'yes'. If any critical point is contradicted, unsupported, or unclear, respond with 'no'.

Conclude your analysis by stating: 'Therefore, the final answer is: Answer: $$ANSWER' (without quotes), where $$ANSWER is your final answer. Think step by step before answering.
""")

OPEN_ENDED_COT_PROMPT4 = OPEN_ENDED_COT_PROMPT2  # trong code gốc prompt 4 giống hệt prompt 2

COT_PROMPT = {
    "direct": OPEN_ENDED_COT_PROMPT1,
    "analytical": OPEN_ENDED_COT_PROMPT3,
    "parallel": OPEN_ENDED_COT_PROMPT2,
    "sequential": OPEN_ENDED_COT_PROMPT4,
}

# Chỉ áp dụng khi bằng chứng duy nhất (direct/analytical) được đưa vào dưới dạng văn bản
TEXT_MODE_EDITS = [
    ("based on the image and the caption", "based on the table and the caption"),
    ("contradicted by the caption and image", "contradicted by the caption and table"),
    ("describe what you observe in the image and caption", "describe what you observe in the table and caption"),
    ("backed by the caption and image", "backed by the caption and table"),
    ("unsupported or contradicted by the caption and image", "unsupported or contradicted by the caption and table"),
]


def apply_text_mode(prompt_text):
    for a, b in TEXT_MODE_EDITS:
        prompt_text = prompt_text.replace(a, b)
    return prompt_text


# ----------------------------------------------------------------------------- prompt thống nhất
# BẢN NHÁP — [DỪNG & HỎI] H5: cần người duyệt trước khi dùng cho thực nghiệm, sau đó đóng băng.
# Dựa trên prompt 1 của SciVer: đổi "the image" thành "the provided evidence (tables and/or text)",
# giữ yêu cầu hoài nghi và câu kết "Answer: $ANSWER". Không phụ thuộc claim_type.
UNIFIED_V1 = Template("""
Claim: $claim
Evidence:
$evidence

Your task is to critically evaluate the claim based on the provided evidence (tables and/or text). Carefully examine whether the information in the provided evidence truly supports the claim. Be skeptical and cautious: if there is any inconsistency, missing evidence, or ambiguity, consider the claim incorrect.

Start by explaining your reasoning process clearly, focusing on identifying potential contradictions, lack of support, or misleading interpretations. If the claim is unsupported or contradicted by the provided evidence, respond with 'no'. Only respond with 'yes' if the claim is fully and clearly supported.

Conclude your analysis by stating: 'Therefore, the final answer is: Answer: $$ANSWER' (without quotes), where $$ANSWER is your final answer. Think step by step before answering.
""")

# unified_v1 + khối kiểm tra số liệu từ M5: chỉ toán hạng có nguồn và phép tính, KHÔNG có kết quả hay kết luận.
UNIFIED_V1_NUMERIC = Template("""
Claim: $claim
Evidence:
$evidence

Numeric checks to consider:
$numeric_checks

Your task is to critically evaluate the claim based on the provided evidence (tables and/or text). Carefully examine whether the information in the provided evidence truly supports the claim. Be skeptical and cautious: if there is any inconsistency, missing evidence, or ambiguity, consider the claim incorrect.

Start by explaining your reasoning process clearly, focusing on identifying potential contradictions, lack of support, or misleading interpretations. If the claim is unsupported or contradicted by the provided evidence, respond with 'no'. Only respond with 'yes' if the claim is fully and clearly supported.

Conclude your analysis by stating: 'Therefore, the final answer is: Answer: $$ANSWER' (without quotes), where $$ANSWER is your final answer. Think step by step before answering.
""")

# Danh sách prompt được phép (SPEC 8.3). Không prompt nào khác được dùng.
CLAIM_TYPE_TO_ID = {"direct": "sciver_p1", "parallel": "sciver_p2", "analytical": "sciver_p3",
                    "sequential": "sciver_p4"}
PROMPTS = {
    "sciver_p1": OPEN_ENDED_COT_PROMPT1, "sciver_p2": OPEN_ENDED_COT_PROMPT2,
    "sciver_p3": OPEN_ENDED_COT_PROMPT3, "sciver_p4": OPEN_ENDED_COT_PROMPT4,
    "unified_v1": UNIFIED_V1, "unified_v1_numeric": UNIFIED_V1_NUMERIC,
}
TEXT_TABLE_LABEL = "Table content (as text):"


def render(prompt_id: str, **fields) -> str:
    """Dựng prompt. "sciver_pN_text" = prompt gốc + TEXT_MODE_EDITS (chỉ ảnh hưởng prompt 1, 3)."""
    base_id = prompt_id[:-5] if prompt_id.endswith("_text") else prompt_id
    if base_id not in PROMPTS or (prompt_id.endswith("_text") and not base_id.startswith("sciver_p")):
        raise KeyError(f"prompt_id không nằm trong danh sách được phép: {prompt_id}")
    text = PROMPTS[base_id].substitute(**fields)
    return apply_text_mode(text) if prompt_id.endswith("_text") else text


def prompt_hashes() -> dict[str, str]:
    """Hash nội dung từng prompt (ghi vào docs/DEVIATIONS.md)."""
    import hashlib
    out = {pid: hashlib.sha256(t.template.encode("utf-8")).hexdigest()[:16] for pid, t in PROMPTS.items()}
    edits = "".join(a + "\u0000" + b for a, b in TEXT_MODE_EDITS)
    for i in (1, 2, 3, 4):
        out[f"sciver_p{i}_text"] = hashlib.sha256(
            (PROMPTS[f"sciver_p{i}"].template + edits).encode("utf-8")).hexdigest()[:16]
    return out
