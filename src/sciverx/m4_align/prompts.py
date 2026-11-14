"""Prompt align_v1 của M4 (SPEC 8.3). Yêu cầu JSON; ID ô và câu được in sẵn trong đầu vào."""

from __future__ import annotations

import hashlib
from string import Template

ALIGN_V1 = Template("""You are given a scientific claim and candidate evidence from the same paper.
Tables show each cell as [cell_id] value. Paragraph sentences are shown as [sentence_id] text.

Claim: $claim

Candidate evidence:
$evidence

Task: identify exactly which table cells and/or sentences are needed to check the claim. Match the
entity, metric, dataset and condition mentioned in the claim. Do NOT judge whether the claim is true.

Reply with JSON only, in this format:
{"status": "matched" | "partial" | "ambiguous" | "not_found",
 "evidence": [{"evidence_id": "<id from the candidate list>", "cell_ids": ["r1c2"], "sentence_ids": ["s0"],
               "reason": "<entity, metric, dataset, condition>"}],
 "alternatives": [<same format, other equally plausible choices when status is ambiguous>],
 "missing": ["<information the claim needs that is not in the evidence>"]}
Use "partial" when only part of the needed information is present, "ambiguous" when several choices
are equally plausible, "not_found" when nothing relevant is present.
""")

ALIGN_PROMPT_ID = "align_v1"
ALIGN_PROMPT_HASH = hashlib.sha256(ALIGN_V1.template.encode("utf-8")).hexdigest()[:16]
