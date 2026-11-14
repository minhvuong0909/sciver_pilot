"""Tokenizer dùng chung cho query và tài liệu của BM25 (SPEC 7-M3).

rank_bm25 không tự tiền xử lý nên mọi xử lý nằm ở đây: chữ thường, regex
giữ nguyên số thập phân, tên model, chỉ số dạng f1@5, top-1, 12.5%; bỏ
stopword theo danh sách cố định; không stemming.
"""

from __future__ import annotations

import re

TOKENIZER_VERSION = "v1"
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-@/][a-z0-9]+)*%?")

# Danh sách cố định (không đổi giữa các lượt chạy; đổi thì tăng TOKENIZER_VERSION).
STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further
had has have having he her here hers herself him himself his how i if in into is it its itself
just me more most my myself no nor not of off on once only or other our ours ourselves out over
own same she should so some such than that the their theirs them themselves then there these
they this those through to too under until up very was we were what when where which while who
whom why will with would you your yours yourself yourselves
""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]
