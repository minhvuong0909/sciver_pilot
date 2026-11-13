"""TEDS (Tree-Edit-Distance-based Similarity, Zhong et al. 2020 - PubTabNet).

Bản gọn, viết lại theo định nghĩa của PubTabNet: so cây HTML <table><tr><td>,
chi phí đổi tên ô = khoảng cách Levenshtein chuẩn hóa của nội dung (ô cùng loại, cùng span).
TEDS = 1 - TED(A, B) / max(|A|, |B|).  structure_only=True -> TEDS-Struct (bỏ nội dung).

Làm việc trên grid đã trải span (dạng chung của tables.py), nên mọi ô có span = 1.
Với pilot so sánh G và E trên cùng dạng biểu diễn, điều này là nhất quán.
"""
from apted import APTED, Config


def _lev(a, b):
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class Node:
    def __init__(self, tag, content="", children=None):
        self.tag, self.content, self.children = tag, content, children or []


class TEDSConfig(Config):
    def __init__(self, structure_only):
        self.structure_only = structure_only

    def rename(self, n1, n2):
        if n1.tag != n2.tag:
            return 1.0
        if n1.tag == "td" and not self.structure_only:
            a, b = n1.content, n2.content
            if not a and not b:
                return 0.0
            return _lev(a, b) / max(len(a), len(b))
        return 0.0

    def children(self, node):
        return node.children


def _tree(grid, structure_only):
    rows = [Node("tr", children=[Node("td", "" if structure_only else c) for c in r]) for r in grid]
    return Node("table", children=rows)


def _size(n):
    return 1 + sum(_size(c) for c in n.children)


def teds(grid_pred, grid_gold, structure_only=False):
    if not grid_gold and not grid_pred:
        return 1.0
    if not grid_gold or not grid_pred:
        return 0.0
    a, b = _tree(grid_pred, structure_only), _tree(grid_gold, structure_only)
    dist = APTED(a, b, TEDSConfig(structure_only)).compute_edit_distance()
    return max(0.0, 1.0 - dist / max(_size(a), _size(b)))
