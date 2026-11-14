"""TEDS và TEDS-Struct — SPEC M1.5, theo định nghĩa PubTabNet (Zhong et al. 2020).

Cây HTML dựng lại từ `cells` (còn rowspan/colspan, không dùng lưới đã trải):
table > [thead] > tr > td, tbody > tr > td. Chi phí:
- xóa / chèn một nút: 1
- đổi tên: tag khác → 1; hai `td` khác span → 1; hai `td` cùng span → khoảng
  cách Levenshtein chuẩn hóa của nội dung (TEDS-Struct: 0)
TEDS = 1 - TED / max(|T_a|, |T_b|).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from apted import APTED, Config

from sciverx.schemas import SourceCell, TableRecord


@dataclass
class Node:
    tag: str
    rowspan: int = 1
    colspan: int = 1
    content: str = ""
    children: list["Node"] = field(default_factory=list)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalized_levenshtein(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    return levenshtein(a, b) / max(len(a), len(b))


class _TedsConfig(Config):
    valuecls = float

    def __init__(self, structure_only: bool):
        self.structure_only = structure_only

    def rename(self, n1: Node, n2: Node) -> float:
        if n1.tag != n2.tag:
            return 1.0
        if n1.tag != "td":
            return 0.0
        if (n1.rowspan, n1.colspan) != (n2.rowspan, n2.colspan):
            return 1.0
        if self.structure_only:
            return 0.0
        return normalized_levenshtein(n1.content, n2.content)

    def children(self, node: Node) -> list[Node]:
        return node.children


def build_tree(cells: list[SourceCell], n_header_rows: int, use_norm: bool = True) -> Node:
    n_rows = max((c.row + c.rowspan for c in cells), default=0)
    # Hàng có ô bắt đầu; hàng bị rowspan phủ hết vẫn tạo <tr> rỗng như HTML gốc.
    rows: dict[int, list[SourceCell]] = {r: [] for r in range(n_rows)}
    for c in cells:
        rows[c.row].append(c)
    head = Node("thead")
    body = Node("tbody")
    for r in range(n_rows):
        tr = Node("tr")
        for c in sorted(rows[r], key=lambda x: x.col):
            text = c.text_norm if (use_norm and c.text_norm is not None) else c.text_raw
            tr.children.append(Node("td", c.rowspan, c.colspan, text))
        (head if r < n_header_rows else body).children.append(tr)
    root = Node("table")
    if head.children:
        root.children.append(head)
    root.children.append(body)
    return root


def tree_size(node: Node) -> int:
    return 1 + sum(tree_size(ch) for ch in node.children)


def teds_trees(a: Node, b: Node, structure_only: bool = False) -> float:
    dist = APTED(a, b, _TedsConfig(structure_only)).compute_edit_distance()
    return 1.0 - dist / max(tree_size(a), tree_size(b))


def teds(pred: TableRecord, gold: TableRecord, structure_only: bool = False,
         use_norm: bool = True) -> float:
    """TEDS của bảng trích (pred, E*) so với bảng chuẩn (gold, G*)."""
    if not pred.cells or not gold.cells:
        return 1.0 if not pred.cells and not gold.cells else 0.0
    return teds_trees(build_tree(pred.cells, pred.n_header_rows, use_norm),
                      build_tree(gold.cells, gold.n_header_rows, use_norm), structure_only)
