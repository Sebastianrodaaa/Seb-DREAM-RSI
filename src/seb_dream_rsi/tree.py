"""Discovery trees.

A tree is rooted at a workspace that has not been evaluated. Every other node
is one generation-evaluation attempt. Non-root nodes have one parent. In this
implementation each non-root node has at most one child, so a branch is a
chain. New branches are opened from the root, matching the parallel-refine
structure in Zheng et al. 2026, Section 3.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

ROOT_ID = "root"


@dataclass
class Node:
    id: str
    parent_id: str | None
    created_seq: int
    branch_index: int | None
    depth: int
    score: float | None
    artifact: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    valid: bool = True
    error: str | None = None
    proposal: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Node:
        return cls(
            id=data["id"],
            parent_id=data["parent_id"],
            created_seq=data["created_seq"],
            branch_index=data["branch_index"],
            depth=data["depth"],
            score=data["score"],
            artifact=data["artifact"],
            diagnostics=data.get("diagnostics") or {},
            valid=data.get("valid", True),
            error=data.get("error"),
            proposal=data.get("proposal", ""),
        )


@dataclass
class DiscoveryTree:
    nodes: dict[str, Node]
    children: dict[str, list[str]]
    next_seq: int = 1

    @classmethod
    def new(cls) -> DiscoveryTree:
        root = Node(
            id=ROOT_ID,
            parent_id=None,
            created_seq=0,
            branch_index=None,
            depth=0,
            score=None,
            artifact={},
            proposal="initial workspace",
        )
        return cls(nodes={ROOT_ID: root}, children={ROOT_ID: []}, next_seq=1)

    def add_child(
        self,
        parent_id: str,
        *,
        branch_index: int,
        score: float,
        artifact: dict[str, Any],
        diagnostics: dict[str, Any] | None = None,
        valid: bool = True,
        error: str | None = None,
        proposal: str = "",
    ) -> Node:
        if parent_id not in self.nodes:
            raise KeyError(parent_id)
        parent = self.nodes[parent_id]
        if parent_id != ROOT_ID and self.children.get(parent_id):
            raise ValueError(
                f"{parent_id} already has a continuation; non-root nodes are chains"
            )
        node_id = f"n{self.next_seq}"
        inherited = parent.branch_index if parent_id != ROOT_ID else branch_index
        node = Node(
            id=node_id,
            parent_id=parent_id,
            created_seq=self.next_seq,
            branch_index=inherited if inherited is not None else branch_index,
            depth=0 if parent_id == ROOT_ID else parent.depth + 1,
            score=score,
            artifact=artifact,
            diagnostics=dict(diagnostics or {}),
            valid=valid,
            error=error,
            proposal=proposal,
        )
        if parent_id == ROOT_ID:
            node.depth = 1
        self.next_seq += 1
        self.nodes[node_id] = node
        self.children.setdefault(parent_id, []).append(node_id)
        self.children.setdefault(node_id, [])
        return node

    @property
    def attempt_count(self) -> int:
        return len(self.nodes) - 1

    def best_score(self) -> float:
        scores = [
            node.score or 0.0 for node in self.nodes.values() if node.id != ROOT_ID
        ]
        return max(scores) if scores else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "next_seq": self.next_seq,
            "nodes": {key: node.to_dict() for key, node in self.nodes.items()},
            "children": {key: list(value) for key, value in self.children.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiscoveryTree:
        return cls(
            nodes={key: Node.from_dict(value) for key, value in data["nodes"].items()},
            children={key: list(value) for key, value in data["children"].items()},
            next_seq=int(data["next_seq"]),
        )
