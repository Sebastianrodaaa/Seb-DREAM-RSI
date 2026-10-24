"""What an exploration policy is allowed to see.

Online and replay share this view. Unopened branches expose an id and a branch
index only. Scores, artifacts, and diagnostics appear after a node is revealed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from daydream.tree import ROOT_ID, DiscoveryTree, Node


@dataclass(frozen=True)
class NodeView:
    id: str
    parent_id: str | None
    branch_index: int | None
    depth: int
    score: float | None
    valid: bool
    error: str | None
    diagnostics: dict
    artifact: dict
    proposal: str


@dataclass(frozen=True)
class RootOption:
    """An unopened branch. No outcome fields are attached."""

    id: str
    branch_index: int


@dataclass
class TreeView:
    root_id: str
    nodes: dict[str, NodeView]
    children: dict[str, list[str]]
    legal_roots: list[RootOption] = field(default_factory=list)

    def leaves(self) -> list[NodeView]:
        leaves: list[NodeView] = []
        for node_id, node in self.nodes.items():
            if node_id == self.root_id:
                continue
            if not self.children.get(node_id):
                leaves.append(node)
        return leaves

    def eligible_ids(self) -> list[str]:
        ids = [root.id for root in self.legal_roots]
        ids.extend(node.id for node in self.leaves())
        return ids

    def trajectory(self, node_id: str) -> list[NodeView]:
        chain: list[NodeView] = []
        current: str | None = node_id
        seen: set[str] = set()
        while current and current != self.root_id and current not in seen:
            seen.add(current)
            chain.append(self.nodes[current])
            current = self.nodes[current].parent_id
        chain.reverse()
        return chain

    def opened_branch_count(self) -> int:
        return len(self.children.get(self.root_id, []))


def _node_view(node: Node) -> NodeView:
    return NodeView(
        id=node.id,
        parent_id=node.parent_id,
        branch_index=node.branch_index,
        depth=node.depth,
        score=node.score,
        valid=node.valid,
        error=node.error,
        diagnostics=dict(node.diagnostics),
        artifact=dict(node.artifact),
        proposal=node.proposal,
    )


def view_from_revealed(tree: DiscoveryTree, revealed: set[str]) -> TreeView:
    """Project a completed tree onto the nodes a replay policy has opened."""

    if ROOT_ID not in revealed:
        raise ValueError("the root is always observed")
    nodes = {node_id: _node_view(tree.nodes[node_id]) for node_id in revealed}
    children: dict[str, list[str]] = {node_id: [] for node_id in revealed}
    for node_id in revealed:
        for child_id in tree.children.get(node_id, []):
            if child_id in revealed:
                children[node_id].append(child_id)
    legal: list[RootOption] = []
    hidden_roots = [
        child_id
        for child_id in tree.children.get(ROOT_ID, [])
        if child_id not in revealed
    ]
    hidden_roots.sort(key=lambda child_id: tree.nodes[child_id].created_seq)
    for child_id in hidden_roots:
        node = tree.nodes[child_id]
        legal.append(
            RootOption(
                id=child_id,
                branch_index=0 if node.branch_index is None else node.branch_index,
            )
        )
    return TreeView(
        root_id=ROOT_ID, nodes=nodes, children=children, legal_roots=legal
    )


def view_from_online(tree: DiscoveryTree, branch_count: int) -> TreeView:
    """Live tree plus the branch slots the current grid plan has not opened."""

    view = view_from_revealed(tree, set(tree.nodes))
    opened = {
        tree.nodes[child_id].branch_index
        for child_id in tree.children.get(ROOT_ID, [])
    }
    slots: list[RootOption] = []
    for index in range(branch_count):
        if index not in opened:
            slots.append(RootOption(id=f"slot:{index}", branch_index=index))
    view.legal_roots = slots
    return view


def sanitize_batch(batch: list[str], view: TreeView, max_parallelism: int) -> list[str]:
    """Keep ids that are legal before the call, drop duplicates, cap the width."""

    eligible = set(view.eligible_ids())
    chosen: list[str] = []
    limit = max(1, max_parallelism)
    for action in batch:
        if action in eligible and action not in chosen:
            chosen.append(action)
        if len(chosen) >= limit:
            break
    return chosen
