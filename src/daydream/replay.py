"""Replay search environment.

``probe_batch`` is the replay half of ``daydream.env.solve``. Every outcome it
reads was stored when the node was first evaluated. The discovery agent and
the evaluator are not called.

Score, from Section 3 of Zheng et al. 2026:

    quality - lambda_cost * attempts + lambda_parallel * attempts / max(1, rounds)

`quality` is the best revealed score. `attempts` is the number of revealed
non-root nodes. `rounds` is the number of nonempty decision batches.
"""

from __future__ import annotations

from dataclasses import dataclass

from daydream.env import solve
from daydream.policy import ExplorationPolicy
from daydream.tree import ROOT_ID, DiscoveryTree
from daydream.view import TreeView, view_from_revealed


@dataclass(frozen=True)
class ReplayResult:
    score: float
    quality: float
    attempts: int
    rounds: int
    parallelism: float
    revealed_ids: tuple[str, ...]
    best_node_id: str | None


def replay_score(
    quality: float,
    attempts: int,
    rounds: int,
    lambda_cost: float,
    lambda_parallel: float,
) -> float:
    if attempts <= 0:
        return 0.0
    parallelism = attempts / max(1, rounds)
    return quality - lambda_cost * attempts + lambda_parallel * parallelism


def _reveal(tree: DiscoveryTree, revealed: set[str], action: str) -> str | None:
    """Return the recorded node an action uncovers, if one still exists.

    Selecting an unopened branch reveals that branch's first node. Selecting
    a revealed leaf reveals its recorded child. The root's children may be
    opened in any order; a non-root node has a single continuation.
    """

    if action not in revealed:
        node = tree.nodes.get(action)
        if node is None or node.parent_id != ROOT_ID:
            return None
        return action
    for child_id in tree.children.get(action, []):
        if child_id not in revealed:
            return child_id
    return None


class ReplayEnv:
    """Frozen tree. Legal roots are unopened recorded branches, not planned slots."""

    def __init__(self, tree: DiscoveryTree) -> None:
        self.tree = tree
        self.revealed: set[str] = {ROOT_ID}

    def finished(self) -> bool:
        return len(self.revealed) >= len(self.tree.nodes)

    def view(self, branch_count: int) -> TreeView:
        del branch_count
        return view_from_revealed(self.tree, self.revealed)

    def probe_batch(self, batch: list[str]) -> bool:
        revealed_any = False
        for action in batch:
            node_id = _reveal(self.tree, self.revealed, action)
            if node_id is not None and node_id not in self.revealed:
                self.revealed.add(node_id)
                revealed_any = True
        return revealed_any


def replay(
    policy: ExplorationPolicy,
    tree: DiscoveryTree,
    *,
    max_rounds: int = 10_000,
    max_parallelism: int = 8,
    lambda_cost: float = 0.002,
    lambda_parallel: float = 0.005,
) -> ReplayResult:
    env = ReplayEnv(tree)
    rounds = solve(
        policy, env, max_rounds=max_rounds, max_parallelism=max_parallelism
    )
    revealed = env.revealed
    attempts = [tree.nodes[node_id] for node_id in revealed if node_id != ROOT_ID]
    quality = max((node.score or 0.0) for node in attempts) if attempts else 0.0
    best_node_id = None
    if attempts:
        best_node_id = max(attempts, key=lambda node: (node.score or 0.0, -node.created_seq)).id
    parallelism = (len(attempts) / rounds) if rounds else 0.0
    return ReplayResult(
        score=replay_score(
            quality, len(attempts), rounds, lambda_cost, lambda_parallel
        ),
        quality=quality,
        attempts=len(attempts),
        rounds=rounds,
        parallelism=parallelism,
        revealed_ids=tuple(sorted(revealed, key=lambda node_id: tree.nodes[node_id].created_seq)),
        best_node_id=best_node_id,
    )
