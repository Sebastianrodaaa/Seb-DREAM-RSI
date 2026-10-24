"""Fast open policy model.

System 1 reads branch scores that a finished rollout already paid for and
proposes one adaptive spec. It does not replay, and it does not call a model.
The dreamer still accepts the proposal only when its replay score beats the
incumbent, so a quick guess cannot deploy a worse policy.

The guess is the smallest controller that can still uncover every stored
world's best node: open branches up through the winning branch, stop at the
depth where that score first appears, and set patience just high enough that
a flat prefix before the payoff is not pruned.
"""

from __future__ import annotations

from dataclasses import dataclass

from daydream.policy import PolicySpec, _is_stagnant
from daydream.tree import ROOT_ID, DiscoveryTree, Node

_MIN_GAIN = 0.02


@dataclass(frozen=True)
class _Target:
    branch_index: int
    depth: int
    scores: tuple[float, ...]


def _chains(tree: DiscoveryTree) -> list[list[Node]]:
    chains: list[list[Node]] = []
    for child_id in tree.children.get(ROOT_ID, []):
        chain: list[Node] = []
        current: str | None = child_id
        seen: set[str] = set()
        while current and current not in seen:
            seen.add(current)
            node = tree.nodes[current]
            chain.append(node)
            children = tree.children.get(current, [])
            current = children[0] if children else None
        if chain and chain[0].branch_index is not None:
            chains.append(chain)
    return chains


def _target(tree: DiscoveryTree) -> _Target | None:
    """Winning branch with the smallest index, cut at its first best score."""

    best_score = 0.0
    chosen: _Target | None = None
    for chain in _chains(tree):
        scores = [node.score or 0.0 for node in chain]
        peak = max(scores)
        branch_index = chain[0].branch_index
        if branch_index is None:
            continue
        if chosen is not None and peak < best_score - 1e-12:
            continue
        depth_index = next(
            index for index, score in enumerate(scores) if score >= peak - 1e-12
        )
        candidate = _Target(
            branch_index=branch_index,
            depth=depth_index + 1,
            scores=tuple(scores),
        )
        better_score = chosen is None or peak > best_score + 1e-12
        earlier_branch = (
            chosen is not None
            and abs(peak - best_score) <= 1e-12
            and candidate.branch_index < chosen.branch_index
        )
        if better_score or earlier_branch:
            best_score = peak
            chosen = candidate
    return chosen


def _min_gain(scores: tuple[float, ...], depth: int) -> float:
    """Stay under any positive step that happens on the way to the payoff."""

    gains = [
        scores[index] - scores[index - 1] for index in range(1, min(depth, len(scores)))
    ]
    positive = [gain for gain in gains if gain > 0.0]
    if not positive:
        return _MIN_GAIN
    smallest = min(positive)
    if smallest < _MIN_GAIN:
        return smallest / 2
    return _MIN_GAIN


def _patience(scores: tuple[float, ...], depth: int, min_gain: float) -> int:
    """Smallest patience that still extends the branch through `depth`."""

    prefix = list(scores[:depth])
    for patience in range(1, 33):
        if all(
            not _is_stagnant(prefix[:length], patience, min_gain)
            for length in range(1, depth)
        ):
            return patience
    return 32


def propose_spec(incumbent: PolicySpec, trees: list[DiscoveryTree]) -> PolicySpec:
    """One adaptive spec. The incumbent is returned when the trees have no attempts."""

    targets: list[_Target] = []
    for tree in trees:
        target = _target(tree)
        if target is not None:
            targets.append(target)
    if not targets:
        return incumbent
    min_gain = min(_min_gain(target.scores, target.depth) for target in targets)
    min_gain = min(10.0, max(0.0, min_gain))
    patience = max(_patience(target.scores, target.depth, min_gain) for target in targets)
    return PolicySpec(
        family="adaptive",
        branch_count=min(64, max(target.branch_index + 1 for target in targets)),
        refine_depth=min(64, max(target.depth for target in targets)),
        patience=min(32, max(1, patience)),
        min_gain=min_gain,
    )
