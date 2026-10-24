"""One search loop for live rollout and replay.

The released Dream-RSI runner has no second interpreter: online and offline
both call ``policy.solve(env)``. Here the policy still returns a batch from a
tree view. ``solve`` is the only loop. A live environment evaluates the batch.
A replay environment reveals nodes that were already stored.
"""

from __future__ import annotations

from typing import Protocol

from daydream.policy import ExplorationPolicy
from daydream.view import TreeView, sanitize_batch


class SearchEnv(Protocol):
    def finished(self) -> bool: ...

    def view(self, branch_count: int) -> TreeView: ...

    def probe_batch(self, batch: list[str]) -> bool: ...


def solve(
    policy: ExplorationPolicy,
    env: SearchEnv,
    *,
    max_rounds: int,
    max_parallelism: int,
) -> int:
    """Run one policy against one environment. Returns nonempty decision rounds."""

    policy.reset()
    rounds = 0
    for _ in range(max_rounds):
        if env.finished():
            break
        view = env.view(policy.spec.branch_count)
        batch = sanitize_batch(
            list(policy.select_batch(view, max_parallelism)),
            view,
            max_parallelism,
        )
        if not batch:
            break
        probed = env.probe_batch(batch)
        rounds += 1
        if not probed:
            break
    return rounds
