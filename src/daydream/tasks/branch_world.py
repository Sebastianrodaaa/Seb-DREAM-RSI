"""A didactic search world with heterogeneous branches.

Scores are a function of branch index and depth. They are not measurements
from the paper. The shape matches the exploration problem Dream-RSI is about:
some directions are dead, one pays off only after a few flat steps, and a
fixed parallel-refine policy spends the same budget on all of them.

Branch 3 jumps from 0.20 to 0.95 at depth 4. Branches 0, 1, and 5 never move.
Branch 2 climbs steadily. A policy that keeps a branch for four steps, then
drops the flat ones, still reaches 0.95 and reveals fewer nodes.
"""

from __future__ import annotations

from typing import Any

from daydream.online import EvalResult, Proposal
from daydream.policy import PolicySpec


def branch_score(branch_index: int, depth: int) -> float:
    if branch_index == 0:
        return 0.30
    if branch_index == 1:
        return 0.35
    if branch_index == 2:
        return 0.50 + 0.05 * min(depth, 8)
    if branch_index == 3:
        return 0.20 if depth < 4 else 0.95
    if branch_index == 4:
        return 0.40 + 0.01 * depth
    return 0.15


class BranchWorld:
    name = "branch-world"

    def initial_spec(self) -> PolicySpec:
        return PolicySpec(
            family="parallel_refine",
            branch_count=6,
            refine_depth=8,
            name="parallel_refine",
        )

    def scripted_propose(
        self,
        parent_artifact: dict[str, Any],
        *,
        branch_index: int,
        depth: int,
        trajectory: list[dict[str, Any]],
    ) -> Proposal:
        del parent_artifact, trajectory
        return Proposal(
            artifact={"branch_index": branch_index, "depth": depth},
            proposal=f"continue branch {branch_index} to depth {depth}",
        )

    def evaluate(self, artifact: dict[str, Any]) -> EvalResult:
        branch_index = int(artifact["branch_index"])
        depth = int(artifact["depth"])
        score = branch_score(branch_index, depth)
        return EvalResult(
            score=score,
            diagnostics={"branch_index": branch_index, "depth": depth},
        )
