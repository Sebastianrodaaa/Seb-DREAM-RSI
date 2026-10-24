"""Sum-difference constructions.

Gamma(A) = log(|A+A| / |A|) / log(|A-A| / |A|), as in Appendix A of
Zheng et al. 2026. The scripted agent starts from a fixed seed per branch
and then inserts the integer, inside a bounded window, that most improves
gamma. The evaluator recomputes gamma from the set; it does not trust the
proposal.
"""

from __future__ import annotations

import math
from typing import Any

from seb_dream_rsi.online import EvalResult, Proposal
from seb_dream_rsi.policy import PolicySpec

# A published more-sums-than-differences example is included as one seed.
# Other seeds are arithmetic or geometric so the branches are not alike.
SEEDS: dict[int, list[int]] = {
    0: [0, 1, 2, 3, 4, 5, 6, 7],
    1: [-4, -3, -2, -1, 0, 1, 2, 3, 4],
    2: [0, 2, 3, 4, 7, 11, 12, 14],
    3: [0, 1, 3, 8, 12, 16, 18],
    4: [0, 2, 5, 9, 15, 17],
    5: [0, 4, 6, 9, 13],
}


def sum_diff_gamma(values: list[int]) -> float:
    unique = sorted(set(int(value) for value in values))
    size = len(unique)
    if size < 2:
        return 0.0
    sums = {left + right for left in unique for right in unique}
    diffs = {left - right for left in unique for right in unique}
    denominator = math.log(len(diffs) / size)
    numerator = math.log(len(sums) / size)
    if denominator <= 1e-12 or numerator <= 0.0:
        return 0.0
    return numerator / denominator


class SumDiffTask:
    name = "sum-diff"

    def initial_spec(self) -> PolicySpec:
        return PolicySpec(
            family="parallel_refine",
            branch_count=6,
            refine_depth=4,
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
        del trajectory
        if depth <= 1 or "values" not in parent_artifact:
            values = list(SEEDS.get(branch_index % len(SEEDS), SEEDS[0]))
            return Proposal(
                artifact={"values": values, "branch_index": branch_index},
                proposal=f"seed construction for branch {branch_index}",
            )
        current = [int(value) for value in parent_artifact["values"]]
        occupied = set(current)
        best_values = current
        best_gamma = sum_diff_gamma(current)
        for candidate in range(-12, 24):
            if candidate in occupied:
                continue
            trial = current + [candidate]
            gamma = sum_diff_gamma(trial)
            if gamma > best_gamma + 1e-12:
                best_gamma = gamma
                best_values = trial
        return Proposal(
            artifact={"values": best_values, "branch_index": branch_index},
            proposal=f"greedy insertion on branch {branch_index} at depth {depth}",
        )

    def evaluate(self, artifact: dict[str, Any]) -> EvalResult:
        values = [int(value) for value in artifact.get("values", [])]
        gamma = sum_diff_gamma(values)
        return EvalResult(
            score=gamma,
            diagnostics={"size": len(set(values)), "gamma": gamma},
        )
