"""LayerNorm kernel search, without a GPU.

KernelBench scores a kernel by inverse runtime after a correctness check.
This task does the same two steps on a CPU LayerNorm: the variant must match
a reference on a fixed batch, and the search score is the inverse of a
declared operation cost. Incorrect variants score 0.

The cost is algorithmic, not a wall-clock measurement, so the ranking does
not depend on the machine that runs the tests.
"""

from __future__ import annotations

import math
from typing import Any

from seb_dream_rsi.online import EvalResult, Proposal
from seb_dream_rsi.policy import PolicySpec

_EPS = 1e-5
_COSTS = {
    "naive": 80.0,
    "two_pass": 40.0,
    "blocked": 22.0,
    "buggy": 8.0,
}


def _reference(rows: list[list[float]]) -> list[list[float]]:
    normalized: list[list[float]] = []
    for row in rows:
        mean = sum(row) / len(row)
        var = sum((value - mean) ** 2 for value in row) / len(row)
        scale = math.sqrt(var + _EPS)
        normalized.append([(value - mean) / scale for value in row])
    return normalized


def _naive(rows: list[list[float]]) -> list[list[float]]:
    return _reference(rows)


def _two_pass(rows: list[list[float]]) -> list[list[float]]:
    output: list[list[float]] = []
    for row in rows:
        mean = sum(row) / len(row)
        second = sum(value * value for value in row) / len(row)
        var = max(0.0, second - mean * mean)
        scale = math.sqrt(var + _EPS)
        output.append([(value - mean) / scale for value in row])
    return output


def _blocked(rows: list[list[float]]) -> list[list[float]]:
    output: list[list[float]] = []
    for row in rows:
        total = 0.0
        total_sq = 0.0
        block = 4
        index = 0
        while index < len(row):
            chunk = row[index : index + block]
            total += sum(chunk)
            total_sq += sum(value * value for value in chunk)
            index += block
        mean = total / len(row)
        var = max(0.0, total_sq / len(row) - mean * mean)
        scale = math.sqrt(var + _EPS)
        output.append([(value - mean) / scale for value in row])
    return output


def _buggy(rows: list[list[float]]) -> list[list[float]]:
    output: list[list[float]] = []
    for row in rows:
        mean = sum(row) / len(row)
        var = sum((value - mean) ** 2 for value in row) / len(row)
        output.append([(value - mean) / (math.sqrt(var) + 1.0) for value in row])
    return output


_IMPL = {
    "naive": _naive,
    "two_pass": _two_pass,
    "blocked": _blocked,
    "buggy": _buggy,
}

_ORDER = ["naive", "buggy", "two_pass", "blocked"]


def _fixture() -> list[list[float]]:
    rows: list[list[float]] = []
    state = 11
    for _ in range(6):
        row: list[float] = []
        for _column in range(8):
            state = (1664525 * state + 1013904223) % (2**31)
            row.append(state / (2**31 - 1) * 4.0 - 2.0)
        rows.append(row)
    return rows


def _close(left: list[list[float]], right: list[list[float]], tol: float = 1e-8) -> bool:
    if len(left) != len(right):
        return False
    for left_row, right_row in zip(left, right):
        if len(left_row) != len(right_row):
            return False
        for a, b in zip(left_row, right_row):
            if abs(a - b) > tol:
                return False
    return True


class LayerNormTask:
    name = "kernel-layernorm"

    def __init__(self) -> None:
        self.rows = _fixture()
        self.expected = _reference(self.rows)

    def initial_spec(self) -> PolicySpec:
        return PolicySpec(
            family="parallel_refine",
            branch_count=4,
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
        del parent_artifact, trajectory
        # Each branch walks the variant list at a different offset. Deeper
        # steps move toward the blocked kernel; one offset lands on the bug.
        variant = _ORDER[(branch_index + depth - 1) % len(_ORDER)]
        return Proposal(
            artifact={"variant": variant},
            proposal=f"layernorm variant {variant}",
        )

    def evaluate(self, artifact: dict[str, Any]) -> EvalResult:
        variant = str(artifact.get("variant", ""))
        function = _IMPL.get(variant)
        if function is None:
            return EvalResult(score=0.0, valid=False, error="unknown variant")
        produced = function(self.rows)
        if not _close(produced, self.expected):
            return EvalResult(
                score=0.0,
                valid=False,
                error="mismatch versus reference",
                diagnostics={"variant": variant},
            )
        cost = _COSTS[variant]
        return EvalResult(
            score=1.0 / cost,
            diagnostics={"variant": variant, "cost": cost},
        )
