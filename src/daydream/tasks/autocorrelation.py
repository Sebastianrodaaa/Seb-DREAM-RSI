"""First autocorrelation inequality, discretized.

Find a non-negative f supported on [-1/4, 1/4] with integral 1. Appendix A
asks to minimize the maximum autoconvolution on [-1/2, 1/2]. The loop stores
a higher-is-better score, so the node score is the inverse of that peak and
the peak itself is kept on the diagnostics.

The grid is coarse on purpose. This is a working evaluator for the discovery
loop, not a claim about the continuous optimum.
"""

from __future__ import annotations

from typing import Any

from seb_dream_rsi.online import EvalResult, Proposal
from seb_dream_rsi.policy import PolicySpec

BINS = 21
SUPPORT = 0.5  # length of [-1/4, 1/4]
DX = SUPPORT / BINS


def autoconvolution_peak(values: list[float], dx: float = DX) -> float | None:
    cleaned = [max(0.0, float(value)) for value in values]
    mass = sum(cleaned) * dx
    if mass <= 1e-15:
        return None
    density = [value / mass for value in cleaned]
    width = len(density)
    peak = 0.0
    for lag in range(-(width - 1), width):
        acc = 0.0
        for index in range(width):
            other = index + lag
            if 0 <= other < width:
                acc += density[index] * density[other]
        height = acc * dx
        if height > peak:
            peak = height
    return peak


def _shape(branch_index: int, depth: int) -> list[float]:
    center = (BINS - 1) / 2
    width = 2.0 + (branch_index % 5) + depth * 0.35
    values: list[float] = []
    for index in range(BINS):
        distance = abs(index - center) / width
        if branch_index % 2 == 0:
            values.append(max(0.0, 1.0 - distance))
        else:
            values.append(2.0 ** (-distance * distance))
    return values


class AutocorrelationTask:
    name = "autocorrelation"

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
        return Proposal(
            artifact={"values": _shape(branch_index, depth)},
            proposal=f"discretized density branch={branch_index} depth={depth}",
        )

    def evaluate(self, artifact: dict[str, Any]) -> EvalResult:
        values = [float(value) for value in artifact.get("values") or []]
        if len(values) != BINS:
            return EvalResult(score=0.0, valid=False, error=f"expected {BINS} bins")
        peak = autoconvolution_peak(values)
        if peak is None:
            return EvalResult(score=0.0, valid=False, error="density has no mass")
        return EvalResult(
            score=1.0 / peak,
            diagnostics={"peak": peak, "bins": BINS},
        )
