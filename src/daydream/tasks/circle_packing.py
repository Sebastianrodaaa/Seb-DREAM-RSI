"""Circle packing in the unit square.

For a list of circles, the score is the sum of radii when every circle lies
inside the square and no two interiors overlap. An infeasible packing scores
0. This is the n-circle objective in Appendix A, with n chosen by the caller
so a local run stays small. The paper reports n in {26, 32}.
"""

from __future__ import annotations

from typing import Any

from seb_dream_rsi.online import EvalResult, Proposal
from seb_dream_rsi.policy import PolicySpec


def packing_is_valid(circles: list[dict[str, float]], *, slack: float = 1e-9) -> bool:
    for circle in circles:
        radius = float(circle["r"])
        if radius < -slack:
            return False
        if float(circle["x"]) < radius - slack or float(circle["x"]) > 1.0 - radius + slack:
            return False
        if float(circle["y"]) < radius - slack or float(circle["y"]) > 1.0 - radius + slack:
            return False
    for index, left in enumerate(circles):
        for right in circles[index + 1 :]:
            dx = float(left["x"]) - float(right["x"])
            dy = float(left["y"]) - float(right["y"])
            gap = float(left["r"]) + float(right["r"])
            if dx * dx + dy * dy < gap * gap - slack:
                return False
    return True


def packing_score(circles: list[dict[str, float]]) -> float:
    if not circles or not packing_is_valid(circles):
        return 0.0
    return float(sum(float(circle["r"]) for circle in circles))


def _grid_pack(count: int, radius_scale: float) -> list[dict[str, float]]:
    side = max(1, int(count**0.5 + 0.999))
    radius = radius_scale / (2 * side)
    circles: list[dict[str, float]] = []
    for index in range(count):
        row, col = divmod(index, side)
        circles.append(
            {
                "x": (col + 0.5) / side,
                "y": (row + 0.5) / side,
                "r": radius,
            }
        )
    return circles


class CirclePackingTask:
    name = "circle-packing"

    def __init__(self, count: int = 8) -> None:
        if count < 1:
            raise ValueError("count must be positive")
        self.count = count

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
        del trajectory
        # Scales below 1 stay feasible on the grid. Deeper steps nudge upward.
        scale = 0.72 + 0.04 * ((branch_index + depth) % 5) + 0.01 * depth
        circles = _grid_pack(self.count, scale)
        if parent_artifact.get("circles") and depth > 1:
            grown = []
            for circle in parent_artifact["circles"]:
                grown.append(
                    {
                        "x": float(circle["x"]),
                        "y": float(circle["y"]),
                        "r": float(circle["r"]) * (1.0 + 0.01 * ((branch_index % 3) + 1)),
                    }
                )
            if packing_is_valid(grown):
                circles = grown
        return Proposal(
            artifact={"circles": circles, "count": self.count},
            proposal=f"grid packing n={self.count} branch={branch_index} depth={depth}",
        )

    def evaluate(self, artifact: dict[str, Any]) -> EvalResult:
        circles = list(artifact.get("circles") or [])
        score = packing_score(circles)
        return EvalResult(
            score=score,
            valid=score > 0.0,
            error=None if score > 0.0 else "infeasible packing",
            diagnostics={"count": len(circles), "sum_radii": score},
        )
