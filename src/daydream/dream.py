"""Offline policy improvement by dreaming.

Each candidate is scored by replaying it on every tree in the history. The
current policy is candidate zero, so the selected policy's average replay
score is at least as high as the policy it replaces. That is the monotonic
selection rule in Section 3.

The default dreamer asks the system-1 model for one adaptive spec and keeps
it when the replay score wins. The revision grid runs only as a fallback, or
when the caller asks for it. Neither path executes model-written Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from seb_dream_rsi.policy import PolicySpec, build_policy
from seb_dream_rsi.replay import replay
from seb_dream_rsi.system1 import propose_spec
from seb_dream_rsi.tree import DiscoveryTree


@dataclass
class DreamOutcome:
    spec: PolicySpec
    score: float
    trace: list[dict] = field(default_factory=list)


def mean_replay_score(
    spec: PolicySpec,
    trees: list[DiscoveryTree],
    *,
    max_rounds: int,
    max_parallelism: int,
    lambda_cost: float,
    lambda_parallel: float,
) -> float:
    if not trees:
        return 0.0
    total = 0.0
    policy = build_policy(spec)
    for tree in trees:
        result = replay(
            policy,
            tree,
            max_rounds=max_rounds,
            max_parallelism=max_parallelism,
            lambda_cost=lambda_cost,
            lambda_parallel=lambda_parallel,
        )
        total += result.score
    return total / len(trees)


def revision_schedule(incumbent: PolicySpec, limit: int) -> list[PolicySpec]:
    """Deterministic successors the offline phase will score, in order."""

    widths = []
    for width in (
        max(1, incumbent.branch_count - 2),
        incumbent.branch_count,
        max(1, incumbent.branch_count - 1),
    ):
        if width not in widths:
            widths.append(width)
    depths = []
    for depth in (
        max(1, incumbent.refine_depth // 2),
        max(1, incumbent.refine_depth - 4),
        incumbent.refine_depth,
        max(1, incumbent.refine_depth - 2),
    ):
        if depth not in depths:
            depths.append(depth)
    specs: list[PolicySpec] = []
    seen: set[tuple] = set()
    # Mid patience first: short patience can discard a branch before it pays,
    # and the incumbent already covers "never prune".
    for patience in (4, 3, 5, 2, 6):
        for width in widths:
            for depth in depths:
                spec = PolicySpec(
                    family="adaptive",
                    branch_count=width,
                    refine_depth=depth,
                    patience=patience,
                    min_gain=0.02,
                )
                key = (
                    spec.family,
                    spec.branch_count,
                    spec.refine_depth,
                    spec.patience,
                    spec.min_gain,
                )
                if key in seen:
                    continue
                seen.add(key)
                specs.append(spec)
                if len(specs) >= limit:
                    return specs
    return specs


def _decision_key(spec: PolicySpec) -> tuple:
    return (
        spec.family,
        spec.branch_count,
        spec.refine_depth,
        spec.patience,
        spec.min_gain,
    )


def improve_policy(
    incumbent: PolicySpec,
    trees: list[DiscoveryTree],
    *,
    revisions: int = 12,
    max_rounds: int = 10_000,
    max_parallelism: int = 8,
    lambda_cost: float = 0.002,
    lambda_parallel: float = 0.005,
    extra_specs: list[PolicySpec] | None = None,
    dreamer: str = "auto",
) -> DreamOutcome:
    """Score the incumbent, then a fast proposal, then the grid if needed.

    `auto` keeps the system-1 spec when it wins and skips the revision grid.
    `system1` never searches the grid. `grid` is the full schedule.
    """

    if dreamer not in {"auto", "system1", "grid"}:
        raise ValueError("dreamer must be 'auto', 'system1', or 'grid'")
    kwargs = {
        "max_rounds": max_rounds,
        "max_parallelism": max_parallelism,
        "lambda_cost": lambda_cost,
        "lambda_parallel": lambda_parallel,
    }
    best_spec = incumbent
    best_score = mean_replay_score(incumbent, trees, **kwargs)
    trace = [{"spec": incumbent.to_dict(), "score": best_score, "role": "incumbent"}]
    seen = {_decision_key(incumbent)}

    def consider(spec: PolicySpec, role: str) -> None:
        nonlocal best_spec, best_score
        key = _decision_key(spec)
        if key in seen:
            return
        seen.add(key)
        score = mean_replay_score(spec, trees, **kwargs)
        trace.append({"spec": spec.to_dict(), "score": score, "role": role})
        if score > best_score + 1e-12:
            best_score = score
            best_spec = spec

    fast_won = False
    if dreamer in {"auto", "system1"}:
        consider(propose_spec(incumbent, trees), "system1")
        fast_won = best_spec is not incumbent
    for spec in extra_specs or []:
        consider(spec, "revision")
    if dreamer == "grid" or (dreamer == "auto" and not fast_won):
        for spec in revision_schedule(incumbent, revisions):
            consider(spec, "revision")
    return DreamOutcome(spec=best_spec, score=best_score, trace=trace)
