"""Offline policy improvement by dreaming.

Each candidate is scored by replaying it on every tree in the history. The
current policy is candidate zero, so the selected policy's average replay
score is at least as high as the policy it replaces. That is the monotonic
selection rule in Section 3.

The default dreamer is a deterministic revision schedule over the adaptive
family. An optional chat-model dreamer may propose the same JSON spec. It
cannot return executable Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from seb_dream_rsi.policy import PolicySpec, build_policy
from seb_dream_rsi.replay import replay
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
) -> DreamOutcome:
    """Score the incumbent and later revisions. Keep the best average."""

    kwargs = {
        "max_rounds": max_rounds,
        "max_parallelism": max_parallelism,
        "lambda_cost": lambda_cost,
        "lambda_parallel": lambda_parallel,
    }
    best_spec = incumbent
    best_score = mean_replay_score(incumbent, trees, **kwargs)
    trace = [{"spec": incumbent.to_dict(), "score": best_score, "role": "incumbent"}]
    candidates = list(extra_specs or []) + revision_schedule(incumbent, revisions)
    for spec in candidates:
        score = mean_replay_score(spec, trees, **kwargs)
        trace.append({"spec": spec.to_dict(), "score": score, "role": "revision"})
        if score > best_score + 1e-12:
            best_score = score
            best_spec = spec
    return DreamOutcome(spec=best_spec, score=best_score, trace=trace)
