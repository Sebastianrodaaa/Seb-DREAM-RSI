"""Outer recursive self-improvement loop.

Round t deploys the current policy, appends the new discovery tree to the
history, then (unless the baseline is fixed) dreams over that history and
keeps the policy with the best average replay score. The next round uses it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from daydream.dream import DreamOutcome, improve_policy
from daydream.online import DiscoveryAgent, Evaluator, run_online
from daydream.policy import PolicySpec, build_policy
from daydream.storage import save_run
from daydream.tree import DiscoveryTree


@dataclass
class RoundSummary:
    round_index: int
    policy: dict[str, Any]
    agent_calls: int
    cumulative_calls: int
    best_score: float
    dream_score: float | None = None
    selected_policy: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "policy": self.policy,
            "agent_calls": self.agent_calls,
            "cumulative_calls": self.cumulative_calls,
            "best_score": self.best_score,
            "dream_score": self.dream_score,
            "selected_policy": self.selected_policy,
        }


@dataclass
class RunResult:
    mode: str
    task: str
    rounds: list[RoundSummary] = field(default_factory=list)
    best_score: float = 0.0
    agent_calls: int = 0
    final_policy: dict[str, Any] = field(default_factory=dict)
    trees: list[DiscoveryTree] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "task": self.task,
            "best_score": self.best_score,
            "agent_calls": self.agent_calls,
            "final_policy": self.final_policy,
            "rounds": [item.to_dict() for item in self.rounds],
        }


def run_discovery(
    task_name: str,
    agent: DiscoveryAgent,
    evaluator: Evaluator,
    initial_spec: PolicySpec,
    *,
    mode: str = "dream",
    rounds: int = 2,
    max_rounds: int = 64,
    max_parallelism: int = 8,
    revisions: int = 12,
    lambda_cost: float = 0.002,
    lambda_parallel: float = 0.005,
    out_dir: str | None = None,
    extra_specs: list[PolicySpec] | None = None,
    dreamer: str = "auto",
) -> RunResult:
    if mode not in {"dream", "fixed"}:
        raise ValueError("mode must be 'dream' or 'fixed'")
    if rounds < 1:
        raise ValueError("rounds must be >= 1")

    spec = initial_spec
    history: list[DiscoveryTree] = []
    summaries: list[RoundSummary] = []
    cumulative = 0
    best: float | None = None
    last_dream: DreamOutcome | None = None

    for index in range(1, rounds + 1):
        deployed = spec
        tree = run_online(
            build_policy(deployed),
            agent,
            evaluator,
            max_rounds=max_rounds,
            max_parallelism=max_parallelism,
        )
        calls = tree.attempt_count
        cumulative += calls
        round_best = tree.best_score()
        best = round_best if best is None else max(best, round_best)
        history.append(tree)
        dream_score = None
        if mode == "dream":
            last_dream = improve_policy(
                deployed,
                history,
                revisions=revisions,
                max_rounds=max(max_rounds, 10_000),
                max_parallelism=max_parallelism,
                lambda_cost=lambda_cost,
                lambda_parallel=lambda_parallel,
                extra_specs=extra_specs,
                dreamer=dreamer,
            )
            spec = last_dream.spec
            dream_score = last_dream.score
        summaries.append(
            RoundSummary(
                round_index=index,
                policy=deployed.to_dict(),
                agent_calls=calls,
                cumulative_calls=cumulative,
                best_score=round_best,
                dream_score=dream_score,
                selected_policy=spec.to_dict(),
            )
        )

    result = RunResult(
        mode=mode,
        task=task_name,
        rounds=summaries,
        best_score=0.0 if best is None else best,
        agent_calls=cumulative,
        final_policy=spec.to_dict(),
        trees=history,
    )
    if out_dir:
        save_run(out_dir, result, last_dream)
    return result
