"""Executable exploration policies.

The initial policy is the paper's parallel-refine controller: open several
branches, then keep refining each one up to a fixed depth. The adaptive
policy reads only the revealed prefix. It prunes branches that have stopped
improving and keeps a batch of the remaining frontiers.

Both policies are JSON specs. A local Python module can also be loaded when
the caller trusts that file. Model output is not executed as Python.
"""

from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from daydream.view import TreeView


@dataclass(frozen=True)
class PolicySpec:
    family: str
    branch_count: int = 6
    refine_depth: int = 8
    patience: int = 4
    min_gain: float = 0.02
    name: str = ""

    def label(self) -> str:
        if self.name:
            return self.name
        if self.family == "parallel_refine":
            return f"parallel-refine-w{self.branch_count}-d{self.refine_depth}"
        return (
            f"adaptive-w{self.branch_count}-d{self.refine_depth}"
            f"-p{self.patience}-g{self.min_gain}"
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["name"] = self.label()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> PolicySpec:
        family = data.get("family")
        if family not in {"parallel_refine", "adaptive"}:
            raise ValueError(f"unknown policy family: {family}")
        branch_count = int(data.get("branch_count", 6))
        refine_depth = int(data.get("refine_depth", 8))
        patience = int(data.get("patience", 4))
        min_gain = float(data.get("min_gain", 0.02))
        if not 1 <= branch_count <= 64:
            raise ValueError("branch_count must be in [1, 64]")
        if not 1 <= refine_depth <= 64:
            raise ValueError("refine_depth must be in [1, 64]")
        if not 1 <= patience <= 32:
            raise ValueError("patience must be in [1, 32]")
        if not 0.0 <= min_gain <= 10.0:
            raise ValueError("min_gain must be in [0, 10]")
        return cls(
            family=family,
            branch_count=branch_count,
            refine_depth=refine_depth,
            patience=patience,
            min_gain=min_gain,
            name=str(data.get("name") or ""),
        )


class ExplorationPolicy(Protocol):
    spec: PolicySpec

    def reset(self) -> None: ...

    def select_batch(self, view: TreeView, max_parallelism: int) -> list[str]: ...


def _is_stagnant(scores: list[float], patience: int, min_gain: float) -> bool:
    """True when the last `patience` steps all failed to improve by `min_gain`.

    A branch shorter than `patience` gains is kept. That is what lets a late
    jump survive long enough to be observed.
    """

    if len(scores) < patience + 1:
        return False
    gains = [scores[index] - scores[index - 1] for index in range(1, len(scores))]
    if len(gains) < patience:
        return False
    return all(gain < min_gain for gain in gains[-patience:])


class ParallelRefinePolicy:
    """Hand-written initial controller used by both methods in round 1."""

    def __init__(self, spec: PolicySpec) -> None:
        self.spec = spec

    def reset(self) -> None:
        return None

    def select_batch(self, view: TreeView, max_parallelism: int) -> list[str]:
        limit = max(1, max_parallelism)
        batch: list[str] = []
        opened = view.opened_branch_count()
        room = max(0, self.spec.branch_count - opened)
        for root in view.legal_roots[:room]:
            if len(batch) >= limit:
                break
            batch.append(root.id)
        leaves = [
            node for node in view.leaves() if node.depth < self.spec.refine_depth
        ]
        leaves.sort(key=lambda node: (node.depth, node.branch_index or 0, node.id))
        for leaf in leaves:
            if len(batch) >= limit:
                break
            batch.append(leaf.id)
        return batch


class AdaptivePolicy:
    """Prefix-only controller. Thresholds are fixed for the whole rollout."""

    def __init__(self, spec: PolicySpec) -> None:
        self.spec = spec

    def reset(self) -> None:
        return None

    def select_batch(self, view: TreeView, max_parallelism: int) -> list[str]:
        limit = max(1, max_parallelism)
        opened = view.opened_branch_count()
        room = max(0, self.spec.branch_count - opened)
        roots = list(view.legal_roots)[:room]
        frontiers = []
        for leaf in view.leaves():
            if leaf.depth >= self.spec.refine_depth:
                continue
            scores = [
                node.score or 0.0 for node in view.trajectory(leaf.id)
            ]
            if _is_stagnant(scores, self.spec.patience, self.spec.min_gain):
                continue
            frontiers.append(leaf)
        frontiers.sort(
            key=lambda node: (
                -(node.score or 0.0),
                node.depth,
                node.branch_index or 0,
                node.id,
            )
        )
        batch: list[str] = []
        explore_slots = len(roots)
        if frontiers and roots:
            explore_slots = min(len(roots), max(1, limit // 2))
        for root in roots[:explore_slots]:
            batch.append(root.id)
        for leaf in frontiers:
            if len(batch) >= limit:
                break
            if leaf.id not in batch:
                batch.append(leaf.id)
        for root in roots[explore_slots:]:
            if len(batch) >= limit:
                break
            batch.append(root.id)
        return batch[:limit]


def build_policy(spec: PolicySpec) -> ExplorationPolicy:
    if spec.family == "parallel_refine":
        return ParallelRefinePolicy(spec)
    if spec.family == "adaptive":
        return AdaptivePolicy(spec)
    raise ValueError(f"unknown policy family: {spec.family}")


def load_policy_file(path: str | Path) -> ExplorationPolicy:
    """Load a trusted local module that defines POLICY.

    The module is imported in-process. Use this for a policy you wrote.
    Do not point it at model output.
    """

    file_path = Path(path).resolve()
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    module_spec = importlib.util.spec_from_file_location(
        "daydream_user_policy", file_path
    )
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"cannot import {file_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    policy = getattr(module, "POLICY", None)
    if policy is None or not hasattr(policy, "select_batch"):
        raise AttributeError(f"{file_path} must define POLICY with select_batch")
    if not hasattr(policy, "spec"):
        raise AttributeError(f"{file_path} POLICY must define spec: PolicySpec")
    policy.reset() if hasattr(policy, "reset") else None
    return policy
