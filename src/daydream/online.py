"""Online rollout.

The policy's code stays fixed for the rollout. Each decision picks a batch of
parents. The discovery agent proposes one child per parent and the evaluator
scores it. Within a batch the attempts do not see each other.

Batches run in selection order so a seed fully determines the tree. The batch
size is still the parallelism the policy is charged for during replay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from daydream.policy import ExplorationPolicy
from daydream.tree import ROOT_ID, DiscoveryTree
from daydream.view import sanitize_batch, view_from_online


@dataclass(frozen=True)
class Proposal:
    artifact: dict[str, Any]
    proposal: str


@dataclass
class EvalResult:
    score: float
    valid: bool = True
    error: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


class DiscoveryAgent(Protocol):
    def propose(
        self,
        parent_artifact: dict[str, Any],
        *,
        branch_index: int,
        depth: int,
        trajectory: list[dict[str, Any]],
    ) -> Proposal: ...


class Evaluator(Protocol):
    def evaluate(self, artifact: dict[str, Any]) -> EvalResult: ...


def _trajectory_payload(tree: DiscoveryTree, parent_id: str) -> list[dict[str, Any]]:
    if parent_id == ROOT_ID:
        return []
    chain: list[dict[str, Any]] = []
    current: str | None = parent_id
    seen: set[str] = set()
    while current and current != ROOT_ID and current not in seen:
        seen.add(current)
        node = tree.nodes[current]
        chain.append(
            {
                "id": node.id,
                "score": node.score,
                "valid": node.valid,
                "error": node.error,
                "proposal": node.proposal,
                "depth": node.depth,
            }
        )
        current = node.parent_id
    chain.reverse()
    return chain


def _resolve(tree: DiscoveryTree, action: str) -> tuple[str, int, int]:
    if action.startswith("slot:"):
        branch_index = int(action.split(":", 1)[1])
        return ROOT_ID, branch_index, 1
    node = tree.nodes[action]
    if node.branch_index is None:
        raise ValueError(f"{action} is not on a branch")
    return action, node.branch_index, node.depth + 1


def run_online(
    policy: ExplorationPolicy,
    agent: DiscoveryAgent,
    evaluator: Evaluator,
    *,
    max_rounds: int = 64,
    max_parallelism: int = 8,
) -> DiscoveryTree:
    tree = DiscoveryTree.new()
    policy.reset()
    for _ in range(max_rounds):
        view = view_from_online(tree, policy.spec.branch_count)
        batch = sanitize_batch(
            list(policy.select_batch(view, max_parallelism)),
            view,
            max_parallelism,
        )
        if not batch:
            break
        planned: list[tuple[str, int, int, list[dict[str, Any]]]] = []
        for action in batch:
            parent_id, branch_index, depth = _resolve(tree, action)
            planned.append(
                (parent_id, branch_index, depth, _trajectory_payload(tree, parent_id))
            )
        for parent_id, branch_index, depth, trajectory in planned:
            parent_artifact = (
                {}
                if parent_id == ROOT_ID
                else dict(tree.nodes[parent_id].artifact)
            )
            try:
                proposal = agent.propose(
                    parent_artifact,
                    branch_index=branch_index,
                    depth=depth,
                    trajectory=trajectory,
                )
                result = evaluator.evaluate(proposal.artifact)
            except Exception as exc:  # a failed attempt is a node, not a crash
                proposal = Proposal(artifact={"error": str(exc)}, proposal="failed attempt")
                result = EvalResult(score=0.0, valid=False, error=f"{type(exc).__name__}: {exc}")
            tree.add_child(
                parent_id,
                branch_index=branch_index,
                score=float(result.score),
                artifact=dict(proposal.artifact),
                diagnostics=dict(result.diagnostics),
                valid=bool(result.valid) and result.error is None,
                error=result.error,
                proposal=proposal.proposal,
            )
    return tree
