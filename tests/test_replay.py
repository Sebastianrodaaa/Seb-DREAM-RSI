"""Replay semantics from Zheng et al. 2026, Section 3."""

from daydream.agents import ScriptedAgent
from daydream.online import EvalResult, run_online
from daydream.policy import PolicySpec, build_policy
from daydream.replay import replay, replay_score
from daydream.tasks.branch_world import BranchWorld
from daydream.tree import DiscoveryTree
from daydream.view import view_from_revealed


class _Counting(BranchWorld):
    def __init__(self) -> None:
        self.calls = 0

    def scripted_propose(self, parent_artifact, *, branch_index, depth, trajectory):
        self.calls += 1
        return super().scripted_propose(
            parent_artifact,
            branch_index=branch_index,
            depth=depth,
            trajectory=trajectory,
        )

    def evaluate(self, artifact):
        self.calls += 1
        return super().evaluate(artifact)


class _OpenBranch:
    """Open one recorded branch, by index, and follow it to the end."""

    spec = PolicySpec(family="adaptive", branch_count=1, refine_depth=8, name="open-one")

    def __init__(self, branch_index: int) -> None:
        self.branch_index = branch_index
        self.target: str | None = None

    def reset(self) -> None:
        self.target = None

    def select_batch(self, view, max_parallelism: int) -> list[str]:
        del max_parallelism
        if self.target is None:
            for root in view.legal_roots:
                if root.branch_index == self.branch_index:
                    self.target = root.id
                    return [root.id]
            return []
        leaves = [node for node in view.leaves() if node.branch_index == self.branch_index]
        if not leaves:
            return []
        return [leaves[0].id]


def test_replay_score_balances_quality_cost_and_batching():
    # quality 1, 10 attempts, 2 rounds, cost 0.1, parallel bonus 0.5
    # 1 - 1 + 0.5 * (10/2) = 2.5
    assert replay_score(1.0, 10, 2, 0.1, 0.5) == 2.5
    assert replay_score(1.0, 0, 0, 0.1, 0.5) == 0.0


def test_replay_can_skip_recorded_branches_and_does_not_call_the_agent():
    task = _Counting()
    spec = task.initial_spec()
    tree = run_online(
        build_policy(spec),
        ScriptedAgent(task),
        task,
        max_parallelism=8,
    )
    calls_after_online = task.calls
    assert tree.attempt_count == 48
    assert tree.best_score() == 0.95

    only_late = replay(_OpenBranch(3), tree, max_parallelism=8)
    assert task.calls == calls_after_online
    assert only_late.quality == 0.95
    assert only_late.attempts == 8
    assert only_late.attempts < tree.attempt_count


def test_unopened_branches_hide_scores():
    task = BranchWorld()
    tree = run_online(
        build_policy(task.initial_spec()),
        ScriptedAgent(task),
        task,
        max_parallelism=8,
    )
    view = view_from_revealed(tree, {"root"})
    assert view.nodes.keys() == {"root"}
    assert len(view.legal_roots) == 6
    assert not hasattr(view.legal_roots[0], "score")


def test_tree_json_roundtrip():
    task = BranchWorld()
    tree = run_online(
        build_policy(PolicySpec(family="parallel_refine", branch_count=2, refine_depth=2)),
        ScriptedAgent(task),
        task,
        max_parallelism=4,
    )
    restored = DiscoveryTree.from_dict(tree.to_dict())
    assert restored.attempt_count == tree.attempt_count
    assert restored.best_score() == tree.best_score()


def test_evaluator_exception_is_a_failed_node():
    class _Boom:
        def evaluate(self, artifact):
            del artifact
            raise RuntimeError("evaluator blew up")

    task = BranchWorld()
    tree = run_online(
        build_policy(PolicySpec(family="parallel_refine", branch_count=1, refine_depth=1)),
        ScriptedAgent(task),
        _Boom(),
        max_parallelism=2,
    )
    node = next(node for node in tree.nodes.values() if node.id != "root")
    assert node.valid is False
    assert node.score == 0.0
    assert node.error is not None
    assert isinstance(EvalResult(score=0.0), EvalResult)
