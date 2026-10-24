"""The fast policy model proposes one spec and skips the grid when it wins."""

from seb_dream_rsi.agents import ScriptedAgent
from seb_dream_rsi.dream import improve_policy, mean_replay_score
from seb_dream_rsi.online import run_online
from seb_dream_rsi.policy import PolicySpec, build_policy
from seb_dream_rsi.system1 import propose_spec
from seb_dream_rsi.tasks.branch_world import BranchWorld
from seb_dream_rsi.tree import DiscoveryTree


def _tree(branch_count: int, refine_depth: int):
    task = BranchWorld()
    spec = PolicySpec(
        family="parallel_refine",
        branch_count=branch_count,
        refine_depth=refine_depth,
        name="parallel_refine",
    )
    return task, spec, run_online(
        build_policy(spec),
        ScriptedAgent(task),
        task,
        max_parallelism=8,
    )


def _replay(spec, tree):
    return mean_replay_score(
        spec,
        [tree],
        max_rounds=10_000,
        max_parallelism=8,
        lambda_cost=0.002,
        lambda_parallel=0.005,
    )


def test_system1_reaches_the_late_payoff_without_the_grid():
    _task, incumbent, tree = _tree(6, 8)
    spec = propose_spec(incumbent, [tree])
    assert spec.family == "adaptive"
    assert spec.branch_count == 4
    assert spec.refine_depth == 4
    assert spec.patience >= 3

    outcome = improve_policy(incumbent, [tree], revisions=12, dreamer="auto")
    assert [item["role"] for item in outcome.trace] == ["incumbent", "system1"]
    assert outcome.score > _replay(incumbent, tree)
    assert outcome.spec.family == "adaptive"


def test_short_horizon_falls_back_to_the_grid():
    """A two-round tree pays a parallelism bonus for revealing every node."""

    _task, incumbent, tree = _tree(6, 2)
    outcome = improve_policy(incumbent, [tree], revisions=8, dreamer="auto")
    roles = [item["role"] for item in outcome.trace]
    assert "system1" in roles
    assert "revision" in roles
    assert outcome.score + 1e-12 >= _replay(incumbent, tree)


def test_system1_dreamer_never_searches_the_grid():
    _task, incumbent, tree = _tree(6, 2)
    outcome = improve_policy(incumbent, [tree], revisions=8, dreamer="system1")
    assert [item["role"] for item in outcome.trace] == ["incumbent", "system1"]
    assert outcome.score + 1e-12 >= _replay(incumbent, tree)


def test_empty_history_keeps_the_incumbent():
    incumbent = PolicySpec(family="parallel_refine", branch_count=6, refine_depth=8)
    assert propose_spec(incumbent, [DiscoveryTree.new()]) == incumbent
