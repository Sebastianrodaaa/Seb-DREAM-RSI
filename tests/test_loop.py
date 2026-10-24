"""The outer loop, including the fixed-exploration baseline."""

import json
from pathlib import Path

from daydream.agents import ScriptedAgent, parse_json_object
from daydream.dream import improve_policy, mean_replay_score
from daydream.loop import run_discovery
from daydream.online import EvalResult, Proposal, run_online
from daydream.policy import PolicySpec, build_policy, load_policy_file
from daydream.tasks.branch_world import BranchWorld


def _pair(rounds: int = 2):
    task = BranchWorld()
    kwargs = {
        "rounds": rounds,
        "max_parallelism": 8,
        "revisions": 8,
    }
    fixed = run_discovery(
        task.name,
        ScriptedAgent(task),
        task,
        task.initial_spec(),
        mode="fixed",
        **kwargs,
    )
    dreamed = run_discovery(
        task.name,
        ScriptedAgent(task),
        task,
        task.initial_spec(),
        mode="dream",
        **kwargs,
    )
    return fixed, dreamed


def test_round_one_matches_and_later_rounds_spend_less():
    fixed, dreamed = _pair()
    assert fixed.rounds[0].agent_calls == dreamed.rounds[0].agent_calls == 48
    assert fixed.rounds[0].best_score == dreamed.rounds[0].best_score == 0.95
    assert dreamed.agent_calls < fixed.agent_calls
    assert dreamed.best_score >= 0.95
    assert dreamed.final_policy["family"] == "adaptive"


def test_selected_replay_score_is_at_least_the_incumbent():
    task = BranchWorld()
    tree = run_online(
        build_policy(task.initial_spec()),
        ScriptedAgent(task),
        task,
        max_parallelism=8,
    )
    incumbent = task.initial_spec()
    outcome = improve_policy(incumbent, [tree], revisions=8, max_parallelism=8)
    incumbent_score = mean_replay_score(
        incumbent,
        [tree],
        max_rounds=10_000,
        max_parallelism=8,
        lambda_cost=0.002,
        lambda_parallel=0.005,
    )
    assert outcome.score + 1e-12 >= incumbent_score
    assert outcome.score > incumbent_score


def test_run_directory_contains_every_tree(tmp_path: Path):
    task = BranchWorld()
    out = tmp_path / "run"
    result = run_discovery(
        task.name,
        ScriptedAgent(task),
        task,
        task.initial_spec(),
        mode="dream",
        rounds=2,
        max_parallelism=8,
        revisions=4,
        out_dir=str(out),
    )
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["agent_calls"] == result.agent_calls
    assert (out / "history" / "tree_001.json").is_file()
    assert (out / "history" / "tree_002.json").is_file()
    assert "api_key" not in (out / "summary.json").read_text(encoding="utf-8").lower()


def test_load_trusted_policy_file():
    path = Path(__file__).resolve().parents[1] / "examples" / "narrow_policy.py"
    policy = load_policy_file(path)
    task = BranchWorld()
    tree = run_online(policy, ScriptedAgent(task), task, max_parallelism=4)
    assert tree.attempt_count == 4
    assert policy.spec.branch_count == 2


def test_best_score_keeps_a_negative_objective():
    class NegativeTask:
        name = "negative"

        def scripted_propose(self, parent_artifact, *, branch_index, depth, trajectory):
            del parent_artifact, branch_index, trajectory
            return Proposal(artifact={"depth": depth}, proposal="negative")

        def evaluate(self, artifact):
            del artifact
            return EvalResult(score=-2.5)

    spec = PolicySpec(
        family="parallel_refine", branch_count=1, refine_depth=1, name="parallel_refine"
    )
    result = run_discovery(
        "negative",
        ScriptedAgent(NegativeTask()),
        NegativeTask(),
        spec,
        mode="fixed",
        rounds=1,
        max_parallelism=2,
    )
    assert result.best_score == -2.5
    assert result.rounds[0].best_score == -2.5


def test_parse_policy_json_from_a_chatty_model():
    payload = parse_json_object(
        'Sure.\n{"family": "adaptive", "branch_count": 4, "refine_depth": 4, '
        '"patience": 4, "min_gain": 0.02}\nDone.'
    )
    spec = PolicySpec.from_dict(payload)
    assert spec.family == "adaptive"
    assert spec.branch_count == 4
