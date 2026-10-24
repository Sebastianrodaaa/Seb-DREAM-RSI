"""Command line for a local Dream-RSI run."""

from __future__ import annotations

import argparse
import json
import sys

from daydream.agents import ScriptedAgent
from daydream.loop import RunResult, run_discovery
from daydream.tasks import build_task, task_names


def _print_result(result: RunResult) -> None:
    print(f"task={result.task} mode={result.mode}")
    print(f"{'round':>5}  {'calls':>6}  {'cum':>6}  {'best':>8}  policy")
    for item in result.rounds:
        policy = item.policy.get("name") or item.policy.get("family")
        print(
            f"{item.round_index:5d}  {item.agent_calls:6d}  "
            f"{item.cumulative_calls:6d}  {item.best_score:8.4f}  {policy}"
        )
    final = result.final_policy.get("name") or result.final_policy.get("family")
    print(
        f"best={result.best_score:.4f}  agent_calls={result.agent_calls}  "
        f"final_policy={final}"
    )


def _run(args: argparse.Namespace) -> int:
    task = build_task(args.task)
    result = run_discovery(
        task.name,
        ScriptedAgent(task),
        task,
        task.initial_spec(),
        mode=args.mode,
        rounds=args.rounds,
        max_parallelism=args.parallelism,
        revisions=args.revisions,
        out_dir=args.out,
        dreamer=args.dreamer,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        _print_result(result)
    return 0


def _compare(args: argparse.Namespace) -> int:
    task = build_task(args.task)
    spec = task.initial_spec()
    fixed = run_discovery(
        task.name,
        ScriptedAgent(task),
        task,
        spec,
        mode="fixed",
        rounds=args.rounds,
        max_parallelism=args.parallelism,
        revisions=args.revisions,
        dreamer=args.dreamer,
    )
    dreamed = run_discovery(
        task.name,
        ScriptedAgent(task),
        task,
        spec,
        mode="dream",
        rounds=args.rounds,
        max_parallelism=args.parallelism,
        revisions=args.revisions,
        out_dir=args.out,
        dreamer=args.dreamer,
    )
    print("fixed exploration")
    _print_result(fixed)
    print()
    print("dream-rsi")
    _print_result(dreamed)
    if dreamed.agent_calls < fixed.agent_calls:
        ratio = fixed.agent_calls / max(1, dreamed.agent_calls)
        print(f"\ndream used {ratio:.2f}x fewer discovery-agent calls")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="daydream",
        description="Replay discovery trees and improve the exploration policy.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="run one discovery loop")
    run_parser.add_argument("--task", default="branch-world", choices=task_names())
    run_parser.add_argument("--mode", default="dream", choices=["dream", "fixed"])
    run_parser.add_argument("--rounds", type=int, default=2)
    run_parser.add_argument("--parallelism", type=int, default=8)
    run_parser.add_argument("--revisions", type=int, default=12)
    run_parser.add_argument(
        "--dreamer",
        default="auto",
        choices=["auto", "system1", "grid"],
        help="auto keeps a fast system-1 policy when it wins, and searches the grid otherwise",
    )
    run_parser.add_argument("--out", default=None, help="directory for trees and summary.json")
    run_parser.add_argument("--json", action="store_true")
    run_parser.set_defaults(func=_run)

    compare_parser = sub.add_parser(
        "compare",
        help="run fixed exploration and Dream-RSI on the same task",
    )
    compare_parser.add_argument("--task", default="branch-world", choices=task_names())
    compare_parser.add_argument("--rounds", type=int, default=2)
    compare_parser.add_argument("--parallelism", type=int, default=8)
    compare_parser.add_argument("--revisions", type=int, default=12)
    compare_parser.add_argument(
        "--dreamer",
        default="auto",
        choices=["auto", "system1", "grid"],
        help="auto keeps a fast system-1 policy when it wins, and searches the grid otherwise",
    )
    compare_parser.add_argument("--out", default=None, help="write the Dream-RSI run here")
    compare_parser.set_defaults(func=_compare)

    tasks_parser = sub.add_parser("tasks", help="list discovery tasks")
    tasks_parser.set_defaults(func=lambda _args: _print_tasks())

    return parser


def _print_tasks() -> int:
    for name in task_names():
        print(name)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
