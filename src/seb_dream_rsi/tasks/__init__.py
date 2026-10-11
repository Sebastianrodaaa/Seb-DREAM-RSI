"""Task registry."""

from __future__ import annotations

from seb_dream_rsi.tasks.autocorrelation import AutocorrelationTask
from seb_dream_rsi.tasks.base import Task
from seb_dream_rsi.tasks.branch_world import BranchWorld
from seb_dream_rsi.tasks.circle_packing import CirclePackingTask
from seb_dream_rsi.tasks.kernels import LayerNormTask
from seb_dream_rsi.tasks.lasso import LassoPathTask
from seb_dream_rsi.tasks.sum_diff import SumDiffTask

__all__ = ["Task", "build_task", "task_names"]


def task_names() -> list[str]:
    return [
        "branch-world",
        "sum-diff",
        "circle-packing",
        "autocorrelation",
        "lasso",
        "kernel-layernorm",
    ]


def build_task(name: str) -> Task:
    if name == "branch-world":
        return BranchWorld()
    if name == "sum-diff":
        return SumDiffTask()
    if name == "circle-packing":
        return CirclePackingTask()
    if name == "autocorrelation":
        return AutocorrelationTask()
    if name == "lasso":
        return LassoPathTask()
    if name == "kernel-layernorm":
        return LayerNormTask()
    known = ", ".join(task_names())
    raise KeyError(f"unknown task {name!r}; expected one of: {known}")
