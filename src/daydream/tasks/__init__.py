"""Task registry."""

from __future__ import annotations

from daydream.tasks.autocorrelation import AutocorrelationTask
from daydream.tasks.base import Task
from daydream.tasks.branch_world import BranchWorld
from daydream.tasks.circle_packing import CirclePackingTask
from daydream.tasks.kernels import LayerNormTask
from daydream.tasks.lasso import LassoPathTask
from daydream.tasks.sum_diff import SumDiffTask

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
