"""Shared task types.

`scripted_propose` is the fixed discovery agent for offline runs. `evaluate`
is the fixed evaluator. Dream-RSI is allowed to change the exploration policy
only; these two stay put across rounds.
"""

from __future__ import annotations

from typing import Any, Protocol

from seb_dream_rsi.online import EvalResult, Proposal
from seb_dream_rsi.policy import PolicySpec


class Task(Protocol):
    name: str

    def initial_spec(self) -> PolicySpec: ...

    def scripted_propose(
        self,
        parent_artifact: dict[str, Any],
        *,
        branch_index: int,
        depth: int,
        trajectory: list[dict[str, Any]],
    ) -> Proposal: ...

    def evaluate(self, artifact: dict[str, Any]) -> EvalResult: ...
