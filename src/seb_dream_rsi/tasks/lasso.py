"""Small Lasso-path search.

The paper's task is to discover a fast, correct regularization-path solver.
Here the discovery agent proposes a solver configuration (iteration cap and
whether to use an active set). The evaluator runs that configuration on one
fixed synthetic design and scores it only when every lambda on the path
matches a reference objective within tolerance. Among correct configs, fewer
arithmetic operations score higher.

Wall-clock timing is intentionally not the search score: it is too noisy for
a regression test. Operation count is the deterministic stand-in.
"""

from __future__ import annotations

from typing import Any

from seb_dream_rsi.online import EvalResult, Proposal
from seb_dream_rsi.policy import PolicySpec

_TOL = 1e-4


def _lcg(seed: int, count: int) -> list[float]:
    state = seed % (2**31)
    values: list[float] = []
    for _ in range(count):
        state = (1103515245 * state + 12345) % (2**31)
        values.append((state / (2**31 - 1)) * 2.0 - 1.0)
    return values


def _design() -> tuple[list[list[float]], list[float], list[float]]:
    """Sparse, correlated design. One pass of coordinate descent is not enough."""

    rows, cols = 36, 20
    raw = _lcg(2026, rows * cols)
    matrix: list[list[float]] = []
    for row in range(rows):
        values = raw[row * cols : (row + 1) * cols]
        values[1] = 0.92 * values[0] + 0.08 * values[1]
        matrix.append(values)
    signal = [1.5, -1.1, 0.6] + [0.0] * (cols - 3)
    response = []
    noise = _lcg(7, rows)
    for row in range(rows):
        prediction = sum(matrix[row][col] * signal[col] for col in range(cols))
        response.append(prediction + 0.02 * noise[row])
    lambdas = [0.4, 0.15, 0.05]
    return matrix, response, lambdas


def _objective(
    matrix: list[list[float]],
    response: list[float],
    beta: list[float],
    lam: float,
) -> float:
    rows = len(response)
    residual = 0.0
    for row in range(rows):
        prediction = sum(matrix[row][col] * beta[col] for col in range(len(beta)))
        gap = response[row] - prediction
        residual += gap * gap
    penalty = sum(abs(value) for value in beta)
    return residual / (2.0 * rows) + lam * penalty


def lasso_path(
    matrix: list[list[float]],
    response: list[float],
    lambdas: list[float],
    *,
    max_iter: int,
    use_active_set: bool,
    tol: float = 1e-8,
) -> tuple[list[list[float]], int]:
    """Coordinate descent with an optional active set. Returns (path, op count)."""

    rows = len(response)
    cols = len(matrix[0])
    columns = [[matrix[row][col] for row in range(rows)] for col in range(cols)]
    norm2 = [sum(value * value for value in column) / rows for column in columns]
    beta = [0.0] * cols
    residual = list(response)
    ops = 0
    path: list[list[float]] = []

    def soft(value: float, threshold: float) -> float:
        if value > threshold:
            return value - threshold
        if value < -threshold:
            return value + threshold
        return 0.0

    def gradient(col: int) -> float:
        nonlocal ops
        dot = 0.0
        column = columns[col]
        for row in range(rows):
            dot += column[row] * residual[row]
        ops += rows
        return dot / rows

    def sweep(coordinates: list[int], lam: float) -> float:
        nonlocal ops
        max_delta = 0.0
        for col in coordinates:
            rho = gradient(col) + norm2[col] * beta[col]
            updated = 0.0 if norm2[col] <= 1e-18 else soft(rho, lam) / norm2[col]
            delta = updated - beta[col]
            if delta != 0.0:
                column = columns[col]
                for row in range(rows):
                    residual[row] -= column[row] * delta
                ops += rows
            beta[col] = updated
            max_delta = max(max_delta, abs(delta))
        return max_delta

    for lam in lambdas:
        if use_active_set:
            # Strong-rule screen: work on coefficients that are already nonzero
            # or whose correlation is not far below lambda. One KKT pass at the
            # end of the lambda puts back anything the screen dropped.
            active: set[int] = set()
            for col in range(cols):
                if abs(beta[col]) > 1e-10 or abs(gradient(col)) >= 0.5 * lam:
                    active.add(col)
            if not active:
                active.add(max(range(cols), key=lambda col: abs(gradient(col))))
        else:
            active = set(range(cols))

        for _ in range(max_iter):
            if sweep(list(active), lam) < tol:
                break
            if use_active_set:
                active = {col for col in active if abs(beta[col]) > 1e-12}

        if use_active_set:
            violated = [
                col
                for col in range(cols)
                if col not in active and abs(gradient(col)) > lam + 1e-8
            ]
            if violated:
                active.update(violated)
                for _ in range(max_iter):
                    if sweep(list(active), lam) < tol:
                        break
        path.append(list(beta))
    return path, ops


class LassoPathTask:
    name = "lasso"

    def __init__(self) -> None:
        self.matrix, self.response, self.lambdas = _design()
        self._reference: list[float] | None = None

    def initial_spec(self) -> PolicySpec:
        return PolicySpec(
            family="parallel_refine",
            branch_count=4,
            refine_depth=3,
            name="parallel_refine",
        )

    def reference_objectives(self) -> list[float]:
        if self._reference is None:
            path, _ops = lasso_path(
                self.matrix,
                self.response,
                self.lambdas,
                max_iter=400,
                use_active_set=False,
            )
            self._reference = [
                _objective(self.matrix, self.response, beta, lam)
                for beta, lam in zip(path, self.lambdas)
            ]
        return self._reference

    def scripted_propose(
        self,
        parent_artifact: dict[str, Any],
        *,
        branch_index: int,
        depth: int,
        trajectory: list[dict[str, Any]],
    ) -> Proposal:
        del parent_artifact, trajectory
        # Branch 0 stays under-iterated. Others buy enough iterations to match
        # the reference, and the later ones turn the active set on.
        if branch_index % 4 == 0:
            max_iter = 2
            use_active_set = False
        elif branch_index % 4 == 1:
            max_iter = 40 + 20 * depth
            use_active_set = False
        else:
            max_iter = 30 + 15 * depth
            use_active_set = True
        return Proposal(
            artifact={"max_iter": max_iter, "use_active_set": use_active_set},
            proposal=f"lasso config iters={max_iter} active_set={use_active_set}",
        )

    def evaluate(self, artifact: dict[str, Any]) -> EvalResult:
        max_iter = int(artifact.get("max_iter", 1))
        use_active_set = bool(artifact.get("use_active_set", False))
        if max_iter < 1 or max_iter > 5_000:
            return EvalResult(score=0.0, valid=False, error="max_iter out of range")
        path, ops = lasso_path(
            self.matrix,
            self.response,
            self.lambdas,
            max_iter=max_iter,
            use_active_set=use_active_set,
        )
        objectives = [
            _objective(self.matrix, self.response, beta, lam)
            for beta, lam in zip(path, self.lambdas)
        ]
        reference = self.reference_objectives()
        worst_gap = max(found - truth for found, truth in zip(objectives, reference))
        if worst_gap > _TOL:
            return EvalResult(
                score=0.0,
                valid=False,
                error="objective outside tolerance",
                diagnostics={"ops": ops, "gap": worst_gap},
            )
        score = 1.0 / (1.0 + ops / 10_000.0)
        return EvalResult(score=score, diagnostics={"ops": ops, "gap": worst_gap})
