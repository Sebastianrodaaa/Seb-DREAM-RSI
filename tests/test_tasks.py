"""Evaluators for the three domains in the paper, at a local scale."""

import pytest

from daydream.policy import PolicySpec
from daydream.tasks.autocorrelation import BINS, autoconvolution_peak
from daydream.tasks.circle_packing import packing_is_valid, packing_score
from daydream.tasks.kernels import LayerNormTask
from daydream.tasks.lasso import LassoPathTask
from daydream.tasks.sum_diff import sum_diff_gamma


def test_arithmetic_progression_has_gamma_one():
    assert sum_diff_gamma(list(range(8))) == pytest.approx(1.0)


def test_mstd_seed_is_a_real_finite_set():
    gamma = sum_diff_gamma([0, 2, 3, 4, 7, 11, 12, 14])
    assert gamma > 1.0


def test_touching_circles_score_their_radii_and_overlaps_score_zero():
    circles = [
        {"x": 0.25, "y": 0.25, "r": 0.25},
        {"x": 0.25, "y": 0.75, "r": 0.25},
        {"x": 0.75, "y": 0.25, "r": 0.25},
        {"x": 0.75, "y": 0.75, "r": 0.25},
    ]
    assert packing_is_valid(circles)
    assert packing_score(circles) == pytest.approx(1.0)
    overlap = [{"x": 0.5, "y": 0.5, "r": 0.2}, {"x": 0.5, "y": 0.5, "r": 0.2}]
    assert packing_score(overlap) == 0.0


def test_a_spike_has_a_higher_autoconvolution_peak_than_a_flat_density():
    flat = [1.0] * BINS
    spike = [0.0] * BINS
    spike[BINS // 2] = 1.0
    flat_peak = autoconvolution_peak(flat)
    spike_peak = autoconvolution_peak(spike)
    assert flat_peak is not None and spike_peak is not None
    assert spike_peak > flat_peak


def test_lasso_rejects_an_under_iterated_solver_and_accepts_a_converged_one():
    task = LassoPathTask()
    short = task.evaluate({"max_iter": 1, "use_active_set": False})
    long = task.evaluate({"max_iter": 200, "use_active_set": False})
    screened = task.evaluate({"max_iter": 200, "use_active_set": True})
    assert short.score == 0.0
    assert long.score > 0.0
    assert screened.score > 0.0
    assert screened.valid
    assert screened.diagnostics["ops"] < long.diagnostics["ops"]


def test_buggy_layernorm_scores_zero_and_blocked_beats_naive():
    task = LayerNormTask()
    buggy = task.evaluate({"variant": "buggy"})
    naive = task.evaluate({"variant": "naive"})
    blocked = task.evaluate({"variant": "blocked"})
    assert buggy.score == 0.0
    assert blocked.score > naive.score > 0.0


def test_policy_spec_rejects_an_unknown_family():
    with pytest.raises(ValueError):
        PolicySpec.from_dict({"family": "random-walk"})
