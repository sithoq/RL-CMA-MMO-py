from __future__ import annotations

import numpy as np
import pandas as pd

from rlmmo.algorithms.online_individual_mpdqn_v2_de_cmaes import (
    OptimizerConfig,
    _should_switch_phase_v2,
    run_optimizer,
)
from rlmmo.optim.cmaes_refine import refine_with_cmaes
from rlmmo.runners.run_one_func import run_one_func


class QuadraticProblem:
    def evaluate(self, x):
        x = np.asarray(x, dtype=float)
        return -float(np.sum((x - 0.5) ** 2))


def test_phase_switch_returns_reason_for_diagnostics():
    cfg = OptimizerConfig(min_phase1=0.8, hard_phase2=0.9, coverage_threshold=0.7)

    should_switch, reason = _should_switch_phase_v2(
        progress=0.5,
        diversity_hist=[],
        success_hist=[],
        archive_hist=[],
        coverage_proxy=0.0,
        cfg=cfg,
    )

    assert should_switch is False
    assert reason == "not_allowed_before_min_phase1"


def test_cmaes_refine_returns_per_seed_diagnostics_when_enabled():
    problem = QuadraticProblem()
    seeds = np.array([[0.0], [1.0]], dtype=float)
    seed_fit = np.array([problem.evaluate(x) for x in seeds], dtype=float)

    refined_pop, refined_fit, used, log_rows = refine_with_cmaes(
        problem,
        seeds,
        seed_fit,
        remaining_fes=20,
        lb=np.array([0.0]),
        ub=np.array([1.0]),
        rng=np.random.default_rng(1),
        diagnostics=True,
        seed_metadata=[{"source_cluster_id": 3, "cluster_size": 4}, {"source_cluster_id": 8, "cluster_size": 2}],
    )

    assert refined_pop.shape == seeds.shape
    assert refined_fit.shape == seed_fit.shape
    assert used <= 20
    assert len(log_rows) == 2
    assert {"seed_id", "source_cluster_id", "cluster_size", "fitness_gain", "used_fes", "budget", "improved"} <= set(
        log_rows[0]
    )


def test_v2_diagnostics_result_contains_phase1_and_phase2_summary():
    result = run_optimizer(
        func_num=1,
        seed=12,
        np_size=20,
        max_fes=1000,
        init_method="random",
        config={"diagnostics": True, "diagnostic_interval": 1},
    )

    assert result["FES"] <= 1000
    assert "phase_switch_reason" in result
    assert "phase1_PR_pop_archive" in result
    assert "phase2_seed_count" in result
    assert "diagnostics_recorder" in result
    assert result["diagnostics_recorder"].generation_rows
    assert result["diagnostics_recorder"].checkpoint_rows
    assert {"action_entropy", "operator_hist", "selected_head_hist"} <= set(
        result["diagnostics_recorder"].generation_rows[0]
    )


def test_runner_writes_diagnostics_csv_files(tmp_path):
    csv_path = run_one_func(
        func_num=1,
        runs=1,
        seed_offset=50,
        init_method="random",
        out_dir=tmp_path,
        algorithm="online_individual_mpdqn_v2_de_cmaes",
        np_size=20,
        max_fes=1000,
        diagnostics=True,
        diagnostic_interval=1,
    )

    rows = pd.read_csv(csv_path)
    diagnostics_dir = tmp_path / "diagnostics"
    generation_logs = list(diagnostics_dir.glob("F01_run001_*_generation_log.csv"))
    checkpoint_logs = list(diagnostics_dir.glob("F01_run001_*_checkpoint_log.csv"))
    phase2_logs = list(diagnostics_dir.glob("F01_run001_*_phase2_cmaes_log.csv"))

    assert not rows.empty
    assert "phase1_PR_pop_archive" in rows.columns
    assert generation_logs
    assert checkpoint_logs
    assert phase2_logs
    assert {"FES", "action_entropy", "operator_hist", "selected_head_hist"} <= set(
        pd.read_csv(generation_logs[0]).columns
    )


def test_runner_accepts_algorithm_config_overrides(tmp_path):
    csv_path = run_one_func(
        func_num=1,
        runs=1,
        seed_offset=70,
        init_method="random",
        out_dir=tmp_path,
        algorithm="online_individual_mpdqn_v2_de_cmaes",
        np_size=20,
        max_fes=1000,
        diagnostics=True,
        diagnostic_interval=1,
        algorithm_config={"coverage_injection": False},
    )

    rows = pd.read_csv(csv_path)
    generation_log = rows.loc[0, "generation_log_path"]
    gen = pd.read_csv(generation_log)

    assert "generation_log_path" in rows.columns
    assert int(gen["injection_count"].sum()) == 0
