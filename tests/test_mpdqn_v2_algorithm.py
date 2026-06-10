import numpy as np

from rlmmo.algorithms.online_individual_mpdqn_v2_de_cmaes import (
    _coverage_proxy_from_archive,
    _select_injection_replacement_indices,
    _should_inject_coverage,
    _target_coverage,
    _individual_reward_vectors_v2,
    _select_head_by_state,
    select_refinement_seeds,
    run_optimizer,
)
from rlmmo.core.archive import PeakArchive
from rlmmo.core.niching import NicheResult
from rlmmo.runners.run_one_func import ALGORITHMS


def test_v2_head_selection_uses_search_state():
    rng = np.random.default_rng(4)
    state = np.zeros(16, dtype=float)

    far_head = _select_head_by_state(
        state=state,
        progress=0.50,
        local_archive_dist=0.90,
        local_success=0.0,
        archive_stall=True,
        rng=rng,
        exploration=0.0,
    )

    quality_state = state.copy()
    quality_state[14] = 0.95  # individual fitness norm
    quality_state[6] = 0.05  # neighborhood stagnation
    quality_head = _select_head_by_state(
        state=quality_state,
        progress=0.70,
        local_archive_dist=0.05,
        local_success=0.5,
        archive_stall=False,
        rng=rng,
        exploration=0.0,
    )

    assert far_head in {"coverage", "diversity"}
    assert quality_head == "quality"


def test_v2_reward_vector_prefers_novel_and_rank_improving_trials():
    parent_fit = np.array([1.0, 2.0, 3.0])
    trial_fit = np.array([2.5, 1.0, 4.0])
    owner = np.array([0, 1, 2])
    replacement_success = np.array([0.0, 0.0, 1.0])
    improvement = np.array([0.0, 0.0, 1.0])
    archive_gain = np.array([0.0, 0.0, 1.0])
    novelty_gain = np.array([0.9, 0.1, 0.2])

    rewards = _individual_reward_vectors_v2(
        replacement_success=replacement_success,
        improvement=improvement,
        archive_gain=archive_gain,
        novelty_gain=novelty_gain,
        trial_fit=trial_fit,
        owner=owner,
        parent_fit=parent_fit,
    )

    assert rewards.shape == (3, 5)
    assert np.all(np.isfinite(rewards))
    assert rewards[0, 2] > rewards[1, 2]
    assert rewards[0, 4] > rewards[1, 4]


def test_v2_coverage_proxy_counts_effective_archive_clusters():
    archive = PeakArchive(radius=0.001, max_size=10, trim_mode="sparse_quality")
    for x in [0.0, 0.01, 0.02, 1.0]:
        archive.add_or_update(np.array([x]), 1.0)

    proxy, clusters = _coverage_proxy_from_archive(archive, expected_peaks=4, lb=np.array([0.0]), ub=np.array([1.0]))

    assert clusters >= 2
    assert 0.0 < proxy <= 1.0


def test_v2_coverage_injection_trigger_and_elite_protection():
    cfg = run_optimizer.__globals__["OptimizerConfig"](
        coverage_injection=True,
        injection_min_progress=0.10,
        elite_frac=0.10,
    )

    should_inject, reason = _should_inject_coverage(
        progress=0.50,
        success_hist=[0.0] * 10,
        archive_cluster_hist=[2] * 20,
        coverage_proxy=0.10,
        individual_stag=np.full(20, 50.0),
        np_size=20,
        cfg=cfg,
    )
    fitness = np.arange(20, dtype=float)
    stag = np.arange(20, dtype=float)
    replace_idx = _select_injection_replacement_indices(fitness, stag, injection_count=5, elite_frac=0.10)

    assert _target_coverage(0.10) == 0.20
    assert np.isclose(_target_coverage(0.90), 0.80)
    assert should_inject is True
    assert reason in {"coverage_below_target", "success_archive_stalled", "individual_stagnated"}
    assert len(replace_idx) == 5
    assert set(replace_idx).isdisjoint({18, 19})


def test_v2_multi_seed_refinement_selects_multiple_sparse_seeds_from_one_cluster():
    cfg = run_optimizer.__globals__["OptimizerConfig"](
        multi_seed_cmaes=True,
        seed_cluster_size=4,
        seed_per_cluster_cap=6,
        seed_global_multiplier=2.0,
        seed_min_budget=100,
    )
    pop = np.column_stack([np.linspace(0.0, 1.0, 20), np.zeros(20)])
    fitness = np.linspace(0.0, 1.0, 20)
    niches = NicheResult(
        members=[np.arange(20, dtype=int)],
        seeds=np.array([19], dtype=int),
        assigned=np.zeros(20, dtype=int),
        eps=0.5,
    )

    seed_idx, metadata = select_refinement_seeds(
        pop,
        fitness,
        niches,
        lb=np.array([0.0, 0.0]),
        ub=np.array([1.0, 1.0]),
        expected_peaks=10,
        remaining_fes=2000,
        population_size=10,
        config=cfg,
    )

    assert 1 < len(seed_idx) <= 6
    assert len(metadata) == len(seed_idx)
    assert {"source_cluster_id", "cluster_size", "seed_rank_in_cluster", "seed_score", "seed_source"} <= set(
        metadata[0]
    )


def test_v2_runner_registered_and_smoke_f1_small_budget():
    assert "online_individual_mpdqn_v2_de_cmaes" in ALGORITHMS

    result = run_optimizer(func_num=1, seed=222, np_size=20, max_fes=1000, init_method="random")

    assert result["algorithm"] == "online_individual_mpdqn_v2_de_cmaes"
    assert result["FES"] <= 1000
    assert "coverage_proxy" in result
    assert "effective_archive_clusters" in result
    assert "mp_head_hist" in result
    assert "reward_vector_mean" in result
