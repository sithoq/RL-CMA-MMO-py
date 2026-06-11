import numpy as np

from rlmmo.algorithms.online_individual_mpdqn_v2_de_cmaes import (
    _apply_archive_reseed,
    _basin_edge_injection_points,
    _coverage_proxy_from_archive,
    _select_archive_reseed_entries,
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
        injection_min_progress=0.35,
        elite_frac=0.10,
    )

    coverage_only, coverage_only_reason = _should_inject_coverage(
        progress=0.50,
        success_hist=[0.50] * 10,
        archive_cluster_hist=list(range(20)),
        coverage_proxy=0.10,
        individual_stag=np.full(20, 5.0),
        np_size=20,
        expected_peaks=6,
        dim=3,
        cfg=cfg,
    )
    should_inject, reason = _should_inject_coverage(
        progress=0.50,
        success_hist=[0.0] * 10,
        archive_cluster_hist=[2] * 20,
        coverage_proxy=0.10,
        individual_stag=np.full(20, 50.0),
        np_size=20,
        expected_peaks=6,
        dim=3,
        cfg=cfg,
    )
    high_peak_inject, high_peak_reason = _should_inject_coverage(
        progress=0.80,
        success_hist=[0.0] * 10,
        archive_cluster_hist=[1] * 20,
        coverage_proxy=0.0,
        individual_stag=np.full(20, 100.0),
        np_size=20,
        expected_peaks=81,
        dim=3,
        cfg=cfg,
    )
    fitness = np.arange(20, dtype=float)
    stag = np.arange(20, dtype=float)
    replace_idx = _select_injection_replacement_indices(fitness, stag, injection_count=5, elite_frac=0.10)

    assert _target_coverage(0.10) == 0.20
    assert np.isclose(_target_coverage(0.90), 0.80)
    assert coverage_only is False
    assert coverage_only_reason == "coverage_only"
    assert should_inject is True
    assert reason in {"conservative_basin_recovery", "light_basin_recovery"}
    assert high_peak_inject is False
    assert high_peak_reason == "high_peak_disabled"
    assert len(replace_idx) == 5
    assert set(replace_idx).isdisjoint({18, 19})


def test_v2_injection_archive_gate_rejects_low_quality_points():
    cfg = run_optimizer.__globals__["OptimizerConfig"](injection_archive_quality_quantile=0.30)
    fitness = np.array([10.0, 20.0, 30.0, 40.0])

    low_ok = run_optimizer.__globals__["_injection_archive_allowed"](15.0, fitness, archive_gain=1.0, cfg=cfg)
    high_ok = run_optimizer.__globals__["_injection_archive_allowed"](35.0, fitness, archive_gain=1.0, cfg=cfg)
    replacement_ok = run_optimizer.__globals__["_injection_archive_allowed"](15.0, fitness, archive_gain=0.25, cfg=cfg)

    assert low_ok is False
    assert high_ok is True
    assert replacement_ok is True


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


def test_v2_seed_score_prefers_quality_for_non_high_peak_functions():
    cfg = run_optimizer.__globals__["OptimizerConfig"](
        multi_seed_cmaes=True,
        seed_cluster_size=3,
        seed_per_cluster_cap=2,
        seed_global_multiplier=1.0,
        seed_min_budget=100,
        seed_sparsity_weight=0.25,
    )
    pop = np.array([[0.0], [0.01], [0.5], [0.51], [1.0]], dtype=float)
    fitness = np.array([0.0, 100.0, 1.0, 2.0, 3.0], dtype=float)
    niches = NicheResult(
        members=[np.arange(5, dtype=int)],
        seeds=np.array([1], dtype=int),
        assigned=np.zeros(5, dtype=int),
        eps=0.5,
    )

    seed_idx, metadata = select_refinement_seeds(
        pop,
        fitness,
        niches,
        lb=np.array([0.0]),
        ub=np.array([1.0]),
        expected_peaks=2,
        remaining_fes=1000,
        population_size=5,
        config=cfg,
    )

    assert int(seed_idx[0]) == 1
    assert metadata[0]["seed_score"] > metadata[-1]["seed_score"]


def test_v2_refinement_seed_selection_prioritizes_archive_representatives():
    cfg = run_optimizer.__globals__["OptimizerConfig"](
        multi_seed_cmaes=True,
        seed_cluster_size=1,
        seed_per_cluster_cap=2,
        seed_global_multiplier=2.0,
        seed_min_budget=100,
        seed_sparsity_weight=0.25,
        seed_archive_bonus=0.20,
    )
    pop = np.array([[0.0], [0.02], [0.01], [0.03]], dtype=float)
    fitness = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
    niches = NicheResult(
        members=[np.arange(4, dtype=int)],
        seeds=np.array([0], dtype=int),
        assigned=np.zeros(4, dtype=int),
        eps=0.5,
    )

    seed_idx, metadata = select_refinement_seeds(
        pop,
        fitness,
        niches,
        lb=np.array([0.0]),
        ub=np.array([1.0]),
        expected_peaks=2,
        remaining_fes=1000,
        population_size=2,
        config=cfg,
    )

    assert int(seed_idx[0]) >= 2
    assert metadata[0]["seed_source"] == "archive"


def test_v2_basin_edge_injection_samples_near_archive_boundary():
    archive = PeakArchive(radius=0.05, max_size=10)
    archive.add_or_update(np.array([0.45, 0.45]), 10.0)
    archive.add_or_update(np.array([0.55, 0.55]), 9.0)
    pop = np.array([[0.45, 0.45], [0.55, 0.55], [0.50, 0.50]], dtype=float)

    points = _basin_edge_injection_points(
        pop=pop,
        archive=archive,
        lb=np.zeros(2),
        ub=np.ones(2),
        injection_count=8,
        candidate_multiplier=8,
        rng=np.random.default_rng(123),
    )
    archive_pop, _ = archive.as_arrays()
    nearest_archive = np.min(np.linalg.norm(points[:, None, :] - archive_pop[None, :, :], axis=2), axis=1)

    assert points.shape == (8, 2)
    assert np.all(points >= 0.0)
    assert np.all(points <= 1.0)
    assert float(np.mean(nearest_archive)) < 0.35
    assert float(np.mean(nearest_archive)) > 0.03


def test_v2_archive_reseed_selects_underrepresented_quality_entries():
    cfg = run_optimizer.__globals__["OptimizerConfig"](
        archive_reseed_quality_quantile=0.50,
        archive_reseed_min_distance_factor=1.0,
    )
    archive = PeakArchive(radius=0.05, max_size=10)
    archive.add_or_update(np.array([0.05, 0.05]), 10.0)
    archive.add_or_update(np.array([0.80, 0.80]), 50.0)
    archive.add_or_update(np.array([0.90, 0.90]), 20.0)
    pop = np.array([[0.04, 0.05], [0.05, 0.04], [0.10, 0.10]], dtype=float)

    selected = _select_archive_reseed_entries(
        archive=archive,
        pop=pop,
        lb=np.zeros(2),
        ub=np.ones(2),
        reseed_count=2,
        cfg=cfg,
    )

    archive_pop, _ = archive.as_arrays()
    selected_points = archive_pop[selected]
    assert selected.size >= 1
    assert np.any(np.all(np.isclose(selected_points, np.array([0.80, 0.80])), axis=1))
    assert not np.any(np.all(np.isclose(selected_points, np.array([0.05, 0.05])), axis=1))


def test_v2_archive_reseed_replaces_stagnated_non_elites_without_fes():
    cfg = run_optimizer.__globals__["OptimizerConfig"](
        archive_reseed_frac=0.20,
        archive_reseed_quality_quantile=0.0,
        archive_reseed_min_distance_factor=0.1,
        elite_frac=0.10,
    )
    archive = PeakArchive(radius=0.01, max_size=10)
    archive.add_or_update(np.array([0.80, 0.80]), 50.0)
    pop = np.zeros((10, 2), dtype=float)
    pop[9] = np.array([0.99, 0.99])
    fitness = np.arange(10, dtype=float)
    personal_best = fitness.copy()
    stag = np.arange(10, dtype=float)

    count, mean_fit, reason = _apply_archive_reseed(
        pop=pop,
        fitness=fitness,
        personal_best=personal_best,
        individual_stag=stag,
        archive=archive,
        lb=np.zeros(2),
        ub=np.ones(2),
        np_size=10,
        cfg=cfg,
    )

    assert count == 1
    assert mean_fit == 50.0
    assert reason == "underrepresented_archive"
    assert np.allclose(pop[9], np.array([0.99, 0.99]))
    assert np.any(np.all(np.isclose(pop[:9], np.array([0.80, 0.80])), axis=1))


def test_v2_runner_registered_and_smoke_f1_small_budget():
    assert "online_individual_mpdqn_v2_de_cmaes" in ALGORITHMS

    result = run_optimizer(func_num=1, seed=222, np_size=20, max_fes=1000, init_method="random")

    assert result["algorithm"] == "online_individual_mpdqn_v2_de_cmaes"
    assert result["FES"] <= 1000
    assert "coverage_proxy" in result
    assert "effective_archive_clusters" in result
    assert "mp_head_hist" in result
    assert "reward_vector_mean" in result
