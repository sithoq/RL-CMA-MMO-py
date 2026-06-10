import numpy as np

from rlmmo.algorithms.online_individual_mpdqn_v2_de_cmaes import (
    _coverage_proxy_from_archive,
    _individual_reward_vectors_v2,
    _select_head_by_state,
    run_optimizer,
)
from rlmmo.core.archive import PeakArchive
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


def test_v2_runner_registered_and_smoke_f1_small_budget():
    assert "online_individual_mpdqn_v2_de_cmaes" in ALGORITHMS

    result = run_optimizer(func_num=1, seed=222, np_size=20, max_fes=1000, init_method="random")

    assert result["algorithm"] == "online_individual_mpdqn_v2_de_cmaes"
    assert result["FES"] <= 1000
    assert "coverage_proxy" in result
    assert "effective_archive_clusters" in result
    assert "mp_head_hist" in result
    assert "reward_vector_mean" in result
