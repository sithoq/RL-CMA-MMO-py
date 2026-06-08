"""V2 individual-level multi-preference Double DQN-DE + DBSCAN-CMA-ES.

This version keeps the existing KNN-DE and DBSCAN+CMA-ES pipeline, but adds
state-aware preference-head selection, novelty-aware rewards, sparse-quality
archive trimming, and a coverage-aware phase switch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np

from rlmmo.benchmarks.cec2013 import CEC2013Problem
from rlmmo.core.archive import PeakArchive
from rlmmo.core.initialization import init_population
from rlmmo.core.neighborhood import knn_indices
from rlmmo.core.niching import NicheResult, build_dbscan_niches, pairwise_distances
from rlmmo.optim.cmaes_refine import refine_with_cmaes
from rlmmo.optim.de import ACTION_DIM, decode_action
from rlmmo.optim.individual_de import generate_individual_offspring
from rlmmo.rl.multi_preference_dqn import MultiPreferenceDQNConfig, MultiPreferenceDoubleDQNAgent


STATE_DIM = 16


@dataclass
class OptimizerConfig:
    """个体级算法超参数。"""

    accuracy: float = 1e-4
    min_phase1: float = 0.80
    hard_phase2: float = 0.90
    archive_multiplier: int = 2
    coverage_threshold: float = 0.70
    archive_novelty_weight: float = 1.0
    high_peak_novelty_weight: float = 2.0
    head_exploration: float = 0.05
    state_dim: int = STATE_DIM
    dqn: MultiPreferenceDQNConfig = field(default_factory=MultiPreferenceDQNConfig)


def run_optimizer(
    func_num: int,
    seed: int,
    np_size: int | None = None,
    max_fes: int | None = None,
    init_method: str = "random",
    config: dict[str, Any] | OptimizerConfig | None = None,
) -> dict[str, Any]:
    """运行一次个体级在线 RL 多峰优化。"""

    cfg = _make_config(config)
    rng = np.random.default_rng(int(seed))
    problem = CEC2013Problem(func_num)
    info = problem.info()
    dim = problem.dim
    lb = problem.lb
    ub = problem.ub
    max_fes_total = int(max_fes or problem.max_fes)
    np_size = int(np_size or problem.recommended_np)
    diag = float(np.linalg.norm(ub - lb) + 1e-12)
    archive_radius = max(float(problem.rho), 1e-6 * diag)
    archive_max = max(problem.expected_peaks * cfg.archive_multiplier, np_size)

    start_time = perf_counter()
    pop = init_population(np_size, dim, lb, ub, init_method, rng)
    fitness = np.empty(np_size, dtype=float)
    fes = 0
    for i in range(np_size):
        if fes >= max_fes_total:
            fitness[i:] = -np.inf
            break
        fitness[i] = problem.evaluate(pop[i])
        fes += 1

    novelty_weight = cfg.high_peak_novelty_weight if problem.expected_peaks >= 50 else cfg.archive_novelty_weight
    archive = PeakArchive(
        radius=archive_radius,
        max_size=archive_max,
        trim_mode="sparse_quality",
        novelty_weight=novelty_weight,
    )
    for i in range(np_size):
        if np.isfinite(fitness[i]):
            archive.add_or_update(pop[i], fitness[i], source=0, fes=fes)

    agent = MultiPreferenceDoubleDQNAgent(cfg.dqn)
    personal_best = fitness.copy()
    individual_stag = np.zeros(np_size, dtype=float)
    global_best = float(np.max(fitness))
    global_stag = 0
    diversity_hist: list[float] = []
    success_hist: list[float] = []
    archive_hist: list[int] = []
    coverage_proxy = 0.0
    effective_archive_clusters = 0
    phase_switch_fes = max_fes_total

    # ========================= 阶段一：个体级 DQN-DE 找峰 =========================
    while fes < max_fes_total:
        progress = fes / max_fes_total
        coverage_proxy, effective_archive_clusters = _coverage_proxy_from_archive(
            archive, problem.expected_peaks, lb, ub
        )
        if _should_switch_phase_v2(progress, diversity_hist, success_hist, archive_hist, coverage_proxy, cfg):
            phase_switch_fes = fes
            break

        niches = build_dbscan_niches(pop, fitness, lb, ub)
        states = extract_individual_states(
            pop, fitness, niches, lb, ub, individual_stag, global_stag, progress
        )
        local_archive_dist = _archive_distances(pop, archive, diag)
        local_success = np.clip(1.0 - individual_stag / max(1, np_size), 0.0, 1.0)
        archive_stall = len(archive_hist) >= 10 and archive_hist[-1] <= archive_hist[-10]
        action_ids = []
        for i in range(np_size):
            head_name = _select_head_by_state(
                states[i],
                progress,
                local_archive_dist[i],
                local_success[i],
                archive_stall,
                rng,
                cfg.head_exploration,
            )
            action_ids.append(agent.select_action(states[i], progress, rng, head_name=head_name))
        actions = [decode_action(aid) for aid in action_ids]

        parent_pop = pop.copy()
        parent_fit = fitness.copy()
        old_niches = niches
        old_cluster_best = _cluster_best_values(parent_fit, old_niches)
        old_cluster_count = len(old_niches.members)

        offspring, owner = generate_individual_offspring(parent_pop, parent_fit, niches, actions, lb, ub, rng)
        novelty_by_trial = _archive_distances(offspring, archive, diag)
        offspring_fit = np.full(np_size, -np.inf, dtype=float)
        evaluated = np.zeros(np_size, dtype=bool)
        for i in range(np_size):
            if fes >= max_fes_total:
                break
            offspring_fit[i] = problem.evaluate(offspring[i])
            evaluated[i] = True
            fes += 1

        replacement_success = np.zeros(np_size, dtype=float)
        improvement = np.zeros(np_size, dtype=float)
        archive_gain = np.zeros(np_size, dtype=float)
        novelty_gain = np.zeros(np_size, dtype=float)
        replaced_parent = np.full(np_size, -1, dtype=int)

        # 竞争性替换仍使用最近父代，保持与多峰 niching 的空间局部性一致。
        cross_dists = pairwise_distances(offspring, parent_pop)
        for trial_idx in np.flatnonzero(evaluated):
            source_i = int(owner[trial_idx])
            novelty_gain[source_i] = max(novelty_gain[source_i], float(novelty_by_trial[trial_idx]))
            parent_idx = int(np.argmin(cross_dists[trial_idx]))
            replaced_parent[source_i] = parent_idx
            if offspring_fit[trial_idx] > fitness[parent_idx]:
                imp = float(offspring_fit[trial_idx] - fitness[parent_idx])
                pop[parent_idx] = offspring[trial_idx]
                fitness[parent_idx] = offspring_fit[trial_idx]
                replacement_success[source_i] = 1.0
                improvement[source_i] = max(0.0, imp)
                archive_gain[source_i] = archive.add_or_update(
                    offspring[trial_idx],
                    offspring_fit[trial_idx],
                    source=1,
                    niche_id=int(old_niches.assigned[source_i]),
                    action_id=action_ids[source_i],
                    fes=fes,
                )

        # 更新个体停滞：被替换且提高 personal best 的个体清零，否则累加。
        for i in range(np_size):
            parent_idx = replaced_parent[i]
            observed_fit = fitness[parent_idx] if parent_idx >= 0 else fitness[i]
            if replacement_success[i] > 0 and observed_fit > personal_best[i] + 1e-12:
                personal_best[i] = observed_fit
                individual_stag[i] = 0.0
            else:
                individual_stag[i] += 1.0

        # 每代保存当前 DBSCAN seed，给阶段二保留候选峰。
        new_niches = build_dbscan_niches(pop, fitness, lb, ub)
        for seed_idx in new_niches.seeds:
            archive.add_or_update(pop[int(seed_idx)], fitness[int(seed_idx)], source=0, fes=fes)

        new_cluster_best = _cluster_best_values(fitness, new_niches)
        cluster_quality_gain = _individual_cluster_quality_gain(old_cluster_best, new_cluster_best, old_niches, new_niches)
        cluster_count_gain = max(0.0, len(new_niches.members) - old_cluster_count) / max(1.0, old_cluster_count)
        reward_vectors = _individual_reward_vectors_v2(
            replacement_success,
            improvement,
            archive_gain,
            novelty_gain,
            offspring_fit,
            owner,
            parent_fit,
        )

        next_states = extract_individual_states(
            pop, fitness, new_niches, lb, ub, individual_stag, global_stag, fes / max_fes_total
        )
        for i in range(np_size):
            agent.remember(states[i], action_ids[i], reward_vectors[i], next_states[i])
        agent.train_step(rng)

        current_best = float(np.max(fitness))
        if current_best > global_best + 1e-12:
            global_best = current_best
            global_stag = 0
        else:
            global_stag += 1

        diversity_hist.append(_population_diversity(pop, lb, ub))
        success_hist.append(float(np.mean(replacement_success[np.flatnonzero(evaluated)])) if np.any(evaluated) else 0.0)
        archive_hist.append(len(archive))

    if phase_switch_fes == max_fes_total and fes < max_fes_total:
        phase_switch_fes = fes
    coverage_proxy, effective_archive_clusters = _coverage_proxy_from_archive(
        archive, problem.expected_peaks, lb, ub
    )

    # ========================= 阶段二：DBSCAN + 完整 CMA-ES 精搜 =========================
    archive_pop, archive_fit = archive.as_arrays()
    if archive_pop.size > 0:
        combo_pop = np.vstack([pop, archive_pop])
        combo_fit = np.concatenate([fitness, archive_fit])
    else:
        combo_pop = pop.copy()
        combo_fit = fitness.copy()

    if fes < max_fes_total and combo_pop.shape[0] > 0:
        combo_niches = build_dbscan_niches(combo_pop, combo_fit, lb, ub)
        seed_idx = combo_niches.seeds
        seeds = combo_pop[seed_idx]
        seed_fit = combo_fit[seed_idx]
        refined_pop, refined_fit, used = refine_with_cmaes(
            problem, seeds, seed_fit, max_fes_total - fes, lb, ub, rng
        )
        fes += used
        final_pop = np.vstack([refined_pop, combo_pop])
        final_fit = np.concatenate([refined_fit, combo_fit])
    else:
        final_pop = combo_pop
        final_fit = combo_fit

    found_peaks, _ = problem.count_goptima(final_pop, cfg.accuracy)
    runtime = perf_counter() - start_time
    pr = found_peaks / max(1, problem.expected_peaks)
    return {
        "algorithm": "online_individual_mpdqn_v2_de_cmaes",
        "func_num": problem.func_num,
        "func_name": info.func_name,
        "family": info.family,
        "dimension": dim,
        "bounds": f"[{float(np.min(lb))},{float(np.max(ub))}]",
        "expected_peaks": problem.expected_peaks,
        "NP": np_size,
        "maxFES": max_fes_total,
        "seed": int(seed),
        "init_method": init_method,
        "found_peaks": int(found_peaks),
        "PR": float(pr),
        "SR": float(pr >= 1.0),
        "FES": int(fes),
        "runtime": float(runtime),
        "archive_size": int(len(archive)),
        "phase_switch_fes": int(phase_switch_fes),
        "phase_switch_ratio": float(phase_switch_fes / max_fes_total),
        "coverage_proxy": float(coverage_proxy),
        "effective_archive_clusters": int(effective_archive_clusters),
        "dqn_action_hist": ";".join(map(str, agent.action_hist.tolist())),
        "mp_head_hist": ";".join(map(str, agent.head_hist.tolist())),
        "reward_vector_mean": ";".join(map(str, _format_reward_vector_mean(agent))),
        "final_pop": final_pop,
        "final_fitness": final_fit,
    }


def _make_config(config: dict[str, Any] | OptimizerConfig | None) -> OptimizerConfig:
    if config is None:
        cfg = OptimizerConfig()
    elif isinstance(config, OptimizerConfig):
        cfg = config
    else:
        cfg = OptimizerConfig()
        for key, value in config.items():
            if key == "dqn" and isinstance(value, dict):
                for dqn_key, dqn_value in value.items():
                    if hasattr(cfg.dqn, dqn_key):
                        setattr(cfg.dqn, dqn_key, dqn_value)
            elif hasattr(cfg, key):
                setattr(cfg, key, value)
    cfg.state_dim = STATE_DIM
    cfg.dqn.state_dim = STATE_DIM
    cfg.dqn.action_dim = ACTION_DIM
    cfg.dqn.buffer_capacity = max(int(cfg.dqn.buffer_capacity), 20000)
    cfg.dqn.batch_size = max(int(cfg.dqn.batch_size), 128)
    return cfg


def extract_individual_states(
    pop: np.ndarray,
    fitness: np.ndarray,
    niches: NicheResult,
    lb: np.ndarray,
    ub: np.ndarray,
    individual_stag: np.ndarray,
    global_stag: int,
    progress: float,
    knn_k: int = 4,
) -> np.ndarray:
    """提取 RLEMMO 风格的个体级状态。

    KNN 负责表示个体局部搜索邻域；DBSCAN 只保留 cluster rank / size 这类
    多样性上下文，不再作为 neighborhood 统计的主要来源。
    """

    pop = np.asarray(pop, dtype=float)
    fitness = np.asarray(fitness, dtype=float)
    n, dim = pop.shape
    diag = float(np.linalg.norm(ub - lb) + 1e-12)
    fit_min = float(np.min(fitness))
    fit_scale = float(np.ptp(fitness) + 1e-12)
    dmat = pairwise_distances(pop)
    center = np.mean(pop, axis=0)
    global_div = float(np.mean(np.linalg.norm(pop - center, axis=1)) / diag)
    global_fit_std = float(np.std(fitness) / fit_scale)
    best_global_idx = int(np.argmax(fitness))
    best_global_x = pop[best_global_idx]
    best_global_fit = float(fitness[best_global_idx])
    global_stag_norm = float(np.clip(global_stag / max(1, n), 0.0, 1.0))

    # DBSCAN cluster 只用于描述区域质量/规模，不作为 KNN neighborhood 的替代。
    niche_rank = np.zeros(len(niches.members), dtype=float)
    niche_best_fit = np.zeros(len(niches.members), dtype=float)
    cluster_scores = []
    for k, members in enumerate(niches.members):
        idx = np.asarray(members, dtype=int)
        local_fit = fitness[idx]
        niche_best_fit[k] = float(np.max(local_fit))
        cluster_scores.append(niche_best_fit[k])
    if cluster_scores:
        order = np.argsort(np.asarray(cluster_scores))[::-1]
        for rank_pos, k in enumerate(order):
            niche_rank[int(k)] = rank_pos / max(1, len(order) - 1)

    states = np.zeros((n, STATE_DIM), dtype=np.float32)
    for i in range(n):
        cluster_id = int(niches.assigned[i]) if niches.assigned.size else 0
        cluster_members = niches.members[cluster_id] if niches.members else np.arange(n, dtype=int)
        cluster_size_ratio = len(cluster_members) / max(1, n)

        # KNN neighborhood 统计：状态和普通 DE 变异使用同一类局部邻域。
        neighbor_idx = knn_indices(dmat, i, max(1, int(knn_k)), include_self=True)
        if neighbor_idx.size == 0:
            neighbor_idx = np.asarray([i], dtype=int)
        neighbor_pop = pop[neighbor_idx]
        neighbor_fit = fitness[neighbor_idx]
        neighbor_center = np.mean(neighbor_pop, axis=0)
        neigh_div = float(np.mean(np.linalg.norm(neighbor_pop - neighbor_center, axis=1)) / diag)
        neigh_fit_std = float(np.std(neighbor_fit) / fit_scale)
        neigh_stag = float(np.clip(np.mean(individual_stag[neighbor_idx]) / max(1, n), 0.0, 1.0))
        neigh_best_idx = int(neighbor_idx[int(np.argmax(neighbor_fit))])
        neigh_best_fit = float(fitness[neigh_best_idx])
        mean_dist_neighbors = float(np.mean(dmat[i, neighbor_idx]) / diag)

        state = np.array(
            [
                global_div,
                global_fit_std,
                progress,
                global_stag_norm,
                neigh_div,
                neigh_fit_std,
                neigh_stag,
                niche_rank[cluster_id] if niche_rank.size else 0.0,
                cluster_size_ratio,
                float(np.linalg.norm(pop[i] - best_global_x) / diag),
                float(np.linalg.norm(pop[i] - pop[neigh_best_idx]) / diag),
                float((best_global_fit - fitness[i]) / fit_scale),
                float((neigh_best_fit - fitness[i]) / fit_scale),
                float(np.clip(individual_stag[i] / max(1, n), 0.0, 1.0)),
                float((fitness[i] - fit_min) / fit_scale),
                mean_dist_neighbors,
            ],
            dtype=float,
        )
        states[i] = _safe_state(state)
    return states

def _safe_state(state: np.ndarray) -> np.ndarray:
    state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(state, 0.0, 1.0).astype(np.float32)


def _cluster_best_values(fitness: np.ndarray, niches: NicheResult) -> np.ndarray:
    if not niches.members:
        return np.empty((0,), dtype=float)
    return np.asarray([float(np.max(fitness[np.asarray(m, dtype=int)])) for m in niches.members], dtype=float)


def _individual_cluster_quality_gain(
    old_best: np.ndarray,
    new_best: np.ndarray,
    old_niches: NicheResult,
    new_niches: NicheResult,
) -> np.ndarray:
    gains = np.zeros(old_niches.assigned.shape[0], dtype=float)
    if old_best.size == 0 or new_best.size == 0:
        return gains
    scale = float(max(np.ptp(np.concatenate([old_best, new_best])), 1e-12))
    for i in range(gains.size):
        old_k = int(old_niches.assigned[i])
        new_k = int(new_niches.assigned[i]) if new_niches.assigned.size > i else old_k
        if old_k < old_best.size and new_k < new_best.size:
            gains[i] = max(0.0, float(new_best[new_k] - old_best[old_k])) / scale
    return np.tanh(gains)


def _select_head_by_state(
    state: np.ndarray,
    progress: float,
    local_archive_dist: float,
    local_success: float,
    archive_stall: bool,
    rng: np.random.Generator,
    exploration: float = 0.05,
) -> str:
    """Select a preference head from local search state.

    The rule is intentionally simple and deterministic under exploration=0
    so it can be tested and interpreted in mechanism analysis.
    """

    if rng.random() < float(np.clip(exploration, 0.0, 1.0)):
        return str(rng.choice(["coverage", "quality", "diversity", "balanced"]))

    state = np.asarray(state, dtype=float)
    local_fit = float(state[14]) if state.size > 14 else 0.0
    neigh_stag = float(state[6]) if state.size > 6 else 0.0
    local_archive_dist = float(np.clip(local_archive_dist, 0.0, 1.0))
    local_success = float(np.clip(local_success, 0.0, 1.0))

    if archive_stall and local_archive_dist > 0.55:
        return "coverage"
    if local_archive_dist > 0.75 or (local_success < 0.10 and progress < 0.85):
        return "diversity"
    if local_fit > 0.85 and neigh_stag < 0.20 and local_archive_dist < 0.25:
        return "quality"
    return "balanced"


def _archive_distances(points: np.ndarray, archive: PeakArchive, diag: float) -> np.ndarray:
    """Return normalized nearest distance from each point to the archive."""

    points = np.asarray(points, dtype=float)
    archive_pop, _ = archive.as_arrays()
    if archive_pop.size == 0:
        return np.ones(points.shape[0], dtype=float)
    dists = np.linalg.norm(points[:, None, :] - archive_pop[None, :, :], axis=2)
    nearest = np.min(dists, axis=1)
    return np.clip(nearest / float(diag + 1e-12), 0.0, 1.0)


def _coverage_proxy_from_archive(
    archive: PeakArchive,
    expected_peaks: int,
    lb: np.ndarray,
    ub: np.ndarray,
) -> tuple[float, int]:
    """Estimate peak coverage from the number of DBSCAN archive clusters."""

    archive_pop, archive_fit = archive.as_arrays()
    if archive_pop.size == 0:
        return 0.0, 0
    niches = build_dbscan_niches(archive_pop, archive_fit, lb, ub)
    clusters = int(len(niches.members))
    proxy = min(1.0, clusters / max(1, int(expected_peaks)))
    return float(proxy), clusters


def _individual_reward_vectors_v2(
    replacement_success: np.ndarray,
    improvement: np.ndarray,
    archive_gain: np.ndarray,
    novelty_gain: np.ndarray,
    trial_fit: np.ndarray,
    owner: np.ndarray,
    parent_fit: np.ndarray,
) -> np.ndarray:
    """Build the v2 reward vector shared by all preference heads.

    Fixed dimension order:
    [archive_gain, fitness_improvement, novelty_gain, replacement_success, local_rank_gain]
    """

    fit_scale = float(np.ptp(parent_fit) + 1e-12)
    archive_component = np.tanh(archive_gain)
    improvement_component = np.tanh(improvement / fit_scale)
    novelty_component = np.tanh(novelty_gain)
    success_component = np.asarray(replacement_success, dtype=float)

    owner = np.asarray(owner, dtype=int)
    trial_fit = np.asarray(trial_fit, dtype=float)
    local_rank_component = np.zeros_like(success_component, dtype=float)
    n = max(1, parent_fit.size)
    for trial_idx, source_i in enumerate(owner):
        if source_i < 0 or source_i >= local_rank_component.size or not np.isfinite(trial_fit[trial_idx]):
            continue
        old_better = np.sum(parent_fit > parent_fit[source_i])
        new_better = np.sum(parent_fit > trial_fit[trial_idx])
        local_rank_component[source_i] = max(
            local_rank_component[source_i],
            max(0.0, float(old_better - new_better) / n),
        )

    reward_vectors = np.column_stack(
        [
            archive_component,
            improvement_component,
            novelty_component,
            success_component,
            local_rank_component,
        ]
    )
    return np.clip(reward_vectors, -1.0, 1.0).astype(np.float32)


def _format_reward_vector_mean(agent: MultiPreferenceDoubleDQNAgent) -> list[str]:
    """Format the mean reward vector as CSV-friendly string values."""

    if agent.buffer.size <= 0:
        mean_reward = np.zeros(agent.config.reward_dim, dtype=float)
    else:
        mean_reward = np.mean(agent.buffer.reward_vectors[: agent.buffer.size], axis=0)
    return [f"{float(x):.6g}" for x in mean_reward]

def _population_diversity(pop: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> float:
    center = np.mean(pop, axis=0)
    diag = float(np.linalg.norm(ub - lb) + 1e-12)
    return float(np.mean(np.linalg.norm(pop - center, axis=1)) / diag)


def _should_switch_phase_v2(
    progress: float,
    diversity_hist: list[float],
    success_hist: list[float],
    archive_hist: list[int],
    coverage_proxy: float,
    cfg: OptimizerConfig,
) -> bool:
    if progress >= cfg.hard_phase2:
        return True
    if progress < cfg.min_phase1:
        return False
    if coverage_proxy < cfg.coverage_threshold:
        return False
    if len(diversity_hist) < 12 or len(success_hist) < 12 or len(archive_hist) < 12:
        return False
    div_recent = float(np.mean(diversity_hist[-5:]))
    div_prev = float(np.mean(diversity_hist[-10:-5]))
    div_stalling = abs(div_recent - div_prev) < 1e-4
    rate_low = float(np.mean(success_hist[-5:])) < 0.02
    archive_stalling = archive_hist[-1] <= archive_hist[-10]
    return bool(div_stalling and rate_low and archive_stalling)


