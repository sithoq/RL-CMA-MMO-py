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
from rlmmo.core.diagnostics import DiagnosticsRecorder, action_entropy, hist_string
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
    coverage_head_min_prob: float = 0.25
    coverage_injection: bool = True
    injection_mode: str = "conservative"
    injection_min_progress: float = 0.35
    injection_interval: int = 30
    injection_frac: float = 0.03
    high_peak_injection_frac: float = 0.0
    injection_candidate_multiplier: int = 8
    injection_archive_quality_quantile: float = 0.30
    basin_edge_step_min: float = 0.04
    basin_edge_step_max: float = 0.18
    archive_reseed_enabled: bool = True
    archive_reseed_min_progress: float = 0.35
    archive_reseed_interval: int = 20
    archive_reseed_frac: float = 0.03
    archive_reseed_quality_quantile: float = 0.50
    archive_reseed_min_distance_factor: float = 1.5
    elite_frac: float = 0.05
    multi_seed_cmaes: bool = True
    seed_cluster_size: int = 12
    seed_per_cluster_cap: int = 6
    seed_global_multiplier: float = 1.25
    seed_min_budget: int = 300
    seed_sparsity_weight: float = 0.25
    seed_archive_bonus: float = 0.15
    high_peak_phase2_threshold: float = 0.60
    high_peak_hard_phase2: float = 0.95
    phase2_min_budget_ratio: float = 0.03
    diagnostics: bool = False
    diagnostic_interval: int = 1
    checkpoint_ratios: list[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    save_pop_snapshots: bool = False
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
    archive_cluster_hist: list[int] = []
    coverage_proxy = 0.0
    effective_archive_clusters = 0
    phase_switch_fes = max_fes_total
    phase_switch_reason = "max_fes_exhausted"
    generation_index = 0
    diagnostic_interval = max(1, int(cfg.diagnostic_interval))
    checkpoint_ratios = sorted(float(np.clip(x, 0.0, 1.0)) for x in cfg.checkpoint_ratios)
    next_checkpoint = 0
    diagnostics = DiagnosticsRecorder(
        enabled=bool(cfg.diagnostics),
        save_pop_snapshots=bool(cfg.save_pop_snapshots),
    )

    # ========================= 阶段一：个体级 DQN-DE 找峰 =========================
    while fes < max_fes_total:
        progress = fes / max_fes_total
        coverage_proxy, effective_archive_clusters = _coverage_proxy_from_archive(
            archive, problem.expected_peaks, lb, ub
        )
        should_switch, switch_reason = _should_switch_phase_v2(
            progress,
            diversity_hist,
            success_hist,
            archive_hist,
            coverage_proxy,
            cfg,
            expected_peaks=problem.expected_peaks,
            effective_archive_clusters=effective_archive_clusters,
            fes=fes,
            max_fes_total=max_fes_total,
        )
        if should_switch:
            phase_switch_fes = fes
            phase_switch_reason = switch_reason
            _log_checkpoint(
                diagnostics,
                "phase_switch",
                problem,
                pop,
                archive,
                fitness,
                lb,
                ub,
                fes,
                max_fes_total,
                cfg.accuracy,
                coverage_proxy,
            )
            break

        niches = build_dbscan_niches(pop, fitness, lb, ub)
        states = extract_individual_states(
            pop, fitness, niches, lb, ub, individual_stag, global_stag, progress
        )
        local_archive_dist = _archive_distances(pop, archive, diag)
        local_success = np.clip(1.0 - individual_stag / max(1, np_size), 0.0, 1.0)
        archive_stall = len(archive_hist) >= 10 and archive_hist[-1] <= archive_hist[-10]
        action_ids = []
        selected_heads = []
        for i in range(np_size):
            head_name = _select_head_by_state(
                states[i],
                progress,
                local_archive_dist[i],
                local_success[i],
                archive_stall,
                rng,
                cfg.head_exploration,
                coverage_proxy=coverage_proxy,
                coverage_target=_target_coverage(progress),
                coverage_head_min_prob=cfg.coverage_head_min_prob,
            )
            selected_heads.append(head_name)
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
            agent.update_action_credit(selected_heads[i], action_ids[i], reward_vectors[i])
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
        generation_index += 1

        coverage_proxy, effective_archive_clusters = _coverage_proxy_from_archive(
            archive, problem.expected_peaks, lb, ub
        )
        archive_cluster_hist.append(effective_archive_clusters)

        injection_count = 0
        injection_archive_gain = 0.0
        injection_to_archive_count = 0
        injection_replaced_mean_fitness = 0.0
        injection_candidate_mean_fitness = 0.0
        injection_best_fitness = 0.0
        injection_reason = "none"
        archive_reseed_count = 0
        archive_reseed_mean_fitness = 0.0
        archive_reseed_reason = "none"
        if generation_index % max(1, int(cfg.injection_interval)) == 0 and fes < max_fes_total:
            should_inject, injection_reason = _should_inject_coverage(
                progress=fes / max_fes_total,
                success_hist=success_hist,
                archive_cluster_hist=archive_cluster_hist,
                coverage_proxy=coverage_proxy,
                individual_stag=individual_stag,
                np_size=np_size,
                expected_peaks=problem.expected_peaks,
                dim=dim,
                cfg=cfg,
            )
            if should_inject:
                (
                    injection_count,
                    injection_archive_gain,
                    injection_to_archive_count,
                    injection_replaced_mean_fitness,
                    injection_candidate_mean_fitness,
                    injection_best_fitness,
                    fes,
                ) = _apply_coverage_injection(
                    problem=problem,
                    pop=pop,
                    fitness=fitness,
                    personal_best=personal_best,
                    individual_stag=individual_stag,
                    archive=archive,
                    lb=lb,
                    ub=ub,
                    rng=rng,
                    max_fes_total=max_fes_total,
                    fes=fes,
                    np_size=np_size,
                    cfg=cfg,
                )
                coverage_proxy, effective_archive_clusters = _coverage_proxy_from_archive(
                    archive, problem.expected_peaks, lb, ub
                )
                archive_cluster_hist[-1] = effective_archive_clusters
                new_niches = build_dbscan_niches(pop, fitness, lb, ub)
                diversity_hist[-1] = _population_diversity(pop, lb, ub)
        elif cfg.coverage_injection:
            injection_reason = "interval_skip"
        if generation_index % max(1, int(cfg.archive_reseed_interval)) == 0:
            should_reseed, archive_reseed_reason = _should_archive_reseed(
                progress=fes / max_fes_total,
                archive=archive,
                pop=pop,
                lb=lb,
                ub=ub,
                individual_stag=individual_stag,
                np_size=np_size,
                expected_peaks=problem.expected_peaks,
                cfg=cfg,
            )
            if should_reseed:
                (
                    archive_reseed_count,
                    archive_reseed_mean_fitness,
                    archive_reseed_reason,
                ) = _apply_archive_reseed(
                    pop=pop,
                    fitness=fitness,
                    personal_best=personal_best,
                    individual_stag=individual_stag,
                    archive=archive,
                    lb=lb,
                    ub=ub,
                    np_size=np_size,
                    cfg=cfg,
                )
                if archive_reseed_count > 0:
                    new_niches = build_dbscan_niches(pop, fitness, lb, ub)
                    diversity_hist[-1] = _population_diversity(pop, lb, ub)
        elif cfg.archive_reseed_enabled:
            archive_reseed_reason = "interval_skip"
        if injection_count > 0 or archive_reseed_count > 0:
            injected_best = float(np.max(fitness))
            if injected_best > global_best + 1e-12:
                global_best = injected_best
                global_stag = 0

        _, candidate_reason = _should_switch_phase_v2(
            fes / max_fes_total,
            diversity_hist,
            success_hist,
            archive_hist,
            coverage_proxy,
            cfg,
            expected_peaks=problem.expected_peaks,
            effective_archive_clusters=effective_archive_clusters,
            fes=fes,
            max_fes_total=max_fes_total,
        )
        if cfg.diagnostics and generation_index % diagnostic_interval == 0:
            diagnostics.log_generation(
                _generation_diagnostics_row(
                    fes=fes,
                    max_fes_total=max_fes_total,
                    pop=pop,
                    fitness=fitness,
                    lb=lb,
                    ub=ub,
                    success_rate=success_hist[-1],
                    archive_size=len(archive),
                    coverage_proxy=coverage_proxy,
                    effective_archive_clusters=effective_archive_clusters,
                    niches=new_niches,
                    individual_stag=individual_stag,
                    action_ids=action_ids,
                    actions=actions,
                    selected_heads=selected_heads,
                    head_names=agent.head_names,
                    reward_vectors=reward_vectors,
                    head_action_credit_top=agent.head_action_credit_top(),
                    injection_count=injection_count,
                    injection_archive_gain=injection_archive_gain,
                    injection_to_archive_count=injection_to_archive_count,
                    injection_replaced_mean_fitness=injection_replaced_mean_fitness,
                    injection_candidate_mean_fitness=injection_candidate_mean_fitness,
                    injection_best_fitness=injection_best_fitness,
                    injection_reason=injection_reason,
                    archive_reseed_count=archive_reseed_count,
                    archive_reseed_mean_fitness=archive_reseed_mean_fitness,
                    archive_reseed_reason=archive_reseed_reason,
                    phase_switch_candidate_reason=candidate_reason,
                )
            )
        while next_checkpoint < len(checkpoint_ratios) and fes / max_fes_total >= checkpoint_ratios[next_checkpoint]:
            ratio = checkpoint_ratios[next_checkpoint]
            _log_checkpoint(
                diagnostics,
                f"{int(round(ratio * 100)):03d}pct",
                problem,
                pop,
                archive,
                fitness,
                lb,
                ub,
                fes,
                max_fes_total,
                cfg.accuracy,
                coverage_proxy,
            )
            next_checkpoint += 1

    if phase_switch_fes == max_fes_total and fes < max_fes_total:
        phase_switch_fes = fes
        phase_switch_reason = "max_fes_exhausted"
    coverage_proxy, effective_archive_clusters = _coverage_proxy_from_archive(
        archive, problem.expected_peaks, lb, ub
    )
    phase1_checkpoint = _checkpoint_metrics(
        problem, pop, archive, fitness, lb, ub, fes, max_fes_total, cfg.accuracy, coverage_proxy
    )
    _log_checkpoint(
        diagnostics,
        "phase1_end",
        problem,
        pop,
        archive,
        fitness,
        lb,
        ub,
        fes,
        max_fes_total,
        cfg.accuracy,
        coverage_proxy,
    )

    # ========================= 阶段二：DBSCAN + 完整 CMA-ES 精搜 =========================
    archive_pop, archive_fit = archive.as_arrays()
    if archive_pop.size > 0:
        combo_pop = np.vstack([pop, archive_pop])
        combo_fit = np.concatenate([fitness, archive_fit])
    else:
        combo_pop = pop.copy()
        combo_fit = fitness.copy()

    phase2_log: list[dict[str, Any]] = []
    phase2_seed_count = 0
    phase2_cmaes_used_fes = 0
    refined_pop = np.empty((0, dim), dtype=float)

    if fes < max_fes_total and combo_pop.shape[0] > 0:
        combo_niches = build_dbscan_niches(combo_pop, combo_fit, lb, ub)
        seed_idx, seed_metadata = select_refinement_seeds(
            combo_pop=combo_pop,
            combo_fit=combo_fit,
            combo_niches=combo_niches,
            lb=lb,
            ub=ub,
            expected_peaks=problem.expected_peaks,
            remaining_fes=max_fes_total - fes,
            population_size=np_size,
            config=cfg,
        )
        seeds = combo_pop[seed_idx]
        seed_fit = combo_fit[seed_idx]
        phase2_seed_count = int(seeds.shape[0])
        if cfg.diagnostics:
            refined_pop, refined_fit, used, phase2_log = refine_with_cmaes(
                problem,
                seeds,
                seed_fit,
                max_fes_total - fes,
                lb,
                ub,
                rng,
                diagnostics=True,
                seed_metadata=seed_metadata,
            )
        else:
            refined_pop, refined_fit, used = refine_with_cmaes(
                problem, seeds, seed_fit, max_fes_total - fes, lb, ub, rng
            )
        phase2_cmaes_used_fes = int(used)
        fes += used
        final_pop = np.vstack([refined_pop, combo_pop])
        final_fit = np.concatenate([refined_fit, combo_fit])
    else:
        final_pop = combo_pop
        final_fit = combo_fit

    found_peaks, _ = problem.count_goptima(final_pop, cfg.accuracy)
    phase2_pr_gain = float(found_peaks / max(1, problem.expected_peaks) - phase1_checkpoint["PR_pop_archive"])
    _mark_phase2_peak_contributions(problem, phase2_log, refined_pop, cfg.accuracy)
    diagnostics.extend_phase2(phase2_log)
    _log_checkpoint(
        diagnostics,
        "phase2_end",
        problem,
        final_pop,
        archive,
        final_fit,
        lb,
        ub,
        fes,
        max_fes_total,
        cfg.accuracy,
        coverage_proxy,
    )
    while next_checkpoint < len(checkpoint_ratios):
        ratio = checkpoint_ratios[next_checkpoint]
        _log_checkpoint(
            diagnostics,
            f"{int(round(ratio * 100)):03d}pct",
            problem,
            final_pop,
            archive,
            final_fit,
            lb,
            ub,
            fes,
            max_fes_total,
            cfg.accuracy,
            coverage_proxy,
        )
        next_checkpoint += 1
    runtime = perf_counter() - start_time
    pr = found_peaks / max(1, problem.expected_peaks)
    result = {
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
        "phase_switch_reason": phase_switch_reason,
        "phase1_PR_pop": float(phase1_checkpoint["PR_pop"]),
        "phase1_PR_archive": float(phase1_checkpoint["PR_archive"]),
        "phase1_PR_pop_archive": float(phase1_checkpoint["PR_pop_archive"]),
        "phase1_archive_size": int(phase1_checkpoint["archive_size"]),
        "phase1_cluster_count": int(phase1_checkpoint["cluster_count"]),
        "phase2_seed_count": int(phase2_seed_count),
        "phase2_cmaes_used_fes": int(phase2_cmaes_used_fes),
        "phase2_improved_seed_count": int(sum(1 for row in phase2_log if bool(row.get("improved", False)))),
        "phase2_mean_fitness_gain": float(np.mean([row["fitness_gain"] for row in phase2_log]))
        if phase2_log
        else 0.0,
        "phase2_PR_gain": float(phase2_pr_gain),
        "coverage_proxy": float(coverage_proxy),
        "effective_archive_clusters": int(effective_archive_clusters),
        "dqn_action_hist": ";".join(map(str, agent.action_hist.tolist())),
        "mp_head_hist": ";".join(map(str, agent.head_hist.tolist())),
        "reward_vector_mean": ";".join(map(str, _format_reward_vector_mean(agent))),
        "final_pop": final_pop,
        "final_fitness": final_fit,
    }
    if cfg.diagnostics:
        result["diagnostics_recorder"] = diagnostics
    return result


def _generation_diagnostics_row(
    fes: int,
    max_fes_total: int,
    pop: np.ndarray,
    fitness: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    success_rate: float,
    archive_size: int,
    coverage_proxy: float,
    effective_archive_clusters: int,
    niches: NicheResult,
    individual_stag: np.ndarray,
    action_ids: list[int],
    actions: list[Any],
    selected_heads: list[str],
    head_names: list[str],
    reward_vectors: np.ndarray,
    head_action_credit_top: str,
    injection_count: int,
    injection_archive_gain: float,
    injection_to_archive_count: int,
    injection_replaced_mean_fitness: float,
    injection_candidate_mean_fitness: float,
    injection_best_fitness: float,
    injection_reason: str,
    archive_reseed_count: int,
    archive_reseed_mean_fitness: float,
    archive_reseed_reason: str,
    phase_switch_candidate_reason: str,
) -> dict[str, Any]:
    """生成一代诊断日志行。

    中文说明：这里只做统计，不读写算法状态，避免诊断开关影响优化轨迹。
    """

    head_to_id = {name: i for i, name in enumerate(head_names)}
    head_ids = [head_to_id.get(name, 0) for name in selected_heads]
    f_ids = [int(action.action_id) // 9 for action in actions]
    cr_ids = [(int(action.action_id) // 3) % 3 for action in actions]
    op_ids = [int(action.operator) for action in actions]
    cluster_sizes = [len(members) for members in niches.members]
    reward_mean = np.mean(reward_vectors, axis=0) if reward_vectors.size else np.zeros(5, dtype=float)
    return {
        "FES": int(fes),
        "progress": float(fes / max(1, max_fes_total)),
        "phase": "phase1",
        "best_fitness": float(np.max(fitness)),
        "mean_fitness": float(np.mean(fitness)),
        "std_fitness": float(np.std(fitness)),
        "diversity": float(_population_diversity(pop, lb, ub)),
        "success_rate": float(success_rate),
        "archive_size": int(archive_size),
        "coverage_proxy": float(coverage_proxy),
        "effective_archive_clusters": int(effective_archive_clusters),
        "dbscan_cluster_count": int(len(niches.members)),
        "mean_cluster_size": float(np.mean(cluster_sizes)) if cluster_sizes else 0.0,
        "individual_stag_mean": float(np.mean(individual_stag)) if individual_stag.size else 0.0,
        "individual_stag_max": float(np.max(individual_stag)) if individual_stag.size else 0.0,
        "action_entropy": float(action_entropy(action_ids, ACTION_DIM)),
        "selected_head_hist": hist_string(head_ids, len(head_names)),
        "action_hist": hist_string(action_ids, ACTION_DIM),
        "F_hist": hist_string(f_ids, 3),
        "CR_hist": hist_string(cr_ids, 3),
        "operator_hist": hist_string(op_ids, 3),
        "reward_vector_mean": ";".join(f"{float(x):.6g}" for x in reward_mean),
        "head_action_credit_top": head_action_credit_top,
        "injection_count": int(injection_count),
        "injection_archive_gain": float(injection_archive_gain),
        "injection_to_archive_count": int(injection_to_archive_count),
        "injection_replaced_mean_fitness": float(injection_replaced_mean_fitness),
        "injection_candidate_mean_fitness": float(injection_candidate_mean_fitness),
        "injection_best_fitness": float(injection_best_fitness),
        "injection_reason": injection_reason,
        "archive_reseed_count": int(archive_reseed_count),
        "archive_reseed_mean_fitness": float(archive_reseed_mean_fitness),
        "archive_reseed_reason": archive_reseed_reason,
        "phase_switch_candidate_reason": phase_switch_candidate_reason,
    }


def _checkpoint_metrics(
    problem: CEC2013Problem,
    pop: np.ndarray,
    archive: PeakArchive,
    fitness: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    fes: int,
    max_fes_total: int,
    accuracy: float,
    coverage_proxy: float,
) -> dict[str, Any]:
    archive_pop, archive_fit = archive.as_arrays()
    found_pop, _ = problem.count_goptima(pop, accuracy) if pop.size else (0, np.empty((0, 0)))
    if archive_pop.size:
        found_archive, _ = problem.count_goptima(archive_pop, accuracy)
        combo_pop = np.vstack([pop, archive_pop])
        combo_fit = np.concatenate([fitness, archive_fit])
    else:
        found_archive = 0
        combo_pop = pop
        combo_fit = fitness
    found_combo, _ = problem.count_goptima(combo_pop, accuracy) if combo_pop.size else (0, np.empty((0, 0)))
    cluster_count = 0
    if combo_pop.size:
        cluster_count = len(build_dbscan_niches(combo_pop, combo_fit, lb, ub).members)
    expected = max(1, int(problem.expected_peaks))
    return {
        "FES": int(fes),
        "progress": float(fes / max(1, max_fes_total)),
        "found_peaks_pop": int(found_pop),
        "PR_pop": float(found_pop / expected),
        "found_peaks_archive": int(found_archive),
        "PR_archive": float(found_archive / expected),
        "found_peaks_pop_archive": int(found_combo),
        "PR_pop_archive": float(found_combo / expected),
        "archive_size": int(len(archive)),
        "cluster_count": int(cluster_count),
        "coverage_proxy": float(coverage_proxy),
    }


def _log_checkpoint(
    recorder: DiagnosticsRecorder,
    label: str,
    problem: CEC2013Problem,
    pop: np.ndarray,
    archive: PeakArchive,
    fitness: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    fes: int,
    max_fes_total: int,
    accuracy: float,
    coverage_proxy: float,
) -> dict[str, Any]:
    if not recorder.enabled:
        return {}
    row = _checkpoint_metrics(problem, pop, archive, fitness, lb, ub, fes, max_fes_total, accuracy, coverage_proxy)
    row["label"] = label
    archive_pop, _ = archive.as_arrays()
    recorder.log_checkpoint(row, pop=pop, archive_pop=archive_pop)
    return row


def _seed_metadata(niches: NicheResult, seed_idx: np.ndarray, population_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in np.asarray(seed_idx, dtype=int):
        cluster_id = int(niches.assigned[idx]) if niches.assigned.size > idx else -1
        cluster_size = len(niches.members[cluster_id]) if 0 <= cluster_id < len(niches.members) else 1
        rows.append(
            {
                "source_cluster_id": cluster_id,
                "cluster_size": int(cluster_size),
                "seed_source": "archive" if int(idx) >= int(population_size) else "population",
            }
        )
    return rows


def _mark_phase2_peak_contributions(
    problem: CEC2013Problem,
    rows: list[dict[str, Any]],
    refined_pop: np.ndarray,
    accuracy: float,
) -> None:
    if not rows or refined_pop.size == 0:
        return
    for i, row in enumerate(rows):
        if i >= refined_pop.shape[0]:
            break
        found, _ = problem.count_goptima(refined_pop[i : i + 1], accuracy)
        row["final_found_peak_contribution"] = int(found > 0)


def _target_coverage(progress: float) -> float:
    """Linear coverage target used by injection and head scheduling."""

    progress = float(np.clip(progress, 0.0, 1.0))
    if progress <= 0.10:
        return 0.20
    if progress >= 0.90:
        return 0.80
    return float(0.20 + (progress - 0.10) * (0.60 / 0.80))


def _should_inject_coverage(
    progress: float,
    success_hist: list[float],
    archive_cluster_hist: list[int],
    coverage_proxy: float,
    individual_stag: np.ndarray,
    np_size: int,
    expected_peaks: int,
    dim: int,
    cfg: OptimizerConfig,
) -> tuple[bool, str]:
    """Decide whether phase-one forced exploration should run."""

    if not cfg.coverage_injection or progress < cfg.injection_min_progress:
        return False, "not_allowed"

    # 中文说明：F8/F9 这类极高峰数函数的 DBSCAN coverage proxy 明显失真，
    # 默认不触发 injection，避免远距离低质量点污染 archive。
    if int(expected_peaks) >= 50 or float(cfg.high_peak_injection_frac) <= 0.0 and int(expected_peaks) >= 50:
        return False, "high_peak_disabled"

    coverage_low = coverage_proxy < _target_coverage(progress)
    if not coverage_low:
        return False, "coverage_ready"

    recent_success = float(np.mean(success_hist[-10:])) if len(success_hist) >= 10 else 1.0
    archive_stalled = False
    if len(success_hist) >= 10 and len(archive_cluster_hist) >= 20:
        cluster_growth = int(archive_cluster_hist[-1] - archive_cluster_hist[-20])
        archive_stalled = cluster_growth <= 1
    stag_mean = float(np.mean(individual_stag)) if individual_stag.size else 0.0
    stagnated = stag_mean > max(1, int(np_size))
    success_low = recent_success < 0.02

    if not (success_low and archive_stalled and stagnated):
        return False, "coverage_only"

    if int(dim) >= 3 and int(expected_peaks) <= 8:
        return True, "conservative_basin_recovery"
    if int(dim) <= 2 and int(expected_peaks) <= 36:
        return True, "light_basin_recovery"
    return False, "unsupported_landscape"


def _select_injection_replacement_indices(
    fitness: np.ndarray,
    individual_stag: np.ndarray,
    injection_count: int,
    elite_frac: float,
) -> np.ndarray:
    """Select stagnated non-elite individuals for forced exploration."""

    fitness = np.asarray(fitness, dtype=float)
    individual_stag = np.asarray(individual_stag, dtype=float)
    n = fitness.size
    count = int(np.clip(injection_count, 0, n))
    if count <= 0:
        return np.empty((0,), dtype=int)
    elite_count = int(np.ceil(float(np.clip(elite_frac, 0.0, 1.0)) * n))
    elite_count = min(max(0, elite_count), n)
    elite = set(np.argsort(fitness)[::-1][:elite_count].astype(int).tolist())
    candidates = [idx for idx in range(n) if idx not in elite]
    if not candidates:
        candidates = list(range(n))
    order = sorted(candidates, key=lambda idx: (float(individual_stag[idx]), -float(fitness[idx])), reverse=True)
    return np.asarray(order[:count], dtype=int)


def _should_archive_reseed(
    progress: float,
    archive: PeakArchive,
    pop: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    individual_stag: np.ndarray,
    np_size: int,
    expected_peaks: int,
    cfg: OptimizerConfig,
) -> tuple[bool, str]:
    """Decide whether archive representatives should be copied back to population."""

    if not cfg.archive_reseed_enabled or progress < cfg.archive_reseed_min_progress:
        return False, "not_allowed"
    if int(expected_peaks) >= 50:
        return False, "high_peak_disabled"
    archive_pop, _ = archive.as_arrays()
    if archive_pop.size == 0:
        return False, "empty_archive"
    selected = _select_archive_reseed_entries(
        archive=archive,
        pop=pop,
        lb=lb,
        ub=ub,
        reseed_count=max(1, int(np.ceil(float(cfg.archive_reseed_frac) * max(1, int(np_size))))),
        cfg=cfg,
    )
    if selected.size == 0:
        return False, "archive_represented"
    return True, "underrepresented_archive"


def _select_archive_reseed_entries(
    archive: PeakArchive,
    pop: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    reseed_count: int,
    cfg: OptimizerConfig,
) -> np.ndarray:
    """Select quality archive entries that are underrepresented in current population."""

    count = int(max(0, reseed_count))
    if count <= 0:
        return np.empty((0,), dtype=int)
    archive_pop, archive_fit = archive.as_arrays()
    if archive_pop.size == 0:
        return np.empty((0,), dtype=int)

    diag = float(np.linalg.norm(np.asarray(ub, dtype=float) - np.asarray(lb, dtype=float)) + 1e-12)
    if np.asarray(pop).size:
        nearest_pop = np.min(pairwise_distances(archive_pop, pop), axis=1)
    else:
        nearest_pop = np.full(archive_pop.shape[0], diag, dtype=float)
    distance_threshold = max(
        float(archive.radius) * float(max(0.0, cfg.archive_reseed_min_distance_factor)),
        0.005 * diag,
    )
    quality_threshold = float(
        np.quantile(
            archive_fit,
            float(np.clip(cfg.archive_reseed_quality_quantile, 0.0, 1.0)),
        )
    )
    eligible = np.flatnonzero((nearest_pop > distance_threshold) & (archive_fit >= quality_threshold))
    if eligible.size == 0:
        eligible = np.flatnonzero(nearest_pop > distance_threshold)
    if eligible.size == 0:
        return np.empty((0,), dtype=int)

    score = 0.65 * _normalize01(nearest_pop[eligible] / diag) + 0.35 * _normalize01(archive_fit[eligible])
    order = eligible[np.argsort(score)[::-1]]
    return np.asarray(order[:count], dtype=int)


def _apply_archive_reseed(
    pop: np.ndarray,
    fitness: np.ndarray,
    personal_best: np.ndarray,
    individual_stag: np.ndarray,
    archive: PeakArchive,
    lb: np.ndarray,
    ub: np.ndarray,
    np_size: int,
    cfg: OptimizerConfig,
) -> tuple[int, float, str]:
    """Copy already evaluated archive representatives back into stagnant slots."""

    requested = int(np.ceil(float(cfg.archive_reseed_frac) * max(1, int(np_size))))
    requested = min(max(1, requested), pop.shape[0])
    archive_idx = _select_archive_reseed_entries(
        archive=archive,
        pop=pop,
        lb=lb,
        ub=ub,
        reseed_count=requested,
        cfg=cfg,
    )
    if archive_idx.size == 0:
        return 0, 0.0, "archive_represented"
    replace_idx = _select_injection_replacement_indices(fitness, individual_stag, archive_idx.size, cfg.elite_frac)
    if replace_idx.size == 0:
        return 0, 0.0, "no_replacement_slot"

    archive_pop, archive_fit = archive.as_arrays()
    used = int(min(replace_idx.size, archive_idx.size))
    copied_fit: list[float] = []
    for local_pos in range(used):
        dst = int(replace_idx[local_pos])
        src = int(archive_idx[local_pos])
        pop[dst] = archive_pop[src]
        fitness[dst] = archive_fit[src]
        personal_best[dst] = archive_fit[src]
        individual_stag[dst] = 0.0
        copied_fit.append(float(archive_fit[src]))
    return used, float(np.mean(copied_fit)) if copied_fit else 0.0, "underrepresented_archive"


def _coverage_injection_points(
    pop: np.ndarray,
    archive: PeakArchive,
    lb: np.ndarray,
    ub: np.ndarray,
    injection_count: int,
    candidate_multiplier: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate far-from-archive and far-from-population injection points."""

    count = int(max(0, injection_count))
    if count <= 0:
        return np.empty((0, pop.shape[1]), dtype=float)
    candidate_count = max(50, int(candidate_multiplier) * count)
    candidates = rng.uniform(lb, ub, size=(candidate_count, pop.shape[1]))
    diag = float(np.linalg.norm(ub - lb) + 1e-12)

    pop_dists = pairwise_distances(candidates, pop)
    pop_nearest = np.min(pop_dists, axis=1) / diag if pop.size else np.ones(candidate_count, dtype=float)
    archive_pop, _ = archive.as_arrays()
    if archive_pop.size:
        archive_dists = pairwise_distances(candidates, archive_pop)
        archive_nearest = np.min(archive_dists, axis=1) / diag
    else:
        archive_nearest = np.ones(candidate_count, dtype=float)

    score = 0.7 * _normalize01(archive_nearest) + 0.3 * _normalize01(pop_nearest)
    selected: list[int] = []
    for idx in np.argsort(score)[::-1]:
        if len(selected) >= count:
            break
        selected.append(int(idx))
    return candidates[np.asarray(selected, dtype=int)]


def _basin_edge_injection_points(
    pop: np.ndarray,
    archive: PeakArchive,
    lb: np.ndarray,
    ub: np.ndarray,
    injection_count: int,
    candidate_multiplier: int,
    rng: np.random.Generator,
    step_min: float = 0.04,
    step_max: float = 0.18,
) -> np.ndarray:
    """Generate conservative points around archive basin boundaries.

    中文说明：这个采样不追求全局最远点，而是从已有 archive 代表向外扩展，
    用于 F12-F20 这类“已找到部分 basin、缺少相邻 basin”的场景。
    """

    count = int(max(0, injection_count))
    dim = int(np.asarray(lb).size)
    if count <= 0:
        return np.empty((0, dim), dtype=float)
    archive_pop, archive_fit = archive.as_arrays()
    if archive_pop.size == 0:
        return _coverage_injection_points(pop, archive, lb, ub, count, candidate_multiplier, rng)

    candidate_count = max(50, int(candidate_multiplier) * count)
    span = np.maximum(ub - lb, 1e-12)
    x_norm = (archive_pop - lb) / span
    center = np.mean(x_norm, axis=0)
    fit_weight = _normalize01(archive_fit) + 0.05
    fit_weight = fit_weight / np.sum(fit_weight)

    candidates = np.empty((candidate_count, dim), dtype=float)
    for i in range(candidate_count):
        anchor_idx = int(rng.choice(archive_pop.shape[0], p=fit_weight))
        anchor_norm = x_norm[anchor_idx]
        direction = anchor_norm - center
        if np.linalg.norm(direction) <= 1e-12:
            direction = rng.normal(0.0, 1.0, size=dim)
        direction = direction + 0.35 * rng.normal(0.0, 1.0, size=dim)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            direction = np.ones(dim, dtype=float) / np.sqrt(dim)
        else:
            direction = direction / norm
        step = float(rng.uniform(step_min, step_max))
        jitter = rng.normal(0.0, 0.015, size=dim)
        cand_norm = np.clip(anchor_norm + step * direction + jitter, 0.0, 1.0)
        candidates[i] = lb + cand_norm * span

    diag = float(np.linalg.norm(ub - lb) + 1e-12)
    pop_nearest = np.min(pairwise_distances(candidates, pop), axis=1) / diag if pop.size else np.ones(candidate_count)
    archive_nearest = np.min(pairwise_distances(candidates, archive_pop), axis=1) / diag
    # 倾向选择 archive 边缘点：离 archive 不能太近，也不能变成全局随机远点。
    edge_band = np.exp(-np.square((archive_nearest - 0.10) / 0.10))
    score = 0.55 * edge_band + 0.45 * _normalize01(pop_nearest)
    selected: list[int] = []
    for idx in np.argsort(score)[::-1]:
        if len(selected) >= count:
            break
        selected.append(int(idx))
    return candidates[np.asarray(selected, dtype=int)]


def _apply_coverage_injection(
    problem: CEC2013Problem,
    pop: np.ndarray,
    fitness: np.ndarray,
    personal_best: np.ndarray,
    individual_stag: np.ndarray,
    archive: PeakArchive,
    lb: np.ndarray,
    ub: np.ndarray,
    rng: np.random.Generator,
    max_fes_total: int,
    fes: int,
    np_size: int,
    cfg: OptimizerConfig,
) -> tuple[int, float, int, float, float, float, int]:
    """Force-restart stagnated non-elite individuals in sparse regions."""

    remaining = int(max_fes_total - fes)
    if remaining <= 0:
        return 0, 0.0, 0, 0.0, 0.0, 0.0, int(fes)
    frac = cfg.high_peak_injection_frac if problem.expected_peaks >= 50 else cfg.injection_frac
    requested = int(np.ceil(float(frac) * max(1, int(np_size))))
    injection_count = int(min(max(1, requested), remaining, pop.shape[0]))
    replace_idx = _select_injection_replacement_indices(fitness, individual_stag, injection_count, cfg.elite_frac)
    if replace_idx.size == 0:
        return 0, 0.0, 0, 0.0, 0.0, 0.0, int(fes)

    replaced_fit = fitness[replace_idx].copy()
    if str(cfg.injection_mode) == "conservative":
        points = _basin_edge_injection_points(
            pop,
            archive,
            lb,
            ub,
            replace_idx.size,
            cfg.injection_candidate_multiplier,
            rng,
            cfg.basin_edge_step_min,
            cfg.basin_edge_step_max,
        )
    else:
        points = _coverage_injection_points(
            pop,
            archive,
            lb,
            ub,
            replace_idx.size,
            cfg.injection_candidate_multiplier,
            rng,
        )
    archive_gain = 0.0
    archive_count = 0
    best_fit = -np.inf
    candidate_fits: list[float] = []
    used = 0
    for local_pos, idx in enumerate(replace_idx):
        if fes >= max_fes_total or local_pos >= points.shape[0]:
            break
        fit = float(problem.evaluate(points[local_pos]))
        candidate_fits.append(fit)
        fes += 1
        used += 1
        pop[int(idx)] = points[local_pos]
        fitness[int(idx)] = fit
        personal_best[int(idx)] = fit
        individual_stag[int(idx)] = 0.0
        best_fit = max(best_fit, fit)
        probe_gain = _archive_probe_gain(archive, points[local_pos], fit)
        if _injection_archive_allowed(fit, fitness, probe_gain, cfg):
            gain = archive.add_or_update(points[local_pos], fit, source=2, fes=fes)
            archive_gain += gain
            if gain > 0:
                archive_count += 1
    if used <= 0:
        return 0, 0.0, 0, 0.0, 0.0, 0.0, int(fes)
    return (
        int(used),
        float(archive_gain),
        int(archive_count),
        float(np.mean(replaced_fit[:used])) if replaced_fit.size else 0.0,
        float(np.mean(candidate_fits)) if candidate_fits else 0.0,
        float(best_fit),
        int(fes),
    )


def _archive_probe_gain(archive: PeakArchive, x: np.ndarray, fitness: float) -> float:
    """Predict PeakArchive gain without mutating it."""

    if not archive.entries:
        return 1.0
    pop, _ = archive.as_arrays()
    dists = np.linalg.norm(pop - np.asarray(x, dtype=float), axis=1)
    nearest = int(np.argmin(dists))
    if float(dists[nearest]) > archive.radius:
        return 1.0
    if float(fitness) > archive.entries[nearest].fitness:
        return 0.25
    return 0.0


def _injection_archive_allowed(
    fitness_value: float,
    population_fitness: np.ndarray,
    archive_gain: float,
    cfg: OptimizerConfig,
) -> bool:
    """Quality gate for writing forced-injection points into the archive."""

    if archive_gain == 0.25:
        return True
    if archive_gain <= 0:
        return False
    threshold = float(
        np.quantile(
            np.asarray(population_fitness, dtype=float),
            float(np.clip(cfg.injection_archive_quality_quantile, 0.0, 1.0)),
        )
    )
    return bool(float(fitness_value) >= threshold)


def select_refinement_seeds(
    combo_pop: np.ndarray,
    combo_fit: np.ndarray,
    combo_niches: NicheResult,
    lb: np.ndarray,
    ub: np.ndarray,
    expected_peaks: int,
    remaining_fes: int,
    population_size: int,
    config: OptimizerConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Select multiple sparse and high-quality CMA-ES seeds from each DBSCAN cluster."""

    combo_pop = np.asarray(combo_pop, dtype=float)
    combo_fit = np.asarray(combo_fit, dtype=float)
    if combo_pop.size == 0 or not combo_niches.members:
        return np.empty((0,), dtype=int), []
    if not config.multi_seed_cmaes:
        seed_idx = np.asarray(combo_niches.seeds, dtype=int)
        return seed_idx, _seed_metadata(combo_niches, seed_idx, population_size)

    global_cap = min(
        int(np.ceil(max(1, expected_peaks) * float(config.seed_global_multiplier))),
        max(1, int(remaining_fes) // max(1, int(config.seed_min_budget))),
    )
    if global_cap <= 0:
        return np.empty((0,), dtype=int), []

    span = np.maximum(ub - lb, 1e-12)
    x_norm = (combo_pop - lb) / span
    candidates: list[dict[str, Any]] = []
    for cluster_id, members in enumerate(combo_niches.members):
        members = np.asarray(members, dtype=int)
        if members.size == 0:
            continue
        quota = min(int(config.seed_per_cluster_cap), int(np.ceil(members.size / max(1, config.seed_cluster_size))))
        fit_norm = _normalize01(combo_fit[members])
        sparsity = _cluster_local_sparsity(x_norm[members])
        sparsity_norm = _normalize01(sparsity)
        sparsity_weight = float(np.clip(config.seed_sparsity_weight, 0.0, 1.0))
        archive_bonus = (members >= int(population_size)).astype(float) * float(config.seed_archive_bonus)
        score = (1.0 - sparsity_weight) * fit_norm + sparsity_weight * sparsity_norm + archive_bonus
        order = members[np.argsort(score)[::-1]]
        score_by_idx = {int(idx): float(score[pos]) for pos, idx in enumerate(members)}
        selected = _greedy_diverse_indices(order, x_norm, quota)
        if selected.size < quota:
            missing = [int(idx) for idx in order if int(idx) not in set(selected.tolist())]
            selected = np.asarray(selected.tolist() + missing[: quota - selected.size], dtype=int)
        for rank, idx in enumerate(selected[:quota]):
            candidates.append(
                {
                    "idx": int(idx),
                    "source_cluster_id": int(cluster_id),
                    "cluster_size": int(members.size),
                    "seed_source": "archive" if int(idx) >= int(population_size) else "population",
                    "seed_rank_in_cluster": int(rank),
                    "seed_score": float(score_by_idx[int(idx)]),
                }
            )

    candidates.sort(key=lambda item: item["seed_score"], reverse=True)
    kept = candidates[:global_cap]
    seed_idx = np.asarray([item.pop("idx") for item in kept], dtype=int)
    return seed_idx, kept


def _cluster_local_sparsity(x_norm: np.ndarray) -> np.ndarray:
    if x_norm.shape[0] <= 1:
        return np.ones(x_norm.shape[0], dtype=float)
    dmat = pairwise_distances(x_norm)
    np.fill_diagonal(dmat, np.inf)
    return np.min(dmat, axis=1)


def _greedy_diverse_indices(order: np.ndarray, x_norm: np.ndarray, quota: int) -> np.ndarray:
    order = np.asarray(order, dtype=int)
    quota = int(max(0, quota))
    if quota <= 0 or order.size == 0:
        return np.empty((0,), dtype=int)
    cluster_sparsity = _cluster_local_sparsity(x_norm[order])
    positive = cluster_sparsity[np.isfinite(cluster_sparsity) & (cluster_sparsity > 0)]
    min_sep = float(np.median(positive) * 0.5) if positive.size else 0.0
    selected: list[int] = []
    for idx in order:
        idx = int(idx)
        if not selected:
            selected.append(idx)
        else:
            d = np.linalg.norm(x_norm[np.asarray(selected)] - x_norm[idx], axis=1)
            if np.min(d) >= min_sep:
                selected.append(idx)
        if len(selected) >= quota:
            break
    return np.asarray(selected, dtype=int)


def _normalize01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    scale = float(np.ptp(values))
    if scale <= 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - float(np.min(values))) / scale


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
    coverage_proxy: float | None = None,
    coverage_target: float | None = None,
    coverage_head_min_prob: float = 0.25,
) -> str:
    """Select a preference head from local search state.

    The rule is intentionally simple and deterministic under exploration=0
    so it can be tested and interpreted in mechanism analysis.
    """

    if rng.random() < float(np.clip(exploration, 0.0, 1.0)):
        return str(rng.choice(["coverage", "quality", "diversity", "balanced"]))

    if coverage_proxy is not None and coverage_target is not None and coverage_proxy < coverage_target:
        min_cov = float(np.clip(coverage_head_min_prob, 0.0, 0.8))
        probs = {
            "coverage": max(0.35, min_cov),
            "diversity": 0.35,
            "balanced": 0.25,
            "quality": 0.05,
        }
        return _sample_head_from_probs(probs, rng)

    if progress >= 0.85:
        return _sample_head_from_probs(
            {"quality": 0.35, "balanced": 0.45, "coverage": 0.10, "diversity": 0.10},
            rng,
        )

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


def _sample_head_from_probs(probs: dict[str, float], rng: np.random.Generator) -> str:
    names = ["coverage", "quality", "diversity", "balanced"]
    weights = np.asarray([float(probs.get(name, 0.0)) for name in names], dtype=float)
    if np.sum(weights) <= 0:
        weights = np.ones(len(names), dtype=float)
    weights = weights / np.sum(weights)
    return str(rng.choice(names, p=weights))


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
    expected_peaks: int = 0,
    effective_archive_clusters: int = 0,
    fes: int | None = None,
    max_fes_total: int | None = None,
) -> tuple[bool, str]:
    if fes is not None and max_fes_total is not None:
        remaining = int(max_fes_total - fes)
        min_phase2 = _phase2_min_budget(max_fes_total, expected_peaks, effective_archive_clusters, cfg)
        if remaining <= min_phase2 and progress >= cfg.min_phase1:
            return True, "phase2_budget_floor"
    if progress >= _effective_hard_phase2(expected_peaks, coverage_proxy, cfg):
        return True, "hard_phase2"
    if progress < cfg.min_phase1:
        return False, "not_allowed_before_min_phase1"
    if coverage_proxy < cfg.coverage_threshold:
        return False, "archive_stalled"
    if len(diversity_hist) < 12 or len(success_hist) < 12 or len(archive_hist) < 12:
        return False, "success_stalled"
    div_recent = float(np.mean(diversity_hist[-5:]))
    div_prev = float(np.mean(diversity_hist[-10:-5]))
    div_stalling = abs(div_recent - div_prev) < 1e-4
    rate_low = float(np.mean(success_hist[-5:])) < 0.02
    archive_stalling = archive_hist[-1] <= archive_hist[-10]
    if div_stalling and rate_low and archive_stalling:
        return True, "coverage_ready_and_stalled"
    if not div_stalling:
        return False, "diversity_stalled"
    if not rate_low:
        return False, "success_stalled"
    return False, "archive_stalled"


def _effective_hard_phase2(expected_peaks: int, coverage_proxy: float, cfg: OptimizerConfig) -> float:
    if int(expected_peaks) >= 50 and coverage_proxy < float(cfg.high_peak_phase2_threshold):
        return float(cfg.high_peak_hard_phase2)
    return float(cfg.hard_phase2)


def _phase2_min_budget(
    max_fes_total: int,
    expected_peaks: int,
    effective_archive_clusters: int,
    cfg: OptimizerConfig,
) -> int:
    seed_estimate = max(1, int(effective_archive_clusters) if effective_archive_clusters > 0 else int(expected_peaks))
    seed_estimate = min(max(1, int(expected_peaks)), seed_estimate)
    ratio_budget = int(np.ceil(float(cfg.phase2_min_budget_ratio) * int(max_fes_total)))
    seed_budget = int(cfg.seed_min_budget) * seed_estimate
    return int(max(ratio_budget, seed_budget))


