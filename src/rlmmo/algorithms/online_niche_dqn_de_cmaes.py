"""在线小生境 Double DQN-DE + DBSCAN-CMA-ES 主算法。

论文主线：
1. 阶段一用在线训练的共享 Double DQN 控制每个小生境的 F、CR 和 DE 变异策略，
   重点扩大峰覆盖率；
2. 阶段二用 DBSCAN 汇总候选峰，再调用完整 CMA-ES 做局部精搜，提高峰定位精度。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np

from rlmmo.benchmarks.cec2013 import CEC2013Problem
from rlmmo.core.archive import PeakArchive
from rlmmo.core.initialization import init_population
from rlmmo.core.niching import NicheResult, build_dbscan_niches, pairwise_distances
from rlmmo.optim.cmaes_refine import refine_with_cmaes
from rlmmo.optim.de import ACTION_DIM, Action, decode_action, generate_offspring
from rlmmo.rl.dqn import DQNConfig, DoubleDQNAgent


@dataclass
class OptimizerConfig:
    """算法超参数。第一版保持少量关键项可配置，避免工程过度膨胀。"""

    accuracy: float = 1e-4
    min_phase1: float = 0.80
    hard_phase2: float = 0.90
    archive_multiplier: int = 2
    state_dim: int = 7
    dqn: DQNConfig = field(default_factory=DQNConfig)


def run_optimizer(
    func_num: int,
    seed: int,
    np_size: int | None = None,
    max_fes: int | None = None,
    init_method: str = "random",
    config: dict[str, Any] | OptimizerConfig | None = None,
) -> dict[str, Any]:
    """运行单次优化并返回完整结果字典。"""

    cfg = _make_config(config)
    rng = np.random.default_rng(int(seed))
    problem = CEC2013Problem(func_num)
    info = problem.info()
    dim = problem.dim
    lb = problem.lb
    ub = problem.ub
    max_fes_total = int(max_fes or problem.max_fes)
    np_size = int(np_size or problem.recommended_np)
    diag = float(np.linalg.norm(ub - lb))
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

    archive = PeakArchive(radius=archive_radius, max_size=archive_max)
    for i in range(np_size):
        if np.isfinite(fitness[i]):
            archive.add_or_update(pop[i], fitness[i], source=0, fes=fes)

    agent = DoubleDQNAgent(cfg.dqn)
    memory = _empty_memory(dim)
    diversity_hist: list[float] = []
    success_hist: list[float] = []
    archive_hist: list[int] = []
    phase_switch_fes = max_fes_total
    gen = 0

    # ========================= 阶段一：DQN-DE 找峰 =========================
    while fes < max_fes_total:
        progress = fes / max_fes_total
        if _should_switch_phase(progress, diversity_hist, success_hist, archive_hist, cfg):
            phase_switch_fes = fes
            break

        gen += 1
        niches = build_dbscan_niches(pop, fitness, lb, ub)
        states, matched, new_memory = _extract_niche_states(pop, fitness, niches, lb, ub, memory, progress)
        actions = [decode_action(agent.select_action(states[k], progress, rng)) for k in range(len(niches.members))]

        parent_pop = pop.copy()
        parent_fit = fitness.copy()
        offspring, owner = generate_offspring(parent_pop, parent_fit, niches, actions, lb, ub, rng)
        offspring_fit = np.full(np_size, -np.inf, dtype=float)
        evaluated = np.zeros(np_size, dtype=bool)
        for i in range(np_size):
            if fes >= max_fes_total:
                break
            offspring_fit[i] = problem.evaluate(offspring[i])
            evaluated[i] = True
            fes += 1

        succ = np.zeros(len(niches.members), dtype=float)
        imp_sum = np.zeros(len(niches.members), dtype=float)
        trial_count = np.zeros(len(niches.members), dtype=float)
        archive_gain = np.zeros(len(niches.members), dtype=float)

        cross_dists = pairwise_distances(offspring, parent_pop)
        for i in np.flatnonzero(evaluated):
            k = int(owner[i])
            trial_count[k] += 1.0
            parent_idx = int(np.argmin(cross_dists[i]))
            if offspring_fit[i] > fitness[parent_idx]:
                improvement = float(offspring_fit[i] - fitness[parent_idx])
                pop[parent_idx] = offspring[i]
                fitness[parent_idx] = offspring_fit[i]
                succ[k] += 1.0
                imp_sum[k] += max(0.0, improvement)
                archive_gain[k] += archive.add_or_update(
                    offspring[i], offspring_fit[i], source=1, niche_id=k, action_id=actions[k].action_id, fes=fes
                )

        # 每代把当前种群中较好的代表也扫一遍，避免未替换但有价值的 seed 丢失。
        for seed_idx in niches.seeds:
            archive.add_or_update(pop[int(seed_idx)], fitness[int(seed_idx)], source=0, fes=fes)

        rewards = _niche_rewards(succ, trial_count, imp_sum, archive_gain, parent_fit)
        next_niches = build_dbscan_niches(pop, fitness, lb, ub)
        next_states, _, next_memory = _extract_niche_states(pop, fitness, next_niches, lb, ub, new_memory, fes / max_fes_total)
        for k in range(len(niches.members)):
            next_state = next_states[_nearest_seed_state(k, niches, next_niches)] if len(next_states) else states[k]
            agent.remember(states[k], actions[k].action_id, rewards[k], next_state)
        agent.train_step(rng)
        memory = next_memory

        diversity_hist.append(_population_diversity(pop, lb, ub))
        success_hist.append(float(np.sum(succ) / max(1.0, np.sum(trial_count))))
        archive_hist.append(len(archive))

    # 如果阶段一一直到 90% 强制点仍未触发，上面的循环会在 hard_phase2 触发；若预算耗尽则无阶段二。
    if phase_switch_fes == max_fes_total and fes < max_fes_total:
        phase_switch_fes = fes

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
    result = {
        "algorithm": "online_niche_dqn_de_cmaes",
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
        "dqn_action_hist": ";".join(map(str, agent.action_hist.tolist())),
        "final_pop": final_pop,
        "final_fitness": final_fit,
    }
    return result


def _make_config(config: dict[str, Any] | OptimizerConfig | None) -> OptimizerConfig:
    if config is None:
        return OptimizerConfig()
    if isinstance(config, OptimizerConfig):
        return config
    cfg = OptimizerConfig()
    for key, value in config.items():
        if key == "dqn" and isinstance(value, dict):
            for dqn_key, dqn_value in value.items():
                if hasattr(cfg.dqn, dqn_key):
                    setattr(cfg.dqn, dqn_key, dqn_value)
        elif hasattr(cfg, key):
            setattr(cfg, key, value)
    cfg.dqn.state_dim = cfg.state_dim
    cfg.dqn.action_dim = ACTION_DIM
    return cfg


def _empty_memory(dim: int) -> dict[str, np.ndarray]:
    return {
        "seeds_pos": np.empty((0, dim), dtype=float),
        "div": np.empty((0,), dtype=float),
        "fit": np.empty((0,), dtype=float),
        "stag": np.empty((0,), dtype=float),
    }


def _extract_niche_states(
    pop: np.ndarray,
    fitness: np.ndarray,
    niches: NicheResult,
    lb: np.ndarray,
    ub: np.ndarray,
    memory: dict[str, np.ndarray],
    progress: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    diag = float(np.linalg.norm(ub - lb) + 1e-12)
    fit_min = float(np.min(fitness))
    fit_range = float(np.ptp(fitness) + 1e-12)
    states = np.zeros((len(niches.members), 7), dtype=np.float32)
    matched = np.full(len(niches.members), -1, dtype=int)
    new_div = np.zeros(len(niches.members), dtype=float)
    new_fit = np.zeros(len(niches.members), dtype=float)
    new_stag = np.zeros(len(niches.members), dtype=float)

    for k, members in enumerate(niches.members):
        idx = np.asarray(members, dtype=int)
        seed_idx = int(niches.seeds[k])
        old = _match_old_seed(pop[seed_idx], memory, diag)
        matched[k] = old
        local_pop = pop[idx]
        local_fit = fitness[idx]
        best_local = float(np.max(local_fit))
        center = np.mean(local_pop, axis=0)
        local_div = float(np.mean(np.linalg.norm(local_pop - center, axis=1)) / diag)
        prev_div = float(memory["div"][old]) if old >= 0 else local_div
        prev_fit = float(memory["fit"][old]) if old >= 0 else best_local
        prev_stag = float(memory["stag"][old]) if old >= 0 else 0.0
        stag = prev_stag + 1.0 if best_local <= prev_fit + 1e-10 else 0.0

        fdc = _local_fdc(local_pop, local_fit)
        delta_div = float(np.tanh((local_div - prev_div) * 20.0))
        fit_norm = float((np.mean(local_fit) - fit_min) / fit_range)
        niche_size_ratio = float(len(idx) / pop.shape[0])
        states[k] = np.array(
            [fdc, local_div, 0.5 + 0.5 * delta_div, fit_norm, min(stag / 50.0, 1.0), niche_size_ratio, progress],
            dtype=np.float32,
        )
        new_div[k] = local_div
        new_fit[k] = best_local
        new_stag[k] = stag

    new_memory = {
        "seeds_pos": pop[niches.seeds].copy() if len(niches.seeds) else np.empty((0, pop.shape[1])),
        "div": new_div,
        "fit": new_fit,
        "stag": new_stag,
    }
    return states, matched, new_memory


def _match_old_seed(seed_pos: np.ndarray, memory: dict[str, np.ndarray], diag: float) -> int:
    old_pos = memory["seeds_pos"]
    if old_pos.size == 0:
        return -1
    d = np.linalg.norm(old_pos - seed_pos, axis=1)
    j = int(np.argmin(d))
    return j if float(d[j]) <= 0.05 * diag else -1


def _local_fdc(local_pop: np.ndarray, local_fit: np.ndarray) -> float:
    if local_pop.shape[0] < 3 or float(np.ptp(local_fit)) < 1e-12:
        return 0.5
    best = local_pop[int(np.argmax(local_fit))]
    dist = np.linalg.norm(local_pop - best, axis=1)
    if float(np.ptp(dist)) < 1e-12:
        return 0.5
    corr = float(np.corrcoef(dist, local_fit)[0, 1])
    if not np.isfinite(corr):
        return 0.5
    # fitness 越高通常离峰越近，因此 -corr 越大代表局部梯度越清晰。
    return float(np.clip((1.0 - corr) * 0.5, 0.0, 1.0))


def _niche_rewards(
    succ: np.ndarray,
    trial_count: np.ndarray,
    imp_sum: np.ndarray,
    archive_gain: np.ndarray,
    parent_fit: np.ndarray,
) -> np.ndarray:
    success_rate = succ / np.maximum(1.0, trial_count)
    fit_scale = float(np.ptp(parent_fit) + 1e-12)
    imp_norm = np.tanh(imp_sum / fit_scale)
    gain_norm = np.tanh(archive_gain)
    rewards = 0.35 * gain_norm + 0.30 * success_rate + 0.25 * imp_norm
    ineffective = (succ <= 0) & (archive_gain <= 0)
    rewards[ineffective] -= 0.05
    return np.clip(rewards, -1.0, 1.0)


def _nearest_seed_state(k: int, old_niches: NicheResult, new_niches: NicheResult) -> int:
    if len(new_niches.seeds) == 0:
        return 0
    old_seed = old_niches.seeds[min(k, len(old_niches.seeds) - 1)]
    # 这里只做索引层面的近似匹配；真实空间匹配已经在状态 memory 内处理。
    return int(np.argmin(np.abs(new_niches.seeds - old_seed)))


def _population_diversity(pop: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> float:
    center = np.mean(pop, axis=0)
    diag = float(np.linalg.norm(ub - lb) + 1e-12)
    return float(np.mean(np.linalg.norm(pop - center, axis=1)) / diag)


def _should_switch_phase(
    progress: float,
    diversity_hist: list[float],
    success_hist: list[float],
    archive_hist: list[int],
    cfg: OptimizerConfig,
) -> bool:
    if progress >= cfg.hard_phase2:
        return True
    if progress < cfg.min_phase1:
        return False
    if len(diversity_hist) < 12 or len(success_hist) < 12 or len(archive_hist) < 12:
        return False
    div_recent = float(np.mean(diversity_hist[-5:]))
    div_prev = float(np.mean(diversity_hist[-10:-5]))
    div_stalling = abs(div_recent - div_prev) < 1e-4
    rate_low = float(np.mean(success_hist[-5:])) < 0.02
    archive_stalling = archive_hist[-1] <= archive_hist[-10]
    return bool(div_stalling and rate_low and archive_stalling)
