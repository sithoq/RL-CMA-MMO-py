"""小生境受限 DE 算子。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rlmmo.core.niching import NicheResult, pairwise_distances


F_LEVELS = np.array([0.1, 0.5, 0.9], dtype=float)
CR_LEVELS = np.array([0.1, 0.5, 0.9], dtype=float)
OPERATOR_NAMES = (
    "niche-rand/1",
    "niche-current-to-local-pbest/1",
    "niche-current-to-nearest-better/1",
)
ACTION_DIM = 27


@dataclass(frozen=True)
class Action:
    f: float
    cr: float
    operator: int
    action_id: int


def decode_action(action_id: int) -> Action:
    """把 0..26 的 DQN 动作解码成 F、CR 和变异策略。"""

    action_id = int(action_id)
    if not 0 <= action_id < ACTION_DIM:
        raise ValueError(f"action_id must be in [0, {ACTION_DIM - 1}], got {action_id}")
    op = action_id % 3
    tmp = action_id // 3
    cr_idx = tmp % 3
    f_idx = tmp // 3
    return Action(float(F_LEVELS[f_idx]), float(CR_LEVELS[cr_idx]), op, action_id)


def generate_offspring(
    pop: np.ndarray,
    fitness: np.ndarray,
    niches: NicheResult,
    niche_actions: list[Action],
    lb: np.ndarray,
    ub: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """按小生境专属动作生成一代子代。

    所有差分向量优先从当前小生境内抽取；若小生境人数不足 4，使用空间最近邻
    补足候选池，尽量避免跨远峰污染。
    """

    pop = np.asarray(pop, dtype=float)
    fitness = np.asarray(fitness, dtype=float)
    n, dim = pop.shape
    offspring = pop.copy()
    owner = niches.assigned.copy()
    dmat = pairwise_distances(pop)

    for niche_id, members in enumerate(niches.members):
        action = niche_actions[niche_id]
        for i in members:
            pool = _local_pool(int(i), members, dmat, n, min_pool=4)
            f_i = float(np.clip(action.f + 0.05 * rng.standard_normal(), 0.1, 1.0))
            cr_i = float(np.clip(action.cr + 0.05 * rng.standard_normal(), 0.05, 1.0))
            mutant = _mutate(int(i), pool, pop, fitness, dmat, f_i, action.operator, rng)
            mask = rng.random(dim) < cr_i
            mask[int(rng.integers(dim))] = True
            trial = pop[int(i)].copy()
            trial[mask] = mutant[mask]
            offspring[int(i)] = np.clip(trial, lb, ub)
    return offspring, owner


def _local_pool(i: int, members: np.ndarray, dmat: np.ndarray, n: int, min_pool: int) -> np.ndarray:
    pool = np.unique(np.asarray(members, dtype=int))
    if pool.size >= min_pool and np.any(pool != i):
        return pool
    nearest = np.argsort(dmat[i])
    merged = list(pool.tolist())
    for idx in nearest:
        idx = int(idx)
        if idx not in merged:
            merged.append(idx)
        if len(merged) >= min_pool:
            break
    return np.asarray(merged, dtype=int)


def _sample_others(pool: np.ndarray, i: int, k: int, rng: np.random.Generator) -> np.ndarray:
    candidates = np.asarray([idx for idx in pool if int(idx) != int(i)], dtype=int)
    if candidates.size == 0:
        candidates = np.asarray(pool, dtype=int)
    replace = candidates.size < k
    return rng.choice(candidates, size=k, replace=replace).astype(int)


def _mutate(
    i: int,
    pool: np.ndarray,
    pop: np.ndarray,
    fitness: np.ndarray,
    dmat: np.ndarray,
    f_i: float,
    operator: int,
    rng: np.random.Generator,
) -> np.ndarray:
    r = _sample_others(pool, i, 3, rng)
    if operator == 0:
        return pop[r[0]] + f_i * (pop[r[1]] - pop[r[2]])
    if operator == 1:
        pbest = _local_pbest(pool, i, fitness, rng)
        return pop[i] + f_i * (pop[pbest] - pop[i]) + f_i * (pop[r[1]] - pop[r[2]])

    nb = _nearest_better(pool, i, fitness, dmat, rng)
    return pop[i] + f_i * (pop[nb] - pop[i]) + f_i * (pop[r[1]] - pop[r[2]])


def _local_pbest(pool: np.ndarray, i: int, fitness: np.ndarray, rng: np.random.Generator) -> int:
    candidates = np.asarray([idx for idx in pool if int(idx) != int(i)], dtype=int)
    if candidates.size == 0:
        return int(i)
    order = candidates[np.argsort(fitness[candidates])[::-1]]
    p_num = max(1, int(np.ceil(0.2 * order.size)))
    return int(rng.choice(order[:p_num]))


def _nearest_better(
    pool: np.ndarray,
    i: int,
    fitness: np.ndarray,
    dmat: np.ndarray,
    rng: np.random.Generator,
) -> int:
    better = np.asarray([idx for idx in pool if fitness[int(idx)] > fitness[i] and int(idx) != i], dtype=int)
    if better.size == 0:
        return _local_pbest(pool, i, fitness, rng)
    return int(better[int(np.argmin(dmat[i, better]))])
