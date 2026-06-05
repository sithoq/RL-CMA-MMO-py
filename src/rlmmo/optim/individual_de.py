"""个体级 DE 变异算子。

阶段一的普通局部搜索使用 KNN 邻域作为候选池；DBSCAN 不再作为硬性繁殖边界，
只在 cross-niche-best/1 中提供跨区域 seed，从而更接近 RLEMMO 的 KNN 搜索邻域
+ DBSCAN 多样性评估思路。
"""

from __future__ import annotations

import numpy as np

from rlmmo.core.neighborhood import knn_indices
from rlmmo.core.niching import NicheResult, pairwise_distances
from rlmmo.optim.de import Action


OPERATOR_NAMES = (
    "knn-rand/1",
    "current-to-knn-best/1",
    "cross-niche-best/1",
)


def generate_individual_offspring(
    pop: np.ndarray,
    fitness: np.ndarray,
    niches: NicheResult,
    individual_actions: list[Action],
    lb: np.ndarray,
    ub: np.ndarray,
    rng: np.random.Generator,
    knn_k: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """为每个个体生成一个 trial。

    KNN pool 是每个个体的主要搜索邻域；DBSCAN 只用于跨小生境探索动作。
    """

    pop = np.asarray(pop, dtype=float)
    fitness = np.asarray(fitness, dtype=float)
    n, dim = pop.shape
    if len(individual_actions) != n:
        raise ValueError(f"individual_actions length must be {n}, got {len(individual_actions)}")

    offspring = pop.copy()
    owner = np.arange(n, dtype=int)
    dmat = pairwise_distances(pop)
    # DE/rand/1 需要至少 3 个非自身候选。include_self=False 时 k 至少取 3，
    # 但如果 NP 较小，knn_indices 会自动截断并在抽样时允许 replace。
    pool_k = max(3, int(knn_k))

    for i in range(n):
        action = individual_actions[i]
        pool = knn_indices(dmat, i, pool_k, include_self=False)
        if pool.size == 0:
            pool = np.asarray([i], dtype=int)
        f_i = float(np.clip(action.f + 0.05 * rng.standard_normal(), 0.1, 1.0))
        cr_i = float(np.clip(action.cr + 0.05 * rng.standard_normal(), 0.05, 1.0))
        mutant = _mutate_individual(i, pool, pop, fitness, dmat, f_i, action.operator, niches, rng)

        mask = rng.random(dim) < cr_i
        mask[int(rng.integers(dim))] = True
        trial = pop[i].copy()
        trial[mask] = mutant[mask]
        offspring[i] = np.clip(trial, lb, ub)

    return offspring, owner


def _sample_others(pool: np.ndarray, i: int, k: int, rng: np.random.Generator) -> np.ndarray:
    candidates = np.asarray([idx for idx in pool if int(idx) != int(i)], dtype=int)
    if candidates.size == 0:
        candidates = np.asarray(pool, dtype=int)
    replace = candidates.size < k
    return rng.choice(candidates, size=k, replace=replace).astype(int)


def _mutate_individual(
    i: int,
    pool: np.ndarray,
    pop: np.ndarray,
    fitness: np.ndarray,
    dmat: np.ndarray,
    f_i: float,
    operator: int,
    niches: NicheResult,
    rng: np.random.Generator,
) -> np.ndarray:
    """执行三个个体级策略之一。"""

    r = _sample_others(pool, i, 3, rng)
    if operator == 0:
        # KNN-rand/1：差分向量严格来自 KNN pool，偏局部探索。
        return pop[r[0]] + f_i * (pop[r[1]] - pop[r[2]])

    if operator == 1:
        # current-to-KNN-best/1：向 KNN 内最优个体靠近，偏局部开发。
        knn_best = _neighborhood_best(pool, i, fitness)
        return pop[i] + f_i * (pop[knn_best] - pop[i]) + f_i * (pop[r[1]] - pop[r[2]])

    # cross-niche-best/1：吸引点来自其他 DBSCAN cluster seed，差分仍来自 KNN pool。
    cross_best = _cross_niche_best(i, niches, fitness, rng)
    return pop[i] + f_i * (pop[cross_best] - pop[i]) + f_i * (pop[r[1]] - pop[r[2]])


def _neighborhood_best(pool: np.ndarray, i: int, fitness: np.ndarray) -> int:
    candidates = np.asarray([idx for idx in pool if int(idx) != int(i)], dtype=int)
    if candidates.size == 0:
        return int(i)
    return int(candidates[int(np.argmax(fitness[candidates]))])


def _cross_niche_best(i: int, niches: NicheResult, fitness: np.ndarray, rng: np.random.Generator) -> int:
    if len(niches.members) <= 1 or niches.seeds.size == 0 or niches.assigned.size <= i:
        return int(i)
    own = int(niches.assigned[i])
    other_seeds = np.asarray([seed for k, seed in enumerate(niches.seeds) if k != own], dtype=int)
    if other_seeds.size == 0:
        return int(i)
    # 在其他小生境的较优 seed 中随机挑一个，避免所有个体被同一最高峰强吸引。
    order = other_seeds[np.argsort(fitness[other_seeds])[::-1]]
    top = order[: max(1, int(np.ceil(0.3 * order.size)))]
    return int(rng.choice(top))
