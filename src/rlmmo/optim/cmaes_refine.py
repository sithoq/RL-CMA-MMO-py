"""阶段二完整 CMA-ES 局部精搜。"""

from __future__ import annotations

import numpy as np


def refine_with_cmaes(
    problem,
    seeds: np.ndarray,
    seed_fitness: np.ndarray,
    remaining_fes: int,
    lb: np.ndarray,
    ub: np.ndarray,
    rng: np.random.Generator,
    min_budget_per_seed: int = 20,
) -> tuple[np.ndarray, np.ndarray, int]:
    """对每个候选峰 seed 分配预算并做 CMA-ES 精搜。"""

    seeds = np.asarray(seeds, dtype=float)
    seed_fitness = np.asarray(seed_fitness, dtype=float)
    if seeds.size == 0 or remaining_fes <= 0:
        return seeds, seed_fitness, 0

    k = seeds.shape[0]
    if remaining_fes < k:
        return seeds, seed_fitness, 0
    weights = _budget_weights(seeds, seed_fitness)
    base = min_budget_per_seed
    budgets = np.full(k, min(base, max(1, remaining_fes // k)), dtype=int)
    leftover = max(0, remaining_fes - int(np.sum(budgets)))
    budgets += np.floor(leftover * weights).astype(int)

    out_pop = seeds.copy()
    out_fit = seed_fitness.copy()
    used_total = 0
    for i in range(k):
        budget = int(min(budgets[i], remaining_fes - used_total))
        if budget <= 0:
            break
        best_x, best_f, used = _run_one_seed(problem, out_pop[i], out_fit[i], budget, lb, ub, rng)
        out_pop[i] = best_x
        out_fit[i] = best_f
        used_total += used
    return out_pop, out_fit, used_total


def _budget_weights(seeds: np.ndarray, seed_fitness: np.ndarray) -> np.ndarray:
    fit = seed_fitness.astype(float)
    fit_norm = (fit - np.min(fit)) / (np.ptp(fit) + 1e-12)
    if seeds.shape[0] > 1:
        d = np.sqrt(np.sum((seeds[:, None, :] - seeds[None, :, :]) ** 2, axis=2))
        d[d == 0] = np.inf
        sep = np.min(d, axis=1)
        sep_norm = (sep - np.min(sep)) / (np.ptp(sep) + 1e-12)
    else:
        sep_norm = np.ones_like(fit_norm)
    weights = 0.6 * fit_norm + 0.4 * sep_norm + 0.05
    return weights / np.sum(weights)


def _run_one_seed(problem, seed: np.ndarray, seed_fit: float, budget: int, lb: np.ndarray, ub: np.ndarray, rng):
    try:
        import cma
    except Exception:
        return _fallback_gaussian_refine(problem, seed, seed_fit, budget, lb, ub, rng)

    dim = seed.size
    sigma0 = float(np.clip(0.05 * np.max(ub - lb), 1e-8, np.max(ub - lb)))
    popsize = max(4, 4 + int(3 * np.log(dim)))
    opts = {
        "bounds": [lb.tolist(), ub.tolist()],
        "popsize": popsize,
        "verbose": -9,
        "seed": int(rng.integers(1, 2**31 - 1)),
    }
    es = cma.CMAEvolutionStrategy(seed.tolist(), sigma0, opts)
    best_x = seed.copy()
    best_f = float(seed_fit)
    used = 0
    while used + es.popsize <= budget and not es.stop():
        xs = es.ask()
        xs = [np.clip(np.asarray(x, dtype=float), lb, ub) for x in xs]
        vals = [float(problem.evaluate(x)) for x in xs]
        used += len(vals)
        es.tell(xs, [-v for v in vals])
        j = int(np.argmax(vals))
        if vals[j] > best_f:
            best_f = vals[j]
            best_x = xs[j].copy()
    return best_x, best_f, used


def _fallback_gaussian_refine(problem, seed, seed_fit, budget, lb, ub, rng):
    best_x = seed.copy()
    best_f = float(seed_fit)
    used = 0
    sigma = 0.03 * (ub - lb)
    while used < budget:
        x = np.clip(best_x + rng.normal(0.0, sigma), lb, ub)
        fit = float(problem.evaluate(x))
        used += 1
        if fit > best_f:
            best_x = x
            best_f = fit
        sigma *= 0.99
    return best_x, best_f, used
