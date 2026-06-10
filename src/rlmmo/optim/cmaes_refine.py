"""阶段二完整 CMA-ES 局部精搜。"""

from __future__ import annotations

from typing import Any

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
    diagnostics: bool = False,
    seed_metadata: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray, int] | tuple[np.ndarray, np.ndarray, int, list[dict[str, Any]]]:
    """对每个候选峰 seed 分配预算并做 CMA-ES 精搜。"""

    seeds = np.asarray(seeds, dtype=float)
    seed_fitness = np.asarray(seed_fitness, dtype=float)
    if seeds.size == 0 or remaining_fes <= 0:
        return _cmaes_return(seeds, seed_fitness, 0, [], diagnostics)

    k = seeds.shape[0]
    if remaining_fes < k:
        rows = _initial_log_rows(seeds, seed_fitness, np.zeros(k, dtype=int), seed_metadata)
        return _cmaes_return(seeds, seed_fitness, 0, rows, diagnostics)
    weights = _budget_weights(seeds, seed_fitness)
    base = min_budget_per_seed
    budgets = np.full(k, min(base, max(1, remaining_fes // k)), dtype=int)
    leftover = max(0, remaining_fes - int(np.sum(budgets)))
    budgets += np.floor(leftover * weights).astype(int)

    out_pop = seeds.copy()
    out_fit = seed_fitness.copy()
    used_total = 0
    log_rows = _initial_log_rows(seeds, seed_fitness, budgets, seed_metadata)
    for i in range(k):
        budget = int(min(budgets[i], remaining_fes - used_total))
        if budget <= 0:
            break
        best_x, best_f, used = _run_one_seed(problem, out_pop[i], out_fit[i], budget, lb, ub, rng)
        out_pop[i] = best_x
        out_fit[i] = best_f
        used_total += used
        log_rows[i]["budget"] = int(budget)
        log_rows[i]["best_fitness_after"] = float(best_f)
        log_rows[i]["fitness_gain"] = float(best_f - seed_fitness[i])
        log_rows[i]["used_fes"] = int(used)
        log_rows[i]["improved"] = bool(best_f > seed_fitness[i] + 1e-12)
    return _cmaes_return(out_pop, out_fit, used_total, log_rows, diagnostics)


def _cmaes_return(
    pop: np.ndarray,
    fit: np.ndarray,
    used: int,
    rows: list[dict[str, Any]],
    diagnostics: bool,
) -> tuple[np.ndarray, np.ndarray, int] | tuple[np.ndarray, np.ndarray, int, list[dict[str, Any]]]:
    if diagnostics:
        return pop, fit, int(used), rows
    return pop, fit, int(used)


def _initial_log_rows(
    seeds: np.ndarray,
    seed_fitness: np.ndarray,
    budgets: np.ndarray,
    seed_metadata: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    nearest = _nearest_seed_distances(seeds)
    metadata = seed_metadata or []
    cluster_count = len({int(item.get("source_cluster_id", -1)) for item in metadata}) if metadata else seeds.shape[0]
    for i in range(seeds.shape[0]):
        meta = metadata[i] if i < len(metadata) else {}
        rows.append(
            {
                "seed_id": int(i),
                "source_cluster_id": int(meta.get("source_cluster_id", i)),
                "cluster_size": int(meta.get("cluster_size", 1)),
                "seed_source": str(meta.get("seed_source", "unknown")),
                "cluster_count": int(cluster_count),
                "seed_fitness_before": float(seed_fitness[i]),
                "best_fitness_after": float(seed_fitness[i]),
                "fitness_gain": 0.0,
                "used_fes": 0,
                "budget": int(budgets[i]) if i < budgets.size else 0,
                "improved": False,
                "nearest_seed_distance": float(nearest[i]) if i < nearest.size else 0.0,
                "final_found_peak_contribution": 0,
            }
        )
    return rows


def _nearest_seed_distances(seeds: np.ndarray) -> np.ndarray:
    if seeds.shape[0] <= 1:
        return np.zeros(seeds.shape[0], dtype=float)
    d = np.sqrt(np.sum((seeds[:, None, :] - seeds[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(d, np.inf)
    return np.min(d, axis=1)


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
