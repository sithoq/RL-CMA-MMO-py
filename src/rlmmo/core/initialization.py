"""种群初始化方法。"""

from __future__ import annotations

import numpy as np


def init_population(
    np_size: int,
    dim: int,
    lb: np.ndarray,
    ub: np.ndarray,
    method: str,
    rng: np.random.Generator,
) -> np.ndarray:
    method = (method or "random").lower()
    if method == "adaptive":
        method = "lhs" if dim <= 3 else "sobol"
    if method == "sobol":
        sample = _sobol(np_size, dim, rng)
    elif method == "lhs":
        sample = _lhs(np_size, dim, rng)
    elif method == "random":
        sample = rng.random((np_size, dim))
    else:
        raise ValueError(f"unknown init_method: {method}")
    return lb + sample * (ub - lb)


def _sobol(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    try:
        from scipy.stats import qmc

        engine = qmc.Sobol(d=dim, scramble=True, seed=int(rng.integers(1, 2**31 - 1)))
        m = int(np.ceil(np.log2(max(1, n))))
        return engine.random_base2(m)[:n]
    except Exception:
        return rng.random((n, dim))


def _lhs(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    try:
        from scipy.stats import qmc

        engine = qmc.LatinHypercube(d=dim, seed=int(rng.integers(1, 2**31 - 1)))
        return engine.random(n)
    except Exception:
        cut = np.linspace(0.0, 1.0, n + 1)
        u = rng.random((n, dim))
        sample = np.zeros((n, dim), dtype=float)
        for j in range(dim):
            sample[:, j] = cut[:n] + u[:, j] * (cut[1:] - cut[:n])
            rng.shuffle(sample[:, j])
        return sample
