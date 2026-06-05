"""KNN 邻域工具。

RLEMMO 风格的个体级控制更适合用 KNN 表示个体局部邻域；DBSCAN 更适合
作为多样性评估与聚类 reward。这里提供统一的 KNN 索引函数，避免状态提取
和 DE 变异各写一套不一致的邻域逻辑。
"""

from __future__ import annotations

import numpy as np

from rlmmo.core.niching import pairwise_distances


def knn_indices(dmat: np.ndarray, i: int, k: int, include_self: bool = True) -> np.ndarray:
    """返回个体 i 的 KNN 索引。

    Parameters
    ----------
    dmat
        shape=(NP, NP) 的距离矩阵。
    i
        目标个体索引。
    k
        邻域大小。会自动截断到合法范围。
    include_self
        True 时返回结果包含 i；False 时排除 i。
    """

    dmat = np.asarray(dmat, dtype=float)
    n = dmat.shape[0]
    if n == 0:
        return np.empty((0,), dtype=int)
    i = int(np.clip(i, 0, n - 1))
    max_k = n if include_self else max(0, n - 1)
    k = int(np.clip(k, 1 if max_k > 0 else 0, max_k))
    order = np.argsort(dmat[i], kind="stable")
    if not include_self:
        order = order[order != i]
    return order[:k].astype(int)


def knn_matrix(pop: np.ndarray, k: int, include_self: bool = True) -> np.ndarray:
    """为整个种群构造 KNN 索引矩阵。"""

    pop = np.asarray(pop, dtype=float)
    n = pop.shape[0]
    dmat = pairwise_distances(pop)
    rows = [knn_indices(dmat, i, k, include_self=include_self) for i in range(n)]
    if not rows:
        return np.empty((0, 0), dtype=int)
    width = max((row.size for row in rows), default=0)
    out = np.full((n, width), -1, dtype=int)
    for i, row in enumerate(rows):
        out[i, : row.size] = row
    return out
