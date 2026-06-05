"""DBSCAN 小生境划分工具。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from sklearn.cluster import DBSCAN
except Exception:  # pragma: no cover - sklearn 不可用时走 fallback
    DBSCAN = None  # type: ignore


@dataclass
class NicheResult:
    members: list[np.ndarray]
    seeds: np.ndarray
    assigned: np.ndarray
    eps: float


def pairwise_distances(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """计算欧氏距离矩阵，避免把 scipy 作为硬依赖写死在核心逻辑里。"""

    a = np.asarray(a, dtype=float)
    b = a if b is None else np.asarray(b, dtype=float)
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def build_dbscan_niches(
    pop: np.ndarray,
    fitness: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
) -> NicheResult:
    """用归一化空间 DBSCAN 动态划分小生境。

    噪声点不会被丢弃：有簇时分给最近簇，无簇时退化为单点簇。
    """

    pop = np.asarray(pop, dtype=float)
    fitness = np.asarray(fitness, dtype=float)
    n, dim = pop.shape
    span = np.maximum(np.asarray(ub, dtype=float) - np.asarray(lb, dtype=float), 1e-12)
    x_norm = (pop - lb) / span

    if n == 0:
        return NicheResult([], np.empty((0,), dtype=int), np.empty((0,), dtype=int), 0.0)
    if n == 1:
        return NicheResult([np.array([0], dtype=int)], np.array([0], dtype=int), np.array([0]), 0.0)

    min_pts = int(max(4, min(10, 2 * dim, n)))
    dmat = pairwise_distances(x_norm)
    sorted_d = np.sort(dmat, axis=1)
    kth = sorted_d[:, min(min_pts - 1, n - 1)]
    eps = float(np.median(kth) * 1.2)
    eps = float(np.clip(eps, 1e-3, 0.25 * np.sqrt(dim)))

    if DBSCAN is not None:
        labels = DBSCAN(eps=eps, min_samples=min_pts).fit_predict(x_norm)
    else:
        labels = _fallback_dbscan(dmat, eps, min_pts)

    labels = _assign_noise(labels, x_norm)
    unique_labels = np.unique(labels)
    members: list[np.ndarray] = []
    seeds: list[int] = []
    assigned = np.empty(n, dtype=int)

    # 重新压缩标签为 0..K-1，便于后续数组索引。
    for new_id, label in enumerate(unique_labels):
        idx = np.flatnonzero(labels == label).astype(int)
        members.append(idx)
        assigned[idx] = new_id
        best_local = idx[int(np.argmax(fitness[idx]))]
        seeds.append(int(best_local))

    return NicheResult(members, np.asarray(seeds, dtype=int), assigned, eps)


def _assign_noise(labels: np.ndarray, x_norm: np.ndarray) -> np.ndarray:
    labels = labels.copy()
    noise = np.flatnonzero(labels < 0)
    valid_labels = np.unique(labels[labels >= 0])
    if len(noise) == 0:
        return labels
    if len(valid_labels) == 0:
        labels[:] = np.arange(labels.size)
        return labels

    centers = np.vstack([x_norm[labels == lab].mean(axis=0) for lab in valid_labels])
    d_noise = pairwise_distances(x_norm[noise], centers)
    nearest = np.argmin(d_noise, axis=1)
    for pos, center_id in zip(noise, nearest, strict=False):
        labels[pos] = valid_labels[int(center_id)]
    return labels


def _fallback_dbscan(dmat: np.ndarray, eps: float, min_pts: int) -> np.ndarray:
    """小规模 fallback DBSCAN，保证没有 sklearn 时仍能跑。"""

    n = dmat.shape[0]
    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = np.flatnonzero(dmat[i] <= eps).tolist()
        if len(neighbors) < min_pts:
            labels[i] = -1
            continue
        labels[i] = cluster_id
        queue = list(neighbors)
        while queue:
            j = queue.pop(0)
            if not visited[j]:
                visited[j] = True
                j_neighbors = np.flatnonzero(dmat[j] <= eps).tolist()
                if len(j_neighbors) >= min_pts:
                    queue.extend([p for p in j_neighbors if p not in queue])
            if labels[j] < 0:
                labels[j] = cluster_id
        cluster_id += 1
    return labels
