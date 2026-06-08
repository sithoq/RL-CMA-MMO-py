"""外部峰存档。

Archive 的职责不是保存所有历史个体，而是保存彼此距离足够远的候选峰。
它给阶段一提供“覆盖奖励”，也给阶段二 CMA-ES 提供候选 seed。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ArchiveEntry:
    x: np.ndarray
    fitness: float
    source: int = 0
    niche_id: int = -1
    action_id: int = -1
    fes: int = 0


class PeakArchive:
    """按距离去重的候选峰存档。

    add_or_update 返回一个小奖励信号：
    - 1.0：发现一个新的空间区域；
    - 0.25：更新了已有区域的更优代表；
    - 0.0：没有新增信息。
    """

    def __init__(
        self,
        radius: float,
        max_size: int,
        trim_mode: str = "fitness",
        novelty_weight: float = 1.0,
    ):
        self.radius = float(max(radius, 1e-12))
        self.max_size = int(max(1, max_size))
        self.trim_mode = str(trim_mode)
        self.novelty_weight = float(novelty_weight)
        self.entries: list[ArchiveEntry] = []

    def __len__(self) -> int:
        return len(self.entries)

    def add_or_update(
        self,
        x: np.ndarray,
        fitness: float,
        source: int = 0,
        niche_id: int = -1,
        action_id: int = -1,
        fes: int = 0,
    ) -> float:
        x = np.asarray(x, dtype=float).copy()
        fitness = float(fitness)
        if not self.entries:
            self.entries.append(ArchiveEntry(x, fitness, source, niche_id, action_id, fes))
            return 1.0

        pop, _ = self.as_arrays()
        dists = np.linalg.norm(pop - x, axis=1)
        nearest = int(np.argmin(dists))
        if float(dists[nearest]) > self.radius:
            self.entries.append(ArchiveEntry(x, fitness, source, niche_id, action_id, fes))
            self._trim()
            return 1.0

        if fitness > self.entries[nearest].fitness:
            self.entries[nearest] = ArchiveEntry(x, fitness, source, niche_id, action_id, fes)
            return 0.25
        return 0.0

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.entries:
            return np.empty((0, 0), dtype=float), np.empty((0,), dtype=float)
        pop = np.vstack([entry.x for entry in self.entries])
        fit = np.array([entry.fitness for entry in self.entries], dtype=float)
        return pop, fit

    def _trim(self) -> None:
        if len(self.entries) <= self.max_size:
            return
        if self.trim_mode == "sparse_quality":
            self._trim_sparse_quality()
            return
        self.entries.sort(key=lambda item: item.fitness, reverse=True)
        del self.entries[self.max_size :]

    def _trim_sparse_quality(self) -> None:
        pop, fit = self.as_arrays()
        if pop.shape[0] <= self.max_size:
            return

        fit_scale = float(np.ptp(fit) + 1e-12)
        fit_norm = (fit - float(np.min(fit))) / fit_scale

        if pop.shape[0] <= 1:
            sparse_norm = np.zeros(pop.shape[0], dtype=float)
        else:
            dmat = np.linalg.norm(pop[:, None, :] - pop[None, :, :], axis=2)
            np.fill_diagonal(dmat, np.inf)
            nearest = np.min(dmat, axis=1)
            dist_scale = float(np.max(nearest) + 1e-12)
            sparse_norm = nearest / dist_scale

        score = fit_norm + self.novelty_weight * sparse_norm
        keep = np.argsort(score)[::-1][: self.max_size]
        self.entries = [self.entries[int(i)] for i in keep]
