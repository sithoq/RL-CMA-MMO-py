"""Diagnostics helpers for mechanism analysis runs.

The recorder is intentionally passive: it only stores rows supplied by the
optimizer and writes CSV/NPZ files when a run finishes. It must not affect the
optimizer state or FES accounting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


GENERATION_COLUMNS = [
    "FES",
    "progress",
    "phase",
    "best_fitness",
    "mean_fitness",
    "std_fitness",
    "diversity",
    "success_rate",
    "archive_size",
    "coverage_proxy",
    "effective_archive_clusters",
    "dbscan_cluster_count",
    "mean_cluster_size",
    "individual_stag_mean",
    "individual_stag_max",
    "action_entropy",
    "selected_head_hist",
    "action_hist",
    "F_hist",
    "CR_hist",
    "operator_hist",
    "reward_vector_mean",
    "head_action_credit_top",
    "injection_count",
    "injection_archive_gain",
    "injection_best_fitness",
    "injection_reason",
    "phase_switch_candidate_reason",
]

CHECKPOINT_COLUMNS = [
    "label",
    "FES",
    "progress",
    "found_peaks_pop",
    "PR_pop",
    "found_peaks_archive",
    "PR_archive",
    "found_peaks_pop_archive",
    "PR_pop_archive",
    "archive_size",
    "cluster_count",
    "coverage_proxy",
]

PHASE2_COLUMNS = [
    "seed_id",
    "source_cluster_id",
    "cluster_size",
    "seed_source",
    "seed_fitness_before",
    "best_fitness_after",
    "fitness_gain",
    "used_fes",
    "budget",
    "improved",
    "nearest_seed_distance",
    "final_found_peak_contribution",
    "cluster_count",
    "seed_rank_in_cluster",
    "seed_score",
]


@dataclass
class DiagnosticsRecorder:
    """Collect lightweight trace data for one optimizer run.

    中文说明：这个类只负责记录和落盘，不参与算法决策。这样诊断开关打开时能
    解释算法过程，关闭时不会改变原实验口径。
    """

    enabled: bool = False
    save_pop_snapshots: bool = False
    generation_rows: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_rows: list[dict[str, Any]] = field(default_factory=list)
    phase2_rows: list[dict[str, Any]] = field(default_factory=list)
    snapshots: dict[str, np.ndarray] = field(default_factory=dict)

    def log_generation(self, row: dict[str, Any]) -> None:
        if self.enabled:
            self.generation_rows.append(_clean_row(row))

    def log_checkpoint(
        self,
        row: dict[str, Any],
        pop: np.ndarray | None = None,
        archive_pop: np.ndarray | None = None,
        seeds: np.ndarray | None = None,
    ) -> None:
        if not self.enabled:
            return
        row = _clean_row(row)
        self.checkpoint_rows.append(row)
        if self.save_pop_snapshots:
            label = str(row.get("label", f"checkpoint_{len(self.checkpoint_rows)}"))
            if pop is not None:
                self.snapshots[f"{label}_pop"] = np.asarray(pop, dtype=float)
            if archive_pop is not None and archive_pop.size > 0:
                self.snapshots[f"{label}_archive"] = np.asarray(archive_pop, dtype=float)
            if seeds is not None and seeds.size > 0:
                self.snapshots[f"{label}_seeds"] = np.asarray(seeds, dtype=float)

    def extend_phase2(self, rows: list[dict[str, Any]]) -> None:
        if self.enabled:
            self.phase2_rows.extend(_clean_row(row) for row in rows)

    def write(self, out_dir: str | Path, prefix: str) -> dict[str, str]:
        """Write collected diagnostics and return output paths."""

        if not self.enabled:
            return {}
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}
        paths["generation_log_path"] = _write_csv(
            out / f"{prefix}_generation_log.csv", self.generation_rows, GENERATION_COLUMNS
        )
        paths["checkpoint_log_path"] = _write_csv(
            out / f"{prefix}_checkpoint_log.csv", self.checkpoint_rows, CHECKPOINT_COLUMNS
        )
        paths["phase2_cmaes_log_path"] = _write_csv(
            out / f"{prefix}_phase2_cmaes_log.csv", self.phase2_rows, PHASE2_COLUMNS
        )
        if self.save_pop_snapshots and self.snapshots:
            snap_path = out / f"{prefix}_snapshots.npz"
            np.savez_compressed(snap_path, **self.snapshots)
            paths["snapshots_path"] = str(snap_path.resolve())
        return paths


def action_entropy(action_ids: list[int] | np.ndarray, action_dim: int) -> float:
    """Normalized entropy of the action distribution in [0, 1]."""

    ids = np.asarray(action_ids, dtype=int)
    if ids.size == 0 or action_dim <= 1:
        return 0.0
    counts = np.bincount(np.clip(ids, 0, action_dim - 1), minlength=action_dim).astype(float)
    probs = counts / max(1.0, float(np.sum(counts)))
    probs = probs[probs > 0]
    entropy = -float(np.sum(probs * np.log(probs)))
    return float(entropy / np.log(action_dim))


def hist_string(values: list[int] | np.ndarray, bins: int) -> str:
    vals = np.asarray(values, dtype=int)
    if vals.size == 0:
        return ";".join(["0"] * bins)
    counts = np.bincount(np.clip(vals, 0, bins - 1), minlength=bins)
    return ";".join(str(int(x)) for x in counts[:bins])


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> str:
    df = pd.DataFrame(rows)
    # 中文说明：即使某次 run 没有阶段二 seed，也写出完整表头，方便批量读取。
    for column in columns:
        if column not in df.columns:
            df[column] = np.nan
    extra = [column for column in df.columns if column not in columns]
    df = df[columns + extra]
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path.resolve())


def _clean_row(row: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, np.generic):
            clean[key] = value.item()
        elif isinstance(value, np.ndarray):
            clean[key] = ";".join(map(str, value.tolist()))
        else:
            clean[key] = value
    return clean
