"""F1-F20 批量实验 runner。"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

from rlmmo.runners.run_one_func import resolve_output_dir, run_one_func


def run_f1_f20(
    runs: int,
    workers: int = 1,
    seed_offset: int = 0,
    init_method: str = "random",
    out_dir: str | Path = "results/batch",
    algorithm: str = "online_niche_dqn_de_cmaes",
    funcs: list[int] | None = None,
    max_fes: int | None = None,
    diagnostics: bool = False,
    diagnostic_interval: int = 1,
    save_pop_snapshots: bool = False,
    algorithm_config: dict | None = None,
) -> tuple[Path, Path]:
    funcs = funcs or list(range(1, 21))
    out_dir = resolve_output_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_paths: list[Path] = []

    if workers <= 1:
        for func_num in funcs:
            csv_paths.append(
                run_one_func(
                    func_num,
                    runs,
                    seed_offset,
                    init_method,
                    out_dir,
                    algorithm,
                    max_fes=max_fes,
                    diagnostics=diagnostics,
                    diagnostic_interval=diagnostic_interval,
                    save_pop_snapshots=save_pop_snapshots,
                    algorithm_config=algorithm_config,
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futures = {
                pool.submit(
                    run_one_func,
                    func_num,
                    runs,
                    seed_offset + 10000 * func_num,
                    init_method,
                    out_dir,
                    algorithm,
                    None,
                    max_fes,
                    diagnostics,
                    diagnostic_interval,
                    save_pop_snapshots,
                    algorithm_config,
                ): func_num
                for func_num in funcs
            }
            for future in as_completed(futures):
                csv_paths.append(future.result())

    return summarize_results(csv_paths, out_dir)


def summarize_results(csv_paths: list[Path], out_dir: Path) -> tuple[Path, Path]:
    frames = [pd.read_csv(path) for path in csv_paths]
    all_df = pd.concat(frames, ignore_index=True)
    aggregations = {
        "algorithm": ("algorithm", "first"),
        "func_name": ("func_name", "first"),
        "family": ("family", "first"),
        "dimension": ("dimension", "first"),
        "expected_peaks": ("expected_peaks", "first"),
        "NP": ("NP", "first"),
        "maxFES": ("maxFES", "first"),
        "runs": ("run_id", "count"),
        "mean_PR": ("PR", "mean"),
        "std_PR": ("PR", "std"),
        "best_PR": ("PR", "max"),
        "worst_PR": ("PR", "min"),
        "SR": ("SR", "mean"),
        "mean_FES": ("FES", "mean"),
        "mean_runtime": ("runtime", "mean"),
        "mean_archive_size": ("archive_size", "mean"),
        "mean_phase_switch_ratio": ("phase_switch_ratio", "mean"),
    }
    optional_mean_columns = {
        "phase1_PR_pop": "mean_phase1_PR_pop",
        "phase1_PR_archive": "mean_phase1_PR_archive",
        "phase1_PR_pop_archive": "mean_phase1_PR_pop_archive",
        "phase1_archive_size": "mean_phase1_archive_size",
        "phase1_cluster_count": "mean_phase1_cluster_count",
        "phase2_seed_count": "mean_phase2_seed_count",
        "phase2_cmaes_used_fes": "mean_phase2_cmaes_used_fes",
        "phase2_improved_seed_count": "mean_phase2_improved_seed_count",
        "phase2_mean_fitness_gain": "mean_phase2_mean_fitness_gain",
        "phase2_PR_gain": "mean_phase2_PR_gain",
        "injection_total_count": "mean_injection_total_count",
        "injection_total_archive_gain": "mean_injection_total_archive_gain",
        "injection_total_to_archive_count": "mean_injection_total_to_archive_count",
        "archive_reseed_total_count": "mean_archive_reseed_total_count",
        "archive_reseed_mean_fitness": "mean_archive_reseed_mean_fitness",
        "coverage_proxy": "mean_coverage_proxy",
        "effective_archive_clusters": "mean_effective_archive_clusters",
    }
    for source, target in optional_mean_columns.items():
        if source in all_df.columns:
            aggregations[target] = (source, "mean")
    grouped = all_df.groupby("func_num", as_index=False).agg(**aggregations)
    if {"mean_phase1_PR_archive", "mean_phase1_PR_pop"} <= set(grouped.columns):
        grouped["mean_population_archive_gap"] = grouped["mean_phase1_PR_archive"] - grouped["mean_phase1_PR_pop"]
    grouped = grouped[
        [col for col in _summary_column_order() if col in grouped.columns]
        + [col for col in grouped.columns if col not in _summary_column_order()]
    ]
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_csv = out_dir / f"server_summary_{stamp}.csv"
    summary_md = out_dir / f"server_summary_{stamp}.md"
    grouped.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    hard = grouped[grouped["func_num"].isin([7, 8, 9, 14, 15, 16, 17, 18, 19, 20])]
    with summary_md.open("w", encoding="utf-8") as f:
        f.write("# RLMMO-Python 批量实验汇总\n\n")
        f.write(f"- Overall mean PR: {grouped['mean_PR'].mean():.4f}\n")
        f.write(f"- Hard mean PR: {hard['mean_PR'].mean():.4f}\n")
        f.write(f"- Overall mean SR: {grouped['SR'].mean():.4f}\n\n")
        try:
            table = grouped.to_markdown(index=False)
        except Exception:
            table = grouped.to_csv(index=False)
        f.write(table)
        f.write("\n")
    return summary_csv, summary_md


def _summary_column_order() -> list[str]:
    return [
        "func_num",
        "algorithm",
        "func_name",
        "family",
        "dimension",
        "expected_peaks",
        "NP",
        "maxFES",
        "runs",
        "mean_PR",
        "std_PR",
        "best_PR",
        "worst_PR",
        "SR",
        "mean_phase1_PR_pop",
        "mean_phase1_PR_archive",
        "mean_phase1_PR_pop_archive",
        "mean_population_archive_gap",
        "mean_phase2_PR_gain",
        "mean_phase2_seed_count",
        "mean_phase2_cmaes_used_fes",
        "mean_phase2_improved_seed_count",
        "mean_injection_total_count",
        "mean_injection_total_archive_gain",
        "mean_injection_total_to_archive_count",
        "mean_archive_reseed_total_count",
        "mean_archive_reseed_mean_fitness",
        "mean_coverage_proxy",
        "mean_effective_archive_clusters",
        "mean_FES",
        "mean_runtime",
        "mean_archive_size",
        "mean_phase_switch_ratio",
        "mean_phase1_archive_size",
        "mean_phase1_cluster_count",
        "mean_phase2_mean_fitness_gain",
    ]

