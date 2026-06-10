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
                ): func_num
                for func_num in funcs
            }
            for future in as_completed(futures):
                csv_paths.append(future.result())

    return summarize_results(csv_paths, out_dir)


def summarize_results(csv_paths: list[Path], out_dir: Path) -> tuple[Path, Path]:
    frames = [pd.read_csv(path) for path in csv_paths]
    all_df = pd.concat(frames, ignore_index=True)
    grouped = all_df.groupby("func_num", as_index=False).agg(
        algorithm=("algorithm", "first"),
        func_name=("func_name", "first"),
        family=("family", "first"),
        dimension=("dimension", "first"),
        expected_peaks=("expected_peaks", "first"),
        NP=("NP", "first"),
        maxFES=("maxFES", "first"),
        runs=("run_id", "count"),
        mean_PR=("PR", "mean"),
        std_PR=("PR", "std"),
        best_PR=("PR", "max"),
        worst_PR=("PR", "min"),
        SR=("SR", "mean"),
        mean_FES=("FES", "mean"),
        mean_runtime=("runtime", "mean"),
        mean_archive_size=("archive_size", "mean"),
        mean_phase_switch_ratio=("phase_switch_ratio", "mean"),
    )
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

