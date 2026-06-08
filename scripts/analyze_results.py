from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rlmmo.benchmarks.cec2013 import CEC2013Problem
from rlmmo.runners.run_one_func import resolve_output_dir


FUNC_CSV_RE = re.compile(r"^F(?P<func>\d{2})_\d+runs_.*\.csv$")
RUN_NPZ_RE = re.compile(r"^F(?P<func>\d{2})_run(?P<run>\d{3})_seed(?P<seed>\d+)_.*\.npz$")
HARD_FUNCS = {7, 8, 9, 14, 15, 16, 17, 18, 19, 20}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析 RLMMO 批量实验结果和每次运行的最优解。")
    parser.add_argument("result_dir", help="结果目录，例如 results/individual_knn_10runs")
    parser.add_argument("--accuracy", type=float, default=1e-4, help="count_goptima 精度，默认 1e-4")
    parser.add_argument("--out-prefix", default=None, help="输出文件前缀，默认为 result_dir/analysis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = resolve_output_dir(args.result_dir)
    if not result_dir.exists():
        raise FileNotFoundError(f"result_dir not found: {result_dir}")

    run_rows = collect_run_rows(result_dir, args.accuracy)
    if not run_rows:
        raise RuntimeError(f"no analyzable runs found in {result_dir}")

    run_df = pd.DataFrame(run_rows).sort_values(["func_num", "run_id"])
    func_df = summarize_by_function(run_df)
    overall_df = summarize_overall(func_df)

    out_prefix = Path(args.out_prefix) if args.out_prefix else result_dir / "analysis"
    if not out_prefix.is_absolute():
        out_prefix = resolve_output_dir(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    run_csv = out_prefix.with_name(out_prefix.name + "_runs.csv")
    func_csv = out_prefix.with_name(out_prefix.name + "_functions.csv")
    overall_csv = out_prefix.with_name(out_prefix.name + "_overall.csv")
    report_md = out_prefix.with_name(out_prefix.name + "_report.md")

    run_df.to_csv(run_csv, index=False, encoding="utf-8-sig")
    func_df.to_csv(func_csv, index=False, encoding="utf-8-sig")
    overall_df.to_csv(overall_csv, index=False, encoding="utf-8-sig")
    write_report(report_md, run_df, func_df, overall_df)

    print(run_csv)
    print(func_csv)
    print(overall_csv)
    print(report_md)


def collect_run_rows(result_dir: Path, accuracy: float) -> list[dict[str, Any]]:
    csv_rows = load_csv_rows(result_dir)
    rows: list[dict[str, Any]] = []
    for npz_path in sorted(result_dir.glob("F*_run*.npz")):
        match = RUN_NPZ_RE.match(npz_path.name)
        if not match:
            continue
        func_num = int(match.group("func"))
        run_id = int(match.group("run"))
        seed = int(match.group("seed"))
        problem = CEC2013Problem(func_num)
        data = np.load(npz_path)
        final_pop = np.asarray(data["final_pop"], dtype=float)
        final_fit = np.asarray(data["final_fitness"], dtype=float)
        best_idx = int(np.argmax(final_fit))
        best_x = final_pop[best_idx]
        best_fit = float(final_fit[best_idx])
        found_peaks, _ = problem.count_goptima(final_pop, accuracy)
        pr = found_peaks / max(1, problem.expected_peaks)
        csv_data = csv_rows.get((func_num, run_id), {})
        rows.append(
            {
                "func_num": func_num,
                "func_name": f"F{func_num:02d}",
                "family": problem.family,
                "dimension": problem.dim,
                "expected_peaks": problem.expected_peaks,
                "run_id": run_id,
                "seed": seed,
                "found_peaks": int(found_peaks),
                "PR": float(pr),
                "SR": float(pr >= 1.0),
                "best_fitness": best_fit,
                "fopt": float(problem.fopt),
                "fitness_gap": float(problem.fopt - best_fit),
                "best_x": vector_to_string(best_x),
                "best_x_norm": float(np.linalg.norm(best_x)),
                "final_pop_size": int(final_pop.shape[0]),
                "FES": int(float(csv_data.get("FES", np.nan))) if "FES" in csv_data else np.nan,
                "runtime": float(csv_data.get("runtime", np.nan)),
                "archive_size": int(float(csv_data.get("archive_size", np.nan))) if "archive_size" in csv_data else np.nan,
                "phase_switch_ratio": float(csv_data.get("phase_switch_ratio", np.nan)),
                "algorithm": csv_data.get("algorithm", "unknown"),
                "npz_path": str(npz_path.resolve()),
            }
        )
    return rows


def load_csv_rows(result_dir: Path) -> dict[tuple[int, int], dict[str, Any]]:
    out: dict[tuple[int, int], dict[str, Any]] = {}
    for csv_path in sorted(result_dir.glob("F*_*.csv")):
        if csv_path.name.startswith("server_summary") or csv_path.name.startswith("analysis"):
            continue
        if not FUNC_CSV_RE.match(csv_path.name):
            continue
        df = pd.read_csv(csv_path)
        if "func_num" not in df.columns or "run_id" not in df.columns:
            continue
        for row in df.to_dict("records"):
            out[(int(row["func_num"]), int(row["run_id"]))] = row
    return out


def summarize_by_function(run_df: pd.DataFrame) -> pd.DataFrame:
    grouped = run_df.groupby("func_num", as_index=False).agg(
        func_name=("func_name", "first"),
        family=("family", "first"),
        dimension=("dimension", "first"),
        expected_peaks=("expected_peaks", "first"),
        runs=("run_id", "count"),
        mean_found_peaks=("found_peaks", "mean"),
        best_found_peaks=("found_peaks", "max"),
        worst_found_peaks=("found_peaks", "min"),
        mean_PR=("PR", "mean"),
        std_PR=("PR", "std"),
        best_PR=("PR", "max"),
        worst_PR=("PR", "min"),
        SR=("SR", "mean"),
        mean_best_fitness=("best_fitness", "mean"),
        best_fitness=("best_fitness", "max"),
        mean_fitness_gap=("fitness_gap", "mean"),
        mean_FES=("FES", "mean"),
        mean_runtime=("runtime", "mean"),
        mean_archive_size=("archive_size", "mean"),
        mean_phase_switch_ratio=("phase_switch_ratio", "mean"),
    )
    best_rows = run_df.sort_values(["func_num", "PR", "best_fitness"], ascending=[True, False, False]).groupby("func_num").head(1)
    best_rows = best_rows[["func_num", "run_id", "seed", "best_x", "npz_path"]].rename(
        columns={"run_id": "best_run_id", "seed": "best_seed", "best_x": "best_solution", "npz_path": "best_npz_path"}
    )
    return grouped.merge(best_rows, on="func_num", how="left")


def summarize_overall(func_df: pd.DataFrame) -> pd.DataFrame:
    hard = func_df[func_df["func_num"].isin(HARD_FUNCS)]
    easy = func_df[~func_df["func_num"].isin(HARD_FUNCS)]
    rows = [
        make_overall_row("all", func_df),
        make_overall_row("hard_F7_F9_F14_F20", hard),
        make_overall_row("non_hard", easy),
    ]
    return pd.DataFrame(rows)


def make_overall_row(name: str, df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"group": name, "func_count": 0, "mean_PR": np.nan, "mean_SR": np.nan, "mean_runtime": np.nan}
    return {
        "group": name,
        "func_count": int(df.shape[0]),
        "mean_PR": float(df["mean_PR"].mean()),
        "mean_SR": float(df["SR"].mean()),
        "mean_runtime": float(df["mean_runtime"].mean()),
        "mean_archive_size": float(df["mean_archive_size"].mean()),
        "mean_phase_switch_ratio": float(df["mean_phase_switch_ratio"].mean()),
    }


def write_report(report_md: Path, run_df: pd.DataFrame, func_df: pd.DataFrame, overall_df: pd.DataFrame) -> None:
    worst = func_df.sort_values("mean_PR").head(5)
    best = func_df.sort_values("mean_PR", ascending=False).head(5)
    with report_md.open("w", encoding="utf-8") as f:
        f.write("# Experiment Analysis Report\n\n")
        f.write("## Overall\n\n")
        f.write(to_markdown_table(overall_df))
        f.write("\n\n## Function Summary\n\n")
        view_cols = [
            "func_name",
            "dimension",
            "expected_peaks",
            "runs",
            "mean_found_peaks",
            "mean_PR",
            "std_PR",
            "SR",
            "best_PR",
            "worst_PR",
            "mean_runtime",
            "mean_phase_switch_ratio",
        ]
        f.write(to_markdown_table(func_df[view_cols]))
        f.write("\n\n## Worst Functions by Mean PR\n\n")
        f.write(to_markdown_table(worst[["func_name", "mean_PR", "SR", "mean_found_peaks", "expected_peaks", "best_run_id", "best_seed", "best_solution"]]))
        f.write("\n\n## Best Functions by Mean PR\n\n")
        f.write(to_markdown_table(best[["func_name", "mean_PR", "SR", "mean_found_peaks", "expected_peaks", "best_run_id", "best_seed", "best_solution"]]))
        f.write("\n")


def to_markdown_table(df: pd.DataFrame) -> str:
    """生成无额外依赖的 Markdown 表格。

    pandas.to_markdown 依赖 tabulate，服务器环境未必安装。这里用内置逻辑生成
    简洁表格，保证报告在最小依赖环境下也可读。
    """

    if df.empty:
        return "_No data_\n"
    work = df.copy()
    for col in work.columns:
        work[col] = work[col].map(_format_cell)
    headers = [str(col) for col in work.columns]
    rows = work.values.tolist()
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _format_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    text = str(value).replace("\n", " ").replace("|", "/")
    if len(text) > 120:
        return text[:117] + "..."
    return text


def vector_to_string(x: np.ndarray, precision: int = 10) -> str:
    return "[" + ", ".join(f"{float(v):.{precision}g}" for v in np.asarray(x).reshape(-1)) + "]"


if __name__ == "__main__":
    main()

