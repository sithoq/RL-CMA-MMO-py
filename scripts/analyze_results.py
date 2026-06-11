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
CORE_FUNCS = {1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13}
HIGH_PEAK_FUNCS = {8, 9}
HARD_FUNCS = {14, 15, 16, 17, 18, 19, 20}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析 RLMMO 批量实验结果和每次运行的最优解。")
    parser.add_argument("result_dir", help="结果目录，例如 results/individual_knn_10runs")
    parser.add_argument("--accuracy", type=float, default=1e-4, help="count_goptima 精度，默认 1e-4")
    parser.add_argument("--out-prefix", default=None, help="输出文件前缀，默认为 result_dir/analysis")
    parser.add_argument(
        "--mechanism",
        action="store_true",
        help="额外输出机制分析表：phase1/archive/phase2/reseed/injection/action entropy 和失败类型。",
    )
    parser.add_argument(
        "--baseline-dir",
        default=None,
        help="与当前 result_dir 做机制对比的基线目录；需要配合 --mechanism 使用。",
    )
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

    if args.mechanism:
        mechanism_run_df = collect_mechanism_rows(result_dir)
        if mechanism_run_df.empty:
            print("mechanism analysis skipped: no function CSV rows found")
            return
        mechanism_func_df = summarize_mechanism_by_function(mechanism_run_df)
        mech_run_csv = out_prefix.with_name(out_prefix.name + "_mechanism_runs.csv")
        mech_func_csv = out_prefix.with_name(out_prefix.name + "_mechanism_functions.csv")
        mech_report_md = out_prefix.with_name(out_prefix.name + "_mechanism_report.md")
        mechanism_run_df.to_csv(mech_run_csv, index=False, encoding="utf-8-sig")
        mechanism_func_df.to_csv(mech_func_csv, index=False, encoding="utf-8-sig")
        write_mechanism_report(mech_report_md, mechanism_func_df)
        print(mech_run_csv)
        print(mech_func_csv)
        print(mech_report_md)
        if args.baseline_dir:
            baseline_dir = resolve_output_dir(args.baseline_dir)
            baseline_run_df = collect_mechanism_rows(baseline_dir)
            if baseline_run_df.empty:
                print(f"mechanism comparison skipped: no baseline rows found in {baseline_dir}")
            else:
                baseline_func_df = summarize_mechanism_by_function(baseline_run_df)
                compare_df = compare_mechanism_functions(baseline_func_df, mechanism_func_df)
                compare_csv = out_prefix.with_name(out_prefix.name + "_mechanism_compare.csv")
                compare_report_md = out_prefix.with_name(out_prefix.name + "_mechanism_compare_report.md")
                compare_df.to_csv(compare_csv, index=False, encoding="utf-8-sig")
                write_mechanism_compare_report(compare_report_md, compare_df, baseline_dir, result_dir)
                print(compare_csv)
                print(compare_report_md)


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


def collect_mechanism_rows(result_dir: Path) -> pd.DataFrame:
    """Collect per-run mechanism metrics from summary CSV and diagnostics logs.

    中文说明：这个分析不重新评价函数，只读取已经落盘的 summary/diagnostics。
    因此它适合服务器跑完实验后快速判断：问题主要出在阶段一覆盖、population
    覆盖流失、阶段二 seed/CMA-ES，还是 injection/reseed 行为。
    """

    rows: list[dict[str, Any]] = []
    diagnostics_dir = result_dir / "diagnostics"
    for row in load_csv_rows(result_dir).values():
        func_num = int(row["func_num"])
        run_id = int(row["run_id"])
        diag = _load_generation_diagnostics(row, diagnostics_dir)
        phase2 = _load_phase2_diagnostics(row, diagnostics_dir)
        base = {
            "func_num": func_num,
            "func_name": f"F{func_num:02d}",
            "group": _function_group(func_num),
            "run_id": run_id,
            "PR": _safe_float(row.get("PR")),
            "phase1_PR_pop": _safe_float(row.get("phase1_PR_pop")),
            "phase1_PR_archive": _safe_float(row.get("phase1_PR_archive")),
            "phase1_PR_pop_archive": _safe_float(row.get("phase1_PR_pop_archive")),
            "phase2_PR_gain": _safe_float(row.get("phase2_PR_gain")),
            "phase2_seed_count": _safe_float(row.get("phase2_seed_count")),
            "phase2_improved_seed_count": _safe_float(row.get("phase2_improved_seed_count")),
            "phase_switch_ratio": _safe_float(row.get("phase_switch_ratio")),
            "archive_size": _safe_float(row.get("archive_size")),
            "coverage_proxy": _safe_float(row.get("coverage_proxy")),
            "effective_archive_clusters": _safe_float(row.get("effective_archive_clusters")),
            "FES": _safe_float(row.get("FES")),
            "maxFES": _safe_float(row.get("maxFES")),
        }
        base.update(_summarize_generation_log(diag))
        base.update(_summarize_phase2_log(phase2))
        base["population_archive_gap"] = base["phase1_PR_archive"] - base["phase1_PR_pop"]
        base["phase2_effective"] = float(base["phase2_PR_gain"] >= 0.05)
        base["failure_mode"] = classify_failure_mode(base)
        base["next_action"] = recommend_next_action(base)
        rows.append(base)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["func_num", "run_id"])


def _load_generation_diagnostics(row: dict[str, Any], diagnostics_dir: Path) -> pd.DataFrame | None:
    path = _diagnostic_path(row.get("generation_log_path"), diagnostics_dir)
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def _load_phase2_diagnostics(row: dict[str, Any], diagnostics_dir: Path) -> pd.DataFrame | None:
    path = _diagnostic_path(row.get("phase2_cmaes_log_path"), diagnostics_dir)
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def _diagnostic_path(raw: Any, diagnostics_dir: Path) -> Path | None:
    if raw is None or pd.isna(raw):
        return None
    path = Path(str(raw))
    if path.exists():
        return path
    candidate = diagnostics_dir / path.name
    if candidate.exists():
        return candidate
    return path


def _summarize_generation_log(df: pd.DataFrame | None) -> dict[str, Any]:
    if df is None or df.empty:
        return {
            "mean_action_entropy": np.nan,
            "late_success_rate": np.nan,
            "late_individual_stag_mean": np.nan,
            "injection_total": np.nan,
            "injection_to_archive_total": np.nan,
            "archive_reseed_total": np.nan,
            "last_archive_reseed_reason": "",
            "last_injection_reason": "",
        }
    late = df.tail(min(20, len(df)))
    return {
        "mean_action_entropy": _col_mean(df, "action_entropy"),
        "late_success_rate": _col_mean(late, "success_rate"),
        "late_individual_stag_mean": _col_mean(late, "individual_stag_mean"),
        "injection_total": _col_sum(df, "injection_count"),
        "injection_to_archive_total": _col_sum(df, "injection_to_archive_count"),
        "archive_reseed_total": _col_sum(df, "archive_reseed_count"),
        "last_archive_reseed_reason": _last_text(df, "archive_reseed_reason"),
        "last_injection_reason": _last_text(df, "injection_reason"),
    }


def _summarize_phase2_log(df: pd.DataFrame | None) -> dict[str, Any]:
    if df is None or df.empty:
        return {
            "phase2_archive_seed_ratio": np.nan,
            "phase2_mean_used_fes": np.nan,
            "phase2_mean_seed_gain": np.nan,
        }
    if "seed_source" in df.columns:
        archive_ratio = float((df["seed_source"].astype(str) == "archive").mean())
    else:
        archive_ratio = np.nan
    return {
        "phase2_archive_seed_ratio": archive_ratio,
        "phase2_mean_used_fes": _col_mean(df, "used_fes"),
        "phase2_mean_seed_gain": _col_mean(df, "fitness_gain"),
    }


def summarize_mechanism_by_function(run_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "PR",
        "phase1_PR_pop",
        "phase1_PR_archive",
        "phase1_PR_pop_archive",
        "phase2_PR_gain",
        "phase2_seed_count",
        "phase_switch_ratio",
        "coverage_proxy",
        "effective_archive_clusters",
        "population_archive_gap",
        "mean_action_entropy",
        "late_success_rate",
        "late_individual_stag_mean",
        "injection_total",
        "injection_to_archive_total",
        "archive_reseed_total",
        "phase2_archive_seed_ratio",
        "phase2_mean_used_fes",
        "phase2_mean_seed_gain",
    ]
    agg = {col: (col, "mean") for col in numeric_cols if col in run_df.columns}
    grouped = run_df.groupby("func_num", as_index=False).agg(
        func_name=("func_name", "first"),
        group=("group", "first"),
        runs=("run_id", "count"),
        **agg,
    )
    grouped["failure_mode"] = grouped.apply(lambda row: classify_failure_mode(row.to_dict()), axis=1)
    grouped["next_action"] = grouped.apply(lambda row: recommend_next_action(row.to_dict()), axis=1)
    return grouped.sort_values(["group", "func_num"])


def compare_mechanism_functions(baseline_df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
    """Compare two mechanism summaries function by function."""

    keep_cols = [
        "func_num",
        "func_name",
        "group",
        "runs",
        "PR",
        "phase1_PR_pop",
        "phase1_PR_archive",
        "phase1_PR_pop_archive",
        "phase2_PR_gain",
        "population_archive_gap",
        "mean_action_entropy",
        "injection_total",
        "injection_to_archive_total",
        "archive_reseed_total",
        "failure_mode",
        "next_action",
    ]
    base = baseline_df[[col for col in keep_cols if col in baseline_df.columns]].copy()
    curr = current_df[[col for col in keep_cols if col in current_df.columns]].copy()
    merged = base.merge(curr, on="func_num", how="outer", suffixes=("_baseline", "_current"))

    for name in ["func_name", "group"]:
        merged[name] = merged.get(f"{name}_current").combine_first(merged.get(f"{name}_baseline"))

    delta_pairs = [
        ("PR", "delta_PR"),
        ("phase1_PR_pop", "delta_phase1_PR_pop"),
        ("phase1_PR_archive", "delta_phase1_PR_archive"),
        ("phase1_PR_pop_archive", "delta_phase1_PR_pop_archive"),
        ("phase2_PR_gain", "delta_phase2_PR_gain"),
        ("population_archive_gap", "delta_population_archive_gap"),
        ("mean_action_entropy", "delta_action_entropy"),
        ("injection_total", "delta_injection_total"),
        ("injection_to_archive_total", "delta_injection_to_archive_total"),
        ("archive_reseed_total", "delta_archive_reseed_total"),
    ]
    for source, target in delta_pairs:
        merged[target] = _merged_numeric(merged, f"{source}_current") - _merged_numeric(merged, f"{source}_baseline")

    merged["comparison_status"] = merged.apply(classify_comparison_status, axis=1)
    merged["comparison_note"] = merged.apply(make_comparison_note, axis=1)

    ordered_cols = [
        "func_num",
        "func_name",
        "group",
        "PR_baseline",
        "PR_current",
        "delta_PR",
        "phase1_PR_pop_archive_baseline",
        "phase1_PR_pop_archive_current",
        "delta_phase1_PR_pop_archive",
        "population_archive_gap_baseline",
        "population_archive_gap_current",
        "delta_population_archive_gap",
        "phase2_PR_gain_baseline",
        "phase2_PR_gain_current",
        "delta_phase2_PR_gain",
        "injection_total_baseline",
        "injection_total_current",
        "delta_injection_total",
        "archive_reseed_total_baseline",
        "archive_reseed_total_current",
        "delta_archive_reseed_total",
        "failure_mode_baseline",
        "failure_mode_current",
        "comparison_status",
        "comparison_note",
    ]
    existing = [col for col in ordered_cols if col in merged.columns]
    return merged[existing].sort_values(["group", "func_num"])


def classify_comparison_status(row: pd.Series) -> str:
    delta_pr = _safe_float(row.get("delta_PR"))
    delta_gap = _safe_float(row.get("delta_population_archive_gap"))
    current_pr = _safe_float(row.get("PR_current"))
    if current_pr >= 0.999:
        return "solved"
    if delta_pr >= 0.05 and delta_gap <= 0.02:
        return "improved"
    if delta_pr >= 0.02:
        return "slightly_improved"
    if delta_pr <= -0.05:
        return "regressed"
    if delta_gap <= -0.05:
        return "coverage_drift_reduced"
    return "mixed_or_flat"


def make_comparison_note(row: pd.Series) -> str:
    status = classify_comparison_status(row)
    delta_pr = _safe_float(row.get("delta_PR"))
    delta_gap = _safe_float(row.get("delta_population_archive_gap"))
    delta_reseed = _safe_float(row.get("delta_archive_reseed_total"))
    current_mode = str(row.get("failure_mode_current", ""))
    if status == "solved":
        return "PR reached 1.0; keep this function as a regression guard"
    if status in {"improved", "slightly_improved"}:
        return f"PR improved by {delta_pr:.3f}; inspect whether failure mode changed from baseline"
    if status == "regressed":
        return f"PR regressed by {abs(delta_pr):.3f}; check conservative injection/reseed side effects"
    if status == "coverage_drift_reduced":
        return f"population/archive gap reduced by {abs(delta_gap):.3f}; reseed is likely helping"
    if delta_reseed > 0 and current_mode == "population_coverage_drift":
        return "reseed triggered but coverage drift remains; consider stronger or more frequent archive reseed"
    return "no clear PR movement; use current failure_mode to choose the next mechanism"


def classify_failure_mode(row: dict[str, Any]) -> str:
    pr = _safe_float(row.get("PR"))
    if pr >= 0.999:
        return "solved"
    phase1_combo = _safe_float(row.get("phase1_PR_pop_archive"))
    archive_gap = _safe_float(row.get("population_archive_gap"))
    phase2_gain = _safe_float(row.get("phase2_PR_gain"))
    coverage = _safe_float(row.get("coverage_proxy"))
    entropy = _safe_float(row.get("mean_action_entropy"))
    injection_total = _safe_float(row.get("injection_total"))
    if archive_gap >= 0.10:
        return "population_coverage_drift"
    if phase2_gain >= 0.10:
        return "phase2_refinement_helpful"
    if phase1_combo < 0.50 and coverage < 0.55:
        return "missing_basin"
    if injection_total > 500 and phase2_gain < 0.05:
        return "injection_noise_or_no_gain"
    if entropy >= 0.94:
        return "weak_action_preference"
    return "phase1_coverage_limited"


def recommend_next_action(row: dict[str, Any]) -> str:
    mode = classify_failure_mode(row)
    func_num = int(_safe_float(row.get("func_num")))
    if mode == "solved":
        return "keep conservative defaults; use as regression guard"
    if mode == "population_coverage_drift":
        return "check archive_reseed_count; strengthen reseed before adding new exploration"
    if mode == "phase2_refinement_helpful":
        return "preserve CMA-ES budget and archive-prioritized seed selection"
    if mode == "missing_basin":
        if func_num in HARD_FUNCS:
            return "add hard-function missing-basin recovery only after reseed validation"
        return "increase phase-one basin discovery carefully; do not tune on F8/F9"
    if mode == "injection_noise_or_no_gain":
        return "reduce aggressive injection; require quality-gated archive writes"
    if mode == "weak_action_preference":
        return "keep action credit diagnostics; do not rely on DQN alone for coverage"
    return "compare phase1 archive/pop and phase2 gain in next diagnostics run"


def _function_group(func_num: int) -> str:
    if func_num in CORE_FUNCS:
        return "core"
    if func_num in HIGH_PEAK_FUNCS:
        return "high_peak_record_only"
    if func_num in HARD_FUNCS:
        return "hard"
    return "other"


def write_mechanism_report(report_md: Path, func_df: pd.DataFrame) -> None:
    focus = func_df[func_df["func_num"].isin(sorted(CORE_FUNCS | HARD_FUNCS))]
    with report_md.open("w", encoding="utf-8") as f:
        f.write("# Mechanism Analysis Report\n\n")
        f.write("This report reads existing summary/diagnostics CSV files only; it does not call `count_goptima` again.\n\n")
        f.write("## Focus Functions\n\n")
        cols = [
            "func_name",
            "group",
            "runs",
            "PR",
            "phase1_PR_pop",
            "phase1_PR_archive",
            "population_archive_gap",
            "phase2_PR_gain",
            "phase2_seed_count",
            "mean_action_entropy",
            "injection_total",
            "archive_reseed_total",
            "failure_mode",
            "next_action",
        ]
        existing = [col for col in cols if col in focus.columns]
        f.write(to_markdown_table(focus[existing]))
        f.write("\n\n## Failure Mode Counts\n\n")
        counts = func_df["failure_mode"].value_counts().rename_axis("failure_mode").reset_index(name="function_count")
        f.write(to_markdown_table(counts))
        f.write("\n")


def write_mechanism_compare_report(
    report_md: Path,
    compare_df: pd.DataFrame,
    baseline_dir: Path,
    current_dir: Path,
) -> None:
    focus = compare_df[compare_df["func_num"].isin(sorted(CORE_FUNCS | HARD_FUNCS))]
    with report_md.open("w", encoding="utf-8") as f:
        f.write("# Mechanism Comparison Report\n\n")
        f.write(f"- Baseline: `{baseline_dir}`\n")
        f.write(f"- Current: `{current_dir}`\n\n")
        f.write("## Focus Function Deltas\n\n")
        cols = [
            "func_name",
            "group",
            "PR_baseline",
            "PR_current",
            "delta_PR",
            "delta_phase1_PR_pop_archive",
            "delta_population_archive_gap",
            "delta_phase2_PR_gain",
            "delta_injection_total",
            "delta_archive_reseed_total",
            "failure_mode_baseline",
            "failure_mode_current",
            "comparison_status",
            "comparison_note",
        ]
        existing = [col for col in cols if col in focus.columns]
        f.write(to_markdown_table(focus[existing]))
        f.write("\n\n## Comparison Status Counts\n\n")
        counts = compare_df["comparison_status"].value_counts().rename_axis("comparison_status").reset_index(name="function_count")
        f.write(to_markdown_table(counts))
        f.write("\n")


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


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None or pd.isna(value):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _series_float(series: Any) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _merged_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _col_mean(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return np.nan
    values = pd.to_numeric(df[col], errors="coerce")
    return float(values.mean()) if values.notna().any() else np.nan


def _col_sum(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return np.nan
    values = pd.to_numeric(df[col], errors="coerce")
    return float(values.sum()) if values.notna().any() else np.nan


def _last_text(df: pd.DataFrame, col: str) -> str:
    if col not in df.columns or df.empty:
        return ""
    return str(df[col].iloc[-1])


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

