from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rlmmo.runners.run_f1_f20 import run_f1_f20
from rlmmo.runners.run_one_func import resolve_output_dir

from scripts.analyze_results import write_mechanism_outputs


FUNC_GROUPS: dict[str, list[int]] = {
    "all": list(range(1, 21)),
    "core": [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13],
    "hard": list(range(14, 21)),
    "high_peak": [8, 9],
    "focus": [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
}


def _parse_funcs(text: str) -> list[int]:
    if ":" in text:
        a, b = text.split(":", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(",") if x.strip()]


def _resolve_funcs(funcs: str, func_group: str | None) -> list[int]:
    if func_group:
        return FUNC_GROUPS[func_group]
    return _parse_funcs(funcs)


def _dry_run_summary(args: argparse.Namespace, funcs: list[int]) -> str:
    max_fes = "benchmark_default" if args.max_fes is None else str(args.max_fes)
    return "\n".join(
        [
            "DRY RUN: no experiments will be executed.",
            f"algorithm={args.algorithm}",
            f"funcs={','.join(str(x) for x in funcs)}",
            f"runs={args.runs}",
            f"workers={args.workers}",
            f"max_fes={max_fes}",
            f"diagnostics={bool(args.diagnostics)}",
            f"analyze_mechanism={bool(args.analyze_mechanism)}",
            f"out={resolve_output_dir(args.out)}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--init-method", default="random")
    parser.add_argument("--out", default="results/batch")
    parser.add_argument("--algorithm", default="online_niche_dqn_de_cmaes")
    parser.add_argument("--funcs", default="1:20")
    parser.add_argument("--func-group", choices=sorted(FUNC_GROUPS), default=None)
    parser.add_argument("--max-fes", type=int, default=None)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--diagnostic-interval", type=int, default=1)
    parser.add_argument("--save-pop-snapshots", action="store_true")
    parser.add_argument("--algorithm-config-json", default=None)
    parser.add_argument("--analyze-mechanism", action="store_true")
    parser.add_argument("--baseline-dir", default=None)
    parser.add_argument("--analysis-prefix", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    algorithm_config = json.loads(args.algorithm_config_json) if args.algorithm_config_json else None
    funcs = _resolve_funcs(args.funcs, args.func_group)
    if args.dry_run:
        print(_dry_run_summary(args, funcs))
        return
    csv_path, md_path = run_f1_f20(
        runs=args.runs,
        workers=args.workers,
        seed_offset=args.seed_offset,
        init_method=args.init_method,
        out_dir=args.out,
        algorithm=args.algorithm,
        funcs=funcs,
        max_fes=args.max_fes,
        diagnostics=args.diagnostics,
        diagnostic_interval=args.diagnostic_interval,
        save_pop_snapshots=args.save_pop_snapshots,
        algorithm_config=algorithm_config,
    )
    print(csv_path)
    print(md_path)
    if args.analyze_mechanism:
        out_dir = Path(csv_path).parent
        analysis_prefix = Path(args.analysis_prefix) if args.analysis_prefix else out_dir / "analysis"
        if not analysis_prefix.is_absolute():
            analysis_prefix = resolve_output_dir(analysis_prefix)
        write_mechanism_outputs(out_dir, analysis_prefix, args.baseline_dir)


if __name__ == "__main__":
    main()
