from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rlmmo.runners.run_f1_f20 import run_f1_f20


def _parse_funcs(text: str) -> list[int]:
    if ":" in text:
        a, b = text.split(":", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--init-method", default="random")
    parser.add_argument("--out", default="results/batch")
    parser.add_argument("--algorithm", default="online_niche_dqn_de_cmaes")
    parser.add_argument("--funcs", default="1:20")
    parser.add_argument("--max-fes", type=int, default=None)
    args = parser.parse_args()
    csv_path, md_path = run_f1_f20(
        runs=args.runs,
        workers=args.workers,
        seed_offset=args.seed_offset,
        init_method=args.init_method,
        out_dir=args.out,
        algorithm=args.algorithm,
        funcs=_parse_funcs(args.funcs),
        max_fes=args.max_fes,
    )
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
