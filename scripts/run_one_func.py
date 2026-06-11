from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rlmmo.runners.run_one_func import run_one_func


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--func", type=int, required=True)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--init-method", default="random")
    parser.add_argument("--out", default="results")
    parser.add_argument("--algorithm", default="online_niche_dqn_de_cmaes")
    parser.add_argument("--np-size", type=int, default=None)
    parser.add_argument("--max-fes", type=int, default=None)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--diagnostic-interval", type=int, default=1)
    parser.add_argument("--save-pop-snapshots", action="store_true")
    parser.add_argument("--algorithm-config-json", default=None)
    args = parser.parse_args()
    algorithm_config = json.loads(args.algorithm_config_json) if args.algorithm_config_json else None
    path = run_one_func(
        args.func,
        args.runs,
        args.seed_offset,
        args.init_method,
        args.out,
        args.algorithm,
        args.np_size,
        args.max_fes,
        args.diagnostics,
        args.diagnostic_interval,
        args.save_pop_snapshots,
        algorithm_config,
    )
    print(path)


if __name__ == "__main__":
    main()
