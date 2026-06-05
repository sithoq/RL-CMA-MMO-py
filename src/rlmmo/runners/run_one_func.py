"""单函数多次重复实验 runner。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from tqdm import trange

from rlmmo.algorithms.online_individual_dqn_de_cmaes import run_optimizer as run_individual_dqn
from rlmmo.algorithms.online_niche_dqn_de_cmaes import run_optimizer as run_niche_dqn


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ALGORITHMS: dict[str, Callable[..., dict[str, Any]]] = {
    "online_niche_dqn_de_cmaes": run_niche_dqn,
    "online_individual_dqn_de_cmaes": run_individual_dqn,
}


def resolve_output_dir(out_dir: str | Path) -> Path:
    """统一输出目录解析规则。

    命令行从项目根、scripts 目录或任意目录启动时，相对路径都按项目根目录解析。
    例如 ``--out results/batch`` 永远写入 ``<project>/results/batch``，不会误写到
    ``<project>/scripts/results/batch``。
    """

    path = Path(out_dir)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def run_one_func(
    func_num: int,
    runs: int,
    seed_offset: int = 0,
    init_method: str = "random",
    out_dir: str | Path = "results",
    algorithm: str = "online_niche_dqn_de_cmaes",
    np_size: int | None = None,
    max_fes: int | None = None,
) -> Path:
    """运行某个 CEC2013 函数的多次重复，并保存 CSV/NPZ。"""

    if algorithm not in ALGORITHMS:
        available = ", ".join(sorted(ALGORITHMS))
        raise ValueError(f"unsupported algorithm: {algorithm}; available: {available}")
    optimizer = ALGORITHMS[algorithm]
    out_dir = resolve_output_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    rows: list[dict[str, Any]] = []

    for run_id in trange(1, int(runs) + 1, desc=f"F{func_num:02d}"):
        seed = int(seed_offset + run_id)
        result = optimizer(
            func_num=func_num,
            seed=seed,
            np_size=np_size,
            max_fes=max_fes,
            init_method=init_method,
        )
        pop_path = out_dir / f"F{func_num:02d}_run{run_id:03d}_seed{seed}_{algorithm}_{init_method}_{stamp}.npz"
        np.savez_compressed(
            pop_path,
            final_pop=result.pop("final_pop"),
            final_fitness=result.pop("final_fitness"),
        )
        result["run_id"] = run_id
        result["final_pop_path"] = str(pop_path.resolve())
        rows.append(result)

    df = pd.DataFrame(rows)
    csv_path = out_dir / f"F{func_num:02d}_{runs}runs_seed{seed_offset}_{algorithm}_{init_method}_{stamp}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path.resolve()
