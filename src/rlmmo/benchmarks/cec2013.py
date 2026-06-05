"""CEC2013 多峰优化测试集封装。

本模块只做一层很薄的适配：底层函数、全局最优统计和元数据仍来自
``third_party/cec2013`` 中 vendor 的 mikeagn/CEC2013 Python 实现。
这里统一处理路径、边界、FES 上限、期望峰数和 PR/SR 统计入口。
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
import os
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CEC_ROOT = PROJECT_ROOT / "third_party" / "cec2013"
if str(CEC_ROOT) not in sys.path:
    sys.path.insert(0, str(CEC_ROOT))

from cec2013.cec2013 import CEC2013, how_many_goptima  # type: ignore  # noqa: E402


class _ScalarEvaluateProxy:
    """给 how_many_goptima 使用的代理，保证 evaluate 一定返回 Python float。"""

    def __init__(self, func):
        self._func = func

    def evaluate(self, x):
        return float(np.asarray(self._func.evaluate(x)).reshape(-1)[0])

    def __getattr__(self, name):
        return getattr(self._func, name)

# CEC2013 MMO 常用官方/建议种群规模。后续实验也可以在 runner 中覆盖。
NP_TABLE: dict[int, int] = {
    1: 80,
    2: 80,
    3: 80,
    4: 80,
    5: 80,
    6: 100,
    7: 300,
    8: 300,
    9: 300,
    10: 100,
    11: 200,
    12: 200,
    13: 200,
    14: 200,
    15: 200,
    16: 200,
    17: 200,
    18: 200,
    19: 200,
    20: 200,
}

FAMILY_TABLE: dict[int, str] = {
    1: "Five-Uneven-Peak Trap",
    2: "Equal Maxima",
    3: "Uneven Decreasing Maxima",
    4: "Himmelblau",
    5: "Six-Hump Camel Back",
    6: "Shubert 2D",
    7: "Vincent 2D",
    8: "Shubert 3D",
    9: "Vincent 3D",
    10: "Modified Rastrigin",
    11: "Composition 1",
    12: "Composition 2",
    13: "Composition 3",
    14: "Composition 3 3D",
    15: "Composition 4 3D",
    16: "Composition 3 5D",
    17: "Composition 4 5D",
    18: "Composition 3 10D",
    19: "Composition 4 10D",
    20: "Composition 4 20D",
}


@dataclass(frozen=True)
class CEC2013Info:
    """测试函数元数据，便于写入 CSV 和实验日志。"""

    func_num: int
    func_name: str
    family: str
    dimension: int
    lb: list[float]
    ub: list[float]
    max_fes: int
    expected_peaks: int
    recommended_np: int
    rho: float
    fopt: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@contextmanager
def _cec_workdir():
    """CEC vendor 代码用相对路径读取 data，因此调用时临时切到 vendor 根目录。"""

    old = Path.cwd()
    os.chdir(CEC_ROOT)
    try:
        yield
    finally:
        os.chdir(old)


class CEC2013Problem:
    """CEC2013 单个函数实例。

    注意：本类不自动累计 FES。优化算法负责每次 evaluate 后自己增加 FES，
    这样 count_goptima 的后验统计不会污染优化过程预算。
    """

    def __init__(self, func_num: int):
        if not 1 <= int(func_num) <= 20:
            raise ValueError(f"func_num must be in [1, 20], got {func_num}")
        self.func_num = int(func_num)
        with _cec_workdir():
            self._func = CEC2013(self.func_num)
        self.dim = int(self._func.get_dimension())
        self.lb = np.array([self._func.get_lbound(i) for i in range(self.dim)], dtype=float)
        self.ub = np.array([self._func.get_ubound(i) for i in range(self.dim)], dtype=float)
        self.max_fes = int(self._func.get_maxfes())
        self.expected_peaks = int(self._func.get_no_goptima())
        self.rho = float(self._func.get_rho())
        self.fopt = float(self._func.get_fitness_goptima())
        self.recommended_np = int(NP_TABLE[self.func_num])
        self.family = FAMILY_TABLE[self.func_num]

    def evaluate(self, x: np.ndarray) -> float:
        """评价单个个体。CEC 原实现最大化 fitness。"""

        arr = np.asarray(x, dtype=float).reshape(-1)
        if arr.size != self.dim:
            raise ValueError(f"x dimension mismatch: expected {self.dim}, got {arr.size}")
        with _cec_workdir():
            return float(np.asarray(self._func.evaluate(arr)).reshape(-1)[0])

    def evaluate_batch(self, pop: np.ndarray) -> np.ndarray:
        """逐个评价种群。保留显式循环，便于算法层精确控制 FES。"""

        pop = np.asarray(pop, dtype=float)
        return np.array([self.evaluate(row) for row in pop], dtype=float)

    def count_goptima(self, pop: np.ndarray, accuracy: float = 1e-4) -> tuple[int, np.ndarray]:
        """统计找到的全局峰数量。返回 (count, seeds)。"""

        pop = np.asarray(pop, dtype=float)
        with _cec_workdir():
            count, seeds = how_many_goptima(pop.copy(), _ScalarEvaluateProxy(self._func), float(accuracy))
        return int(np.asarray(count).reshape(-1)[0]), np.asarray(seeds, dtype=float)

    def info(self) -> CEC2013Info:
        return CEC2013Info(
            func_num=self.func_num,
            func_name=f"F{self.func_num:02d}",
            family=self.family,
            dimension=self.dim,
            lb=self.lb.tolist(),
            ub=self.ub.tolist(),
            max_fes=self.max_fes,
            expected_peaks=self.expected_peaks,
            recommended_np=self.recommended_np,
            rho=self.rho,
            fopt=self.fopt,
        )


def get_problem(func_num: int) -> CEC2013Problem:
    return CEC2013Problem(func_num)


def get_info(func_num: int) -> CEC2013Info:
    return CEC2013Problem(func_num).info()




