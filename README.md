# RLMMO-Python

在线强化学习辅助多峰优化 Python 研究框架。

当前主线：

```text
阶段一：DBSCAN 小生境 + 共享 Double DQN 辅助 DE，用于提高峰覆盖率
阶段二：DBSCAN 候选峰 + pycma 完整 CMA-ES 精搜，用于提高峰定位精度
```

## 安装

```bash
cd Q:/work/Code/RLMMO-Python
pip install -e .[dev]
```

如果本机无法联网安装 `cma`，阶段二会退化为一个小预算 Gaussian fallback；论文实验应安装 `pycma` 后运行。

## 快速运行

```bash
python scripts/run_one_func.py --func 1 --runs 1 --max-fes 5000
python scripts/run_one_func.py --func 7 --runs 1 --max-fes 10000
python scripts/run_f1_f20.py --runs 10 --workers 12 --out results/online_dqn_de_cmaes_10runs
```

## 当前 PR 改进实验流程

当前主算法与分组实验、机制分析、验收 gate 见：

```text
docs/pr_improvement_workflow.md
```

## 第三方代码

`third_party/cec2013/` 来自 mikeagn/CEC2013 的 Python3 版本，并保留原始 `LICENSE.txt`。项目通过 `rlmmo.benchmarks.cec2013` 做薄封装，不直接修改第三方 benchmark 逻辑。
