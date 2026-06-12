# RLMMO-Python

在线强化学习辅助多峰优化 Python 研究框架。

当前主线：

```text
阶段一：个体级 Multi-Preference Double DQN 辅助 KNN-DE，用于提高峰覆盖率
阶段一辅助：PeakArchive、conservative injection、archive reseed，用于降低覆盖流失
阶段二：population + archive 经 DBSCAN 提取多 seed，再用 pycma CMA-ES 精搜
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
python scripts/run_f1_f20.py --algorithm online_individual_mpdqn_v2_de_cmaes --func-group core --runs 5 --diagnostics --out results/core_reseed_diagnostics --analyze-mechanism --baseline-dir results/f1_f20_diagnostics --dry-run
```

## 当前 PR 改进实验流程

当前主算法与分组实验、机制分析、验收 gate 见：

```text
docs/pr_improvement_workflow.md
```

## 第三方代码

`third_party/cec2013/` 来自 mikeagn/CEC2013 的 Python3 版本，并保留原始 `LICENSE.txt`。项目通过 `rlmmo.benchmarks.cec2013` 做薄封装，不直接修改第三方 benchmark 逻辑。
