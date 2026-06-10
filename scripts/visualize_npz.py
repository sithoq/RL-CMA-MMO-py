#!/usr/bin/env python
"""可视化 .npz 实验结果：种群分布 与 最终最优解比对。

支持:
  - 1D 函数 (F01-F03): 函数曲线 + 种群散点 + 已知全局最优
  - 2D 函数 (F04-F07, F10-F13): 等高线/热力图 + 种群散点 + 已知全局最优
  - 3D+ 函数: PCA 投影 + 适应度分布 + 平行坐标

用法:
  python scripts/visualize_npz.py results/smoke/F01_*.npz              # 单文件
  python scripts/visualize_npz.py results/individual_knn_1run/          # 目录下所有 npz
  python scripts/visualize_npz.py results/batch/ --compare              # 多文件对比
  python scripts/visualize_npz.py results/batch/ --func F01             # 筛选指定函数
  python scripts/visualize_npz.py results/batch/ --output my_plots/     # 指定输出目录
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm
from scipy.spatial import ConvexHull

# ---- 项目路径 ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "cec2013"))

# ---- 全局最优解位置（F01-F10 解析解，F11-F20 从 data/*_opt.dat 读取） ----

# F01-F10 已知全局最优解 (来自 CEC2013 论文)
KNOWN_OPTIMA: dict[int, np.ndarray] = {
    # F01: Five-Uneven-Peak Trap, x in [0, 30], 2 global peaks at x=0 and x=30
    1: np.array([[0.0], [30.0]]),
    # F02: Equal Maxima, x in [0, 1], 5 global peaks
    2: np.array([[0.1], [0.3], [0.5], [0.7], [0.9]]),
    # F03: Uneven Decreasing Maxima, x in [0, 1], 1 global peak
    3: np.array([[0.0797]]),
    # F04: Himmelblau, 4 global peaks
    4: np.array([
        [3.0, 2.0],
        [3.584428, -1.848126],
        [-2.805118, 3.131312],
        [-3.779310, -3.283186],
    ]),
    # F05: Six-Hump Camel Back, 2 global peaks
    5: np.array([
        [0.089842, -0.712656],
        [-0.089842, 0.712656],
    ]),
    # F06-F09: too many peaks, use how_many_goptima to find them
    # F10: Modified Rastrigin 2D, 12 global peaks (3*4)
    10: np.array([
        [i/6, j/8] for i in range(0, 3) for j in range(0, 4)
    ]),
}

# F11-F20 从 data/*_opt.dat 文件加载
def _load_optima_data(func_num: int, dim: int) -> Optional[np.ndarray]:
    """从 vendor data 目录加载已知全局最优。"""
    data_dir = PROJECT_ROOT / "third_party" / "cec2013" / "data"
    mapping = {
        (11, 2): "CF1_M_D2_opt.dat",
        (12, 2): "CF2_M_D2_opt.dat",
        (13, 2): "CF3_M_D2_opt.dat",
        (14, 3): "CF3_M_D3_opt.dat",
        (15, 3): "CF4_M_D3_opt.dat",
        (16, 5): "CF3_M_D5_opt.dat",
        (17, 5): "CF4_M_D5_opt.dat",
        (18, 10): "CF3_M_D10_opt.dat",
        (19, 10): "CF4_M_D10_opt.dat",
        (20, 20): "CF4_M_D20_opt.dat",
    }
    fname = mapping.get((func_num, dim))
    if fname is None:
        return None
    path = data_dir / fname
    if not path.exists():
        return None
    try:
        arr = np.loadtxt(path, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr
    except Exception:
        return None

for fnum, dim in [(11, 2), (12, 2), (13, 2), (14, 3), (15, 3),
                   (16, 5), (17, 5), (18, 10), (19, 10), (20, 20)]:
    opt = _load_optima_data(fnum, dim)
    if opt is not None:
        KNOWN_OPTIMA[fnum] = opt


# ---- 函数景观评估 ----

def evaluate_1d_landscape(func_num: int, xs: np.ndarray) -> np.ndarray:
    """对 1D 函数 (F01-F03) 评估景观值。"""
    if func_num == 1:
        return np.array([_f1_five_uneven_peak_trap(x) for x in xs])
    elif func_num == 2:
        return np.sin(5.0 * np.pi * xs) ** 6
    elif func_num == 3:
        return (
            np.exp(-2.0 * np.log(2) * ((xs - 0.08) / 0.854) ** 2)
            * (np.sin(5 * np.pi * (xs ** 0.75 - 0.05))) ** 6
        )
    else:
        # fallback via CEC2013 evaluator
        from cec2013.cec2013 import CEC2013
        import os as _os
        old = _os.getcwd()
        _os.chdir(PROJECT_ROOT / "third_party" / "cec2013")
        try:
            f = CEC2013(func_num)
            return np.array([float(f.evaluate(np.array([x]))) for x in xs])
        finally:
            _os.chdir(old)


def evaluate_2d_grid(func_num: int, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """对 2D 函数 (F04-F13) 评估网格景观值。直接使用解析式或调用 CEC。"""
    from cec2013.cec2013 import CEC2013
    import os as _os

    old = _os.getcwd()
    _os.chdir(PROJECT_ROOT / "third_party" / "cec2013")
    try:
        f = CEC2013(func_num)
        nx, ny = len(xs), len(ys)
        zs = np.zeros((ny, nx))
        for i, y in enumerate(ys):
            for j, x_val in enumerate(xs):
                zs[i, j] = float(f.evaluate(np.array([x_val, y])))
        return zs
    finally:
        _os.chdir(old)


def _f1_five_uneven_peak_trap(x):
    if x < 2.5:
        return 80 * (2.5 - x)
    elif x < 5:
        return 64 * (x - 2.5)
    elif x < 7.5:
        return 64 * (7.5 - x)
    elif x < 12.5:
        return 28 * (x - 7.5)
    elif x < 17.5:
        return 28 * (17.5 - x)
    elif x < 22.5:
        return 32 * (x - 17.5)
    elif x < 27.5:
        return 32 * (27.5 - x)
    else:
        return 80 * (x - 27.5)


# ---- NPZ 加载与解析 ----

def parse_filename(filepath: Path) -> dict:
    """从文件名提取元数据。"""
    name = filepath.stem
    parts = name.split("_")
    info = {"file": filepath}
    # F01_run001_seed1_online_niche_dqn_de_cmaes_random_2026-06-04_15-43-22
    for part in parts:
        if part.startswith("F") and len(part) >= 3:
            info["func_name"] = part
            info["func_num"] = int(part[1:])
        elif part.startswith("run"):
            info["run_id"] = int(part[3:])
        elif part.startswith("seed"):
            info["seed"] = int(part[4:])
    # 提取算法名
    algo_parts = []
    in_algo = False
    for p in parts:
        if p in ("online", "niche", "individual", "dqn", "de", "cmaes"):
            in_algo = True
        if in_algo and p not in ("random", "lhs", "sobol") and not p.startswith("202"):
            algo_parts.append(p)
        else:
            if in_algo and algo_parts:
                break
    info["algorithm"] = "_".join(algo_parts) if algo_parts else "unknown"
    info["init_method"] = "random" if "random" in parts else "unknown"
    return info


def load_npz_data(filepath: Path) -> dict:
    """加载单个 npz 文件。"""
    data = np.load(filepath)
    info = parse_filename(filepath)
    info["final_pop"] = data["final_pop"]
    info["final_fitness"] = data["final_fitness"]
    info["np_size"] = len(data["final_fitness"])
    info["dim"] = data["final_pop"].shape[1]
    return info


# ---- 可视化核心 ----

# 中文字体设置
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_1d(ax, data_list: list[dict], func_num: int):
    """1D 函数可视化：函数曲线 + 种群散点 + 全局最优标记。"""
    info = data_list[0]
    lb = 0.0
    ub = 30.0 if func_num == 1 else 1.0

    # 函数景观
    xs = np.linspace(lb, ub, 1000)
    ys = evaluate_1d_landscape(func_num, xs)
    ax.plot(xs, ys, "k-", linewidth=1.0, alpha=0.5, label="函数景观")

    # 已知全局最优
    if func_num in KNOWN_OPTIMA:
        opts = KNOWN_OPTIMA[func_num].ravel()
        opt_vals = evaluate_1d_landscape(func_num, opts)
        ax.scatter(opts, opt_vals, c="red", marker="*", s=150,
                   zorder=5, edgecolors="darkred", linewidths=0.5,
                   label=f"已知最优 (n={len(opts)})")

    # 每个文件的种群
    colors = cm.tab10(np.linspace(0, 1, max(len(data_list), 1)))
    for i, d in enumerate(data_list):
        pop = d["final_pop"].ravel()
        fit = d["final_fitness"]
        # 对适应度做微小抖动用 jitter 区分重叠点
        fit_jitter = fit + np.random.default_rng(42 + i).uniform(-0.01, 0.01, len(fit))
        ax.scatter(pop, fit_jitter, c=[colors[i]], marker="o", s=25,
                   alpha=0.7, edgecolors="k", linewidths=0.3,
                   label=f"{d.get('run_id', '?')} | seed={d.get('seed', '?')}")

    ax.set_xlabel("x")
    ax.set_ylabel("fitness")
    ax.set_title(f"F{func_num:02d} — {get_func_family(func_num)}")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)


def plot_2d(ax, data_list: list[dict], func_num: int):
    """2D 函数可视化：等高线 + 种群散点 + 全局最优。"""
    info = data_list[0]
    dim = info["dim"]

    # 确定边界
    bounds = {
        4: (-6, 6), 5: ((-1.9, -1.1), (1.9, 1.1)),
        6: (-10, 10), 7: (0.25, 10),
        10: (0, 1), 11: (-5, 5), 12: (-5, 5), 13: (-5, 5),
    }
    b = bounds.get(func_num, (-5, 5))
    if isinstance(b[0], tuple):
        xlim, ylim = b
    else:
        xlim = ylim = b

    # 函数景观等高线
    grid_n = 100
    xs = np.linspace(xlim[0], xlim[1], grid_n)
    ys = np.linspace(ylim[0], ylim[1], grid_n)
    try:
        zs = evaluate_2d_grid(func_num, xs, ys)
        levels = 15
        contour = ax.contourf(xs, ys, zs, levels=levels, cmap="viridis", alpha=0.7)
        plt.colorbar(contour, ax=ax, shrink=0.8, label="fitness")
    except Exception as e:
        print(f"  [WARN] 无法计算 2D 景观: {e}")

    # 已知全局最优
    if func_num in KNOWN_OPTIMA:
        opts = KNOWN_OPTIMA[func_num]
        ax.scatter(opts[:, 0], opts[:, 1], c="red", marker="*", s=120,
                   zorder=5, edgecolors="darkred", linewidths=0.5,
                   label=f"已知最优 (n={len(opts)})")

    # 每个文件的种群
    colors = cm.tab10(np.linspace(0, 1, max(len(data_list), 1)))
    for i, d in enumerate(data_list):
        pop = d["final_pop"]
        ax.scatter(pop[:, 0], pop[:, 1], c=[colors[i]], marker="o", s=20,
                   alpha=0.7, edgecolors="k", linewidths=0.3,
                   label=f"run={d.get('run_id', '?')} seed={d.get('seed', '?')}")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(f"F{func_num:02d} — {get_func_family(func_num)} (dim={dim})")
    ax.legend(fontsize=7, loc="best")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def plot_high_dim(ax, data_list: list[dict], func_num: int):
    """高维可视化：PCA 投影到 2D + 适应度颜色映射。"""
    from sklearn.decomposition import PCA

    all_pops = []
    all_fits = []
    labels = []
    colors_list = []

    base_colors = cm.tab10(np.linspace(0, 1, max(len(data_list), 1)))

    for i, d in enumerate(data_list):
        all_pops.append(d["final_pop"])
        all_fits.append(d["final_fitness"])
        labels.append(f"run={d.get('run_id', '?')}")
        colors_list.append(np.full(len(d["final_fitness"]), i))

    all_pops = np.vstack(all_pops)
    all_fits = np.concatenate(all_fits)
    all_colors = np.concatenate(colors_list)

    # PCA
    pca = PCA(n_components=2)
    proj = pca.fit_transform(all_pops)

    scatter = ax.scatter(proj[:, 0], proj[:, 1], c=all_fits, cmap="plasma",
                         s=20, alpha=0.7, edgecolors="k", linewidths=0.2)
    plt.colorbar(scatter, ax=ax, shrink=0.8, label="fitness")

    # 如果已知最优在 2D 内, 也投影
    if func_num in KNOWN_OPTIMA:
        opts = KNOWN_OPTIMA[func_num]
        if opts.shape[1] == all_pops.shape[1]:
            opts_proj = pca.transform(opts)
            ax.scatter(opts_proj[:, 0], opts_proj[:, 1], c="red", marker="*",
                       s=120, zorder=5, edgecolors="darkred", linewidths=0.5,
                       label=f"已知最优 (n={len(opts)})")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title(f"F{func_num:02d} — {get_func_family(func_num)} (PCA 投影)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)


def plot_fitness_distribution(ax, data_list: list[dict], func_num: int):
    """适应度分布直方图 / KDE 对比。"""
    colors = cm.tab10(np.linspace(0, 1, max(len(data_list), 1)))
    for i, d in enumerate(data_list):
        fits = d["final_fitness"]
        ax.hist(fits, bins=30, alpha=0.4, color=colors[i],
                label=f"run={d.get('run_id', '?')} (n={len(fits)})",
                edgecolor="k", linewidth=0.3)

    # 标记已知最优适应度
    opt_fit = None
    if func_num in KNOWN_OPTIMA:
        from cec2013.cec2013 import CEC2013
        import os as _os
        old = _os.getcwd()
        _os.chdir(PROJECT_ROOT / "third_party" / "cec2013")
        try:
            f = CEC2013(func_num)
            opt_fit = float(f.get_fitness_goptima())
        finally:
            _os.chdir(old)

    if opt_fit is not None:
        ax.axvline(opt_fit, color="red", linestyle="--", linewidth=1.5,
                   label=f"全局最优值 = {opt_fit:.4f}")

    ax.set_xlabel("fitness")
    ax.set_ylabel("频次")
    func_family = get_func_family(func_num)
    ax.set_title(f"F{func_num:02d} — {func_family} 适应度分布")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3, axis="y")


def plot_fitness_scatter(ax, data_list: list[dict], func_num: int):
    """种群个体适应度散点图 (按排名)。"""
    colors = cm.tab10(np.linspace(0, 1, max(len(data_list), 1)))
    for i, d in enumerate(data_list):
        fits = np.sort(d["final_fitness"])[::-1]  # 降序
        ax.scatter(range(len(fits)), fits, c=[colors[i]], s=8, alpha=0.6,
                   label=f"run={d.get('run_id', '?')} (top3: {fits[:3].round(4)})")

    opt_fit = get_opt_fitness(func_num)
    if opt_fit is not None:
        ax.axhline(opt_fit, color="red", linestyle="--", linewidth=1.0, alpha=0.6)

    ax.set_xlabel("个体排名")
    ax.set_ylabel("fitness")
    ax.set_title(f"F{func_num:02d} 适应度排序")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)


def get_func_family(func_num: int) -> str:
    table = {
        1: "Five-Uneven-Peak Trap", 2: "Equal Maxima",
        3: "Uneven Decreasing Maxima", 4: "Himmelblau",
        5: "Six-Hump Camel Back", 6: "Shubert 2D", 7: "Vincent 2D",
        8: "Shubert 3D", 9: "Vincent 3D", 10: "Modified Rastrigin",
        11: "Composition 1", 12: "Composition 2", 13: "Composition 3",
        14: "Composition 3 3D", 15: "Composition 4 3D",
        16: "Composition 3 5D", 17: "Composition 4 5D",
        18: "Composition 3 10D", 19: "Composition 4 10D",
        20: "Composition 4 20D",
    }
    return table.get(func_num, "Unknown")


def get_opt_fitness(func_num: int) -> Optional[float]:
    """获取函数的全局最优点适应度值。"""
    opts = {
        1: 200.0, 2: 1.0, 3: 1.0, 4: 200.0,
        5: 1.031628453489877, 6: 186.7309088310239,
        7: 1.0, 8: 2709.093505572820, 9: 1.0,
        10: -2.0, 11: 0.0, 12: 0.0, 13: 0.0,
        14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0,
        18: 0.0, 19: 0.0, 20: 0.0,
    }
    return opts.get(func_num)


# ---- 主可视化入口 ----

def visualize_files(npz_files: list[Path], output_dir: Optional[Path] = None):
    """对所有 npz 文件进行可视化。

    按函数编号分组，每组生成一张图。
    """
    if not npz_files:
        print("错误: 没有找到 npz 文件。")
        return

    # 加载所有数据
    all_data = []
    for fp in npz_files:
        try:
            d = load_npz_data(fp)
            all_data.append(d)
            print(f"  加载: {fp.name}  →  F{d['func_num']:02d}, dim={d['dim']}, "
                  f"NP={d['np_size']}, fitness∈[{d['final_fitness'].min():.4f}, {d['final_fitness'].max():.4f}]")
        except Exception as e:
            print(f"  [错误] 无法加载 {fp}: {e}")

    if not all_data:
        print("没有成功加载任何数据。")
        return

    # 按函数编号分组
    from collections import defaultdict
    groups = defaultdict(list)
    for d in all_data:
        groups[d["func_num"]].append(d)

    print(f"\n共 {len(all_data)} 个文件，{len(groups)} 个函数。\n")

    if output_dir is None:
        output_dir = PROJECT_ROOT / "results" / "plots"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for func_num, data_list in sorted(groups.items()):
        dim = data_list[0]["dim"]
        n_files = len(data_list)
        func_family = get_func_family(func_num)

        print(f"  绘制 F{func_num:02d} ({func_family}, dim={dim}, {n_files} 个文件)")

        if dim == 1:
            fig = plt.figure(figsize=(14, 8))
            gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
            ax1 = fig.add_subplot(gs[0, :])      # 函数景观+种群
            ax2 = fig.add_subplot(gs[1, 0])       # 适应度分布
            ax3 = fig.add_subplot(gs[1, 1])       # 适应度排序
            plot_1d(ax1, data_list, func_num)
            plot_fitness_distribution(ax2, data_list, func_num)
            plot_fitness_scatter(ax3, data_list, func_num)
            fig.suptitle(f"F{func_num:02d} — {func_family} (1D)  种群与最优解可视化",
                         fontsize=14, fontweight="bold")

        elif dim == 2:
            fig = plt.figure(figsize=(16, 12))
            gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
            ax1 = fig.add_subplot(gs[0, :2])      # 等高线+种群
            ax2 = fig.add_subplot(gs[0, 2])        # 适应度分布
            ax3 = fig.add_subplot(gs[1, 0])        # 适应度排序
            ax4 = fig.add_subplot(gs[1, 1])        # 维度1散点
            ax5 = fig.add_subplot(gs[1, 2])        # 维度2散点
            plot_2d(ax1, data_list, func_num)
            plot_fitness_distribution(ax2, data_list, func_num)
            plot_fitness_scatter(ax3, data_list, func_num)
            # 边际散点
            colors = cm.tab10(np.linspace(0, 1, max(len(data_list), 1)))
            for i, d in enumerate(data_list):
                pop = d["final_pop"]
                fits = d["final_fitness"]
                ax4.scatter(pop[:, 0], fits, c=[colors[i]], s=8, alpha=0.5)
                ax5.scatter(pop[:, 1], fits, c=[colors[i]], s=8, alpha=0.5)
            opt_fit = get_opt_fitness(func_num)
            for ax_ in (ax4, ax5):
                if opt_fit is not None:
                    ax_.axhline(opt_fit, color="red", linestyle="--", linewidth=1, alpha=0.5)
                ax_.set_ylabel("fitness")
                ax_.grid(True, alpha=0.3)
            ax4.set_xlabel("x1")
            ax4.set_title("x1 vs fitness")
            ax5.set_xlabel("x2")
            ax5.set_title("x2 vs fitness")
            fig.suptitle(f"F{func_num:02d} — {func_family} (2D)  种群与最优解可视化",
                         fontsize=14, fontweight="bold")

        else:
            fig = plt.figure(figsize=(16, 10))
            gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
            ax1 = fig.add_subplot(gs[0, :2])      # PCA投影
            ax2 = fig.add_subplot(gs[0, 2])        # 适应度分布
            ax3 = fig.add_subplot(gs[1, 0])        # 适应度排序
            ax4 = fig.add_subplot(gs[1, 1])        # Top-N 适应度柱状图
            ax5 = fig.add_subplot(gs[1, 2])        # 统计信息文本
            plot_high_dim(ax1, data_list, func_num)
            plot_fitness_distribution(ax2, data_list, func_num)
            plot_fitness_scatter(ax3, data_list, func_num)

            # Top-N 柱状图
            colors = cm.tab10(np.linspace(0, 1, max(len(data_list), 1)))
            bar_width = 0.8 / len(data_list)
            top_n = min(10, min(len(d["final_fitness"]) for d in data_list))
            for i, d in enumerate(data_list):
                top_fits = np.sort(d["final_fitness"])[::-1][:top_n]
                x_pos = np.arange(top_n) + i * bar_width
                ax4.bar(x_pos, top_fits, bar_width, color=colors[i], alpha=0.7,
                        label=f"run={d.get('run_id', '?')}")
            opt_fit = get_opt_fitness(func_num)
            if opt_fit is not None:
                ax4.axhline(opt_fit, color="red", linestyle="--", linewidth=1.0)
            ax4.set_xlabel("排名")
            ax4.set_ylabel("fitness")
            ax4.set_title(f"Top-{top_n} 适应度")
            ax4.set_xticks(np.arange(top_n) + bar_width * (len(data_list) - 1) / 2)
            ax4.set_xticklabels([str(i + 1) for i in range(top_n)])
            ax4.legend(fontsize=7)
            ax4.grid(True, alpha=0.3, axis="y")

            # 统计信息
            ax5.axis("off")
            stats_text = f"F{func_num:02d} - {func_family}\n"
            stats_text += f"dim: {dim}\n"
            stats_text += f"fopt: {get_opt_fitness(func_num)}\n"
            if func_num in KNOWN_OPTIMA:
                stats_text += f"known optima: {len(KNOWN_OPTIMA[func_num])}\n"
            stats_text += "\n"
            for i, d in enumerate(data_list):
                fits = d["final_fitness"]
                stats_text += f"run={d.get('run_id', '?')}:\n"
                stats_text += f"  NP={d['np_size']}, best={fits.max():.4f}\n"
                stats_text += f"  mean={fits.mean():.4f}, std={fits.std():.4f}\n"
                n_top = np.sum(fits >= (get_opt_fitness(func_num) or 0) - 1e-3)
                stats_text += f"  >=opt: {n_top}/{len(fits)}\n"
            ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes,
                     fontsize=9, verticalalignment="top", fontfamily="monospace",
                     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

            fig.suptitle(f"F{func_num:02d} — {func_family} ({dim}D)  种群与最优解可视化",
                         fontsize=14, fontweight="bold")

        out_path = output_dir / f"F{func_num:02d}_{func_family.replace(' ', '_')}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"    保存: {out_path}")

    print(f"\n完成！所有图片已保存到: {output_dir}")


# ---- 对比模式：多文件/多算法对比 ----

def compare_algorithms(npz_files: list[Path], output_dir: Optional[Path] = None):
    """对比不同算法/运行在相同函数上的表现。"""
    if not npz_files:
        return

    all_data = [load_npz_data(fp) for fp in npz_files]

    from collections import defaultdict
    groups = defaultdict(list)
    for d in all_data:
        groups[d["func_num"]].append(d)

    if output_dir is None:
        output_dir = PROJECT_ROOT / "results" / "plots"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for func_num, data_list in sorted(groups.items()):
        dim = data_list[0]["dim"]
        func_family = get_func_family(func_num)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"F{func_num:02d} — {func_family}  多文件对比",
                     fontsize=14, fontweight="bold")

        # 1. 适应度箱线图
        ax = axes[0, 0]
        labels = [f"run{d.get('run_id','?')}" for d in data_list]
        fit_data = [d["final_fitness"] for d in data_list]
        bp = ax.boxplot(fit_data, tick_labels=labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], cm.tab10(np.linspace(0, 1, len(data_list)))):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        opt_fit = get_opt_fitness(func_num)
        if opt_fit is not None:
            ax.axhline(opt_fit, color="red", linestyle="--", linewidth=1.5)
        ax.set_ylabel("fitness")
        ax.set_title("适应度箱线图")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3, axis="y")

        # 2. 最佳个体对比
        ax = axes[0, 1]
        x_pos = np.arange(len(data_list))
        bests = [d["final_fitness"].max() for d in data_list]
        means = [d["final_fitness"].mean() for d in data_list]
        width = 0.35
        ax.bar(x_pos - width/2, bests, width, label="最佳", color="steelblue", alpha=0.8)
        ax.bar(x_pos + width/2, means, width, label="均值", color="coral", alpha=0.8)
        if opt_fit is not None:
            ax.axhline(opt_fit, color="red", linestyle="--", linewidth=1.0)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_ylabel("fitness")
        ax.set_title("最佳 vs 均值适应度")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # 3. 适应度分布 (重叠)
        ax = axes[0, 2]
        for i, d in enumerate(data_list):
            ax.hist(d["final_fitness"], bins=25, alpha=0.4, color=cm.tab10(i),
                    label=labels[i], edgecolor="k", linewidth=0.3)
        if opt_fit is not None:
            ax.axvline(opt_fit, color="red", linestyle="--", linewidth=1.5)
        ax.set_xlabel("fitness")
        ax.set_ylabel("频次")
        ax.set_title("适应度分布重叠")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

        # 4. 如果 dim<=2，绘制种群对比散点
        if dim == 1:
            ax = axes[1, 0]
            for i, d in enumerate(data_list):
                pop = d["final_pop"].ravel()
                fits = d["final_fitness"]
                ax.scatter(pop, fits, c=[cm.tab10(i)], s=15, alpha=0.6,
                           edgecolors="k", linewidths=0.2, label=labels[i])
            if func_num in KNOWN_OPTIMA:
                opts = KNOWN_OPTIMA[func_num].ravel()
                opt_fit_val = get_opt_fitness(func_num)
                if opt_fit_val is not None:
                    ax.scatter(opts, [opt_fit_val]*len(opts), c="red", marker="*", s=100,
                               zorder=5, label="全局最优")
            ax.set_xlabel("x")
            ax.set_ylabel("fitness")
            ax.set_title("种群分布 (1D)")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        elif dim == 2:
            ax = axes[1, 0]
            for i, d in enumerate(data_list):
                pop = d["final_pop"]
                ax.scatter(pop[:, 0], pop[:, 1], c=[cm.tab10(i)], s=15, alpha=0.6,
                           edgecolors="k", linewidths=0.2, label=labels[i])
            if func_num in KNOWN_OPTIMA:
                opts = KNOWN_OPTIMA[func_num]
                ax.scatter(opts[:, 0], opts[:, 1], c="red", marker="*", s=100,
                           zorder=5, label="全局最优")
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.set_title("种群分布 (2D)")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        else:
            # PCA 投影
            from sklearn.decomposition import PCA
            ax = axes[1, 0]
            all_pops = np.vstack([d["final_pop"] for d in data_list])
            all_labels = np.concatenate([np.full(len(d["final_pop"]), i) for i, d in enumerate(data_list)])
            pca = PCA(n_components=2)
            proj = pca.fit_transform(all_pops)
            for i in range(len(data_list)):
                mask = all_labels == i
                ax.scatter(proj[mask, 0], proj[mask, 1], c=[cm.tab10(i)], s=15,
                           alpha=0.6, edgecolors="k", linewidths=0.2, label=labels[i])
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("PCA 投影")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        # 5. 与最优值差距
        ax = axes[1, 1]
        if opt_fit is not None:
            gaps = [np.abs(d["final_fitness"] - opt_fit) for d in data_list]
            for i, gap in enumerate(gaps):
                ax.hist(gap, bins=25, alpha=0.4, color=cm.tab10(i),
                        label=labels[i], edgecolor="k", linewidth=0.3)
            ax.set_xlabel("|fitness - opt|")
            ax.set_ylabel("频次")
            ax.set_title("与全局最优的差距分布")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3, axis="y")
        else:
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center", va="center")
            ax.set_title("与全局最优的差距 (无数据)")

        # 6. 统计信息表
        ax = axes[1, 2]
        ax.axis("off")
        stats_text = f"F{func_num:02d} - {func_family} (dim={dim})\n"
        stats_text += f"fopt (global optimum): {opt_fit}\n"
        stats_text += f"known optima count: {len(KNOWN_OPTIMA.get(func_num, []))}\n\n"
        for i, d in enumerate(data_list):
            fits = d["final_fitness"]
            stats_text += f"run={d.get('run_id','?')} seed={d.get('seed','?')}:\n"
            stats_text += f"  algo={d.get('algorithm','?')[:30]}\n"
            stats_text += f"  NP={d['np_size']}, dim={d['dim']}\n"
            stats_text += f"  best={fits.max():.6f}\n"
            stats_text += f"  mean={fits.mean():.4f} +- {fits.std():.4f}\n"
            stats_text += f"  #near_opt(tol 1e-3)={np.sum(np.abs(fits-(opt_fit or 0))<1e-3)}/{len(fits)}\n\n"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

        out_path = output_dir / f"compare_F{func_num:02d}_{func_family.replace(' ', '_')}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  对比图保存: {out_path}")


# ---- CLI ----

def main():
    parser = argparse.ArgumentParser(
        description="可视化 RLMMO .npz 实验结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/visualize_npz.py results/smoke/F01_*.npz
  python scripts/visualize_npz.py results/individual_knn_1run/
  python scripts/visualize_npz.py results/individual_knn_1run/ --func F01
  python scripts/visualize_npz.py results/batch/ --compare
        """,
    )
    parser.add_argument("inputs", nargs="+", type=str,
                        help="npz 文件路径 或 包含 npz 文件的目录")
    parser.add_argument("--func", type=str, default=None,
                        help="仅处理指定函数 (如 F01, F02)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出目录 (默认: results/plots/)")
    parser.add_argument("--compare", "-c", action="store_true",
                        help="多文件对比模式")
    parser.add_argument("--dpi", type=int, default=150,
                        help="输出图片 DPI (默认: 150)")

    args = parser.parse_args()

    # 收集所有 npz 文件
    npz_files: list[Path] = []
    for inp in args.inputs:
        p = Path(inp)
        if p.is_dir():
            npz_files.extend(sorted(p.glob("*.npz")))
        elif "*" in inp or "?" in inp:
            # glob pattern
            npz_files.extend(sorted(Path(".").glob(inp)))
        elif p.is_file():
            npz_files.append(p)
        else:
            print(f"  警告: 找不到 {inp}")

    if not npz_files:
        print("错误: 未找到任何 .npz 文件。")
        sys.exit(1)

    # 去重
    npz_files = sorted(set(npz_files))

    # 按函数筛选
    if args.func:
        func_num = int(args.func.replace("F", "").replace("f", ""))
        npz_files = [f for f in npz_files if f"F{func_num:02d}" in f.name]
        if not npz_files:
            print(f"错误: 未找到 F{func_num:02d} 的文件。")
            sys.exit(1)

    print(f"找到 {len(npz_files)} 个 npz 文件")

    output_dir = Path(args.output) if args.output else None

    if args.compare:
        compare_algorithms(npz_files, output_dir)
    else:
        visualize_files(npz_files, output_dir)


if __name__ == "__main__":
    main()
