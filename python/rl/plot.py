"""训练与评估结果的可视化。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans CJK SC",
                                          "Noto Serif CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else []


def plot_training(output: Path, logs):
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 6.4))
    for label, path in logs:
        history = load_json(path)
        if not history:
            continue
        iterations = [item["iteration"] for item in history]
        axes[0, 0].plot(iterations, [item["mean_error"] for item in history], label=label)
        axes[0, 1].plot(iterations, [item.get("alpha", np.nan) for item in history], label=label)
        evaluated = [item for item in history if "evaluation_early_error" in item]
        if evaluated:
            axes[1, 0].plot([item["iteration"] for item in evaluated],
                            [item["evaluation_early_error"] for item in evaluated],
                            marker="o", markersize=3, label=label + " 早段")
            axes[1, 0].plot([item["iteration"] for item in evaluated],
                            [item["evaluation_late_error"] for item in evaluated],
                            marker="s", markersize=3, linestyle="--", label=label + " 末段")
        axes[1, 1].plot(iterations, [item.get("stability", 0.0) for item in history], label=label)
        axes[1, 1].plot(iterations, [item.get("context_entropy", np.nan) / 100.0
                                     for item in history], ":", label=label + " 熵/100")
    axes[0, 0].set_title("训练期平均误差")
    axes[0, 1].set_title("温度系数")
    axes[1, 0].set_title("评估误差（前四分之一与后四分之一）")
    axes[1, 1].set_title("上下文稳定项与熵")
    for axis in axes.ravel():
        axis.set_xlabel("迭代")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    print("已保存", output)


def plot_episode(output: Path, archive: Path):
    data = np.load(archive)
    angle = data["angle"]
    reference = data["reference"]
    error = data["error"]
    names = ["大 yaw", "roll", "小 yaw", "pitch"]
    figure, axes = plt.subplots(4, 2, figsize=(12.0, 2.6 * 4))
    for joint in range(4):
        axes[joint, 0].plot(reference[0][:, joint], color="C1", linestyle="--", label="参考")
        axes[joint, 0].plot(angle[0][:, joint], color="C0", label="策略")
        axes[joint, 0].set_ylabel(names[joint] + " (rad)")
        axes[joint, 0].grid(True, alpha=0.3)
        axes[joint, 0].legend(fontsize=7)
        axes[joint, 1].plot(np.abs(error[0][:, joint]), color="C3")
        axes[joint, 1].set_ylabel("误差 (rad)")
        axes[joint, 1].grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("时间步")
    axes[-1, 1].set_xlabel("时间步")
    figure.suptitle("跟踪曲线与误差（随机参数回合）")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    print("已保存", output)


def plot_adaptation(output: Path, archives, labels):
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for archive, label in zip(archives, labels):
        data = np.load(archive)
        error = np.abs(data["error"])
        mean = error.mean(axis=(0, 2)) if error.ndim == 3 else error.mean(axis=0)
        axes[0].plot(mean, label=label)
        steps = np.arange(mean.size)
        window = max(1, mean.size // 30)
        smoothed = np.convolve(mean, np.ones(window) / window, mode="valid")
        axes[1].plot(smoothed, label=label)
    axes[0].set_title("回合内平均绝对误差")
    axes[1].set_title("平滑后的适应过程")
    for axis in axes:
        axis.set_xlabel("时间步")
        axis.set_ylabel("平均绝对误差 (rad)")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    print("已保存", output)


def plot_context(output: Path, archive: Path):
    data = np.load(archive)
    context = data["context"]
    parameters = data["parameters"]
    total_mass = parameters[:, 0:4].sum(axis=1)
    centered = context - context.mean(axis=0)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ components[:2].T
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    scatter = axes[0].scatter(projection[:, 0], projection[:, 1], c=total_mass, cmap="viridis")
    figure.colorbar(scatter, ax=axes[0], label="总质量 (kg)")
    axes[0].set_title("收敛上下文的前两个主成分")
    axes[0].set_xlabel("主成分 1")
    axes[0].set_ylabel("主成分 2")
    correlation = []
    for dimension in range(parameters.shape[1]):
        value = parameters[:, dimension]
        if value.std() < 1.0e-9:
            correlation.append(0.0)
            continue
        correlation.append(float(np.corrcoef(value, context[:, 0])[0, 1]))
    axes[1].hist(np.abs(correlation), bins=20, color="C0")
    axes[1].set_title("各参数与上下文第一维的相关系数绝对值")
    axes[1].grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    print("已保存", output)


def main() -> None:
    parser = argparse.ArgumentParser(description="结果绘图")
    parser.add_argument("--mode", default="training")
    parser.add_argument("--output", type=Path, default=Path("data/rl_plot.png"))
    parser.add_argument("--logs", nargs="*", default=[])
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--archives", nargs="*", default=[])
    parser.add_argument("--labels", nargs="*", default=[])
    arguments = parser.parse_args()
    if arguments.mode == "training":
        logs = []
        for item in arguments.logs:
            label, path = item.split("=", 1)
            logs.append((label, Path(path)))
        plot_training(arguments.output, logs)
    elif arguments.mode == "episode":
        plot_episode(arguments.output, arguments.archive)
    elif arguments.mode == "adaptation":
        plot_adaptation(arguments.output, [Path(item) for item in arguments.archives],
                        arguments.labels)
    elif arguments.mode == "context":
        plot_context(arguments.output, arguments.archive)


if __name__ == "__main__":
    main()
