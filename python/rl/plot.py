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


def plot_training(output: Path, logs, scales):
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 6.8))
    for label, path in logs:
        history = load_json(path)
        if not history:
            continue
        iterations = [item["iteration"] for item in history]
        smoothed = [item.get("mean_error_smoothed", item["mean_error"]) for item in history]
        axes[0, 0].plot(iterations, smoothed, label=label)
        evaluated = [item for item in history if "evaluation_0.0" in item]
        if evaluated:
            axes[0, 0].plot([item["iteration"] for item in evaluated],
                            [item["evaluation_0.0"]["in_range_error"] for item in evaluated],
                            "o-", markersize=3, label=label + " 固定参数评估")
        axes[0, 1].plot(iterations, [item.get("alpha", np.nan) for item in history],
                        label=label + " 温度")
        axes[0, 1].plot(iterations, [item.get("scale_max", np.nan) for item in history],
                        label=label + " 尺度上限")
        axes[1, 0].plot(iterations, [item.get("behavior_loss", np.nan) for item in history],
                        label=label + " 克隆")
        axes[1, 0].plot(iterations, [item.get("dynamics_loss", np.nan) for item in history],
                        label=label + " 动力学")
        axes[1, 0].plot(iterations, [item.get("parameter_loss", np.nan) for item in history],
                        label=label + " 参数")
        axes[1, 0].plot(iterations, [item.get("load_loss", np.nan) for item in history],
                        label=label + " 负载")
        axes[1, 1].plot(iterations, [item.get("mean_hits", np.nan) for item in history],
                        label=label + " 平均撞限位")
        axes[1, 1].plot(iterations, [item.get("max_hits", np.nan) for item in history],
                        ":", label=label + " 最大撞限位")
    axes[0, 0].set_title("训练与固定参数评估的限位内误差")
    axes[0, 1].set_title("温度系数与参数尺度上限")
    axes[1, 0].set_title("辅助损失")
    axes[1, 1].set_title("每回合撞限位次数")
    for axis in axes.ravel():
        axis.set_xlabel("迭代")
        axis.set_yscale("log")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=6)
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
    figure.suptitle("跟踪曲线与误差")
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


def plot_scale_sweep(output: Path, reports):
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for label, path in reports:
        data = load_json(path)
        if "parameter_scale" not in data:
            continue
        scales = sorted(float(key) for key in data["parameter_scale"])
        errors = [data["parameter_scale"][str(scale)]["in_range_error"] for scale in scales]
        hits = [data["parameter_scale"][str(scale)]["max_hits"] for scale in scales]
        axes[0].plot(scales, errors, "o-", label=label)
        axes[1].plot(scales, hits, "s-", label=label)
        if "compare_parameter_scale" in data:
            errors = [data["compare_parameter_scale"][str(scale)]["in_range_error"]
                      for scale in scales]
            hits = [data["compare_parameter_scale"][str(scale)]["max_hits"] for scale in scales]
            axes[0].plot(scales, errors, "o--", label=label + " 对照基线")
            axes[1].plot(scales, hits, "s--", label=label + " 对照基线")
    axes[0].set_title("参数尺度扫描：限位内误差")
    axes[1].set_title("参数尺度扫描：单回合最大撞限位")
    for axis in axes:
        axis.set_xlabel("参数尺度")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    print("已保存", output)


def main() -> None:
    parser = argparse.ArgumentParser(description="结果绘图")
    parser.add_argument("--mode", default="training")
    parser.add_argument("--output", type=Path, default=Path("data/rl_plot.png"))
    parser.add_argument("--logs", nargs="*", default=[])
    parser.add_argument("--reports", nargs="*", default=[])
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--archives", nargs="*", default=[])
    parser.add_argument("--labels", nargs="*", default=[])
    parser.add_argument("--scales", nargs="*", type=float, default=[])
    arguments = parser.parse_args()

    def pairs(items):
        result = []
        for item in items:
            label, path = item.split("=", 1)
            result.append((label, Path(path)))
        return result

    if arguments.mode == "training":
        plot_training(arguments.output, pairs(arguments.logs), arguments.scales)
    elif arguments.mode == "episode":
        plot_episode(arguments.output, arguments.archive)
    elif arguments.mode == "adaptation":
        plot_adaptation(arguments.output, [Path(item) for item in arguments.archives],
                        arguments.labels)
    elif arguments.mode == "sweep":
        plot_scale_sweep(arguments.output, pairs(arguments.reports))


if __name__ == "__main__":
    main()
