"""对比标准协议与低激励协议的力矩与速度时间曲线（取一条代表段）。"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans CJK SC",
                                          "Noto Serif CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np

LABELS = ["大 yaw", "roll", "小 yaw", "pitch"]


def main() -> None:
    parser = argparse.ArgumentParser(description="协议对比")
    parser.add_argument("--standard", type=Path, default=Path("data/closed_loop_data_noisy.npz"))
    parser.add_argument("--low", type=Path, default=Path("data/low_excitation_data_noisy.npz"))
    parser.add_argument("--segment", type=int, default=-1,
                        help="段号，-1 表示自动选择速度最大的一段")
    parser.add_argument("--output", type=Path, default=Path("data/low_excitation_timeseries.png"))
    arguments = parser.parse_args()
    standard = np.load(arguments.standard)
    low = np.load(arguments.low)
    segment = arguments.segment
    if segment < 0:
        segment = int(np.argmax(np.abs(low["qdot_clean"]).max(axis=(1, 2))))
    time = np.arange(300) * 0.01
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 6.4), sharex=True)
    for joint in range(4):
        axes[0, 0].plot(time, standard["tau_clean"][segment, :, joint], color="C%d" % joint,
                        linewidth=1.3, label=LABELS[joint])
        axes[0, 1].plot(time, low["tau_clean"][segment, :, joint], color="C%d" % joint,
                        linewidth=1.3, label=LABELS[joint])
        axes[1, 0].plot(time, standard["qdot_clean"][segment, :, joint], color="C%d" % joint,
                        linewidth=1.3)
        axes[1, 1].plot(time, low["qdot_clean"][segment, :, joint], color="C%d" % joint,
                        linewidth=1.3)
    axes[0, 0].set_title("标准协议：力矩")
    axes[0, 1].set_title("低激励协议：力矩")
    axes[1, 0].set_title("标准协议：速度")
    axes[1, 1].set_title("低激励协议：速度")
    for axis in axes.reshape(-1):
        axis.grid(True, alpha=0.3)
        axis.set_xlabel("时间 (s)")
    axes[0, 0].set_ylabel("力矩 (N·m)")
    axes[1, 0].set_ylabel("速度 (rad/s)")
    axes[1, 1].set_ylabel("速度 (rad/s)")
    axes[0, 0].legend(fontsize=8, ncols=2)
    figure.tight_layout()
    figure.savefig(arguments.output, dpi=170)
    print("已保存", arguments.output)


if __name__ == "__main__":
    main()