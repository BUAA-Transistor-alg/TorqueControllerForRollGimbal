"""生成更强的采集协议数据：用于高噪声工况下的对照。"""
from __future__ import annotations

import math
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import collect_low_excitation as collector


def main() -> None:
    collector.SLOW_AMPLITUDES = [1.20, 0.32, 0.45, 0.45]
    collector.SLOW_FREQUENCIES = [1.0, 1.0, 1.0, 1.0]
    collector.FAST_AMPLITUDES = [0.28, 0.16, 0.20, 0.20]
    collector.FAST_FREQUENCIES = [5.0, 5.0, 5.0, 5.0]
    collector.ROLL_CENTER = -0.20
    clean = Path("data/strong_data.npz")
    noisy = Path("data/strong_data_noisy.npz")
    collector.collect(100, 20261220, 1.0, 1.0, 0.25, math.radians(20.0), 1.0, 0.3,
                      clean, noisy, 20261221)
    payload = np.load(clean)
    print("强协议峰值速度 %.2f rad/s，峰值力矩 %.2f N·m" % (
        np.abs(payload["qdot"]).max(), np.abs(payload["tau"]).max()))
    for joint in range(4):
        print("  关节 %d 位置 %.2f 到 %.2f" % (
            joint + 1, payload["q"][:, :, joint].min(), payload["q"][:, :, joint].max()))


if __name__ == "__main__":
    main()
