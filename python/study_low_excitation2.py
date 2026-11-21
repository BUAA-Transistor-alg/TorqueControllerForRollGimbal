"""低激励协议第二轮：慢分量负责姿态行程、快分量负责加速度，分别控制幅值。

每组配置：生成 100 段 3 s 数据 -> 拟合 -> 在标准 100 段上评估 0.5 s 开环 RMSE。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

import collect_low_excitation as collector
import eval_openloop as open_loop

CONFIGS = [
    {"name": "F1", "slow_amp": (0.45, 0.035, 0.06, 0.06), "slow_freq": (0.5, 0.5, 0.5, 0.5),
     "fast_amp": (0.031, 0.019, 0.025, 0.025), "fast_freq": (9.0, 9.0, 9.0, 9.0),
     "tilt": 10.0, "roll_center": -0.07, "static": 0.06, "widths": (21, 31)},
    {"name": "F2", "slow_amp": (0.45, 0.025, 0.04, 0.04), "slow_freq": (0.5, 0.5, 0.5, 0.5),
     "fast_amp": (0.031, 0.019, 0.025, 0.025), "fast_freq": (9.0, 9.0, 9.0, 9.0),
     "tilt": 10.0, "roll_center": -0.06, "static": 0.05, "widths": (21, 31)},
    {"name": "F3", "slow_amp": (0.45, 0.05, 0.08, 0.08), "slow_freq": (0.5, 0.5, 0.5, 0.5),
     "fast_amp": (0.031, 0.019, 0.025, 0.025), "fast_freq": (9.0, 9.0, 9.0, 9.0),
     "tilt": 14.0, "roll_center": -0.08, "static": 0.08, "widths": (21, 31)},
]


def run(config, segments=100, seed=20261220):
    collector.SLOW_AMPLITUDES = list(config["slow_amp"])
    collector.SLOW_FREQUENCIES = list(config["slow_freq"])
    collector.FAST_AMPLITUDES = list(config["fast_amp"])
    collector.FAST_FREQUENCIES = list(config["fast_freq"])
    collector.ROLL_CENTER = config["roll_center"]
    clean = Path("data/low2_%s.npz" % config["name"])
    noisy = Path("data/low2_%s_noisy.npz" % config["name"])
    collector.collect(segments, seed, 1.0, 1.0, config["static"],
                      math.radians(config["tilt"]), 1.0, 0.3, clean, noisy, seed + 1)
    payload = np.load(noisy)
    velocity_peak = float(np.abs(payload["qdot_clean"]).max())
    torque_peak = float(np.abs(payload["tau_clean"]).max())
    acceleration = np.abs(np.gradient(payload["qdot_clean"], 0.01, axis=1)).max()
    import regressor_full as rf
    best = None
    for width in config["widths"]:
        save = Path("data/low2_%s_w%d.npz" % (config["name"], width))
        sys.argv = ["regressor", "--data", str(noisy), "--segments", str(segments),
                    "--save", str(save), "--mode", "corrected", "--width", str(width),
                    "--order", "5", "--stride", "3"]
        try:
            rf.main()
        except AssertionError as error:
            print("  宽度 %d 拟合失败：%s" % (width, error), flush=True)
            continue
        metrics = open_loop.evaluate(np.load(save)["model"])
        record = {"width": width, "rmse50": metrics["rmse"][50].tolist(),
                  "max50": metrics["maximum"][50].tolist(),
                  "direction": metrics["direction_wrong"]}
        if best is None or max(record["rmse50"]) < max(best["rmse50"]):
            best = record
    return {"config": config["name"], "velocity_peak": velocity_peak,
            "torque_peak": torque_peak, "acceleration_peak": float(acceleration),
            "best": best}


def main() -> None:
    results = []
    for config in CONFIGS:
        print("### 配置 %s" % config["name"], flush=True)
        record = run(config)
        results.append(record)
        if record["best"] is not None:
            print("  峰值速度 %.3f rad/s，峰值力矩 %.3f N·m，峰值加速度 %.1f rad/s^2" % (
                record["velocity_peak"], record["torque_peak"], record["acceleration_peak"]))
            print("  宽度 %d：50 步 RMSE %s，方向错 %d" % (
                record["best"]["width"],
                " ".join("%.2e" % v for v in record["best"]["rmse50"]),
                record["best"]["direction"]), flush=True)
    Path("data/low_excitation_study5.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()