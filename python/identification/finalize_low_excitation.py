"""生成低激励推荐协议的数据、模型与对比报告。

两档协议：
  minimal —— 峰值速度约 0.6 rad/s、峰值力矩约 0.72 到 0.79 N·m
  safe    —— 峰值速度约 1.0 rad/s、峰值力矩约 0.83 到 0.94 N·m，精度余量更大

用法：python3 finalize_low_excitation.py --protocol minimal|safe
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans CJK SC",
                                          "Noto Serif CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np

from acquisition import collect_low_excitation as collector
import eval_openloop as open_loop
import identify_physical as ip

PROTOCOLS = {
    "minimal": {"name": "低激励-最小", "slow_amp": (0.45, 0.05, 0.08, 0.08),
                "roll_center": -0.08, "static": 0.08, "tilt": 14.0,
                "fast_amp": (0.031, 0.019, 0.025, 0.025), "fast_freq": 9.0},
    "safe": {"name": "低激励-稳健", "slow_amp": (0.45, 0.10, 0.15, 0.15),
             "roll_center": -0.12, "static": 0.15, "tilt": 10.0,
             "fast_amp": (0.10, 0.06, 0.08, 0.08), "fast_freq": 5.0},
}
LABELS = ["大 yaw", "roll", "小 yaw", "pitch"]
WIDTH = 31
SEED = 20261220


def main() -> None:
    parser = argparse.ArgumentParser(description="低激励协议交付")
    parser.add_argument("--protocol", choices=("minimal", "safe"), default="minimal")
    arguments = parser.parse_args()
    config = PROTOCOLS[arguments.protocol]
    collector.SLOW_AMPLITUDES = list(config["slow_amp"])
    collector.SLOW_FREQUENCIES = [0.5] * 4
    collector.FAST_AMPLITUDES = list(config["fast_amp"])
    collector.FAST_FREQUENCIES = [config["fast_freq"]] * 4
    collector.ROLL_CENTER = config["roll_center"]
    clean = Path("data/low_excitation_data.npz")
    noisy = Path("data/low_excitation_data_noisy.npz")
    collector.collect(100, SEED, 1.0, 1.0, config["static"], math.radians(config["tilt"]),
                      1.0, 0.3, clean, noisy, SEED + 1)
    model_path = Path("data/low_excitation_model.npz")
    subprocess.run([sys.executable, str(Path(__file__).with_name("regressor_full.py")),
                    "--data", str(noisy),
                    "--segments", "100", "--save", str(model_path), "--mode", "corrected",
                    "--width", str(WIDTH), "--order", "5", "--stride", "3"], check=True,
                   stdout=subprocess.DEVNULL)
    raw = np.load(model_path)["model"]
    try:
        ip.build_model(raw)
    except RuntimeError:
        repaired = Path("data/low_excitation_model_repaired.npz")
        subprocess.run([sys.executable, str(Path(__file__).with_name("finalize_model.py")),
                        "--identified", str(model_path),
                        "--save", str(repaired), "--total-mass", "0",
                        "--inertia-floor", "3e-5"], check=True, stdout=subprocess.DEVNULL)
        raw = np.load(repaired)["model"]
        np.savez_compressed(model_path, model=raw)
        ip.build_model(raw)
    payload = np.load(noisy)
    reference = np.load("data/closed_loop_data_noisy.npz")
    metrics = open_loop.evaluate(raw)
    baseline = open_loop.evaluate(np.load("data/identified_final.npz")["model"])
    record = {
        "protocol": {"slow_amp": list(config["slow_amp"]), "slow_freq": [0.5] * 4,
                     "fast_amp": list(config["fast_amp"]), "fast_freq": config["fast_freq"],
                     "roll_center": config["roll_center"], "static": config["static"],
                     "tilt_degree": config["tilt"], "segments": 100, "points": 300,
                     "dt": 0.01, "fit_width": WIDTH, "seed": SEED},
        "clean_peak_velocity": float(np.abs(payload["qdot_clean"]).max()),
        "clean_peak_torque": float(np.abs(payload["tau_clean"]).max()),
        "clean_peak_acceleration": float(np.abs(np.gradient(payload["qdot_clean"], 0.01,
                                                            axis=1)).max()),
        "reference_clean_peak_velocity": float(np.abs(reference["qdot_clean"]).max()),
        "reference_clean_peak_torque": float(np.abs(reference["tau_clean"]).max()),
        "reference_clean_peak_acceleration": float(
            np.abs(np.gradient(reference["qdot_clean"], 0.01, axis=1)).max()),
        "rmse25": metrics["rmse"][25].tolist(),
        "rmse50": metrics["rmse"][50].tolist(),
        "max50": metrics["maximum"][50].tolist(),
        "direction_wrong": metrics["direction_wrong"],
        "static_direction": {str(key): list(value)
                             for key, value in metrics["static_direction"].items()},
        "baseline_rmse50": baseline["rmse"][50].tolist(),
    }
    Path("data/low_excitation_metrics.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False))
    print(json.dumps(record, indent=2, ensure_ascii=False))
    print(open_loop.format_report(metrics))

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    width = 0.36
    index = np.arange(4)
    for axis, key, label in ((axes[0], "torque", "峰值力矩 (N·m)"),
                             (axes[1], "velocity", "峰值速度 (rad/s)")):
        if key == "torque":
            reference_values = np.abs(reference["tau_clean"]).max(axis=(0, 1))
            low_values = np.abs(payload["tau_clean"]).max(axis=(0, 1))
        else:
            reference_values = np.abs(reference["qdot_clean"]).max(axis=(0, 1))
            low_values = np.abs(payload["qdot_clean"]).max(axis=(0, 1))
        axis.bar(index - width / 2, reference_values, width, label="标准协议", color="C0")
        axis.bar(index + width / 2, low_values, width, label=config["name"], color="C1")
        axis.set_xticks(index)
        axis.set_xticklabels(LABELS)
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3, axis="y")
        axis.legend(fontsize=8)
    figure.suptitle("协议峰值对比")
    figure.tight_layout()
    figure.subplots_adjust(top=0.88)
    figure.savefig("data/low_excitation_comparison.png", dpi=170)
    print("已保存 data/low_excitation_comparison.png")


if __name__ == "__main__":
    main()