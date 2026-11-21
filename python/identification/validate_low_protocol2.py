"""对候选协议做多种子验证（修复 Path 类型）。"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from acquisition import collect_low_excitation as collector
import eval_openloop as open_loop

CANDIDATES = {
    "F3": {"slow_amp": (0.45, 0.05, 0.08, 0.08), "roll_center": -0.08, "static": 0.08,
           "tilt": 14.0},
    "E4": {"slow_amp": (0.45, 0.10, 0.15, 0.15), "roll_center": -0.12, "static": 0.15,
           "tilt": 10.0},
}


def apply(config):
    collector.SLOW_AMPLITUDES = list(config["slow_amp"])
    collector.SLOW_FREQUENCIES = [0.5] * 4
    collector.FAST_AMPLITUDES = [0.031, 0.019, 0.025, 0.025]
    collector.FAST_FREQUENCIES = [9.0] * 4
    collector.ROLL_CENTER = config["roll_center"]


def run(name, config, seed, width=21):
    apply(config)
    clean = Path("data/val2_%s_%d.npz" % (name, seed))
    noisy = Path("data/val2_%s_%d_noisy.npz" % (name, seed))
    collector.collect(100, seed, 1.0, 1.0, config["static"], math.radians(config["tilt"]),
                      1.0, 0.3, clean, noisy, seed + 1)
    payload = np.load(noisy)
    print("%s 种子 %d：v %.3f  tau %.3f" % (
        name, seed, np.abs(payload["qdot_clean"]).max(), np.abs(payload["tau_clean"]).max()),
        flush=True)
    import subprocess
    save = Path("data/val2_%s_%d_w%d.npz" % (name, seed, width))
    subprocess.run([sys.executable, str(Path(__file__).with_name("refit_relaxed.py")),
                    "--data", str(noisy),
                    "--segments", "100", "--save", str(save), "--mode", "corrected",
                    "--width", str(width), "--order", "5", "--stride", "3"],
                   check=True, stdout=subprocess.DEVNULL)
    metrics = open_loop.evaluate(np.load(save)["model"])
    print("    宽度 %d：RMSE50 %s  方向 %d" % (
        width, " ".join("%.2e" % v for v in metrics["rmse"][50]),
        metrics["direction_wrong"]), flush=True)


def main() -> None:
    for seed in (20261260, 20261270):
        run("F3", CANDIDATES["F3"], seed)
        run("E4", CANDIDATES["E4"], seed)


if __name__ == "__main__":
    main()