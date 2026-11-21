"""多种子稳健性验证：松弛拟合 + 物理修复回退，报告每个种子的峰值与 0.5 s 开环 RMSE。"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from acquisition import collect_low_excitation as collector
import eval_openloop as open_loop
import identify_physical as ip

CANDIDATES = {
    "B1": {"slow_amp": (0.45, 0.10, 0.15, 0.15), "roll_center": -0.12, "static": 0.15,
           "tilt": 10.0, "fast_amp": (0.10, 0.06, 0.08, 0.08), "fast_freq": 5.0},
    "F3": {"slow_amp": (0.45, 0.05, 0.08, 0.08), "roll_center": -0.08, "static": 0.08,
           "tilt": 14.0, "fast_amp": (0.031, 0.019, 0.025, 0.025), "fast_freq": 9.0},
    "E4": {"slow_amp": (0.45, 0.10, 0.15, 0.15), "roll_center": -0.12, "static": 0.15,
           "tilt": 10.0, "fast_amp": (0.031, 0.019, 0.025, 0.025), "fast_freq": 9.0},
}
SEEDS = (20261260, 20261270)
WIDTH = 31


def apply(config):
    collector.SLOW_AMPLITUDES = list(config["slow_amp"])
    collector.SLOW_FREQUENCIES = [0.5] * 4
    collector.FAST_AMPLITUDES = list(config["fast_amp"])
    collector.FAST_FREQUENCIES = [config["fast_freq"]] * 4
    collector.ROLL_CENTER = config["roll_center"]


def load_usable_model(path: Path):
    raw = np.load(path)["model"]
    try:
        ip.build_model(raw)
        return raw
    except RuntimeError:
        repaired = path.with_name(path.stem + "_repaired.npz")
        subprocess.run([sys.executable, str(Path(__file__).with_name("finalize_model.py")),
                        "--identified", str(path),
                        "--save", str(repaired), "--total-mass", "0",
                        "--inertia-floor", "3e-5"], check=True, stdout=subprocess.DEVNULL)
        return np.load(repaired)["model"]


def run(name, config, seed):
    apply(config)
    clean = Path("data/rv_%s_%d.npz" % (name, seed))
    noisy = Path("data/rv_%s_%d_noisy.npz" % (name, seed))
    collector.collect(100, seed, 1.0, 1.0, config["static"], math.radians(config["tilt"]),
                      1.0, 0.3, clean, noisy, seed + 1)
    payload = np.load(noisy)
    save = Path("data/rv_%s_%d_model.npz" % (name, seed))
    subprocess.run([sys.executable, str(Path(__file__).with_name("refit_relaxed.py")),
                    "--data", str(noisy),
                    "--segments", "100", "--save", str(save), "--mode", "corrected",
                    "--width", str(WIDTH), "--order", "5", "--stride", "3"],
                   check=True, stdout=subprocess.DEVNULL)
    raw = load_usable_model(save)
    metrics = open_loop.evaluate(raw)
    record = {"name": name, "seed": seed,
              "velocity": float(np.abs(payload["qdot_clean"]).max()),
              "torque": float(np.abs(payload["tau_clean"]).max()),
              "acceleration": float(np.abs(np.gradient(payload["qdot_clean"], 0.01,
                                                       axis=1)).max()),
              "rmse50": metrics["rmse"][50].tolist(),
              "direction": metrics["direction_wrong"]}
    print("%-3s 种子 %d：v %.3f  tau %.3f  a %.1f  RMSE50 %s  方向 %d" % (
        name, seed, record["velocity"], record["torque"], record["acceleration"],
        " ".join("%.2e" % v for v in record["rmse50"]), record["direction"]), flush=True)
    return record


def main() -> None:
    records = []
    for seed in SEEDS:
        for name, config in CANDIDATES.items():
            try:
                records.append(run(name, config, seed))
            except Exception as error:
                print("%-3s 种子 %d 失败：%s" % (name, seed, error), flush=True)
    Path("data/low_excitation_seeds.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
