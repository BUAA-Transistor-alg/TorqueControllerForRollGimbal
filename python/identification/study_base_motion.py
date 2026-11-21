"""底盘晃动与滑移容忍性扫描。"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import eval_openloop as open_loop
import identify_physical as ip

CONFIGS = [
    ("baseline", 0.0, 2.0, 0.0, 1.0),
    ("yaw0.5", 0.5, 2.0, 0.0, 1.0),
    ("yaw1", 1.0, 2.0, 0.0, 1.0),
    ("yaw2", 2.0, 2.0, 0.0, 1.0),
    ("yaw5", 5.0, 2.0, 0.0, 1.0),
    ("slip001", 0.0, 2.0, 0.01, 1.0),
    ("slip005", 0.0, 2.0, 0.05, 1.0),
    ("slip01", 0.0, 2.0, 0.10, 1.0),
    ("slip03", 0.0, 2.0, 0.30, 1.0),
    ("mixed", 2.0, 2.0, 0.05, 1.0),
    ("yaw10", 10.0, 2.0, 0.0, 1.0),
    ("slip05", 0.0, 2.0, 0.50, 1.0),
    ("yaw10_slow", 10.0, 0.6, 0.0, 1.0),
]


def usable(raw):
    try:
        ip.build_model(raw)
        return raw
    except RuntimeError:
        return None


def run(tag, yaw_amplitude, yaw_frequency, slip_amplitude, slip_frequency, width):
    clean = Path("data/bm_%s.npz" % tag)
    noisy = Path("data/bm_%s_noisy.npz" % tag)
    subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "acquisition"
                                     / "collect_base_motion.py"), "--yaw-amplitude",
                    str(yaw_amplitude), "--yaw-frequency", str(yaw_frequency),
                    "--slip-amplitude", str(slip_amplitude), "--slip-frequency",
                    str(slip_frequency), "--clean", str(clean), "--noisy", str(noisy)],
                   check=True, stdout=subprocess.DEVNULL)
    save = Path("data/bm_%s_model.npz" % tag)
    subprocess.run([sys.executable, str(Path(__file__).with_name("regressor_full.py")),
                    "--data", str(noisy),
                    "--segments", "100", "--save", str(save), "--mode", "corrected",
                    "--width", str(width), "--order", "5", "--stride", "3"], check=True,
                   stdout=subprocess.DEVNULL)
    raw = usable(np.load(save)["model"])
    if raw is None:
        repaired = save.with_name(save.stem + "_repaired.npz")
        subprocess.run([sys.executable, str(Path(__file__).with_name("finalize_model.py")),
                        "--identified", str(save),
                        "--save", str(repaired), "--total-mass", "0",
                        "--inertia-floor", "3e-5"], check=True, stdout=subprocess.DEVNULL)
        raw = np.load(repaired)["model"]
    metrics = open_loop.evaluate(raw)
    return {"tag": tag, "yaw_amplitude_deg": yaw_amplitude, "slip_amplitude": slip_amplitude,
            "rmse50": metrics["rmse"][50].tolist(),
            "direction": metrics["direction_wrong"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="底盘运动容忍性扫描")
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--width", type=int, default=51)
    arguments = parser.parse_args()
    configs = [config for index, config in enumerate(CONFIGS)
               if index % arguments.parts == arguments.part]
    results = []
    for tag, yaw_amplitude, yaw_frequency, slip_amplitude, slip_frequency in configs:
        record = run(tag, yaw_amplitude, yaw_frequency, slip_amplitude, slip_frequency,
                     arguments.width)
        print("%-9s 偏航摆幅 %4.1f 度，滑移加速度 %.3f m/s^2：RMSE50 %s 方向 %d" % (
            tag, yaw_amplitude, slip_amplitude,
            " ".join("%.2e" % v for v in record["rmse50"]), record["direction"]), flush=True)
        results.append(record)
    Path("data/base_motion_study_%d.json" % arguments.part).write_text(
        json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()