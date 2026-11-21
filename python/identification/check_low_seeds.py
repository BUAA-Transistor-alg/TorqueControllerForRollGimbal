"""用交付管线（严格物理约束 + 宽度 31）检查低激励协议在不同种子下的精度。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import eval_openloop as open_loop
import identify_physical as ip

SEEDS = (20261260, 20261270)
NAME = "F3"


def main() -> None:
    for seed in SEEDS:
        noisy = Path("data/rv_%s_%d_noisy.npz" % (NAME, seed))
        save = Path("data/rv_%s_%d_strict.npz" % (NAME, seed))
        if not noisy.exists():
            print("缺少", noisy)
            continue
        subprocess.run([sys.executable, str(Path(__file__).with_name("regressor_full.py")),
                        "--data", str(noisy),
                        "--segments", "100", "--save", str(save), "--mode", "corrected",
                        "--width", "31", "--order", "5", "--stride", "3"], check=True,
                       stdout=subprocess.DEVNULL)
        raw = np.load(save)["model"]
        try:
            ip.build_model(raw)
        except RuntimeError:
            repaired = save.with_name(save.stem + "_repaired.npz")
            subprocess.run([sys.executable, str(Path(__file__).with_name("finalize_model.py")),
                            "--identified", str(save),
                            "--save", str(repaired), "--total-mass", "0",
                            "--inertia-floor", "3e-5"], check=True, stdout=subprocess.DEVNULL)
            raw = np.load(repaired)["model"]
        metrics = open_loop.evaluate(raw)
        payload = np.load(noisy)
        print("种子 %d：v %.3f  tau %.3f  RMSE50 %s  方向 %d" % (
            seed, np.abs(payload["qdot_clean"]).max(), np.abs(payload["tau_clean"]).max(),
            " ".join("%.2e" % v for v in metrics["rmse"][50]), metrics["direction_wrong"]),
            flush=True)


if __name__ == "__main__":
    main()
