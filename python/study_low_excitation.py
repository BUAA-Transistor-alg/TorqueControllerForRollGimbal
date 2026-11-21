"""低激励协议扫描：对每组幅值/频率/坡度生成数据、拟合、并在标准 100 段上评估。

评估口径与任务一致：辨识参数与真实参数在同一初始状态、同一记录力矩下的 0.5 s 开环位置 RMSE。
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
    {"name": "A2f2_t20", "amplitude": 2.0, "frequency": 2.0, "tilt": 20.0,
     "static_amplitude": 0.2, "gain": 1.0, "width": 51},
    {"name": "A3f3_t20", "amplitude": 3.0, "frequency": 3.0, "tilt": 20.0,
     "static_amplitude": 0.3, "gain": 1.0, "width": 51},
    {"name": "A1f3_t20", "amplitude": 1.0, "frequency": 3.0, "tilt": 20.0,
     "static_amplitude": 0.2, "gain": 1.0, "width": 51},
    {"name": "A2f3_t20", "amplitude": 2.0, "frequency": 3.0, "tilt": 20.0,
     "static_amplitude": 0.2, "gain": 1.0, "width": 51},
    {"name": "A2f3_t10", "amplitude": 2.0, "frequency": 3.0, "tilt": 10.0,
     "static_amplitude": 0.2, "gain": 1.0, "width": 51},
]


def run(config: dict, segments: int, seed: int, widths=(51, 71)) -> dict:
    clean = Path("data/low_%s.npz" % config["name"])
    noisy = Path("data/low_%s_noisy.npz" % config["name"])
    collector.collect(segments, seed, config["amplitude"], config["frequency"],
                      config["static_amplitude"], math.radians(config["tilt"]),
                      config["gain"], 0.3, clean, noisy, seed + 1)
    payload = np.load(noisy)
    velocity_peak = float(np.abs(payload["qdot"]).max())
    torque_peak = float(np.abs(payload["tau"]).max())
    best = None
    import regressor_full as rf
    for width in widths:
        save = Path("data/low_%s_w%d.npz" % (config["name"], width))
        sys.argv = ["regressor", "--data", str(noisy), "--segments", str(segments),
                    "--save", str(save), "--mode", "corrected", "--width", str(width),
                    "--order", "5", "--stride", "3"]
        try:
            rf.main()
        except AssertionError as error:
            print("  宽度 %d 拟合失败：%s" % (width, error))
            continue
        metrics = open_loop.evaluate(np.load(save)["model"])
        record = {"width": width,
                  "rmse50": metrics["rmse"][50].tolist(),
                  "max50": metrics["maximum"][50].tolist(),
                  "direction": metrics["direction_wrong"]}
        if best is None or max(record["rmse50"]) < max(best["rmse50"]):
            best = record
    return {"config": config, "velocity_peak": velocity_peak, "torque_peak": torque_peak,
            "best": best}


def main() -> None:
    segments = 100
    seed = 20261220
    results = []
    for config in CONFIGS:
        print("### 配置 %s" % config["name"], flush=True)
        results.append(run(config, segments, seed))
        record = results[-1]
        if record["best"] is not None:
            print("  峰值速度 %.3f rad/s，峰值力矩 %.3f N·m，50 步 RMSE 最大 %.2e，方向错 %d"
                  % (record["velocity_peak"], record["torque_peak"],
                     max(record["best"]["rmse50"]), record["best"]["direction"]), flush=True)
    Path("data/low_excitation_study.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
