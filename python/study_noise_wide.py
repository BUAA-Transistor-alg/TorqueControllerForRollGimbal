"""高噪声工况下加宽平滑窗口的补充实验。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import study_noise as sn

CONFIGS = {
    "efficient_iso5": ("efficient", {"position": 0.005, "velocity": 0.5, "torque": 0.05}),
    "efficient_iso10": ("efficient", {"position": 0.01, "velocity": 1.0, "torque": 0.1}),
    "efficient_pos5": ("efficient", {"position": 0.005, "velocity": 0.1, "torque": 0.01}),
    "efficient_pos10": ("efficient", {"position": 0.01, "velocity": 0.1, "torque": 0.01}),
    "strong_iso10": ("strong", {"position": 0.01, "velocity": 1.0, "torque": 0.1}),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="加宽窗口补充实验")
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=2)
    arguments = parser.parse_args()
    items = list(CONFIGS.items())
    items = [item for index, item in enumerate(items) if index % arguments.parts == arguments.part]
    results = []
    for tag, (protocol_name, noise) in items:
        protocol = sn.PROTOCOLS[protocol_name]
        print("###", tag, flush=True)
        record = sn.evaluate_config(tag + "_wide", protocol["clean"], noise, (71, 91, 111),
                                    20261220)
        best = record["best"]
        if best is None:
            print("  失败", flush=True)
        else:
            print("  宽度 %d RMSE50 %s 方向 %d" % (
                best["width"], " ".join("%.2e" % v for v in best["rmse50"]),
                best["direction"]), flush=True)
        results.append({"tag": tag, **record})
    Path("data/noise_wide_%d.json" % arguments.part).write_text(
        json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
