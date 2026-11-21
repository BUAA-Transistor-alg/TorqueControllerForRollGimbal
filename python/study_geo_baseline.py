"""几何噪声下不做几何估计的基线（标准协议）。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import study_noise as sn


def main() -> None:
    results = []
    for scale in (5.0, 10.0, 15.0, 20.0):
        noise = sn.NOMINAL.copy()
        noise["slope"] = np.radians(scale)
        noise["direction"] = np.radians(2.0 * scale)
        noise["yaw"] = np.radians(3.0 * scale)
        tag = "strong_geoBaseline%g" % scale
        print("###", tag, flush=True)
        record = sn.evaluate_config(tag, sn.PROTOCOLS["strong"]["clean"], noise, (51, 71),
                                    20261220)
        best = record["best"]
        if best is None:
            print("  失败", flush=True)
        else:
            print("  坡度 %.1f 度：RMSE50 %s 方向 %d" % (
                scale, " ".join("%.2e" % v for v in best["rmse50"]), best["direction"]),
                flush=True)
        results.append({"protocol": "strong", **record})
    Path("data/noise_geo_baseline_strong.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
