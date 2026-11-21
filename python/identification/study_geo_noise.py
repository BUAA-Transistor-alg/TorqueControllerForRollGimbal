"""补充几何噪声细扫：坡度/坡向/偏航噪声在 0.5 到 2 度量级的容差。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import study_noise as sn


def main() -> None:
    parser = argparse.ArgumentParser(description="几何噪声细扫")
    parser.add_argument("--protocol", default="efficient")
    parser.add_argument("--scales", default="0.5,1,2")
    arguments = parser.parse_args()
    protocol = sn.PROTOCOLS[arguments.protocol]
    results = []
    for scale in [float(value) for value in arguments.scales.split(",")]:
        noise = sn.NOMINAL.copy()
        noise["slope"] = np.radians(scale)
        noise["direction"] = np.radians(2.0 * scale)
        noise["yaw"] = np.radians(3.0 * scale)
        tag = "%s_geoFine%g" % (arguments.protocol, scale)
        print("###", tag, flush=True)
        record = sn.evaluate_config(tag, protocol["clean"], noise, protocol["widths"],
                                    20261220)
        best = record["best"]
        if best is None:
            print("  失败", flush=True)
        else:
            print("  坡度 %.1f 度 坡向 %.1f 度 偏航 %.1f 度 | 宽度 %d RMSE50 %s 方向 %d" % (
                scale, 2.0 * scale, 3.0 * scale, best["width"],
                " ".join("%.2e" % v for v in best["rmse50"]), best["direction"]), flush=True)
        results.append({"protocol": arguments.protocol, **record})
    Path("data/noise_geo_%s.json" % arguments.protocol).write_text(
        json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
