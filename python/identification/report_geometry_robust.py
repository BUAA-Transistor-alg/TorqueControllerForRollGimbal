"""几何量偏差稳健性汇总：基线（直接用记录几何）与逐段几何估计的对比。"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans CJK SC",
                                          "Noto Serif CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    return json.loads(Path(path).read_text())


MEASURED_BASELINE = {
    "strong": {5.0: 4.48e-2, 10.0: 7.98e-2, 15.0: 1.32e-1, 20.0: 1.67e-1},
    "efficient": {},
}


def baseline_points(protocol):
    points = dict(MEASURED_BASELINE.get(protocol, {}))
    for pattern in ("data/noise_geo_%s.json", "data/noise_study_[0-9].json"):
        for path in sorted(glob.glob(pattern % protocol if "%s" in pattern else pattern)):
            try:
                records = load(path)
            except json.JSONDecodeError:
                continue
            for record in records:
                if record.get("protocol") != protocol or record.get("best") is None:
                    continue
                tag = record["tag"].replace("_wide", "")
                prefix = protocol + "_"
                if tag.startswith(prefix):
                    tag = tag[len(prefix):]
                if tag.startswith("geoFine"):
                    scale = float(tag[len("geoFine"):])
                elif tag.startswith("geo") and tag[3:].replace(".", "").isdigit():
                    scale = float(tag[3:])
                else:
                    continue
                points[scale] = max(record["best"]["rmse50"])
    return points


def robust_points(name):
    reports = load("data/geo_robust_reports_%s.json" % name)
    best = min(reports["iterations"], key=lambda item: max(item["rmse50"]))
    return max(best["rmse50"]), best["rmse50"], best["geometry_error_median_deg"], \
        best["geometry_error_max_deg"]


def main() -> None:
    print("%-22s %-14s %-14s %s" % ("协议 / 几何误差", "基线 RMSE50max", "几何估计后", "恢复几何误差"))
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    for protocol, label, color, robust_names in (
            ("strong", "标准协议", "C1", ("std_s10", "std_s15", "std_s20")),
            ("efficient", "低激励协议", "C0", ("s5", "s10_strong"))):
        base = baseline_points(protocol)
        if base:
            pairs = sorted(base.items())
            axes[0].plot([item[0] for item in pairs], [item[1] for item in pairs],
                         marker="o", linestyle="--", color=color,
                         label="%s 基线" % label)
        for name in robust_names:
            try:
                worst, per_axis, median, maximum = robust_points(name)
            except FileNotFoundError:
                continue
            sigma = {"std_s10": 10.0, "std_s15": 15.0, "std_s20": 20.0,
                     "s5": 5.0, "s10_strong": 10.0}[name]
            print("%-10s 坡度 %.0f 度      %-14s %-14.2e 中位数 %.2f 度，最大 %.2f 度" % (
                label, sigma,
                "%.2e" % (max(base.get(sigma, np.nan), 0.0) if sigma in base else float("nan")),
                worst, median, maximum))
            axes[0].scatter([sigma], [worst], marker="*", s=150, color=color,
                            zorder=5)
    axes[0].axhline(5.0e-3, color="C3", linestyle="--", linewidth=1.2, label="目标 5e-3")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("记录几何量的坡度噪声 (度)")
    axes[0].set_ylabel("0.5 s 开环 RMSE 最大值 (rad)")
    axes[0].grid(True, alpha=0.3, which="both")
    axes[0].legend(fontsize=8)
    axes[0].set_title("几何噪声：基线（虚线）与逐段估计（星号）")

    # 右图：标准协议下逐段几何恢复误差
    errors = []
    for name in ("std_s10", "s10_strong"):
        path = "data/geo_robust_reports_%s.json" % name
        if Path(path).exists():
            reports = load(path)
            best = min(reports["iterations"], key=lambda item: max(item["rmse50"]))
            errors.append((name, reports["corrupted_median_deg"],
                           best["geometry_error_median_deg"]))
    labels = [item[0] for item in errors]
    corrupted = [item[1] for item in errors]
    recovered = [item[2] for item in errors]
    position = np.arange(len(errors))
    axes[1].bar(position - 0.2, corrupted, 0.4, label="加噪后中位数", color="C3")
    axes[1].bar(position + 0.2, recovered, 0.4, label="估计后中位数", color="C0")
    axes[1].set_yscale("log")
    axes[1].set_xticks(position)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("几何误差中位数 (度)")
    axes[1].grid(True, alpha=0.3, axis="y", which="both")
    axes[1].legend(fontsize=8)
    axes[1].set_title("逐段几何估计的恢复精度")
    figure.tight_layout()
    figure.savefig("data/geometry_robust.png", dpi=170)
    print("已保存 data/geometry_robust.png")


if __name__ == "__main__":
    main()