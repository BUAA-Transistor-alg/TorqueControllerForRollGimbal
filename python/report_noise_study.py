"""汇总噪声敏感性研究：表格与曲线（含几何细扫与加宽窗口补充实验）。"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans CJK SC",
                                          "Noto Serif CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np

PROTOCOL_LABELS = {"efficient": "低激励协议", "strong": "标准协议"}
FIELD_LABELS = {"pos": "仅位置", "vel": "仅速度", "tor": "仅力矩", "geo": "仅几何"}
FIELD_COLORS = {"pos": "C0", "vel": "C1", "tor": "C2", "geo": "C3"}
TARGET = 5.0e-3


def short_tag(record: dict) -> str:
    tag = record.get("tag", "?").replace("_wide", "")
    for prefix in ("efficient_", "strong_"):
        if tag.startswith(prefix):
            return tag[len(prefix):]
    return tag


def load_all():
    merged = {}
    for pattern in ("data/noise_study_[0-9].json", "data/noise_geo_*.json",
                    "data/noise_wide_[0-9].json"):
        for path in sorted(glob.glob(pattern)):
            for record in json.loads(Path(path).read_text()):
                record.setdefault("tag", "?")
                if "protocol" not in record:
                    record["protocol"] = ("efficient" if record["tag"].startswith("efficient")
                                          else "strong")
                key = (record["protocol"], short_tag(record))
                current = merged.get(key)
                if current is None:
                    merged[key] = record
                elif record["best"] is not None and (current["best"] is None or (
                        max(record["best"]["rmse50"]) < max(current["best"]["rmse50"]))):
                    merged[key] = record
    return list(merged.values())


def scale_of(tag: str):
    for field in ("iso", "pos", "vel", "tor", "geoFine", "geo"):
        if tag.startswith(field):
            suffix = tag[len(field):]
            try:
                return field, float(suffix)
            except ValueError:
                return field, None
    return None, None


def main() -> None:
    records = load_all()
    rows = []
    for record in records:
        if record["best"] is None:
            continue
        rows.append((record["protocol"], short_tag(record), record["noise"], record["best"]))
    order = {"efficient": 0, "strong": 1}
    rows.sort(key=lambda row: (order.get(row[0], 9), row[1]))
    print("%-10s %-12s %-32s %-9s %-5s %s" % (
        "协议", "配置", "噪声（位置/速度/力矩/坡度）", "RMSE50max", "方向", "RMSE50 四轴"))
    for protocol, tag, noise, best in rows:
        print("%-10s %-12s %.4f/%.3f/%.4f/%5.1f°  %-9.2e %-5d %s" % (
            protocol, tag, noise["position"], noise["velocity"], noise["torque"],
            np.degrees(noise.get("slope", 0.0)), max(best["rmse50"]), best["direction"],
            " ".join("%.2e" % v for v in best["rmse50"])))

    figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    for protocol in ("efficient", "strong"):
        pairs = []
        for record in records:
            if record["protocol"] != protocol or record["best"] is None:
                continue
            field, scale = scale_of(short_tag(record))
            if field != "iso" or scale is None:
                continue
            pairs.append((scale, max(record["best"]["rmse50"])))
        if pairs:
            pairs.sort()
            axes[0].plot([item[0] for item in pairs], [item[1] for item in pairs],
                         marker="o", color=("C0" if protocol == "efficient" else "C1"),
                         label=PROTOCOL_LABELS[protocol])
    axes[0].axhline(TARGET, color="C3", linestyle="--", linewidth=1.2, label="目标 5e-3")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("噪声倍数（位置/速度/力矩同时放大）")
    axes[0].set_ylabel("0.5 s 开环 RMSE 最大值 (rad)")
    axes[0].grid(True, alpha=0.3, which="both")
    axes[0].legend(fontsize=8)
    axes[0].set_title("三通道同倍噪声")

    for protocol in ("efficient", "strong"):
        for field in ("pos", "vel", "tor", "geo"):
            pairs = []
            for record in records:
                if record["protocol"] != protocol or record["best"] is None:
                    continue
                record_field, scale = scale_of(short_tag(record))
                if record_field != field:
                    continue
                if field == "geo":
                    scale = np.degrees(record["noise"].get("slope", 0.0))
                if scale is None:
                    continue
                pairs.append((scale, max(record["best"]["rmse50"])))
            if not pairs:
                continue
            pairs.sort()
            axes[1].plot([item[0] for item in pairs], [item[1] for item in pairs],
                         marker=("o" if protocol == "efficient" else "s"),
                         linestyle=("-" if protocol == "efficient" else "--"),
                         color=FIELD_COLORS[field],
                         label="%s %s" % (PROTOCOL_LABELS[protocol], FIELD_LABELS[field]))
    axes[1].axhline(TARGET, color="C3", linestyle="--", linewidth=1.2)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("单字段噪声倍数（几何字段用坡度角度）")
    axes[1].set_ylabel("0.5 s 开环 RMSE 最大值 (rad)")
    axes[1].grid(True, alpha=0.3, which="both")
    axes[1].legend(fontsize=7, ncols=2)
    axes[1].set_title("逐字段加噪")
    figure.tight_layout()
    figure.savefig("data/noise_study.png", dpi=170)
    print("已保存 data/noise_study.png")


if __name__ == "__main__":
    main()
