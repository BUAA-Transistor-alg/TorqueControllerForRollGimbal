"""汇总低激励协议扫描结果：峰值速度、峰值力矩、峰值加速度与 0.5 s 开环 RMSE。

从已保存的数据集与模型文件重新计算，保证口径一致（峰值取干净轨迹）。
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import eval_openloop as open_loop

REFERENCE = {"标准协议": (1.857, 1.521)}


def parse_name(path: str):
    match = re.match(r"data/(.+)_w(\d+)\.npz", path)
    return (match.group(1), int(match.group(2))) if match else (None, None)


def main() -> None:
    rows = []
    for model_path in sorted(glob.glob("data/low*_w*.npz")):
        if model_path.endswith(".json"):
            continue
        tag, width = parse_name(model_path)
        if tag is None:
            continue
        data_path = Path("data/%s_noisy.npz" % tag)
        if not data_path.exists():
            continue
        data = np.load(data_path)
        velocity = float(np.abs(data["qdot_clean"]).max())
        torque = float(np.abs(data["tau_clean"]).max())
        acceleration = float(np.abs(np.gradient(data["qdot_clean"], 0.01, axis=1)).max())
        try:
            metrics = open_loop.evaluate(np.load(model_path)["model"])
        except Exception as error:
            print("跳过 %s：%s" % (model_path, error))
            continue
        rows.append({"tag": tag, "width": width, "velocity": velocity, "torque": torque,
                     "acceleration": acceleration, "rmse": metrics["rmse"][50],
                     "rmse_max": float(np.max(metrics["rmse"][50])),
                     "direction": metrics["direction_wrong"]})

    rows.sort(key=lambda row: (row["rmse_max"], row["torque"]))
    print("%-16s %4s %8s %8s %7s %9s  %s  %s" % (
        "配置", "宽度", "v峰值", "tau峰值", "a峰值", "RMSE50max", "RMSE50 四轴", "方向"))
    for row in rows:
        print("%-16s %4d %8.3f %8.3f %7.1f %9.2e  %s  %d" % (
            row["tag"], row["width"], row["velocity"], row["torque"], row["acceleration"],
            row["rmse_max"], " ".join("%.2e" % v for v in row["rmse"]), row["direction"]))

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for axis, key, label in ((axes[0], "torque", "峰值力矩 (N·m)"),
                             (axes[1], "velocity", "峰值速度 (rad/s)")):
        axis.scatter([row[key] for row in rows], [row["rmse_max"] for row in rows],
                     s=36, color="C0", label="低激励协议")
        axis.axhline(5.0e-3, color="C3", linestyle="--", linewidth=1.2, label="目标 5e-3")
        axis.scatter([REFERENCE["标准协议"][0 if key == "velocity" else 1]],
                     [9.5e-4], marker="*", s=180, color="C2", label="标准协议")
        axis.set_xlabel(label)
        axis.set_ylabel("0.5 s 开环 RMSE 最大值 (rad)")
        axis.set_yscale("log")
        axis.grid(True, alpha=0.3, which="both")
        axis.legend(fontsize=8)
    figure.suptitle("低激励协议的精度与峰值力矩 / 峰值速度")
    figure.tight_layout()
    figure.subplots_adjust(top=0.88)
    figure.savefig("data/low_excitation_pareto.png", dpi=170)
    print("已保存 data/low_excitation_pareto.png")


if __name__ == "__main__":
    main()
