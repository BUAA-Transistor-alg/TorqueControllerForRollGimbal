"""最终开环对比图：真值参数与辨识参数在同一干净初始状态与同一力矩序列下的轨迹。"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import eval_openloop as open_loop
import gimbal_dynamics as gd
import identify_physical as ip

LABELS = ["big yaw", "roll", "small yaw", "pitch"]


def main() -> None:
    parser = argparse.ArgumentParser(description="最终开环对比图")
    parser.add_argument("--identified", type=Path, default=Path("data/identified_regressor.npz"))
    parser.add_argument("--data", type=Path, default=Path("data/closed_loop_data_noisy.npz"))
    parser.add_argument("--output", type=Path, default=Path("data/final_open_loop.png"))
    parser.add_argument("--steps", type=int, default=100)
    arguments = parser.parse_args()

    data = np.load(arguments.data)
    identified = ip.build_model(np.load(arguments.identified)["model"])
    truth = gd.make_model()
    metrics = open_loop.evaluate(np.load(arguments.identified)["model"], arguments.data)
    error = np.max(np.abs(metrics["difference"][:, :50, :]), axis=(1, 2))
    chosen = [int(np.argmin(error)), int(np.argmax(error)), 0, 1]
    context = gd.Context(1)
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), sharex=True)
    time = np.arange(arguments.steps) * 0.01
    try:
        for panel, segment in enumerate(chosen):
            terrain = gd.make_terrain(float(data["terrain"][segment, 0]),
                                      float(data["terrain"][segment, 1]))
            base = gd.make_base(yaw=float(data["base_yaw"][segment]))
            truth_angles, _ = open_loop.simulate(context, truth, terrain, base,
                                                 data["q_clean"][segment, 0],
                                                 data["qdot_clean"][segment, 0],
                                                 data["tau_clean"][segment, :arguments.steps])
            ident_angles, _ = open_loop.simulate(context, identified, terrain, base,
                                                 data["q_clean"][segment, 0],
                                                 data["qdot_clean"][segment, 0],
                                                 data["tau_clean"][segment, :arguments.steps])
            axis = axes.reshape(-1)[panel]
            for joint in range(4):
                axis.plot(time, truth_angles[:, joint], linewidth=1.6,
                          color="C%d" % joint, label=LABELS[joint] if panel == 0 else None)
                axis.plot(time, ident_angles[:, joint], linewidth=1.4, linestyle="--",
                          color="C%d" % joint)
            axis.set_title("segment %d (0.5 s max deviation %.2e rad)" % (
                segment, np.max(np.abs(truth_angles[:50] - ident_angles[:50]))))
            axis.set_ylabel("angle (rad)")
            axis.grid(True, alpha=0.3)
    finally:
        context.close()
    for axis in axes[-1, :]:
        axis.set_xlabel("time (s)")
    axes[0, 0].legend(loc="best", ncols=2)
    figure.suptitle("Open loop: solid = true parameters, dashed = identified parameters "
                    "(same initial state and torque sequence)")
    figure.tight_layout()
    figure.subplots_adjust(top=0.90)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=170)
    print("已保存", arguments.output)


if __name__ == "__main__":
    main()