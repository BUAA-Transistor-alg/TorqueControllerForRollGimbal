"""长时域开环示例：挑几条段把 3 s 整段画出来（只做少量仿真，秒级完成）。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

matplotlib_cache = Path(__file__).resolve().parent / "data" / ".mplconfig"
matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import gimbal_dynamics as gd
import identify_physical as ip

LABELS = ["大 yaw", "roll", "小 yaw", "pitch"]


def simulate(context, model, terrain, base, initial_q, initial_v, torques):
    state = gd.make_state(initial_q, initial_v)
    angles = np.zeros((len(torques), 4))
    for k in range(len(torques)):
        angles[k] = [state.angle[i] for i in range(4)]
        state = context.forward_step(model, terrain, base, state, torques[k], 0.01, 2)
    return angles


def main() -> None:
    parser = argparse.ArgumentParser(description="长时域开环示例")
    parser.add_argument("--identified", type=Path, default=Path("data/identified_final.npz"))
    parser.add_argument("--data", type=Path, default=Path("data/closed_loop_data_noisy.npz"))
    parser.add_argument("--output", type=Path, default=Path("data/long_open_loop.png"))
    parser.add_argument("--segments", type=str, default="62,89,0,1",
                        help="逗号分隔的段号，也支持 best/worst/median（需要扫描全部段）")
    arguments = parser.parse_args()

    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans CJK SC",
                                              "Noto Serif CJK SC", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    data = np.load(arguments.data)
    identified = ip.build_model(np.load(arguments.identified)["model"])
    truth = gd.make_model()
    context = gd.Context(1)
    steps = data["q_clean"].shape[1]
    time = np.arange(steps) * 0.01
    tokens = [token.strip() for token in arguments.segments.split(",") if token.strip()]
    scan = any(token in ("best", "worst", "median") for token in tokens)
    if scan:
        endpoint = np.zeros(data["q_clean"].shape[0])
        truth_all = np.zeros((data["q_clean"].shape[0], steps, 4))
        ident_all = np.zeros_like(truth_all)
        try:
            for segment in range(data["q_clean"].shape[0]):
                terrain = gd.make_terrain(float(data["terrain"][segment, 0]),
                                          float(data["terrain"][segment, 1]))
                base = gd.make_base(yaw=float(data["base_yaw"][segment]))
                truth_all[segment] = simulate(context, truth, terrain, base,
                                              data["q_clean"][segment, 0],
                                              data["qdot_clean"][segment, 0],
                                              data["tau_clean"][segment])
                ident_all[segment] = simulate(context, identified, terrain, base,
                                              data["q_clean"][segment, 0],
                                              data["qdot_clean"][segment, 0],
                                              data["tau_clean"][segment])
        finally:
            context.close()
        endpoint = np.max(np.abs(ident_all[:, :50, :] - truth_all[:, :50, :]), axis=(1, 2))
        order = np.argsort(endpoint)
    chosen = []
    for token in tokens:
        if token == "best":
            chosen.append(int(order[0]))
        elif token == "worst":
            chosen.append(int(order[-1]))
        elif token == "median":
            chosen.append(int(order[len(order) // 2]))
        else:
            chosen.append(int(token))
    truth_all = np.zeros((len(chosen), steps, 4))
    ident_all = np.zeros_like(truth_all)
    context = gd.Context(1)
    try:
        for index, segment in enumerate(chosen):
            terrain = gd.make_terrain(float(data["terrain"][segment, 0]),
                                      float(data["terrain"][segment, 1]))
            base = gd.make_base(yaw=float(data["base_yaw"][segment]))
            truth_all[index] = simulate(context, truth, terrain, base,
                                        data["q_clean"][segment, 0],
                                        data["qdot_clean"][segment, 0],
                                        data["tau_clean"][segment])
            ident_all[index] = simulate(context, identified, terrain, base,
                                        data["q_clean"][segment, 0],
                                        data["qdot_clean"][segment, 0],
                                        data["tau_clean"][segment])
    finally:
        context.close()
    print("各段偏差：")
    for index, segment in enumerate(chosen):
        print("  段 %3d：0.5 s 最大 %.2e rad，1 s 最大 %.2e rad，3 s 最大 %.2e rad" % (
            segment,
            np.max(np.abs(ident_all[index, :50] - truth_all[index, :50])),
            np.max(np.abs(ident_all[index, :100] - truth_all[index, :100])),
            np.max(np.abs(ident_all[index] - truth_all[index]))))

    figure, axes = plt.subplots(len(chosen), 1, figsize=(11.0, 2.7 * len(chosen)), sharex=True)
    axes = np.atleast_1d(axes)
    for panel, segment in enumerate(chosen):
        axis = axes[panel]
        for joint in range(4):
            axis.plot(time, truth_all[panel, :, joint], color="C%d" % joint,
                      linewidth=1.6, label=LABELS[joint] if panel == 0 else None)
            axis.plot(time, ident_all[panel, :, joint], color="C%d" % joint,
                      linewidth=1.2, linestyle="--")
        axis.set_title("第 %d 段：0.5 s 最大偏差 %.2e rad，3 s 最大偏差 %.2e rad" % (
            segment,
            np.max(np.abs(ident_all[panel, :50] - truth_all[panel, :50])),
            np.max(np.abs(ident_all[panel] - truth_all[panel]))))
        axis.set_ylabel("角度 (rad)")
        axis.grid(True, alpha=0.3)
        for mark in (0.25, 0.5, 1.0):
            axis.axvline(mark, color="0.6", linewidth=0.8, linestyle=":")
    axes[0].legend(loc="upper left", ncols=4)
    axes[-1].set_xlabel("时间 (s)")
    figure.suptitle("整段 3 s 开环：实线为真值参数，虚线为辨识参数（同一初始状态与力矩序列）")
    figure.tight_layout()
    figure.subplots_adjust(top=0.94)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=170)
    print("已保存", arguments.output)


if __name__ == "__main__":
    main()