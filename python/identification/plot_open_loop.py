"""在同一初始状态与同一组控制力矩下，对比真实参数与辨识参数的开环关节角。"""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import gimbal_dynamics as gd
import identify_physical as ip


def simulate(context, model, terrain, base, initial, torques, steps):
    state = gd.make_state(initial.angle, initial.velocity)
    angles = np.zeros((steps, 4))
    for k in range(steps):
        angles[k] = [state.angle[i] for i in range(4)]
        state = context.forward_step(model, terrain, base, state, torques[k], 0.01, 2)
    return angles


def main() -> None:
    parser = argparse.ArgumentParser(description="开环关节角对比")
    parser.add_argument("--data", type=Path, default=Path("data/identification_data.npz"))
    parser.add_argument("--identified", type=Path, default=Path("data/identified_beta.npz"))
    parser.add_argument("--segment", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--training-steps", type=int, default=80)
    parser.add_argument("--output", type=Path, default=Path("open_loop_comparison.png"))
    arguments = parser.parse_args()

    q, qdot, qdd, tau, terrain, base_yaw = ip.load_data(arguments.data)
    identified_x = np.load(arguments.identified)["model"]
    identified = ip.build_model(identified_x)
    truth = gd.make_model()

    sl = slice(0, arguments.steps)
    terrain_state = gd.make_terrain(float(terrain[arguments.segment, 0]),
                                    float(terrain[arguments.segment, 1]))
    base = gd.make_base(yaw=float(base_yaw[arguments.segment]))
    initial = gd.make_state(q[arguments.segment, 0], qdot[arguments.segment, 0])
    torques = tau[arguments.segment, :arguments.steps]

    context = gd.Context(1)
    try:
        truth_angles = simulate(context, truth, terrain_state, base, initial, torques, arguments.steps)
        identified_angles = simulate(context, identified, terrain_state, base, initial, torques, arguments.steps)
    finally:
        context.close()

    print("truth angle range:", np.min(truth_angles, axis=0), np.max(truth_angles, axis=0))
    print("identified angle range:", np.min(identified_angles, axis=0), np.max(identified_angles, axis=0))
    print("max abs difference:", np.max(np.abs(truth_angles - identified_angles), axis=0))

    time = np.arange(arguments.steps) * 0.01
    labels = ["Joint 1 (big yaw)", "Joint 2 (roll)", "Joint 3 (small yaw)", "Joint 4 (pitch)"]
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), sharex=True)
    for index, axis in enumerate(axes.reshape(-1)):
        axis.plot(time, truth_angles[:, index], label="True parameters", linewidth=1.8)
        axis.plot(time, identified_angles[:, index], label="Identified parameters",
                  linewidth=1.5, linestyle="--")
        axis.set_title(labels[index])
        axis.set_ylabel("angle (rad)")
        axis.grid(True, alpha=0.3)
        axis.axvline(arguments.training_steps * 0.01, color="0.35", linestyle=":",
                     linewidth=1.0, label="curriculum horizon" if index == 0 else None)
    for axis in axes[-1, :]:
        axis.set_xlabel("time (s)")
    axes[0, 0].legend(loc="best")
    figure.suptitle(
        "Open-loop joint angles: true vs identified parameters (segment %d, %.2f s)"
        % (arguments.segment, arguments.steps * 0.01),
        y=0.985,
    )
    figure.tight_layout()
    figure.subplots_adjust(top=0.90)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=180)
    print("图像已保存到", arguments.output)


if __name__ == "__main__":
    main()
