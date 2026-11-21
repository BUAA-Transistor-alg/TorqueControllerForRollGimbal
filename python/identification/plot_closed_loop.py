"""同 PID 参数与目标序列下，真实参数和辨识参数的闭环位置与力矩曲线对比。"""

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


def simulate(context, model, data, segment: int):
    steps = data["points"]
    terrain = gd.make_terrain(float(data["terrain"][segment, 0]),
                              float(data["terrain"][segment, 1]))
    base = gd.make_base(yaw=float(data["base_yaw"][segment]))
    state = gd.make_state(data["initial"][segment], data["initial_velocity"][segment])
    kp = data["kp"][segment]
    ki = data["ki"][segment]
    kd = data["kd"][segment]
    limit = float(data["torque_limit"][segment])
    integral_limit = float(data["integral_limit"][segment])
    dt = float(data["dt"])
    integral = np.zeros(4)
    angles = np.zeros((steps, 4))
    torques = np.zeros((steps, 4))
    target = data["target"][segment]
    target_velocity = data["target_velocity"][segment]
    for k in range(steps):
        q = np.array([state.angle[i] for i in range(4)])
        qdot = np.array([state.velocity[i] for i in range(4)])
        angles[k] = q
        error = target[k] - q
        integral = np.clip(integral + error * dt, -integral_limit, integral_limit)
        feedforward = context.bias_torque(model, terrain, base, state)
        command = (np.asarray(feedforward) + kp * error + ki * integral
                   + kd * (target_velocity[k] - qdot))
        command = np.clip(command, -limit, limit)
        torques[k] = command
        if k + 1 < steps:
            state = context.forward_step(model, terrain, base, state, command, dt, 2)
    return angles, torques


def main() -> None:
    parser = argparse.ArgumentParser(description="闭环位置与力矩对比")
    parser.add_argument("--data", type=Path, default=Path("data/closed_loop_data.npz"))
    parser.add_argument("--identified", type=Path, default=Path("data/identified_closed_loop.npz"))
    parser.add_argument("--segment", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("closed_loop_comparison.png"))
    arguments = parser.parse_args()

    data = np.load(arguments.data)
    identified = ip.build_model(np.load(arguments.identified)["model"])
    truth = gd.make_model()
    context = gd.Context(1)
    try:
        true_angle, true_torque = simulate(context, truth, data, arguments.segment)
        identified_angle, identified_torque = simulate(context, identified, data, arguments.segment)
    finally:
        context.close()

    angle_error = identified_angle - true_angle
    torque_error = identified_torque - true_torque
    print("位置 RMSE (rad):", np.sqrt(np.mean(angle_error ** 2, axis=0)))
    print("力矩 RMSE (N·m):", np.sqrt(np.mean(torque_error ** 2, axis=0)))
    print("位置最大误差 (rad):", np.max(np.abs(angle_error), axis=0))
    print("力矩最大误差 (N·m):", np.max(np.abs(torque_error), axis=0))

    time = np.arange(data["points"]) * float(data["dt"])
    labels = ["Joint 1 (big yaw)", "Joint 2 (roll)", "Joint 3 (small yaw)", "Joint 4 (pitch)"]
    figure, axes = plt.subplots(4, 2, figsize=(12.0, 12.0), sharex=True)
    for joint in range(4):
        axes[joint, 0].plot(time, true_angle[:, joint], label="True parameters", linewidth=1.7)
        axes[joint, 0].plot(time, identified_angle[:, joint], label="Identified parameters",
                            linewidth=1.4, linestyle="--")
        axes[joint, 0].set_ylabel("angle (rad)")
        axes[joint, 0].set_title(labels[joint] + " position")
        axes[joint, 0].grid(True, alpha=0.3)
        axes[joint, 1].plot(time, true_torque[:, joint], label="True parameters", linewidth=1.7)
        axes[joint, 1].plot(time, identified_torque[:, joint], label="Identified parameters",
                            linewidth=1.4, linestyle="--")
        axes[joint, 1].set_ylabel("torque (N·m)")
        axes[joint, 1].set_title(labels[joint] + " torque")
        axes[joint, 1].grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    axes[0, 0].legend(loc="best")
    figure.suptitle("Closed-loop PID comparison: true vs identified parameters")
    figure.tight_layout()
    figure.subplots_adjust(top=0.94)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=170)
    print("图像已保存到", arguments.output)


if __name__ == "__main__":
    main()
