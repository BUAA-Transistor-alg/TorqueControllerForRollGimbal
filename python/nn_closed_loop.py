"""神经网络动力学模型的闭环 PID 对比。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import gimbal_dynamics as gd
import train_nn_dynamics as tn


def nn_next(model, means, stds, q, qdot, tau, terrain, yaw):
    features = tn.build_features(q, qdot, tau, terrain, yaw)
    normalized = (features - means["x"]) / stds["x"]
    with torch.no_grad():
        prediction = model(torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0))
    delta = prediction.numpy()[0] * stds["y"] + means["y"]
    return q + delta[:4], qdot + delta[4:]


def simulate_true(data, segment):
    terrain = gd.make_terrain(float(data["terrain"][segment, 0]),
                              float(data["terrain"][segment, 1]))
    base = gd.make_base(yaw=float(data["base_yaw"][segment]))
    state = gd.make_state(data["initial"][segment], data["initial_velocity"][segment])
    kp, ki, kd = data["kp"][segment], data["ki"][segment], data["kd"][segment]
    limit = float(data["torque_limit"][segment])
    integral_limit = float(data["integral_limit"][segment])
    target = data["target"][segment]
    target_velocity = data["target_velocity"][segment]
    integral = np.zeros(4)
    context = gd.Context(1)
    angles = np.zeros((data["points"], 4))
    torques = np.zeros((data["points"], 4))
    try:
        for k in range(data["points"]):
            q = np.array([state.angle[i] for i in range(4)])
            qdot = np.array([state.velocity[i] for i in range(4)])
            angles[k] = q
            error = target[k] - q
            integral = np.clip(integral + error * 0.01, -integral_limit, integral_limit)
            command = kp * error + ki * integral + kd * (target_velocity[k] - qdot)
            command = np.clip(command, -limit, limit)
            torques[k] = command
            if k + 1 < data["points"]:
                state = context.forward_step(gd.make_model(), terrain, base, state,
                                             command, 0.01, 2)
    finally:
        context.close()
    return angles, torques


def simulate_nn(model, means, stds, data, segment):
    terrain = np.array([float(data["terrain"][segment, 0]),
                        float(data["terrain"][segment, 1])])
    yaw = float(data["base_yaw"][segment])
    q = np.array(data["initial"][segment])
    qdot = np.array(data["initial_velocity"][segment])
    kp, ki, kd = data["kp"][segment], data["ki"][segment], data["kd"][segment]
    limit = float(data["torque_limit"][segment])
    integral_limit = float(data["integral_limit"][segment])
    target = data["target"][segment]
    target_velocity = data["target_velocity"][segment]
    integral = np.zeros(4)
    angles = np.zeros((data["points"], 4))
    torques = np.zeros((data["points"], 4))
    for k in range(data["points"]):
        angles[k] = q
        error = target[k] - q
        integral = np.clip(integral + error * 0.01, -integral_limit, integral_limit)
        command = kp * error + ki * integral + kd * (target_velocity[k] - qdot)
        command = np.clip(command, -limit, limit)
        torques[k] = command
        if k + 1 < data["points"]:
            q, qdot = nn_next(model, means, stds, q, qdot, command, terrain, yaw)
    return angles, torques


def main() -> None:
    parser = argparse.ArgumentParser(description="神经网络闭环对比")
    parser.add_argument("--data", type=Path, default=Path("data/closed_loop_data_noisy.npz"))
    parser.add_argument("--model", type=Path, default=Path("data/nn_dynamics.pt"))
    parser.add_argument("--segment", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("nn_closed_loop_comparison.png"))
    arguments = parser.parse_args()

    data = np.load(arguments.data)
    checkpoint = torch.load(arguments.model, map_location="cpu", weights_only=False)
    model = tn.DeltaNet(checkpoint["mean_x"].shape[0])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    means = {"x": checkpoint["mean_x"], "y": checkpoint["mean_y"]}
    stds = {"x": checkpoint["std_x"], "y": checkpoint["std_y"]}
    true_angle, true_torque = simulate_true(data, arguments.segment)
    nn_angle, nn_torque = simulate_nn(model, means, stds, data, arguments.segment)
    angle_error = nn_angle - true_angle
    torque_error = nn_torque - true_torque
    print("位置 RMSE (rad):", np.sqrt(np.mean(angle_error ** 2, axis=0)))
    print("力矩 RMSE (N·m):", np.sqrt(np.mean(torque_error ** 2, axis=0)))
    print("位置最大误差 (rad):", np.max(np.abs(angle_error), axis=0))
    print("力矩最大误差 (N·m):", np.max(np.abs(torque_error), axis=0))

    time = np.arange(data["points"]) * 0.01
    labels = ["Joint 1 (big yaw)", "Joint 2 (roll)", "Joint 3 (small yaw)", "Joint 4 (pitch)"]
    figure, axes = plt.subplots(4, 2, figsize=(12.0, 12.0), sharex=True)
    for joint in range(4):
        axes[joint, 0].plot(time, true_angle[:, joint], label="True dynamics", linewidth=1.7)
        axes[joint, 0].plot(time, nn_angle[:, joint], label="Neural network",
                            linewidth=1.4, linestyle="--")
        axes[joint, 0].set_ylabel("angle (rad)")
        axes[joint, 0].set_title(labels[joint] + " position")
        axes[joint, 0].grid(True, alpha=0.3)
        axes[joint, 1].plot(time, true_torque[:, joint], label="True dynamics", linewidth=1.7)
        axes[joint, 1].plot(time, nn_torque[:, joint], label="Neural network",
                            linewidth=1.4, linestyle="--")
        axes[joint, 1].set_ylabel("torque (N·m)")
        axes[joint, 1].set_title(labels[joint] + " torque")
        axes[joint, 1].grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    axes[0, 0].legend(loc="best")
    figure.suptitle("Closed-loop PID: true dynamics vs neural network dynamics")
    figure.tight_layout()
    figure.subplots_adjust(top=0.94)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=170)
    print("图像已保存到", arguments.output)


if __name__ == "__main__":
    main()
