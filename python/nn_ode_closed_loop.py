"""Neural ODE 动力学模型的闭环 PID 对比。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import gimbal_dynamics as gd
import train_neural_ode as tno


def load_ode(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    function = tno.OdeFunction(16, checkpoint["width"], checkpoint["depth"])
    function.load_state_dict(checkpoint["state_dict"])
    function.eval()
    return function, checkpoint


def true_closed_loop(data, segment, steps):
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
    angles = np.zeros((steps, 4))
    torques = np.zeros((steps, 4))
    try:
        for k in range(steps):
            q = np.array([state.angle[i] for i in range(4)])
            qdot = np.array([state.velocity[i] for i in range(4)])
            angles[k] = q
            error = target[k] - q
            integral = np.clip(integral + error * 0.01, -integral_limit, integral_limit)
            command = kp * error + ki * integral + kd * (target_velocity[k] - qdot)
            command = np.clip(command, -limit, limit)
            torques[k] = command
            if k + 1 < steps:
                state = context.forward_step(gd.make_model(), terrain, base, state, command, 0.01, 2)
    finally:
        context.close()
    return angles, torques


def ode_closed_loop(function, checkpoint, data, segment, steps):
    mean = torch.tensor(checkpoint["mean_x"], dtype=torch.float32)
    std = torch.tensor(checkpoint["std_x"], dtype=torch.float32)
    scale = checkpoint["qdd_scale"]
    terrain = data["terrain"][segment]
    yaw = float(data["base_yaw"][segment])
    terrain_features = torch.tensor(np.array([terrain[0], terrain[1], np.sin(yaw), np.cos(yaw)]),
                                    dtype=torch.float32).unsqueeze(0)
    state = torch.tensor(np.concatenate([data["initial"][segment],
                                         data["initial_velocity"][segment]]),
                         dtype=torch.float32).unsqueeze(0)
    kp, ki, kd = data["kp"][segment], data["ki"][segment], data["kd"][segment]
    limit = float(data["torque_limit"][segment])
    integral_limit = float(data["integral_limit"][segment])
    target = data["target"][segment]
    target_velocity = data["target_velocity"][segment]
    integral = np.zeros(4)
    angles = np.zeros((steps, 4))
    torques = np.zeros((steps, 4))
    for k in range(steps):
        q = state[0, :4].numpy().copy()
        qdot = state[0, 4:].numpy().copy()
        angles[k] = q
        error = target[k] - q
        integral = np.clip(integral + error * 0.01, -integral_limit, integral_limit)
        command = kp * error + ki * integral + kd * (target_velocity[k] - qdot)
        command = np.clip(command, -limit, limit)
        torques[k] = command
        if k + 1 < steps:
            torque_tensor = torch.tensor(command, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                state = tno.integrate_step(function, state, torque_tensor, terrain_features,
                                            0.01, scale, checkpoint.get("integrator", "heun"))
    return angles, torques


def main() -> None:
    parser = argparse.ArgumentParser(description="Neural ODE 闭环对比")
    parser.add_argument("--data", type=Path, default=Path("data/closed_loop_data_noisy.npz"))
    parser.add_argument("--model", type=Path, default=Path("data/nn_ode_large.pt"))
    parser.add_argument("--segment", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("nn_ode_closed_loop.png"))
    arguments = parser.parse_args()

    data = np.load(arguments.data)
    function, checkpoint = load_ode(arguments.model)
    true_angle, true_torque = true_closed_loop(data, arguments.segment, arguments.steps)
    ode_angle, ode_torque = ode_closed_loop(function, checkpoint, data, arguments.segment,
                                            arguments.steps)
    angle_error = ode_angle - true_angle
    torque_error = ode_torque - true_torque
    print("位置 RMSE (rad):", np.sqrt(np.mean(angle_error ** 2, axis=0)))
    print("力矩 RMSE (N·m):", np.sqrt(np.mean(torque_error ** 2, axis=0)))
    print("位置最大误差 (rad):", np.max(np.abs(angle_error), axis=0))
    print("力矩最大误差 (N·m):", np.max(np.abs(torque_error), axis=0))

    time = np.arange(arguments.steps) * 0.01
    labels = ["Joint 1 (big yaw)", "Joint 2 (roll)", "Joint 3 (small yaw)", "Joint 4 (pitch)"]
    figure, axes = plt.subplots(4, 2, figsize=(12.0, 12.0), sharex=True)
    for joint in range(4):
        axes[joint, 0].plot(time, true_angle[:, joint], label="True dynamics", linewidth=1.8)
        axes[joint, 0].plot(time, ode_angle[:, joint], label="Neural ODE",
                            linewidth=1.5, linestyle="--")
        axes[joint, 0].set_ylabel("angle (rad)")
        axes[joint, 0].set_title(labels[joint] + " position")
        axes[joint, 0].grid(True, alpha=0.3)
        axes[joint, 1].plot(time, true_torque[:, joint], label="True dynamics", linewidth=1.8)
        axes[joint, 1].plot(time, ode_torque[:, joint], label="Neural ODE",
                            linewidth=1.5, linestyle="--")
        axes[joint, 1].set_ylabel("torque (N·m)")
        axes[joint, 1].set_title(labels[joint] + " torque")
        axes[joint, 1].grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    axes[0, 0].legend(loc="best")
    figure.suptitle("Closed-loop PID: true dynamics vs Neural ODE")
    figure.tight_layout()
    figure.subplots_adjust(top=0.94)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=170)
    print("图像已保存到", arguments.output)


if __name__ == "__main__":
    main()
