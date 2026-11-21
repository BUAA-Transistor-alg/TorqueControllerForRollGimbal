"""从神经网络动力学模型中采集轨迹数据，供物理参数拟合使用。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import numpy as np
import nn_closed_loop as ncl


def main() -> None:
    parser = argparse.ArgumentParser(description="从神经网络模型采集数据")
    parser.add_argument("--source", type=Path, default=Path("data/closed_loop_data_noisy.npz"))
    parser.add_argument("--model", type=Path, default=Path("data/nn_dynamics.pt"))
    parser.add_argument("--save", type=Path, default=Path("data/nn_collected.npz"))
    parser.add_argument("--segments", type=int, default=100)
    parser.add_argument("--points", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261124)
    arguments = parser.parse_args()

    source = np.load(arguments.source)
    checkpoint = torch.load(arguments.model, map_location="cpu", weights_only=False)
    model = ncl.tn.DeltaNet(checkpoint["mean_x"].shape[0])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    means = {"x": checkpoint["mean_x"], "y": checkpoint["mean_y"]}
    stds = {"x": checkpoint["std_x"], "y": checkpoint["std_y"]}
    rng = np.random.default_rng(arguments.seed)

    q = np.zeros((arguments.segments, arguments.points, 4))
    qdot = np.zeros_like(q)
    tau = np.zeros_like(q)
    target = np.zeros_like(q)
    target_velocity = np.zeros_like(q)
    terrain = np.zeros((arguments.segments, 2))
    base_yaw = np.zeros(arguments.segments)
    initial = np.zeros((arguments.segments, 4))
    initial_velocity = np.zeros((arguments.segments, 4))
    kp = np.zeros((arguments.segments, 4))
    ki = np.zeros((arguments.segments, 4))
    kd = np.zeros((arguments.segments, 4))
    torque_limit = np.zeros(arguments.segments)
    integral_limit = np.zeros(arguments.segments)

    for segment in range(arguments.segments):
        source_segment = int(rng.integers(0, source["q"].shape[0]))
        terrain[segment] = source["terrain"][source_segment]
        base_yaw[segment] = source["base_yaw"][source_segment]
        kp[segment] = source["kp"][source_segment]
        ki[segment] = source["ki"][source_segment]
        kd[segment] = source["kd"][source_segment]
        torque_limit[segment] = source["torque_limit"][source_segment]
        integral_limit[segment] = source["integral_limit"][source_segment]
        target[segment] = source["target"][source_segment, :arguments.points]
        target_velocity[segment] = source["target_velocity"][source_segment, :arguments.points]
        state_q = np.array(source["initial"][source_segment])
        state_qdot = np.array(source["initial_velocity"][source_segment])
        initial[segment] = state_q
        initial_velocity[segment] = state_qdot
        terrain_array = terrain[segment]
        yaw = float(base_yaw[segment])
        command_sequence = source["tau"][source_segment, :arguments.points]
        for k in range(arguments.points):
            q[segment, k] = state_q
            qdot[segment, k] = state_qdot
            tau[segment, k] = command_sequence[k]
            if k + 1 < arguments.points:
                state_q, state_qdot = ncl.nn_next(model, means, stds, state_q, state_qdot,
                                                  command_sequence[k], terrain_array, yaw)
        if not np.all(np.isfinite(state_q)) or not np.all(np.isfinite(state_qdot)):
            raise RuntimeError("神经网络采集轨迹出现非有限值")

    np.savez_compressed(
        arguments.save, q=q, qdot=qdot, tau=tau, target=target,
        target_velocity=target_velocity, terrain=terrain, base_yaw=base_yaw,
        initial=initial, initial_velocity=initial_velocity,
        kp=kp, ki=ki, kd=kd, torque_limit=torque_limit, integral_limit=integral_limit,
        dt=0.01, points=arguments.points,
    )
    print("已保存神经网络采集数据", arguments.save)
    print("q range", q.min(axis=(0, 1)), q.max(axis=(0, 1)))
    print("qdot range", qdot.min(axis=(0, 1)), qdot.max(axis=(0, 1)))
    print("tau range", tau.min(axis=(0, 1)), tau.max(axis=(0, 1)))


if __name__ == "__main__":
    main()
