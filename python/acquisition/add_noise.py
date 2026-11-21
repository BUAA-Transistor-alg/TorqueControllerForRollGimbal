"""给闭环 PID 采集数据加入高斯测量噪声。"""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="为闭环数据加入高斯噪声")
    parser.add_argument("--input", type=Path, default=Path("data/closed_loop_data.npz"))
    parser.add_argument("--output", type=Path, default=Path("data/closed_loop_data_noisy.npz"))
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--torque-std", type=float, default=0.01)
    parser.add_argument("--velocity-std", type=float, default=0.1)
    parser.add_argument("--position-std", type=float, default=0.001)
    arguments = parser.parse_args()

    source = np.load(arguments.input)
    rng = np.random.default_rng(arguments.seed)
    q = source["q"].copy()
    qdot = source["qdot"].copy()
    tau = source["tau"].copy()
    q_noisy = q + rng.normal(0.0, arguments.position_std, q.shape)
    qdot_noisy = qdot + rng.normal(0.0, arguments.velocity_std, qdot.shape)
    tau_noisy = tau + rng.normal(0.0, arguments.torque_std, tau.shape)
    output = {key: source[key] for key in source.files}
    output.update({
        "q": q_noisy,
        "qdot": qdot_noisy,
        "tau": tau_noisy,
        "q_clean": q,
        "qdot_clean": qdot,
        "tau_clean": tau,
        "initial": q_noisy[:, 0].copy(),
        "initial_velocity": qdot_noisy[:, 0].copy(),
        "initial_clean": q[:, 0].copy(),
        "initial_velocity_clean": qdot[:, 0].copy(),
        "noise_position_std": arguments.position_std,
        "noise_velocity_std": arguments.velocity_std,
        "noise_torque_std": arguments.torque_std,
    })
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.output, **output)
    print("已保存含噪数据到", arguments.output)
    print("位置噪声标准差", arguments.position_std)
    print("速度噪声标准差", arguments.velocity_std)
    print("力矩噪声标准差", arguments.torque_std)


if __name__ == "__main__":
    main()
