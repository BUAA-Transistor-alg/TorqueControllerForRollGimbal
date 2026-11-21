"""逐段线性估计底盘系重力方向（几何量），并检验在真值参数下的恢复精度。

底盘静止时重力力矩对底盘系重力方向 ĝ 严格线性：tau_grav = G(q) @ ĝ。
每个采样点：
    tau_ng = 零重力模型的力矩（惯性 + 科氏 + 阻尼，与地形无关）
    G 的三列 = ĝ 取底盘系三个单位向量时的重力力矩
于是 tau_obs - tau_ng = G @ ĝ 是 3 参数线性问题，加 |ĝ| = 1 的约束后逐段求解。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

import gimbal_dynamics as gd
import identify_physical as ip
import params

# 三个参考地形，使底盘系重力方向分别等于 e_x、e_y、e_z（底盘偏航取 0）
REFERENCE = (gd.make_terrain(math.pi / 2, 0.0),
             gd.make_terrain(math.pi / 2, math.pi / 2),
             gd.make_terrain(math.pi, 0.0))


def model_with_gravity(raw: np.ndarray, gravity: float):
    return gd.make_model(raw[0:4], raw[4:16].reshape(4, 3), raw[16:40].reshape(4, 6),
                         raw[40:52].reshape(4, 3), raw[52:56], gravity)


def gravity_matrix(context, model_full, model_zero, state, qdd):
    """4x3：重力力矩对底盘系重力方向的雅可比。"""
    columns = []
    for terrain in REFERENCE:
        full = np.asarray(context.inverse_dynamics(model_full, terrain, gd.make_base(),
                                                   state, qdd))
        columns.append(full)
    zero = np.asarray(context.inverse_dynamics(model_zero, REFERENCE[0], gd.make_base(),
                                               state, qdd))
    return np.column_stack([column - zero for column in columns])


def solve_direction(matrix, target, iterations=200):
    direction = np.array([0.0, 0.0, -1.0])
    hessian = matrix.T @ matrix
    for _ in range(iterations):
        gradient = matrix.T @ (matrix @ direction - target)
        step = np.linalg.solve(hessian + 1.0e-9 * np.eye(3), gradient)
        direction = direction - step
        direction = direction / np.linalg.norm(direction)
    return direction


def solve_direction_aggregate(normal: np.ndarray, moment: np.ndarray,
                              iterations: int = 200) -> np.ndarray:
    """求解 min g^T normal g - 2 moment^T g，约束 |g| = 1。"""
    direction = np.array([0.0, 0.0, -1.0])
    for _ in range(iterations):
        gradient = normal @ direction - moment
        direction = direction - np.linalg.solve(normal + 1.0e-9 * np.eye(3), gradient)
        direction = direction / np.linalg.norm(direction)
    return direction


def rotation_z(angle: float) -> np.ndarray:
    return np.array([[math.cos(angle), -math.sin(angle), 0.0],
                     [math.sin(angle), math.cos(angle), 0.0],
                     [0.0, 0.0, 1.0]])


def rotation_y(angle: float) -> np.ndarray:
    return np.array([[math.cos(angle), 0.0, math.sin(angle)],
                     [0.0, 1.0, 0.0],
                     [-math.sin(angle), 0.0, math.cos(angle)]])


def base_gravity(slope: float, direction: float, yaw: float) -> np.ndarray:
    rotation = rotation_z(direction) @ rotation_y(slope) @ rotation_z(-direction)
    return -(rotation @ rotation_z(yaw)).T @ np.array([0.0, 0.0, 1.0])


def main() -> None:
    parser = argparse.ArgumentParser(description="几何量线性可辨识性检验")
    parser.add_argument("--data", type=Path, default=Path("data/closed_loop_data_noisy.npz"))
    parser.add_argument("--model", type=Path, default=None, help="默认使用真值参数")
    parser.add_argument("--width", type=int, default=51)
    parser.add_argument("--segments", type=int, default=5)
    parser.add_argument("--geo-sigma", type=float, default=5.0)
    arguments = parser.parse_args()

    import regressor_fit as regfit

    class Args:
        pass

    args = Args()
    args.data = arguments.data
    args.method = "sg"
    args.width = arguments.width
    args.order = 5
    args.segments = 100
    args.stride = 3
    args.margin = None
    problem = regfit.Problem(args)
    problem.prepare(args.method, args.width, args.order)
    if arguments.model is None:
        raw = np.concatenate([params.MASSES, np.array(params.COM).reshape(-1),
                              np.array(params.INERTIA).reshape(-1),
                              np.array(params.JOINT_OFFSETS).reshape(-1), params.DAMPING])
    else:
        raw = np.load(arguments.model)["model"]
    model_full = model_with_gravity(raw, params.GRAVITY)
    model_zero = model_with_gravity(raw, 0.0)
    torque = savgol_filter(np.load(arguments.data)["tau"], 9, 3, axis=1)
    rng = np.random.default_rng(20261220)
    corrupted = []
    for segment in range(len(problem.terrains)):
        corrupted.append((problem.terrains[segment].slope_angle
                          + math.radians(rng.normal(0.0, arguments.geo_sigma)),
                          problem.terrains[segment].slope_direction
                          + math.radians(rng.normal(0.0, 2.0 * arguments.geo_sigma)),
                          problem.bases[segment].yaw
                          + math.radians(rng.normal(0.0, 3.0 * arguments.geo_sigma))))
    print("段号  真值倾角(度)  加噪后误差(度)  恢复后误差(度)")
    errors_before = []
    errors_after = []
    for segment in range(arguments.segments):
        true_gravity = base_gravity(problem.terrains[segment].slope_angle,
                                    problem.terrains[segment].slope_direction,
                                    problem.bases[segment].yaw)
        corrupted_gravity = base_gravity(*corrupted[segment])
        estimate = np.zeros(3)
        count = 0
        for s, k in problem.samples:
            if s != segment:
                continue
            state = gd.make_state(problem.states[0, s, k], problem.states[1, s, k])
            qdd = problem.states[2, s, k]
            matrix = gravity_matrix(problem.ctx, model_full, model_zero, state, qdd)
            zero_torque = np.asarray(problem.ctx.inverse_dynamics(
                model_zero, REFERENCE[0], gd.make_base(), state, qdd))
            target = torque[s, k] - zero_torque
            estimate += solve_direction(matrix, target)
            count += 1
        estimate /= count
        estimate /= np.linalg.norm(estimate)
        before = math.degrees(math.acos(np.clip(corrupted_gravity @ true_gravity, -1.0, 1.0)))
        after = math.degrees(math.acos(np.clip(estimate @ true_gravity, -1.0, 1.0)))
        errors_before.append(before)
        errors_after.append(after)
        print("%3d   %8.1f   %10.2f   %10.2f" % (
            segment, math.degrees(problem.terrains[segment].slope_angle), before, after))
    print("中位数：加噪后 %.2f 度，恢复后 %.2f 度" % (
        np.median(errors_before), np.median(errors_after)))
    problem.ctx.close()


if __name__ == "__main__":
    main()