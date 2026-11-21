"""对几何量偏差稳健的辨识：逐段估计底盘系重力方向，与物理参数交替优化。

流程：
  1. 用初始模型（默认取此前在无偏数据上辨识的模型）逐段线性估计底盘系重力方向，
     覆盖记录中的坡度/坡向/底盘偏航；
  2. 用修正后的几何量重新做物理参数回归；
  3. 重复 1 到 2，直到参数稳定。
验收使用几何量无偏差的评估集。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

import eval_openloop as open_loop
import gimbal_dynamics as gd
import identify_physical as ip
import param_basis as pb
import regressor_full as regfull
import estimate_geometry_linear as egl


def geometry_step(problem, model, torque, segments):
    """对每一段估计底盘系重力方向并写回 problem 的 terrain 与 base。"""
    model_full = egl.model_with_gravity(model, 9.81)
    model_zero = egl.model_with_gravity(model, 0.0)
    directions = []
    for segment in range(segments):
        normal = np.zeros((3, 3))
        moment = np.zeros(3)
        for s, k in problem.samples:
            if s != segment:
                continue
            state = gd.make_state(problem.states[0, s, k], problem.states[1, s, k])
            qdd = problem.states[2, s, k]
            matrix = egl.gravity_matrix(problem.ctx, model_full, model_zero, state, qdd)
            zero_torque = np.asarray(problem.ctx.inverse_dynamics(
                model_zero, egl.REFERENCE[0], gd.make_base(), state, qdd))
            target = torque[s, k] - zero_torque
            normal += matrix.T @ matrix
            moment += matrix.T @ target
        direction = egl.solve_direction_aggregate(normal, moment)
        directions.append(direction)
        slope = math.acos(float(np.clip(-direction[2], -1.0, 1.0)))
        azimuth = math.atan2(float(direction[1]), float(direction[0]))
        problem.terrains[segment] = gd.make_terrain(slope, azimuth)
        problem.bases[segment] = gd.make_base(yaw=0.0)
    return np.array(directions)


def main() -> None:
    parser = argparse.ArgumentParser(description="几何量偏差稳健辨识")
    parser.add_argument("--data", type=Path, default=Path("data/low_excitation_data_noisy.npz"))
    parser.add_argument("--init", type=Path, default=Path("data/identified_final.npz"),
                        help="用于第一次几何估计的初始模型")
    parser.add_argument("--prior", type=Path, default=Path("data/identified_analytic.npz"),
                        help="物理参数回归的先验")
    parser.add_argument("--save", type=Path, default=Path("data/geo_robust_model.npz"))
    parser.add_argument("--geometry-sigma", type=float, default=5.0)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--width", type=int, default=31)
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--mode", default="corrected")
    parser.add_argument("--seed", type=int, default=20261220)
    parser.add_argument("--reports", type=Path, default=Path("data/geo_robust_reports.json"))
    arguments = parser.parse_args()

    class Args:
        pass

    args = Args()
    args.data = arguments.data
    args.init = arguments.prior
    args.save = arguments.save
    args.method = "sg"
    args.width = arguments.width
    args.order = arguments.order
    args.mode = arguments.mode
    args.stride = arguments.stride
    args.segments = 100
    args.margin = None
    args.outer = 0
    args.refine = 0
    args.prior_sigma = 0.5
    args.evaluate = False

    import regressor_fit as regfit
    problem = regfit.Problem(args)
    problem.prepare(args.method, args.width, args.order)
    segments = len(problem.terrains)
    truth = [(float(t.slope_angle), float(t.slope_direction),
              float(b.yaw)) for t, b in zip(problem.terrains, problem.bases)]
    rng = np.random.default_rng(arguments.seed)
    if arguments.geometry_sigma > 0.0:
        for segment in range(segments):
            slope = truth[segment][0] + math.radians(rng.normal(
                0.0, arguments.geometry_sigma))
            direction = truth[segment][1] + math.radians(rng.normal(
                0.0, 2.0 * arguments.geometry_sigma))
            yaw = truth[segment][2] + math.radians(rng.normal(
                0.0, 3.0 * arguments.geometry_sigma))
            problem.terrains[segment] = gd.make_terrain(slope, direction)
            problem.bases[segment] = gd.make_base(yaw=yaw)
    corrupted_errors = []
    for segment in range(segments):
        true_gravity = egl.base_gravity(*truth[segment])
        corrupted_gravity = egl.base_gravity(problem.terrains[segment].slope_angle,
                                             problem.terrains[segment].slope_direction,
                                             problem.bases[segment].yaw)
        corrupted_errors.append(math.degrees(math.acos(
            float(np.clip(corrupted_gravity @ true_gravity, -1.0, 1.0)))))
    print("加噪后几何误差中位数 %.2f 度，最大 %.2f 度" % (
        np.median(corrupted_errors), np.max(corrupted_errors)), flush=True)
    torque = savgol_filter(np.load(arguments.data)["tau"], 9, 3, axis=1)
    offsets = pb.NOMINAL_OFFSETS.copy()
    model = np.load(arguments.init)["model"]
    reports = []
    basis = None
    for iteration in range(arguments.iterations):
        directions = geometry_step(problem, model, torque, segments)
        errors = []
        for segment in range(segments):
            true_gravity = egl.base_gravity(*truth[segment])
            errors.append(math.degrees(math.acos(
                float(np.clip(directions[segment] @ true_gravity, -1.0, 1.0)))))
        prior = np.load(arguments.prior)["model"]
        model, loss, basis, coordinate = regfull.solve(problem, offsets, prior,
                                                      arguments.mode)
        metrics = open_loop.evaluate(model)
        reports.append({"iteration": iteration, "loss": float(loss),
                        "geometry_error_median_deg": float(np.median(errors)),
                        "geometry_error_max_deg": float(np.max(errors)),
                        "rmse50": metrics["rmse"][50].tolist(),
                        "rmse25": metrics["rmse"][25].tolist(),
                        "direction_wrong": metrics["direction_wrong"]})
        print("迭代 %d：几何误差中位数 %.2f 度（最大 %.2f），RMSE50 %s，方向 %d" % (
            iteration, np.median(errors), np.max(errors),
            " ".join("%.2e" % v for v in metrics["rmse"][50]),
            metrics["direction_wrong"]), flush=True)
    problem.ctx.close()
    arguments.reports.write_text(json.dumps(
        {"corrupted_median_deg": float(np.median(corrupted_errors)), "iterations": reports},
        indent=2, ensure_ascii=False))
    np.savez_compressed(arguments.save, model=model, basis=basis)
    print("已保存", arguments.save)


if __name__ == "__main__":
    main()
