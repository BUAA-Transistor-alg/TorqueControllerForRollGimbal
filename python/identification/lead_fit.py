"""主辨识脚本：44 维线性参数最小二乘 + 物理可行参数化精化。

原理：
    tau = Y(q, qdot, qdd, 地形, 底盘偏航) @ p，  p 为 44 维参数（40 惯性 + 4 阻尼）
Y 只依赖几何（偏置）与状态，与惯性参数无关，由 tcrg_inertial_basis 给出。
可辨识子空间由 param_basis 的 SVD 得到；零空间分量对动力学无影响，取物理先验。
最终模型在物理可行参数化（质量 softplus、惯量 Cholesky）下直接最小化数据残差，
保证质量为正、惯量正定。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.signal import savgol_filter

import gimbal_dynamics as gd
import identify_physical as ip
import param_basis as pb


def smooth_data(data, width: int, order: int, tau_width: int, derivative_width: int | None = None):
    derivative_width = width if derivative_width is None else derivative_width
    q_smooth = savgol_filter(data["q"], width, order, axis=1)
    velocity = savgol_filter(data["q"], derivative_width, order, deriv=1, delta=0.01, axis=1)
    acceleration = savgol_filter(data["q"], derivative_width, order, deriv=2, delta=0.01, axis=1)
    tau_smooth = savgol_filter(data["tau"], tau_width, 3, axis=1)
    return q_smooth, velocity, acceleration, tau_smooth


def normal_equations(data, q, velocity, acceleration, tau_target, offsets, trim: int,
                     stride: int, use_measured_tau: bool):
    """返回 G = A^T A（44x44）、c = A^T b（44）与残差统计。"""
    segments, points = q.shape[0], q.shape[1]
    samples = [(segment, step) for segment in range(segments)
               for step in range(trim, points - trim, stride)]
    reference = pb.nominal_model()
    reference[40:52] = np.asarray(offsets, dtype=float).reshape(-1)
    model = ip.build_model(reference)
    matrix = np.zeros((44, 4 * len(samples)))
    values = np.zeros(4 * len(samples))
    context = gd.Context(1)
    try:
        for index, (segment, step) in enumerate(samples):
            terrain = gd.make_terrain(float(data["terrain"][segment, 0]),
                                      float(data["terrain"][segment, 1]))
            base = gd.make_base(yaw=float(data["base_yaw"][segment]))
            state = gd.make_state(q[segment, step], velocity[segment, step])
            basis_values = np.array(context.inertial_basis(model, terrain, base, state,
                                                           acceleration[segment, step]))
            matrix[:, 4 * index:4 * index + 4] = basis_values
            values[4 * index:4 * index + 4] = tau_target[segment, step]
    finally:
        context.close()
    gram = matrix @ matrix.T
    moment = matrix @ values
    return gram, moment, matrix, values, len(samples)


def linear_solution(gram, moment, prior_values, basis, ridge: float):
    reduced = basis.T @ gram @ basis
    target = basis.T @ (moment - gram @ prior_values)
    if ridge > 0.0:
        reduced = reduced + ridge * np.eye(reduced.shape[0])
    coefficients = np.linalg.solve(reduced, target)
    values = prior_values + basis @ coefficients
    singular = np.linalg.eigvalsh(reduced)
    return values, coefficients, singular


def physical_refinement(gram, moment, prior, offsets, null_weight: float = 1.0e-2,
                        prior_values: np.ndarray | None = None, max_nfev: int = 200):
    """在物理可行参数化（质量 softplus、惯量 Cholesky）下最小化数据残差。

    数据代价 p^T G p - 2 p^T c 等价于加权残差 ||Lambda^{1/2} V^T (p - p*)||，
    只在 G 的 28 维值域上求解；零空间分量用很小的权重拉向先验，保证物理合理性。
    """
    from scipy.optimize import least_squares
    from param_basis import _raw_from_theta, _theta_from_raw

    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    keep = eigenvalues > eigenvalues[-1] * 1.0e-9
    square_root = np.sqrt(eigenvalues[keep])
    projection = eigenvectors[:, keep].T
    optimum = eigenvectors[:, keep] @ (projection @ moment / eigenvalues[keep])
    null_basis = eigenvectors[:, ~keep]
    prior_values = pb.inertial_from_raw(prior) if prior_values is None else prior_values
    prior_null = null_basis.T @ prior_values

    def residual(theta):
        values = pb.inertial_from_raw(_raw_from_theta(theta, offsets))
        main = square_root * (projection @ (values - optimum))
        null = null_weight * (null_basis.T @ values - prior_null)
        return np.concatenate([main, null])

    start = _theta_from_raw(prior)
    result = least_squares(residual, start, method="lm", max_nfev=max_nfev,
                           x_scale="jac", ftol=1.0e-14, xtol=1.0e-14, gtol=1.0e-14)
    raw = _raw_from_theta(result.x, offsets)
    return raw, pb.inertial_from_raw(raw)


def report_model(raw: np.ndarray) -> None:
    print("质量", np.array2string(raw[0:4], precision=4))
    for body in range(4):
        inertia = np.array([[raw[16 + 6 * body], raw[16 + 6 * body + 3], raw[16 + 6 * body + 4]],
                            [raw[16 + 6 * body + 3], raw[17 + 6 * body], raw[16 + 6 * body + 5]],
                            [raw[16 + 6 * body + 4], raw[16 + 6 * body + 5], raw[18 + 6 * body]]])
        print("  B%d 惯量特征值 %s" % (body + 1, np.array2string(
            np.linalg.eigvalsh(inertia), precision=4)))
    print("偏置", np.array2string(raw[40:52], precision=5))
    print("阻尼", np.array2string(raw[52:56], precision=5))


def main() -> None:
    parser = argparse.ArgumentParser(description="44 维线性参数辨识 + 物理精化")
    parser.add_argument("--data", type=Path, default=Path("data/closed_loop_data_noisy.npz"))
    parser.add_argument("--save", type=Path, default=Path("data/identified_lead.npz"))
    parser.add_argument("--prior", type=Path, default=Path("data/coarse_prior.npz"))
    parser.add_argument("--smooth", type=int, default=31)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--derivative", type=int, default=0)
    parser.add_argument("--tau-smooth", type=int, default=9)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--measured-tau", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--eval-data", type=Path, default=None)
    arguments = parser.parse_args()

    data = np.load(arguments.data)
    width = arguments.smooth
    trim = width // 2 + 1
    q, velocity, acceleration, tau = smooth_data(data, width, arguments.order,
                                                 arguments.tau_smooth,
                                                 arguments.derivative or None)
    prior = pb.prior_model(arguments.prior)
    offsets = pb.NOMINAL_OFFSETS.reshape(-1).copy()
    basis, singular, rank = pb.compute_basis(offsets=offsets)
    prior_values = pb.inertial_from_raw(prior)
    target_tau = tau if not arguments.measured_tau else data["tau"]
    gram, moment, matrix, values, count = normal_equations(
        data, q, velocity, acceleration, target_tau, offsets, trim, arguments.stride,
        arguments.measured_tau)
    estimate_values, coefficients, singular_values = linear_solution(
        gram, moment, prior_values, basis, arguments.ridge)
    residual = matrix.T @ estimate_values - values
    print("平滑宽度 %d，样本 %d，残差 RMS %.4e N·m（逐关节 %s）" % (
        width, count, np.sqrt(np.mean(residual ** 2)),
        np.array2string(np.sqrt(np.mean(residual.reshape(-1, 4) ** 2, axis=0)), precision=4)))
    print("可辨识维数 %d，回归条件数 %.3e" % (rank, singular_values[-1] / singular_values[0]))

    raw, refined_values = physical_refinement(gram, moment, prior, offsets,
                                              prior_values=prior_values)
    residual_refined = matrix.T @ refined_values - values
    print("物理精化后残差 RMS %.4e N·m，可辨识坐标与线性解偏差 %.3e" % (
        np.sqrt(np.mean(residual_refined ** 2)),
        np.linalg.norm(basis.T @ (refined_values - estimate_values))))
    report_model(raw)

    arguments.save.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.save, model=raw, values=refined_values,
                        linear_values=estimate_values, basis=basis,
                        residual_rms=float(np.sqrt(np.mean(residual_refined ** 2))))
    print("已保存", arguments.save)

    if arguments.evaluate:
        import eval_openloop as open_loop
        metrics = open_loop.evaluate(raw, arguments.eval_data or arguments.data)
        print(open_loop.format_report(metrics))


if __name__ == "__main__":
    main()