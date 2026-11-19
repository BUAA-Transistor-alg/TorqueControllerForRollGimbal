"""只用含噪数据和标称几何，用线性最小二乘得到粗略参数初值。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

import gimbal_dynamics as gd
import gradient_fit as gf
import identify_physical as ip


def nominal_model() -> np.ndarray:
    x = np.zeros(56)
    x[0:4] = [1.5, 1.5, 1.5, 1.5]
    for body in range(4):
        x[4 + 3 * body:7 + 3 * body] = [0.0, 0.05, 0.0]
        x[16 + 6 * body:22 + 6 * body] = [0.01, 0.01, 0.01, 0.0, 0.0, 0.0]
    x[40:52] = np.array([[0.0, 0.0, 0.0],
                         [0.0, 0.09, 0.039],
                         [0.0, 0.0, 0.0521],
                         [0.0, 0.0, 0.1579]]).reshape(-1)
    x[52:56] = 0.1
    return x


def project_model(x: np.ndarray) -> np.ndarray:
    x[0:4] = np.clip(x[0:4], 0.1, 6.0)
    for body in range(4):
        x[4 + 3 * body:7 + 3 * body] = np.clip(x[4 + 3 * body:7 + 3 * body], -0.5, 0.5)
        matrix = np.array([[x[16 + 6 * body], x[16 + 6 * body + 3], x[16 + 6 * body + 4]],
                           [x[16 + 6 * body + 3], x[17 + 6 * body], x[16 + 6 * body + 5]],
                           [x[16 + 6 * body + 4], x[16 + 6 * body + 5], x[18 + 6 * body]]])
        values, vectors = np.linalg.eigh(matrix)
        values = np.clip(values, 1.0e-4, 0.5)
        matrix = vectors @ np.diag(values) @ vectors.T
        x[16 + 6 * body] = matrix[0, 0]
        x[17 + 6 * body] = matrix[1, 1]
        x[18 + 6 * body] = matrix[2, 2]
        x[16 + 6 * body + 3] = matrix[0, 1]
        x[16 + 6 * body + 4] = matrix[0, 2]
        x[16 + 6 * body + 5] = matrix[1, 2]
    x[52:56] = np.clip(x[52:56], 0.0, 2.0)
    return x


def main() -> None:
    parser = argparse.ArgumentParser(description="含噪数据的线性最小二乘初值")
    parser.add_argument("--data", type=Path, default=Path("data/closed_loop_data_noisy.npz"))
    parser.add_argument("--save", type=Path, default=Path("data/ls_initial.npz"))
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--smooth", type=int, default=101)
    parser.add_argument("--ridge", type=float, default=1e-3)
    arguments = parser.parse_args()

    data = np.load(arguments.data)
    q = data["q"]
    qdot_smooth = savgol_filter(data["qdot"], arguments.smooth, 3, axis=1)
    dt = float(data["dt"])
    qdd = np.gradient(qdot_smooth, dt, axis=1)
    tau = data["tau"]
    samples = [(segment, index) for segment in range(q.shape[0])
               for index in range(0, q.shape[1], arguments.stride)]
    print("样本数", len(samples))

    template = ip.build_model(nominal_model())
    context = gd.Context(1)
    rows = []
    values = []
    try:
        for segment, index in samples:
            terrain = gd.make_terrain(float(data["terrain"][segment, 0]),
                                      float(data["terrain"][segment, 1]))
            base = gd.make_base(yaw=float(data["base_yaw"][segment]))
            state = gd.make_state(q[segment, index], qdot_smooth[segment, index])
            basis = np.asarray(context.inertial_basis(
                template, terrain, base, state, qdd[segment, index]))
            rows.append(np.transpose(basis))
            values.append(tau[segment, index])
    finally:
        context.close()

    matrix = np.vstack(rows)
    target = np.concatenate(values)
    # 总质量 6 kg 是已知规格，作为约束加入最小二乘
    constraint = np.zeros((1, 44))
    constraint[0, 0:4] = 1.0
    matrix = np.vstack([matrix, 100.0 * constraint])
    target = np.concatenate([target, [600.0]])
    scales = np.array([1.0] * 4 + [0.05] * 12 + [0.01] * 24 + [0.1] * 4)
    for body in range(4):
        # 质量列、h 列、I_o 列统一按惯性参数尺度处理
        scales[body * 10] = 1.0
        scales[body * 10 + 1:body * 10 + 4] = 0.05
        scales[body * 10 + 4:body * 10 + 10] = 0.01
    scales[40:44] = 0.1
    initial = nominal_model()
    theta0 = np.zeros(44)
    for body in range(4):
        m = initial[body]
        c = initial[4 + 3 * body:7 + 3 * body]
        inertia = ip.inertia_matrix(body) if False else np.array(
            [[initial[16 + 6 * body], initial[16 + 6 * body + 3], initial[16 + 6 * body + 4]],
             [initial[16 + 6 * body + 3], initial[17 + 6 * body], initial[16 + 6 * body + 5]],
             [initial[16 + 6 * body + 4], initial[16 + 6 * body + 5], initial[18 + 6 * body]]])
        spatial = inertia + m * ((c @ c) * np.eye(3) - np.outer(c, c))
        theta0[body * 10:body * 10 + 10] = [m, m * c[0], m * c[1], m * c[2],
                                            spatial[0, 0], spatial[1, 1], spatial[2, 2],
                                            spatial[0, 1], spatial[0, 2], spatial[1, 2]]
    theta0[40:44] = initial[52:56]
    scaled = matrix * scales[None, :]
    normal = scaled.T @ scaled
    ridge = arguments.ridge * np.trace(normal) / normal.shape[0]
    delta = np.linalg.solve(normal + ridge * np.eye(44), scaled.T @ (target - matrix @ theta0))
    parameters = theta0 + scales * delta
    residual = matrix @ parameters - target
    print("线性最小二乘残差均方根", np.sqrt(np.mean(residual ** 2)))

    x = nominal_model()
    for body in range(4):
        block = parameters[body * 10:body * 10 + 10]
        mass = max(block[0], 0.02)
        c = block[1:4] / mass
        spatial = np.array([[block[4], block[7], block[8]],
                            [block[7], block[5], block[9]],
                            [block[8], block[9], block[6]]])
        inertia = spatial - mass * ((c @ c) * np.eye(3) - np.outer(c, c))
        x[body] = mass
        x[4 + 3 * body:7 + 3 * body] = c
        x[16 + 6 * body:22 + 6 * body] = [inertia[0, 0], inertia[1, 1], inertia[2, 2],
                                          inertia[0, 1], inertia[0, 2], inertia[1, 2]]
    x[52:56] = np.maximum(parameters[40:44], 0.0)
    x = project_model(x)

    beta = ip.beta_from_model(x)
    reduced = gf.project(beta[gf.KEEP])
    arguments.save.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.save, reduced=reduced, full_beta=gf.expand_beta(reduced),
                        model=ip.map_beta(gf.expand_beta(reduced), True), ls_model=x)
    truth = ip.beta_of_prior()[gf.KEEP]
    error = reduced - truth
    scale = ip.beta_scales()[gf.KEEP]
    print("初值相对真值的归一化误差均方根", np.sqrt(np.mean((error / scale) ** 2)))
    print("初值相对真值的最大归一化误差", np.max(np.abs(error / scale)))
    print("最小二乘初值已保存到", arguments.save)


if __name__ == "__main__":
    main()
