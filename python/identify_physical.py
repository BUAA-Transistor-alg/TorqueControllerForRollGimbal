"""按 PARAMETER_IDENTIFIABILITY.md 的 35 个物理基参数做重参数化辨识。

基参数：
- B1：beta1 = Izz + m(cx^2+cy^2)，beta2 = m cx，beta3 = m cy。
- B2、B3、B4：等效质量 m、一阶矩 h(3)、a^T I_o a、p^T I_o p - q^T I_o q。
- 10 个独立安装偏置，4 个阻尼。
基参数到原始参数的反解选择保持正定的规范：B1 取 c1z=0、小正 Ixx/Iyy，
B2 到 B4 取垂直于关节轴的横向惯量迹为规范值、交叉项为零。o1.z、o2.z 固定为 0；
可用 --fix-o1-xy 把 o1.x、o1.y 也固定为 0，此时独立基参数为 33 个。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

import gimbal_dynamics as gd
import params
import identify

AXES = [np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])]
PLANES = [None, ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
          ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
          ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))]


def load_data(path: Path):
    data = np.load(path)
    q = data["q"]
    qdot = data["qdot"]
    tau = data["tau"]
    dt = float(data["dt"])
    qdd = np.gradient(qdot, dt, axis=1)
    return q, qdot, qdd, tau, data["terrain"], data["base_yaw"]


def inertia_matrix(index: int) -> np.ndarray:
    q = params.INERTIA[index]
    return np.array([[q[0], q[3], q[4]], [q[3], q[1], q[5]], [q[4], q[5], q[2]]])


def spatial_inertia(index: int) -> np.ndarray:
    m = params.MASSES[index]
    c = np.array(params.COM[index])
    return inertia_matrix(index) + m * ((c @ c) * np.eye(3) - np.outer(c, c))


def beta_names() -> list[str]:
    names = ["B1.beta1", "B1.beta2", "B1.beta3"]
    for i in range(1, 4):
        names += [f"B{i+1}.m", f"B{i+1}.hx", f"B{i+1}.hy", f"B{i+1}.hz",
                  f"B{i+1}.a_I_a", f"B{i+1}.diff"]
    names += ["o1.x", "o1.y", "o2.x", "o2.y",
              "o3.x", "o3.y", "o3.z", "o4.x", "o4.y", "o4.z"]
    names += ["d1", "d2", "d3", "d4"]
    return names


def beta_scales() -> np.ndarray:
    scales = [0.05, 0.2, 0.2]
    for _ in range(1, 4):
        scales += [0.5, 0.1, 0.1, 0.1, 0.001, 0.001]
    scales += [0.1] * 10
    scales += [0.1] * 4
    return np.array(scales)


def beta_of_prior() -> np.ndarray:
    beta = [params.INERTIA[0][2] + params.MASSES[0] *
            (params.COM[0][0] ** 2 + params.COM[0][1] ** 2),
            params.MASSES[0] * params.COM[0][0],
            params.MASSES[0] * params.COM[0][1]]
    for i in range(1, 4):
        m = params.MASSES[i]
        c = np.array(params.COM[i])
        h = m * c
        inertia = spatial_inertia(i)
        a = AXES[i]
        p, q = PLANES[i]
        beta += [m, h[0], h[1], h[2], a @ inertia @ a,
                 np.array(p) @ inertia @ np.array(p) - np.array(q) @ inertia @ np.array(q)]
    beta += [params.JOINT_OFFSETS[0][0], params.JOINT_OFFSETS[0][1],
             params.JOINT_OFFSETS[1][0], params.JOINT_OFFSETS[1][1],
             params.JOINT_OFFSETS[2][0], params.JOINT_OFFSETS[2][1],
             params.JOINT_OFFSETS[2][2],
             params.JOINT_OFFSETS[3][0], params.JOINT_OFFSETS[3][1],
             params.JOINT_OFFSETS[3][2]]
    beta += list(params.DAMPING)
    return np.array(beta, dtype=float)


def map_beta(beta: np.ndarray, fix_origin_xy: bool) -> np.ndarray:
    x = np.zeros(56)
    m1 = 1.0  # B1 质量是不可辨识规范，固定为 1 kg
    beta1, beta2, beta3 = beta[0], beta[1], beta[2]
    x[0] = m1
    cx, cy = beta2 / m1, beta3 / m1
    x[4], x[5], x[6] = cx, cy, 0.0
    gauge = 1.0e-2
    x[16:22] = [gauge, gauge, beta1 - m1 * (cx * cx + cy * cy), 0.0, 0.0, 0.0]
    position = 3
    for i in range(1, 4):
        m = beta[position]
        h = beta[position + 1:position + 4]
        position += 4
        c = h / m
        x[i] = m
        x[4 + 3 * i:7 + 3 * i] = c
        a = AXES[i]
        p, q = (np.array(PLANES[i][0]), np.array(PLANES[i][1]))
        longitudinal = beta[position] - m * ((c @ c) - (a @ c) ** 2)
        position += 1
        difference = beta[position] - m * ((q @ c) ** 2 - (p @ c) ** 2)
        position += 1
        gauge_trace = 0.01
        gauge_pq = 0.0
        gauge_ap = 0.0
        gauge_aq = 0.0
        inertia = (longitudinal * np.outer(a, a)
                   + 0.5 * (gauge_trace + difference) * np.outer(p, p)
                   + 0.5 * (gauge_trace - difference) * np.outer(q, q)
                   + gauge_pq * (np.outer(p, q) + np.outer(q, p))
                   + gauge_ap * (np.outer(a, p) + np.outer(p, a))
                   + gauge_aq * (np.outer(a, q) + np.outer(q, a)))
        x[16 + 6 * i:22 + 6 * i] = [inertia[0, 0], inertia[1, 1], inertia[2, 2],
                                    inertia[0, 1], inertia[0, 2], inertia[1, 2]]
    offsets = beta[position:position + 10]
    x[40] = 0.0 if fix_origin_xy else offsets[0]
    x[41] = 0.0 if fix_origin_xy else offsets[1]
    x[42] = 0.0
    x[43] = offsets[2]
    x[44] = offsets[3]
    x[45] = 0.0
    x[46], x[47], x[48] = offsets[4], offsets[5], offsets[6]
    x[49], x[50], x[51] = offsets[7], offsets[8], offsets[9]
    x[52:56] = beta[position + 10:position + 14]
    return x


def beta_from_model(x: np.ndarray) -> np.ndarray:
    beta = [x[18] + x[0] * (x[4] * x[4] + x[5] * x[5]),
            x[0] * x[4], x[0] * x[5]]
    for i in range(1, 4):
        m = x[i]
        c = x[4 + 3 * i:7 + 3 * i]
        h = m * c
        inertia = np.array([[x[16 + 6 * i], x[16 + 6 * i + 3], x[16 + 6 * i + 4]],
                            [x[16 + 6 * i + 3], x[17 + 6 * i], x[16 + 6 * i + 5]],
                            [x[16 + 6 * i + 4], x[16 + 6 * i + 5], x[18 + 6 * i]]])
        spatial = inertia + m * ((c @ c) * np.eye(3) - np.outer(c, c))
        a = AXES[i]
        p = np.array(PLANES[i][0])
        q = np.array(PLANES[i][1])
        beta += [m, h[0], h[1], h[2],
                 float(a @ spatial @ a),
                 float(p @ spatial @ p - q @ spatial @ q)]
    beta += [0.0, 0.0,
             x[43], x[44], x[46], x[47], x[48], x[49], x[50], x[51]]
    beta += list(x[52:56])
    return np.array(beta, dtype=float)


def build_model(x: np.ndarray) -> gd.TcrgModel:
    return gd.make_model(x[0:4], x[4:16].reshape(4, 3), x[16:40].reshape(4, 6),
                         x[40:52].reshape(4, 3), x[52:56])


class Problem:
    def __init__(self, path: Path, stride: int, segments: int, fix_origin_xy: bool):
        self.q, self.qdot, self.qdd, self.tau, self.terrain, self.base_yaw = load_data(path)
        self.samples = [(s, k) for s in range(segments)
                        for k in range(0, self.q.shape[1], stride)]
        self.context = gd.Context(1)
        self.fix_origin_xy = fix_origin_xy

    def close(self):
        self.context.close()

    def residual(self, beta: np.ndarray) -> np.ndarray:
        if self.fix_origin_xy:
            keep = np.ones(35, dtype=bool)
            keep[21] = False
            keep[22] = False
            full = np.zeros(35)
            full[keep] = beta
            beta = full
        x = map_beta(beta, self.fix_origin_xy)
        model = build_model(x)
        values = np.empty((len(self.samples), 4))
        for row, (s, k) in enumerate(self.samples):
            terrain = gd.make_terrain(float(self.terrain[s, 0]), float(self.terrain[s, 1]))
            base = gd.make_base(yaw=float(self.base_yaw[s]))
            state = gd.make_state(self.q[s, k], self.qdot[s, k])
            try:
                predicted = self.context.inverse_dynamics(model, terrain, base, state, self.qdd[s, k])
            except RuntimeError:
                values[row] = 1.0e3
                continue
            values[row] = np.asarray(predicted) - self.tau[s, k]
        return values.reshape(-1)


def fit(problem: Problem, max_nfev: int) -> None:
    from scipy.optimize import least_squares
    beta0 = beta_of_prior()
    scale = beta_scales()
    if problem.fix_origin_xy:
        # 删除 o1.x、o1.y 两个基参数
        keep = np.ones(35, dtype=bool)
        keep[21] = False
        keep[22] = False
        beta0 = beta0[keep]
        scale = scale[keep]
    rng = np.random.default_rng(5)
    start = beta0 + 0.02 * scale * rng.standard_normal(beta0.size)
    lower = beta0 - 2.0 * scale
    upper = beta0 + 2.0 * scale
    if problem.fix_origin_xy:
        # 质量、阻尼等保持物理范围
        pass
    print("基参数维度", beta0.size, "残差", len(problem.samples) * 4)
    result = least_squares(problem.residual, start, bounds=(lower, upper),
                           x_scale=scale, jac="2-point", diff_step=1e-4,
                           max_nfev=max_nfev, verbose=2)
    print("状态", result.status, result.message)
    print("代价 %.6e，残差最大 %.6e" % (result.cost, np.max(np.abs(result.fun))))
    full = beta0.copy()
    if problem.fix_origin_xy:
        full_beta = np.zeros(35)
        keep = np.ones(35, dtype=bool); keep[21] = False; keep[22] = False
        full_beta[keep] = result.x
        full_beta[21] = 0.0; full_beta[22] = 0.0
    else:
        full_beta = result.x
    truth = np.concatenate([params.MASSES, np.array(params.COM).reshape(-1),
                            np.array(params.INERTIA).reshape(-1),
                            np.array(params.JOINT_OFFSETS).reshape(-1), params.DAMPING])
    estimated = map_beta(full_beta, problem.fix_origin_xy)
    names = identify.parameter_names()
    error = estimated - truth
    print("原始参数误差（按尺度归一化）前 20：")
    order = np.argsort(np.abs(error / identify_parameter_scale()))[::-1]
    for index in order[:20]:
        print("  %-18s 真值 % .6e 估计 % .6e 归一化误差 % .3f" % (
            names[index], truth[index], estimated[index],
            error[index] / identify_parameter_scale()[index]))


def identify_parameter_scale() -> np.ndarray:
    return np.array([1.0] * 4 + [0.05] * 12 + [0.01] * 24 + [0.05] * 12 + [0.1] * 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="文档物理基参数重参数化辨识")
    parser.add_argument("--data", type=Path, default=Path("data/identification_data.npz"))
    parser.add_argument("--segments", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-nfev", type=int, default=25)
    parser.add_argument("--no-fix-o1-xy", action="store_false", dest="fix_o1_xy",
                        help="不固定 o1.x、o1.y，恢复为 35 个基参数")
    parser.set_defaults(fix_o1_xy=True)
    arguments = parser.parse_args()
    problem = Problem(arguments.data, arguments.stride, arguments.segments, arguments.fix_o1_xy)
    try:
        fit(problem, arguments.max_nfev)
    finally:
        problem.close()


if __name__ == "__main__":
    main()
