"""用结构基参数重参数化后进行辨识。

基参数的构造：在丰富激励（关节大范围运动、底盘平移加速度与偏航角速度/角加速度、
不同坡角坡向）下，对逆动力学关于 56 个原始参数求数值雅可比，做 SVD，取前 35 个
右奇异向量 U。基参数 beta 与原始参数 x 的关系为：

    x(beta) = x_true + scale * U * (beta - beta0)

其中 scale 为参数尺度，beta0 = U^T (x_true/scale)。这样全部原始参数都表示为 35 个
基参数 beta 的函数，U 的零空间分量固定为 CAD 先验值。辨识数据仍用 collect_data.py
采集的限位序列。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

import gimbal_dynamics as gd
import params
from identify import parameter_names

PARAM_COUNT = 56
BASE_COUNT = 35


def truth_vector() -> np.ndarray:
    return np.concatenate([
        np.array(params.MASSES),
        np.array(params.COM).reshape(-1),
        np.array(params.INERTIA).reshape(-1),
        np.array(params.JOINT_OFFSETS).reshape(-1),
        np.array(params.DAMPING),
    ])


def parameter_scales() -> np.ndarray:
    scale = np.zeros(PARAM_COUNT)
    scale[0:4] = 1.0
    scale[4:16] = 0.05
    scale[16:40] = 0.01
    scale[40:52] = 0.05
    scale[52:56] = 0.1
    return scale


def build_model(vector: np.ndarray) -> gd.TcrgModel:
    return gd.make_model(
        vector[0:4],
        vector[4:16].reshape(4, 3),
        vector[16:40].reshape(4, 6),
        vector[40:52].reshape(4, 3),
        vector[52:56],
    )


class Structure:
    def __init__(self, truth: np.ndarray, samples: int = 128, seed: int = 11):
        self.truth = truth
        self.scale = parameter_scales()
        self.context = gd.Context(1)
        rng = np.random.default_rng(seed)
        model = build_model(truth)
        residuals = []
        states = []
        for _ in range(samples):
            q = rng.uniform(-2.0, 2.0, 4)
            qdot = rng.uniform(-4.0, 4.0, 4)
            qdd = rng.uniform(-20.0, 20.0, 4)
            slope = rng.uniform(0.0, 0.8)
            direction = rng.uniform(0.0, 2.0 * math.pi)
            terrain = gd.make_terrain(slope, direction)
            yaw = rng.uniform(-math.pi, math.pi)
            yaw_rate = rng.uniform(-3.0, 3.0)
            yaw_acceleration = rng.uniform(-8.0, 8.0)
            acceleration = rng.uniform(-8.0, 8.0, 3)
            base = gd.make_base(acceleration, yaw, yaw_rate, yaw_acceleration)
            state = gd.make_state(q, qdot)
            states.append((terrain, base, state, qdd))
            residuals.append(self.context.inverse_dynamics(model, terrain, base, state, qdd))
        self.states = states
        self.residuals = np.asarray(residuals)

        jacobian = np.zeros((samples * 4, PARAM_COUNT))
        for column in range(PARAM_COUNT):
            step = 1e-5 * max(abs(truth[column]), self.scale[column] * 1e-3)
            plus = truth.copy()
            minus = truth.copy()
            plus[column] += step
            minus[column] -= step
            plus_values = []
            minus_values = []
            valid = True
            for terrain, base, state, qdd in states:
                try:
                    plus_values.append(self.context.inverse_dynamics(
                        build_model(plus), terrain, base, state, qdd))
                    minus_values.append(self.context.inverse_dynamics(
                        build_model(minus), terrain, base, state, qdd))
                except RuntimeError:
                    valid = False
                    break
            if not valid:
                # 退化方向，用单边差分
                plus_values = []
                minus_values = []
                for terrain, base, state, qdd in states:
                    plus_values.append(self.context.inverse_dynamics(
                        build_model(plus), terrain, base, state, qdd))
                    minus_values.append(self.residuals[len(plus_values) - 1])
            derivative = (np.asarray(plus_values) - np.asarray(minus_values)) / (2.0 * step)
            jacobian[:, column] = derivative.reshape(-1)
        normalized = jacobian * self.scale[None, :]
        _, singular, vh = np.linalg.svd(normalized, full_matrices=True)
        self.singular = singular
        self.rank = int(np.sum(singular > 1e-8 * singular[0]))
        print("结构雅可比秩 = %d / %d，最大奇异值 %.3e" % (self.rank, PARAM_COUNT, singular[0]))
        self.basis = vh[: min(self.rank, BASE_COUNT)].T
        self.beta0 = self.basis.T @ (truth / self.scale)
        self.context.close()

    def vector(self, beta: np.ndarray) -> np.ndarray:
        return self.truth + self.scale * (self.basis @ (beta - self.beta0))

def load_data(path: Path):
    data = np.load(path)
    q = data["q"]
    qdot = data["qdot"]
    tau = data["tau"]
    dt = float(data["dt"])
    qdd = np.gradient(qdot, dt, axis=1)
    return q, qdot, qdd, tau, data["terrain"], data["base_yaw"]


class DataProblem:
    def __init__(self, path: Path, structure: Structure, stride: int = 4,
                 max_segments: int | None = None):
        self.q, self.qdot, self.qdd, self.tau, self.terrain, self.base_yaw = load_data(path)
        segments = self.q.shape[0]
        if max_segments is not None:
            segments = min(segments, max_segments)
        self.samples = [
            (segment, index)
            for segment in range(segments)
            for index in range(0, self.q.shape[1], stride)
        ]
        self.structure = structure
        self.context = gd.Context(1)
        self.count = len(self.samples)

    def close(self):
        self.context.close()

    def residual(self, beta: np.ndarray) -> np.ndarray:
        vector = self.structure.vector(beta)
        model = build_model(vector)
        values = np.empty((self.count, 4))
        for row, (segment, index) in enumerate(self.samples):
            terrain = gd.make_terrain(float(self.terrain[segment, 0]),
                                      float(self.terrain[segment, 1]))
            base = gd.make_base(yaw=float(self.base_yaw[segment]))
            state = gd.make_state(self.q[segment, index], self.qdot[segment, index])
            try:
                predicted = self.context.inverse_dynamics(
                    model, terrain, base, state, self.qdd[segment, index])
            except RuntimeError:
                values[row] = 1.0e3
                continue
            values[row] = np.asarray(predicted) - self.tau[segment, index]
        return values.reshape(-1)


def fit(problem: DataProblem, structure: Structure, max_nfev: int) -> None:
    from scipy.optimize import least_squares

    beta0 = structure.beta0
    rng = np.random.default_rng(3)
    start = beta0
    for noise in (0.1, 0.05, 0.02, 0.01, 0.005, 0.002):
        candidate = beta0 + noise * rng.standard_normal(beta0.size)
        if np.max(np.abs(problem.residual(candidate))) < 100.0:
            start = candidate
            print("起始基参数扰动幅度 %.3f，模型物理有效" % noise)
            break
    else:
        print("所有扰动均不满足物理可实现性，从基参数真值开始")
    lower = beta0 - 1.0
    upper = beta0 + 1.0
    print("基参数维度 %d，数据残差 %d" % (beta0.size, problem.count * 4))
    result = least_squares(
        problem.residual,
        start,
        bounds=(lower, upper),
        x_scale=0.1,
        jac="2-point",
        diff_step=1e-4,
        max_nfev=max_nfev,
        verbose=2,
    )
    print("优化状态:", result.status, result.message)
    print("初始代价 %.6e，最终代价 %.6e" % (result.cost, result.cost))
    print("最终残差最大绝对值 %.6e" % np.max(np.abs(result.fun)))
    truth = structure.truth
    estimated = structure.vector(result.x)
    scale = structure.scale
    names = parameter_names()
    error = estimated - truth
    print("原始参数误差（按尺度归一化）前 20：")
    order = np.argsort(np.abs(error / scale))[::-1]
    for index in order[:20]:
        print("  %-18s 真值 % .6e 估计 % .6e 归一化误差 % .3f" % (
            names[index], truth[index], estimated[index], error[index] / scale[index]))
    print("基参数误差范数 %.3e，其中可辨识方向误差 %.3e" % (
        np.linalg.norm(result.x - beta0), np.linalg.norm(result.x - beta0)))


def main() -> None:
    parser = argparse.ArgumentParser(description="35 个基参数的重参数化辨识")
    parser.add_argument("--data", type=Path, default=Path("data/identification_data.npz"))
    parser.add_argument("--segments", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-nfev", type=int, default=25)
    parser.add_argument("--structure-samples", type=int, default=96)
    parser.add_argument("--export-mapping", type=Path, default=None)
    arguments = parser.parse_args()
    truth = truth_vector()
    structure = Structure(truth, arguments.structure_samples)
    if arguments.export_mapping is not None:
        arguments.export_mapping.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            arguments.export_mapping,
            basis=structure.basis,
            beta0=structure.beta0,
            scale=structure.scale,
            truth=truth,
            names=np.array(parameter_names()),
        )
        print("基参数映射已保存到", arguments.export_mapping)
    problem = DataProblem(arguments.data, structure, arguments.stride, arguments.segments)
    try:
        fit(problem, structure, arguments.max_nfev)
    finally:
        problem.close()


if __name__ == "__main__":
    main()

