"""用采集到的限位序列做参数辨识尝试与可辨识性分析。

数据只来自 collect_data.py 的仿真采集；辨识使用同一动力学模型的逆动力学残差：
  r = tau_pred(model, q, qdot, qdd) - tau_recorded
不修改模型本身。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

import gimbal_dynamics as gd
import params

PARAM_COUNT = 56


def load(path: Path):
    data = np.load(path)
    q = data["q"]
    qdot = data["qdot"]
    tau = data["tau"]
    dt = float(data["dt"])
    qdd = np.gradient(qdot, dt, axis=1)
    terrain = data["terrain"]
    base_yaw = data["base_yaw"]
    initial_angle = data["initial_angle"]
    initial_velocity = data["initial_velocity"]
    truth = np.concatenate([
        np.array(params.MASSES),
        np.array(params.COM).reshape(-1),
        np.array(params.INERTIA).reshape(-1),
        np.array(params.JOINT_OFFSETS).reshape(-1),
        np.array(params.DAMPING),
    ])
    return q, qdot, qdd, tau, terrain, base_yaw, initial_angle, initial_velocity, truth


def build_model(vector: np.ndarray) -> gd.TcrgModel:
    masses = vector[0:4]
    com = vector[4:16].reshape(4, 3)
    inertia = vector[16:40].reshape(4, 6)
    offsets = vector[40:52].reshape(4, 3)
    damping = vector[52:56]
    return gd.make_model(masses, com, inertia, offsets, damping)


def make_samples(segments: int, points: int, stride: int):
    return [(segment, index) for segment in range(segments) for index in range(0, points, stride)]


class Problem:
    def __init__(self, path: Path, stride: int = 2, max_segments: int | None = None):
        (self.q, self.qdot, self.qdd, self.tau, self.terrain, self.base_yaw,
         self.initial_angle, self.initial_velocity, self.truth) = load(path)
        segments = self.q.shape[0]
        if max_segments is not None:
            segments = min(segments, max_segments)
        self.segments = segments
        self.points = self.q.shape[1]
        self.samples = make_samples(segments, self.points, stride)
        self.context = gd.Context(1)
        self.sample_count = len(self.samples)
        self.residual_count = self.sample_count * 4

    def close(self):
        self.context.close()

    def residual(self, vector: np.ndarray) -> np.ndarray:
        model = build_model(vector)
        values = np.empty((self.sample_count, 4), dtype=float)
        for row, (segment, index) in enumerate(self.samples):
            slope = float(self.terrain[segment, 0])
            direction = float(self.terrain[segment, 1])
            terrain = gd.make_terrain(slope, direction)
            base = gd.make_base(yaw=float(self.base_yaw[segment]))
            state = gd.make_state(self.q[segment, index], self.qdot[segment, index])
            try:
                predicted = self.context.inverse_dynamics(
                    model, terrain, base, state, self.qdd[segment, index]
                )
            except RuntimeError:
                # 候选参数不满足物理可实现性时给固定惩罚，让优化器退回可行域
                values[row] = 1.0e3
                continue
            values[row] = np.asarray(predicted) - self.tau[segment, index]
        return values.reshape(-1)


def parameter_names() -> list[str]:
    names = [f"B{i + 1}.mass" for i in range(4)]
    names += [f"B{i + 1}.com_{axis}" for i in range(4) for axis in "xyz"]
    names += [
        f"B{i + 1}.inertia_{component}"
        for i in range(4)
        for component in ("xx", "yy", "zz", "xy", "xz", "yz")
    ]
    names += [f"o{i + 1}.{axis}" for i in range(4) for axis in "xyz"]
    names += [f"d{i + 1}" for i in range(4)]
    return names


def parameter_scales() -> np.ndarray:
    scale = np.zeros(PARAM_COUNT)
    scale[0:4] = 1.0
    scale[4:16] = 0.05
    scale[16:40] = 0.01
    scale[40:52] = 0.05
    scale[52:56] = 0.1
    return scale


def analyze(problem: Problem) -> None:
    truth = problem.truth
    residual = problem.residual(truth)
    print("真值处残差：最大绝对值 %.6e，均方根 %.6e" % (
        np.max(np.abs(residual)), np.sqrt(np.mean(residual**2))))
    scales = parameter_scales()
    steps = 1e-5 * np.maximum(np.abs(truth), scales)
    jacobian = np.zeros((problem.residual_count, PARAM_COUNT))
    for column in range(PARAM_COUNT):
        plus = truth.copy()
        minus = truth.copy()
        plus[column] += steps[column]
        minus[column] -= steps[column]
        jacobian[:, column] = (problem.residual(plus) - problem.residual(minus)) / (2.0 * steps[column])
    normalized = jacobian * scales[None, :]
    singular = np.linalg.svd(normalized, compute_uv=False)
    largest = singular[0] if singular.size else 0.0
    print("残差雅可比列数 %d，行数 %d" % (PARAM_COUNT, problem.residual_count))
    print("最大奇异值 %.3e，最小奇异值 %.3e" % (largest, singular[-1]))
    for threshold in (1e-10, 1e-8, 1e-6, 1e-4):
        rank = int(np.sum(singular > threshold * largest))
        print("阈值 %.0e：秩 = %d，零空间维数 = %d" % (threshold, rank, PARAM_COUNT - rank))
    # 每个参数在零空间中的参与度（阈值 1e-8）
    rank = int(np.sum(singular > 1e-8 * largest))
    _, _, vh = np.linalg.svd(normalized, full_matrices=True)
    null = vh[rank:]
    participation = np.sum(null * null, axis=0)
    names = parameter_names()
    print("零空间参与度最高的 20 个参数：")
    order = np.argsort(participation)[::-1]
    for index in order[:20]:
        print("  %-18s 参与度 %.3f" % (names[index], participation[index]))
    return jacobian, singular, participation, names


def fit(problem: Problem, max_nfev: int) -> None:
    from scipy.optimize import least_squares

    truth = problem.truth
    rng = np.random.default_rng(7)
    start = truth.copy()
    start[0:4] *= 1.0 + 0.1 * rng.standard_normal(4)
    start[4:16] += 0.005 * rng.standard_normal(12)
    start[16:40] *= 1.0 + 0.1 * rng.standard_normal(24)
    start[40:52] += 0.005 * rng.standard_normal(12)
    start[52:56] += 0.02 * rng.standard_normal(4)
    lower = truth.copy()
    upper = truth.copy()
    lower[0:4] = np.maximum(0.05, truth[0:4] * 0.5)
    upper[0:4] = truth[0:4] * 1.5
    lower[4:16] = truth[4:16] - 0.05
    upper[4:16] = truth[4:16] + 0.05
    tiny = np.abs(truth[16:40]) < 1e-12
    lower[16:40] = np.where(tiny, -1e-3,
                            np.where(truth[16:40] >= 0.0, truth[16:40] * 0.5, truth[16:40] * 2.0))
    upper[16:40] = np.where(tiny, 1e-3,
                            np.where(truth[16:40] >= 0.0, truth[16:40] * 2.0, truth[16:40] * 0.5))
    lower[40:52] = truth[40:52] - 0.05
    upper[40:52] = truth[40:52] + 0.05
    lower[52:56] = 0.0
    upper[52:56] = 0.5
    start = np.clip(start, lower, upper)
    scale = parameter_scales()

    def residual_scaled(values):
        return problem.residual(values) / 1.0

    print("开始最小二乘：参数 %d，残差 %d" % (PARAM_COUNT, problem.residual_count))
    result = least_squares(
        residual_scaled,
        start,
        bounds=(lower, upper),
        x_scale=scale,
        jac="2-point",
        diff_step=1e-5,
        max_nfev=max_nfev,
        verbose=2,
    )
    print("优化状态:", result.status, result.message)
    print("初始代价 %.6e，最终代价 %.6e" % (result.cost, result.cost))
    print("最终残差最大绝对值 %.6e" % np.max(np.abs(result.fun)))
    error = result.x - truth
    print("参数误差（按尺度归一化）前 20：")
    order = np.argsort(np.abs(error / scale))[::-1]
    names = parameter_names()
    for index in order[:20]:
        print("  %-18s 真值 % .6e 估计 % .6e 归一化误差 % .3f" % (
            names[index], truth[index], result.x[index], error[index] / scale[index]))


def main() -> None:
    parser = argparse.ArgumentParser(description="限位序列参数辨识")
    parser.add_argument("--data", type=Path, default=Path("data/identification_data.npz"))
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--segments", type=int, default=None)
    parser.add_argument("--analysis", action="store_true")
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--max-nfev", type=int, default=60)
    arguments = parser.parse_args()
    problem = Problem(arguments.data, arguments.stride, arguments.segments)
    try:
        if arguments.analysis or not arguments.fit:
            analyze(problem)
        if arguments.fit:
            fit(problem, arguments.max_nfev)
    finally:
        problem.close()


if __name__ == "__main__":
    main()
