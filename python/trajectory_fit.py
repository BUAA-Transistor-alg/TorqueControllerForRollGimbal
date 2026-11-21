"""轨迹空间多打靶精化：直接最小化开环轨迹残差，初始状态作为自由变量。

目标函数：
    L(delta) = 1/W * sum_w  1/L * sum_k sum_j (q_sim(w,k) - q_meas(w,k))^2
其中每个窗口的初始状态 (q0, v0) 单独优化（profile likelihood），
参数只沿可辨识子空间移动：p = p_base + basis @ delta。

梯度用 C 库的解析参数灵敏度，配合包络定理：内层初始状态收敛后
dL/ddelta = basis^T (dp/dx)^T dL/dx。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import gimbal_dynamics as gd
import identify_physical as ip
import param_basis as pb


def theta_jacobian(theta: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """56x44：原始参数对无约束物理参数 theta 的雅可比（中心差分）。"""
    base = pb._raw_from_theta(theta, offsets)
    jacobian = np.zeros((56, theta.size))
    for index in range(theta.size):
        step = 1.0e-6 * max(1.0, abs(theta[index]))
        plus = theta.copy()
        minus = theta.copy()
        plus[index] += step
        minus[index] -= step
        jacobian[:, index] = (pb._raw_from_theta(plus, offsets)
                              - pb._raw_from_theta(minus, offsets)) / (2.0 * step)
    return jacobian


class Window:
    __slots__ = ("segment", "start", "length", "terrain", "base", "target", "torques")

    def __init__(self, segment, start, length, terrain, base, target, torques):
        self.segment = segment
        self.start = start
        self.length = length
        self.terrain = terrain
        self.base = base
        self.target = target
        self.torques = torques


def build_windows(data, length: int, per_segment: int, seed: int):
    rng = np.random.default_rng(seed)
    windows = []
    for segment in range(data["q"].shape[0]):
        terrain = gd.make_terrain(float(data["terrain"][segment, 0]),
                                  float(data["terrain"][segment, 1]))
        base = gd.make_base(yaw=float(data["base_yaw"][segment]))
        starts = rng.integers(0, data["q"].shape[1] - length, per_segment)
        for start in starts:
            window = Window(segment, int(start), length, terrain, base,
                            data["q"][segment, start:start + length],
                            data["tau"][segment, start:start + length])
            windows.append(window)
    return windows


def model_from_values(values, offsets):
    return ip.build_model(pb.raw_from_inertial(values, offsets))


def window_residual(context, model, window, initial):
    state = gd.make_state(initial[:4], initial[4:])
    residual = np.zeros((window.length, 4))
    for k in range(window.length):
        residual[k] = [state.angle[i] for i in range(4)] - window.target[k]
        state = context.forward_step(model, window.terrain, window.base, state,
                                     window.torques[k], 0.01, 2)
    return residual


def solve_initial_state(context, model, window, initial, iterations: int):
    """高斯牛顿法优化窗口初始状态（8 维），返回 (初始状态, 残差平方和)。"""
    initial = initial.copy()
    step = 1.0e-6
    scale = np.array([1.0e-6] * 4 + [1.0e-6] * 4)
    residual = window_residual(context, model, window, initial)
    cost = float(np.sum(residual ** 2))
    for _ in range(iterations):
        jacobian = np.zeros((window.length * 4, 8))
        for column in range(8):
            moved = initial.copy()
            moved[column] += step
            jacobian[:, column] = (window_residual(context, model, window, moved)
                                   - residual).reshape(-1) / step
        normal = jacobian.T @ jacobian + 1.0e-8 * np.eye(8)
        gradient = jacobian.T @ residual.reshape(-1)
        update = np.linalg.solve(normal, gradient)
        update = np.clip(update, -1.0, 1.0)
        damp = 1.0
        for _ in range(6):
            candidate = initial - damp * update
            candidate_residual = window_residual(context, model, window, candidate)
            candidate_cost = float(np.sum(candidate_residual ** 2))
            if candidate_cost < cost:
                initial = candidate
                residual = candidate_residual
                cost = candidate_cost
                break
            damp *= 0.5
        else:
            break
    return initial, cost


def window_loss_gradient(context, model, window, initial, raw, jacobian_raw, basis):
    count = window.length
    torques = [list(row) for row in window.torques]
    targets = [list(row) for row in window.target]
    loss, gradient_library, _ = context.parameter_gradient(
        model, window.terrain, window.base,
        gd.make_state(initial[:4], initial[4:]),
        torques, targets, None,
        [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
        0.01, 2)
    return loss, np.array(gradient_library)


def main() -> None:
    parser = argparse.ArgumentParser(description="轨迹空间多打靶精化")
    parser.add_argument("--data", type=Path, default=Path("data/closed_loop_data_noisy.npz"))
    parser.add_argument("--base", type=Path, default=Path("data/regressor_fullscan_07.npz"))
    parser.add_argument("--save", type=Path, default=Path("data/identified_trajectory.npz"))
    parser.add_argument("--length", type=int, default=50)
    parser.add_argument("--per-segment", type=int, default=5)
    parser.add_argument("--inner", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=40)
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--segment-limit", type=int, default=0)
    parser.add_argument("--evaluate", action="store_true")
    arguments = parser.parse_args()

    data = np.load(arguments.data)
    if arguments.segment_limit > 0:
        data = {key: value[:arguments.segment_limit] if hasattr(value, "shape") and
                value.shape[:1] == (data["q"].shape[0],) else value for key, value in data.items()}
    base_raw = np.load(arguments.base)["model"]
    offsets = pb.NOMINAL_OFFSETS.reshape(-1)
    basis, singular, rank = pb.compute_basis(offsets=offsets)
    base_values = pb.inertial_from_raw(base_raw)
    base_theta = pb._theta_from_raw(base_raw)
    null_projection = np.eye(44) - basis @ basis.T
    base_null = null_projection @ base_values
    windows = build_windows(data, arguments.length, arguments.per_segment, 20261203)
    print("窗口数 %d，长度 %d，可辨识维数 %d" % (len(windows), arguments.length, rank))

    context = gd.Context(1)
    initials = []
    for window in windows:
        q0 = window.target[0]
        v0 = data["qdot"][window.segment, window.start]
        value, _ = solve_initial_state(context, model_from_values(base_values, offsets),
                                       window, np.concatenate([q0, v0]), 2)
        initials.append(value)

    history = []

    def objective(theta):
        raw = pb._raw_from_theta(theta, offsets)
        values = pb.inertial_from_raw(raw)
        jacobian_raw = theta_jacobian(theta, offsets)
        total = 0.0
        gradient_raw = np.zeros(56)
        for index, window in enumerate(windows):
            model = ip.build_model(raw)
            initial, _ = solve_initial_state(context, model, window, initials[index],
                                             arguments.inner)
            initials[index] = initial
            loss, part = window_loss_gradient(context, model, window, initial, raw,
                                              jacobian_raw, None)
            total += loss
            gradient_raw += part
        total /= len(windows)
        gradient_raw /= len(windows)
        gradient = jacobian_raw.T @ gradient_raw
        if arguments.ridge > 0.0:
            offset_theta = theta - base_theta
            total += arguments.ridge * float(offset_theta @ offset_theta)
            gradient += 2.0 * arguments.ridge * offset_theta
        history.append(total)
        if len(history) % 2 == 1:
            print("迭代 %3d 损失 %.6e（窗口 RMS %.4e rad）" % (
                len(history), total, np.sqrt(total)))
        return total, gradient

    initial_loss = objective(base_theta)[0]
    print("初始损失 %.6e（窗口 RMS %.4e rad）" % (initial_loss, np.sqrt(initial_loss)))
    result = minimize(objective, base_theta, jac=True, method="L-BFGS-B",
                      options={"maxiter": arguments.maxiter, "ftol": 1e-16, "gtol": 1e-14,
                               "maxls": 30, "disp": True})
    print("优化状态", result.status, result.message)
    raw = pb._raw_from_theta(result.x, offsets)
    values = pb.inertial_from_raw(raw)
    change = basis.T @ (values - base_values)
    print("可辨识坐标变化范数 %.4e（基模型可辨识坐标范数 %.4e）" % (
        np.linalg.norm(change), np.linalg.norm(basis.T @ base_values)))
    context.close()
    np.savez_compressed(arguments.save, model=raw, values=values, basis=basis,
                        theta=result.x, seconds=np.array(history))
    print("已保存", arguments.save)
    if arguments.evaluate:
        import eval_openloop as open_loop
        print(open_loop.format_report(open_loop.evaluate(raw, arguments.data)))


if __name__ == "__main__":
    main()