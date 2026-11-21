"""用库的解析参数灵敏度对含噪闭环数据做多打靶最小二乘拟合。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import gimbal_dynamics as gd
import gradient_fit as gf
import identify_physical as ip


class WindowData:
    def __init__(self, path: Path, smooth: int = 0):
        data = np.load(path)
        from scipy.signal import savgol_filter
        self.q = data["q"]
        self.qdot = data["qdot"]
        self.tau = data["tau"]
        if smooth > 2:
            self.q = savgol_filter(self.q, smooth, 3, axis=1)
            self.qdot = savgol_filter(self.qdot, smooth, 3, axis=1)
            self.tau = savgol_filter(self.tau, smooth, 3, axis=1)
        self.terrain = data["terrain"]
        self.base_yaw = data["base_yaw"]
        self.segments = self.q.shape[0]
        self.points = self.q.shape[1]


def window_loss_gradient(context, data, beta, segment, start, length):
    x = ip.map_beta(gf.expand_beta(beta), True)
    model = ip.build_model(x)
    terrain = gd.make_terrain(float(data.terrain[segment, 0]),
                              float(data.terrain[segment, 1]))
    base = gd.make_base(yaw=float(data.base_yaw[segment]))
    initial = gd.make_state(data.q[segment, start], data.qdot[segment, start])
    loss, gradient_library, _ = context.parameter_gradient(
        model, terrain, base, initial,
        data.tau[segment, start:start + length],
        data.q[segment, start:start + length],
        data.qdot[segment, start:start + length],
        [1.0, 1.0, 1.0, 1.0], [0.1, 0.1, 0.1, 0.1], [0.0, 0.0, 0.0, 0.0],
        0.01, 2,
    )
    gradient = np.zeros(56)
    for grouped in range(56):
        gradient[grouped] = gradient_library[gf.PERM[grouped]]
    T = gf.mapping_jacobian(gf.expand_beta(beta))[:, gf.KEEP]
    return loss, T.T @ gradient


def evaluate(context, data, beta, windows, length):
    total = 0.0
    for segment, start in windows:
        loss, _ = window_loss_gradient(context, data, beta, segment, start, length)
        total += loss
    return total / len(windows)


def main() -> None:
    parser = argparse.ArgumentParser(description="解析梯度多打靶拟合")
    parser.add_argument("--data", type=Path, default=Path("data/closed_loop_data_noisy.npz"))
    parser.add_argument("--init", type=Path, default=Path("data/trajectory_ls_initial.npz"))
    parser.add_argument("--save", type=Path, default=Path("data/identified_analytic.npz"))
    parser.add_argument("--windows", type=int, default=200)
    parser.add_argument("--val-windows", type=int, default=50)
    parser.add_argument("--length", type=int, default=25)
    parser.add_argument("--maxiter", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20261129)
    parser.add_argument("--smooth", type=int, default=0)
    arguments = parser.parse_args()

    data = WindowData(arguments.data, arguments.smooth)
    rng = np.random.default_rng(arguments.seed)
    training_segments = np.arange(0, 80)
    validation_segments = np.arange(80, data.segments)
    train_windows = [(int(rng.choice(training_segments)),
                      int(rng.integers(0, data.points - arguments.length)))
                     for _ in range(arguments.windows)]
    val_windows = [(int(rng.choice(validation_segments)),
                    int(rng.integers(0, data.points - arguments.length)))
                   for _ in range(arguments.val_windows)]
    initial = gf.project(np.load(arguments.init)["reduced"].copy())
    scale = ip.beta_scales()[gf.KEEP]
    reference = initial.copy()
    context = gd.Context(1)

    def objective(z):
        beta = gf.project(z * scale)
        loss = 0.0
        gradient = np.zeros_like(beta)
        for segment, start in train_windows:
            value, part = window_loss_gradient(context, data, beta, segment, start, arguments.length)
            loss += value
            gradient += part
        loss /= len(train_windows)
        gradient /= len(train_windows)
        loss += 1e-5 * float(np.sum(((beta - reference) / scale) ** 2))
        gradient += 2e-5 * (beta - reference) / (scale * scale)
        return loss, scale * gradient

    initial_loss = objective(initial / scale)[0]
    validation_initial = evaluate(context, data, initial, val_windows, arguments.length)
    print("初始训练损失", initial_loss, "验证损失", validation_initial)
    result = minimize(objective, initial / scale, jac=True, method="L-BFGS-B",
                      options={"maxiter": arguments.maxiter, "ftol": 1e-12,
                               "gtol": 1e-10, "maxls": 20, "disp": True})
    beta = gf.project(result.x * scale)
    train_loss = evaluate(context, data, beta, train_windows, arguments.length)
    val_loss = evaluate(context, data, beta, val_windows, arguments.length)
    truth = ip.beta_of_prior()[gf.KEEP]
    error = beta - truth
    print("优化状态", result.message)
    print("训练损失", train_loss, "验证损失", val_loss)
    print("参数归一化误差 RMS", np.sqrt(np.mean((error / scale) ** 2)),
          "最大", np.max(np.abs(error / scale)))
    context.close()
    arguments.save.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.save, reduced=beta, full_beta=gf.expand_beta(beta),
                        model=ip.map_beta(gf.expand_beta(beta), True))
    print("保存到", arguments.save)


if __name__ == "__main__":
    main()
