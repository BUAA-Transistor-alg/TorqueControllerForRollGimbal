"""含噪闭环数据的多打靶输出误差最小二乘（记录力矩驱动）。

不使用真实参数先验；初值来自含噪数据上的轨迹最小二乘。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import gimbal_dynamics as gd
import gradient_fit as gf
import identify_physical as ip


def simulate_window(context, model, data, segment, start, length):
    terrain = gd.make_terrain(float(data["terrain"][segment, 0]),
                              float(data["terrain"][segment, 1]))
    base = gd.make_base(yaw=float(data["base_yaw"][segment]))
    state = gd.make_state(data["q"][segment, start], data["qdot"][segment, start])
    q_values = np.zeros((length, 4))
    v_values = np.zeros((length, 4))
    for k in range(length):
        q_values[k] = [state.angle[i] for i in range(4)]
        v_values[k] = [state.velocity[i] for i in range(4)]
        if k + 1 < length:
            state = context.forward_step(model, terrain, base, state,
                                         data["tau"][segment, start + k], 0.01, 2)
    return q_values, v_values


def main() -> None:
    parser = argparse.ArgumentParser(description="多打靶输出误差最小二乘")
    parser.add_argument("--data", type=Path, default=Path("data/closed_loop_data_noisy.npz"))
    parser.add_argument("--init", type=Path, default=Path("data/trajectory_ls_initial.npz"))
    parser.add_argument("--save", type=Path, default=Path("data/identified_robust.npz"))
    parser.add_argument("--train-windows", type=int, default=140)
    parser.add_argument("--val-windows", type=int, default=40)
    parser.add_argument("--length", type=int, default=25)
    parser.add_argument("--max-nfev", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261128)
    arguments = parser.parse_args()

    data = np.load(arguments.data)
    rng = np.random.default_rng(arguments.seed)
    training_segments = np.arange(0, 80)
    validation_segments = np.arange(80, data["q"].shape[0])

    def sample_windows(segments, count):
        return [(int(rng.choice(segments)),
                 int(rng.integers(0, data["q"].shape[1] - arguments.length)))
                for _ in range(count)]

    train_windows = sample_windows(training_segments, arguments.train_windows)
    val_windows = sample_windows(validation_segments, arguments.val_windows)
    position_std = 0.001
    velocity_std = 0.1
    context = gd.Context(1)

    def residual_for_windows(beta, windows):
        beta = gf.project(beta)
        model = ip.build_model(ip.map_beta(gf.expand_beta(beta), True))
        values = []
        for segment, start in windows:
            q_pred, v_pred = simulate_window(context, model, data, segment, start, arguments.length)
            values.append((q_pred - data["q"][segment, start:start + arguments.length]).reshape(-1)
                          / position_std)
            values.append((v_pred - data["qdot"][segment, start:start + arguments.length]).reshape(-1)
                          / velocity_std)
        return np.concatenate(values)

    def residual(beta):
        return residual_for_windows(beta, train_windows)

    initial = np.load(arguments.init)["reduced"].copy()
    print("训练初始残差 RMS", np.sqrt(np.mean(residual(initial) ** 2)))
    print("验证初始残差 RMS", np.sqrt(np.mean(residual_for_windows(initial, val_windows) ** 2)))
    from scipy.optimize import least_squares
    result = least_squares(residual, initial, x_scale=ip.beta_scales()[gf.KEEP],
                           jac="2-point", diff_step=1e-4, loss="soft_l1", f_scale=1.0,
                           max_nfev=arguments.max_nfev, verbose=2)
    beta = gf.project(result.x)
    context.close()
    train_rms = np.sqrt(np.mean(residual_for_windows(beta, train_windows) ** 2))
    val_rms = np.sqrt(np.mean(residual_for_windows(beta, val_windows) ** 2))
    print("训练残差 RMS", train_rms, "验证残差 RMS", val_rms)
    truth = ip.beta_of_prior()[gf.KEEP]
    error = beta - truth
    scale = ip.beta_scales()[gf.KEEP]
    print("参数归一化误差 RMS", np.sqrt(np.mean((error / scale) ** 2)),
          "最大", np.max(np.abs(error / scale)))
    arguments.save.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.save, reduced=beta, full_beta=gf.expand_beta(beta),
                        model=ip.map_beta(gf.expand_beta(beta), True))
    print("保存到", arguments.save)


if __name__ == "__main__":
    main()
