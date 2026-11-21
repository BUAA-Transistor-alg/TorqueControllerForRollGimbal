"""含噪闭环数据的输出误差最小二乘参数辨识。

候选物理模型 + 记录 PID 参数/目标序列闭环仿真，比较位置速度轨迹。
不使用真实参数先验，初值取标称几何与总质量给出的粗略模型。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import gimbal_dynamics as gd
import gradient_fit as gf
import identify_physical as ip


def nominal_model() -> np.ndarray:
    x = np.zeros(56)
    x[0:4] = [1.5, 1.5, 1.5, 1.5]
    for body in range(4):
        x[4 + 3 * body:7 + 3 * body] = [0.0, 0.05, 0.0]
        x[16 + 6 * body:22 + 6 * body] = [0.01, 0.01, 0.01, 0.0, 0.0, 0.0]
    x[40:52] = np.array([[0.0, 0.0, 0.0], [0.0, 0.09, 0.039],
                         [0.0, 0.0, 0.0521], [0.0, 0.0, 0.1579]]).reshape(-1)
    x[52:56] = 0.1
    return x


def simulate_window(context, model, data, segment, start, length):
    terrain = gd.make_terrain(float(data["terrain"][segment, 0]),
                              float(data["terrain"][segment, 1]))
    base = gd.make_base(yaw=float(data["base_yaw"][segment]))
    state = gd.make_state(data["q"][segment, start], data["qdot"][segment, start])
    kp, ki, kd = data["kp"][segment], data["ki"][segment], data["kd"][segment]
    limit = float(data["torque_limit"][segment])
    integral_limit = float(data["integral_limit"][segment])
    target = data["target"][segment]
    target_velocity = data["target_velocity"][segment]
    integral = np.zeros(4)
    q_values = np.zeros((length, 4))
    v_values = np.zeros((length, 4))
    for k in range(length):
        q = np.array([state.angle[i] for i in range(4)])
        v = np.array([state.velocity[i] for i in range(4)])
        q_values[k] = q
        v_values[k] = v
        error = target[start + k] - q
        integral = np.clip(integral + error * 0.01, -integral_limit, integral_limit)
        feedforward = np.asarray(context.bias_torque(model, terrain, base, state))
        command = (feedforward + kp * error + ki * integral
                   + kd * (target_velocity[start + k] - v))
        command = np.clip(command, -limit, limit)
        if k + 1 < length:
            state = context.forward_step(model, terrain, base, state, command, 0.01, 2)
    return q_values, v_values


def main() -> None:
    parser = argparse.ArgumentParser(description="输出误差参数辨识")
    parser.add_argument("--data", type=Path, default=Path("data/closed_loop_data_noisy.npz"))
    parser.add_argument("--save", type=Path, default=Path("data/identified_output_error.npz"))
    parser.add_argument("--windows", type=int, default=60)
    parser.add_argument("--length", type=int, default=50)
    parser.add_argument("--max-nfev", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20261127)
    parser.add_argument("--init", type=Path, default=None)
    arguments = parser.parse_args()

    data = np.load(arguments.data)
    rng = np.random.default_rng(arguments.seed)
    windows = [(int(rng.integers(0, data["q"].shape[0])),
                int(rng.integers(0, data["q"].shape[1] - arguments.length)))
               for _ in range(arguments.windows)]
    if arguments.init is not None and arguments.init.exists():
        beta_start = gf.project(np.load(arguments.init)["reduced"].copy())
        print("从已有结果继续")
    else:
        beta_start = gf.project(ip.beta_from_model(nominal_model())[gf.KEEP])
    scale = ip.beta_scales()[gf.KEEP]
    context = gd.Context(1)
    position_std = float(data.get("noise_position_std", 0.001)) if hasattr(data, "get") else 0.001
    velocity_std = float(data.get("noise_velocity_std", 0.1)) if hasattr(data, "get") else 0.1

    def residual(beta):
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

    initial = residual(beta_start)
    print("初始加权残差均方根", np.sqrt(np.mean(initial ** 2)))
    from scipy.optimize import least_squares
    result = least_squares(residual, beta_start, x_scale=scale, jac="2-point",
                           diff_step=1e-4, max_nfev=arguments.max_nfev, verbose=2)
    context.close()
    beta = gf.project(result.x)
    truth = ip.beta_of_prior()[gf.KEEP]
    error = beta - truth
    print("最终加权残差均方根", np.sqrt(np.mean(result.fun ** 2)))
    print("参数归一化误差均方根", np.sqrt(np.mean((error / scale) ** 2)),
          "最大", np.max(np.abs(error / scale)))
    arguments.save.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.save, reduced=beta, full_beta=gf.expand_beta(beta),
                        model=ip.map_beta(gf.expand_beta(beta), True))
    print("保存到", arguments.save)


if __name__ == "__main__":
    main()
