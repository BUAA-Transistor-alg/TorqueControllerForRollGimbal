"""用 0.5 s 轨迹的仿真残差做非线性最小二乘，得到不依赖真实参数的初值。"""

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


def main() -> None:
    parser = argparse.ArgumentParser(description="轨迹仿真最小二乘初值")
    parser.add_argument("--data", type=Path, default=Path("data/closed_loop_data_noisy.npz"))
    parser.add_argument("--save", type=Path, default=Path("data/trajectory_ls_initial.npz"))
    parser.add_argument("--windows", type=int, default=24)
    parser.add_argument("--length", type=int, default=50)
    parser.add_argument("--max-nfev", type=int, default=25)
    arguments = parser.parse_args()

    data = np.load(arguments.data)
    q = data["q"]
    qdot = data["qdot"]
    tau = data["tau"]
    rng = np.random.default_rng(20261122)
    windows = [(int(rng.integers(0, q.shape[0])), int(rng.integers(0, q.shape[1] - arguments.length)))
               for _ in range(arguments.windows)]
    scale_beta = ip.beta_scales()[gf.KEEP]
    start = gf.project(ip.beta_from_model(nominal_model())[gf.KEEP])
    context = gd.Context(1)

    expected = arguments.windows * arguments.length * 8

    def residual(beta):
        model = ip.build_model(ip.map_beta(gf.expand_beta(beta), True))
        values = []
        for segment, begin in windows:
            terrain = gd.make_terrain(float(data["terrain"][segment, 0]),
                                      float(data["terrain"][segment, 1]))
            base = gd.make_base(yaw=float(data["base_yaw"][segment]))
            state = gd.make_state(q[segment, begin], qdot[segment, begin])
            for k in range(arguments.length):
                values.extend((np.array([state.angle[i] for i in range(4)])
                               - q[segment, begin + k]))
                values.extend(0.1 * (np.array([state.velocity[i] for i in range(4)])
                                     - qdot[segment, begin + k]))
                if k + 1 < arguments.length:
                    try:
                        state = context.forward_step(model, terrain, base, state,
                                                     tau[segment, begin + k], 0.01, 2)
                    except RuntimeError:
                        return np.full(expected, 1.0e3)
        return np.asarray(values)

    from scipy.optimize import least_squares
    result = least_squares(residual, start, x_scale=scale_beta, jac="2-point",
                           diff_step=1e-4, max_nfev=arguments.max_nfev, verbose=2)
    context.close()
    beta = gf.project(result.x)
    truth = ip.beta_of_prior()[gf.KEEP]
    error = beta - truth
    scale = ip.beta_scales()[gf.KEEP]
    print("轨迹最小二乘初值误差 rms", np.sqrt(np.mean((error / scale) ** 2)),
          "max", np.max(np.abs(error / scale)))
    np.savez_compressed(arguments.save, reduced=beta, full_beta=gf.expand_beta(beta),
                        model=ip.map_beta(gf.expand_beta(beta), True))
    print("保存到", arguments.save)


if __name__ == "__main__":
    main()
