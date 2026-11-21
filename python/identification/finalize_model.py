"""在保持可辨识坐标不变的前提下，用零空间自由度把总质量调整到已知值（6 kg），

关节力矩只依赖 28 维可辨识坐标；其余 16 维零空间分量可以任意取。已知总质量 6 kg
是一条物理先验，把它作为等式约束、把质心惯量正定作为不等式约束，在零空间内求最接近解。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.optimize import minimize

import eval_openloop as open_loop
import identify_physical as ip
import param_basis as pb

OFFSETS = pb.NOMINAL_OFFSETS.reshape(-1)


def centroidal(values: np.ndarray, body: int) -> np.ndarray:
    mass = values[10 * body]
    h = values[10 * body + 1:10 * body + 4]
    spatial = np.array([[values[10 * body + 4], values[10 * body + 7],
                         values[10 * body + 8]],
                        [values[10 * body + 7], values[10 * body + 5],
                         values[10 * body + 9]],
                        [values[10 * body + 8], values[10 * body + 9],
                         values[10 * body + 6]]])
    com = h / mass
    return spatial - mass * ((com @ com) * np.eye(3) - np.outer(com, com))


def main() -> None:
    parser = argparse.ArgumentParser(description="在保持可辨识坐标前提下调整总质量")
    parser.add_argument("--identified", type=Path, default=Path("data/identified_final.npz"))
    parser.add_argument("--save", type=Path, required=True)
    parser.add_argument("--total-mass", type=float, default=6.0,
                        help="总质量约束，0 表示不施加")
    parser.add_argument("--inertia-floor", type=float, default=1.0e-6)
    parser.add_argument("--reference", type=Path, default=None,
                        help="零空间取值的参考模型，默认为待调整模型自身")
    parser.add_argument("--data", type=Path, default=Path("data/closed_loop_data_noisy.npz"))
    arguments = parser.parse_args()

    basis, singular, rank = pb.compute_basis(offsets=OFFSETS)
    start = pb.inertial_from_raw(np.load(arguments.identified)["model"])
    center = start if arguments.reference is None else pb.inertial_from_raw(
        np.load(arguments.reference)["model"])
    target = basis.T @ start
    mass_indices = [0, 10, 20, 30]

    def residual(values):
        return basis.T @ values - target

    def objective(values):
        shifted = values - center
        return float(shifted @ shifted), 2.0 * shifted

    def mass_constraint(values):
        return np.array([np.sum(values[mass_indices]) - arguments.total_mass])

    def mass_gradient(values):
        gradient = np.zeros(44)
        gradient[mass_indices] = 1.0
        return gradient

    def inertia_constraint(values):
        return np.concatenate([np.linalg.eigvalsh(centroidal(values, body))
                               - arguments.inertia_floor for body in range(4)])

    constraints = [{"type": "eq", "fun": residual, "jac": lambda v: basis.T},
                   {"type": "ineq", "fun": inertia_constraint}]
    if arguments.total_mass > 0.0:
        constraints.append({"type": "eq", "fun": mass_constraint,
                            "jac": lambda v: mass_gradient(v)[None, :]})
    result = minimize(objective, start, jac=True, method="SLSQP", constraints=constraints,
                      options={"maxiter": 2000, "ftol": 1.0e-14})
    deviation = np.linalg.norm(residual(result.x))
    print("状态：%s，可辨识坐标偏差 %.3e" % (result.message, deviation))
    if deviation > 1.0e-8:
        raise RuntimeError("无法在保持可辨识坐标的同时满足物理约束")
    raw = pb.raw_from_inertial(result.x, OFFSETS)
    ip.build_model(raw)
    print("质量合计 %.4f kg，各刚体 %s" % (
        np.sum(result.x[mass_indices]), np.array2string(result.x[mass_indices], precision=4)))
    for body in range(4):
        print("  B%d 质心惯量特征值 %s" % (body + 1, np.array2string(
            np.linalg.eigvalsh(centroidal(result.x, body)), precision=6)))
    metrics = open_loop.evaluate(raw, arguments.data)
    print(open_loop.format_report(metrics))
    np.savez_compressed(arguments.save, model=raw, identifiable=basis.T @ pb.inertial_from_raw(raw),
                        basis=basis)
    print("已保存", arguments.save)


if __name__ == "__main__":
    main()