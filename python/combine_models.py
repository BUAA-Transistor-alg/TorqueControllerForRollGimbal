"""把多个独立辨识结果在可辨识坐标上合并，并投影成物理合法的 56 维模型。

不同数据集得到的可辨识坐标误差互不相关时，等权平均可以降低方差。
合并后在最小约束（质量为正、质心惯量正定）下求解可辨识坐标最接近的物理模型。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import eval_openloop as open_loop
import identify_physical as ip
import param_basis as pb


def feasibility(p: np.ndarray) -> np.ndarray:
    values = [np.array(p[40:44])]
    for body in range(4):
        mass = p[body * 10]
        values.append(np.array([mass - 0.02]))
        h = p[body * 10 + 1:body * 10 + 4]
        spatial = np.array([[p[body * 10 + 4], p[body * 10 + 7], p[body * 10 + 8]],
                            [p[body * 10 + 7], p[body * 10 + 5], p[body * 10 + 9]],
                            [p[body * 10 + 8], p[body * 10 + 9], p[body * 10 + 6]]])
        com = h / max(mass, 1.0e-6)
        centroidal = spatial - mass * ((com @ com) * np.eye(3) - np.outer(com, com))
        values.append(np.linalg.eigvalsh(centroidal) - 1.0e-6)
    return np.concatenate(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="合并多个辨识结果")
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--weights", type=float, nargs="+", default=None)
    parser.add_argument("--save", type=Path, default=Path("data/identified_combined.npz"))
    parser.add_argument("--data", type=Path, default=Path("data/closed_loop_data_noisy.npz"))
    arguments = parser.parse_args()

    offsets = pb.NOMINAL_OFFSETS.reshape(-1)
    basis, singular, rank = pb.compute_basis(offsets=offsets)
    coordinates = []
    for path in arguments.models:
        raw = np.load(path)["model"]
        coordinates.append(basis.T @ pb.inertial_from_raw(raw))
        print("%-40s 可辨识坐标范数 %.4e" % (path.name, np.linalg.norm(coordinates[-1])))
    weights = np.ones(len(coordinates)) if arguments.weights is None else np.asarray(
        arguments.weights, dtype=float)
    weights = weights / np.sum(weights)
    target = np.tensordot(weights, np.asarray(coordinates), axes=1)
    print("合并权重", np.array2string(weights, precision=3))

    start = pb.inertial_from_raw(np.load(arguments.models[0])["model"])

    def objective(values):
        residual = basis.T @ values - target
        return float(residual @ residual)

    def gradient(values):
        return 2.0 * basis @ (basis.T @ values - target)

    result = minimize(objective, start, jac=gradient, method="SLSQP",
                      constraints=[{"type": "ineq", "fun": feasibility}],
                      options={"maxiter": 3000, "ftol": 1.0e-14})
    deviation = np.linalg.norm(basis.T @ result.x - target)
    print("投影状态：%s，可辨识坐标偏差 %.3e" % (result.message, deviation))
    if deviation > 1.0e-3:
        raise RuntimeError("物理投影无法保持合并后的可辨识坐标")
    raw = pb.raw_from_inertial(result.x, offsets)
    ip.build_model(raw)
    metrics = open_loop.evaluate(raw, arguments.data)
    print(open_loop.format_report(metrics))
    np.savez_compressed(arguments.save, model=raw, identifiable=basis.T @ pb.inertial_from_raw(raw),
                        basis=basis)
    print("已保存", arguments.save)


if __name__ == "__main__":
    main()
