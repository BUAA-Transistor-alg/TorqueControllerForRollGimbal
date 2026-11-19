"""在闭环 PID 数据的 0.5 s 窗口上做物理基参数梯度下降。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import gimbal_dynamics as gd
import gradient_fit as gf
import identify_physical as ip

WEIGHTS_ANGLE = [1.0, 1.0, 1.0, 1.0]
WEIGHTS_VELOCITY = [0.1, 0.1, 0.1, 0.1]
WEIGHTS_TORQUE = [0.0, 0.0, 0.0, 0.0]


class ClosedLoopData:
    def __init__(self, path: Path):
        data = np.load(path)
        self.q = data["q"]
        self.qdot = data["qdot"]
        self.tau = data["tau"]
        self.terrain = data["terrain"]
        self.base_yaw = data["base_yaw"]
        self.segments = self.q.shape[0]
        self.points = self.q.shape[1]

    def windows(self, length: int, stride: int):
        return [(segment, start) for segment in range(self.segments)
                for start in range(0, self.points - length + 1, stride)]


def window_gradient(context, data: ClosedLoopData, beta: np.ndarray,
                    segment: int, start: int, length: int):
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
        WEIGHTS_ANGLE, WEIGHTS_VELOCITY, WEIGHTS_TORQUE,
        0.01, 2,
    )
    gradient = np.zeros(56)
    for grouped in range(56):
        gradient[grouped] = gradient_library[gf.PERM[grouped]]
    T = gf.mapping_jacobian(gf.expand_beta(beta))[:, gf.KEEP]
    return loss, T.T @ gradient


def evaluate(context, data: ClosedLoopData, beta: np.ndarray, length: int, stride: int):
    windows = data.windows(length, stride)
    total = 0.0
    for segment, start in windows:
        loss, _ = window_gradient(context, data, beta, segment, start, length)
        total += loss
    return total / len(windows)


def main() -> None:
    parser = argparse.ArgumentParser(description="闭环窗口参数辨识")
    parser.add_argument("--data", type=Path, default=Path("data/closed_loop_data.npz"))
    parser.add_argument("--init", type=Path, default=None)
    parser.add_argument("--start-perturb", type=float, default=0.02)
    parser.add_argument("--save", type=Path, default=Path("data/identified_closed_loop.npz"))
    parser.add_argument("--stages", type=str, default="10,20,35,50")
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--decay", type=float, default=0.999)
    parser.add_argument("--regularization", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=20261120)
    arguments = parser.parse_args()

    data = ClosedLoopData(arguments.data)
    context = gd.Context(1)
    rng = np.random.default_rng(arguments.seed)
    reference = ip.beta_of_prior()[gf.KEEP]
    scale = ip.beta_scales()[gf.KEEP]
    if arguments.init is not None and arguments.init.exists():
        beta = np.load(arguments.init)["reduced"].copy()
        print("从已有结果继续")
    else:
        beta = gf.project(reference + arguments.start_perturb * scale)
    beta = gf.project(beta)
    reference = beta.copy()
    moment = np.zeros_like(beta)
    velocity = np.zeros_like(beta)
    beta1, beta2 = 0.9, 0.999
    learning_rate = arguments.learning_rate
    stages = [int(value) for value in arguments.stages.split(",") if value]
    try:
        for stage_index, length in enumerate(stages):
            stride = max(length // 2, 10)
            windows = data.windows(length, stride)
            stage_loss = evaluate(context, data, beta, length, stride)
            print("阶段 %d，窗口长度 %d，窗口数 %d，初始平均损失 %.6e" % (
                stage_index + 1, length, len(windows), stage_loss))
            for iteration in range(arguments.iterations):
                chosen = [windows[index] for index in
                          rng.integers(0, len(windows), size=arguments.batch)]
                total_loss = 0.0
                total_gradient = np.zeros_like(beta)
                for segment, start in chosen:
                    loss, gradient = window_gradient(context, data, beta, segment, start, length)
                    total_loss += loss
                    total_gradient += gradient
                total_loss /= arguments.batch
                total_gradient /= arguments.batch
                total_loss += arguments.regularization * float(
                    np.sum(((beta - reference) / scale) ** 2))
                total_gradient += (2.0 * arguments.regularization
                                   * (beta - reference) / (scale * scale))
                gradient_u = scale * total_gradient
                norm = np.linalg.norm(gradient_u)
                if norm > 1.0:
                    gradient_u = gradient_u / norm
                moment = beta1 * moment + (1.0 - beta1) * gradient_u
                velocity = beta2 * velocity + (1.0 - beta2) * gradient_u * gradient_u
                bias1 = 1.0 - beta1 ** (iteration + 1)
                bias2 = 1.0 - beta2 ** (iteration + 1)
                beta = gf.project(beta - learning_rate * (moment / bias1) / (
                    np.sqrt(velocity / bias2) + 1.0e-8))
                learning_rate = max(learning_rate * arguments.decay, 1.0e-7)
                if (iteration + 1) % 20 == 0:
                    print("  迭代 %d，批损失 %.6e，步长 %.2e" % (
                        iteration + 1, total_loss, learning_rate))
            final_loss = evaluate(context, data, beta, length, stride)
            print("阶段 %d 结束，平均损失 %.6e" % (stage_index + 1, final_loss))
        arguments.save.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(arguments.save, reduced=beta, reference=reference,
                            scale=scale, keep=gf.KEEP,
                            full_beta=gf.expand_beta(beta),
                            model=ip.map_beta(gf.expand_beta(beta), True))
        print("辨识结果已保存到", arguments.save)
    finally:
        context.close()


if __name__ == "__main__":
    main()
