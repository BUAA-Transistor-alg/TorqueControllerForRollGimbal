"""使用库的参数灵敏度梯度，对文档定义的物理基参数做梯度下降。

链式法则：库返回 dL/dx（x 为 56 个原始参数，按每个刚体 10 个连续排列），
基参数 beta 到原始参数 x 的映射由 identify_physical.map_beta 给出，
本文件单独实现其雅可比 T = dx/dbeta，再计算 dL/dbeta = T^T dL/dx。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import gimbal_dynamics as gd
import identify_physical as ip
import params

# 库内参数布局（每个刚体连续 10 个）到本项目分组布局的索引置换
PERM = []
for body in range(4):
    PERM.append(10 * body)
for body in range(4):
    for axis in range(3):
        PERM.append(10 * body + 1 + axis)
for body in range(4):
    for component in range(6):
        PERM.append(10 * body + 4 + component)
PERM += list(range(40, 56))
PERM = np.array(PERM)

FIXED_BETA = (21, 22)  # o1.x、o1.y


def mapping_jacobian(beta: np.ndarray) -> np.ndarray:
    T = np.zeros((56, 35))
    m1 = params.MASSES[0]
    cx, cy = beta[1] / m1, beta[2] / m1
    T[4, 1] = 1.0 / m1
    T[5, 2] = 1.0 / m1
    T[18, 0] = 1.0
    T[18, 1] = -2.0 * cx
    T[18, 2] = -2.0 * cy

    position = 3
    for body in range(1, 4):
        m = beta[position]
        h = beta[position + 1:position + 4]
        c = h / m
        a = ip.AXES[body]
        p = np.array(ip.PLANES[body][0])
        q = np.array(ip.PLANES[body][1])
        dA = 2.0 * c - 2.0 * (a @ c) * a
        dB = 2.0 * (q @ c) * q - 2.0 * (p @ c) * p
        A = c @ c - (a @ c) ** 2
        B = (q @ c) ** 2 - (p @ c) ** 2

        def inertia_derivative(longitudinal: float, difference: float) -> np.ndarray:
            return (longitudinal * np.outer(a, a)
                    + 0.5 * difference * (np.outer(p, p) - np.outer(q, q)))

        # 对质量的导数
        dIaa_dm = -A + dA @ c
        dd_dm = -B + dB @ c
        dI = inertia_derivative(dIaa_dm, dd_dm)
        T[body, position] = 1.0
        T[4 + 3 * body:7 + 3 * body, position] = -c / m
        T[16 + 6 * body:22 + 6 * body, position] = inertia_components(dI)
        for k in range(3):
            T[4 + 3 * body + k, position + 1 + k] = 1.0 / m
            dIaa_dh = -dA[k]
            dd_dh = -dB[k]
            dI = inertia_derivative(dIaa_dh, dd_dh)
            T[16 + 6 * body:22 + 6 * body, position + 1 + k] = inertia_components(dI)
        T[16 + 6 * body:22 + 6 * body, position + 4] = inertia_components(np.outer(a, a))
        dI = inertia_derivative(0.0, 1.0)
        T[16 + 6 * body:22 + 6 * body, position + 5] = inertia_components(dI)
        position += 6

    offsets = [40, 41, 43, 44, 46, 47, 48, 49, 50, 51]
    for index, target in enumerate(offsets):
        T[target, 21 + index] = 1.0
    for index in range(4):
        T[52 + index, 31 + index] = 1.0
    # 固定 o1.x、o1.y 为 0 时，这两列对原始参数的导数也为 0
    for column in FIXED_BETA:
        T[:, column] = 0.0
    return T


def inertia_components(matrix: np.ndarray) -> np.ndarray:
    return np.array([matrix[0, 0], matrix[1, 1], matrix[2, 2],
                     matrix[0, 1], matrix[0, 2], matrix[1, 2]])


def check_mapping_jacobian(beta: np.ndarray) -> None:
    T = mapping_jacobian(beta)
    maximum = 0.0
    for column in range(T.shape[1]):
        step = 1e-6 * max(1.0, abs(beta[column]))
        plus = beta.copy()
        minus = beta.copy()
        plus[column] += step
        minus[column] -= step
        fd = (ip.map_beta(plus, True) - ip.map_beta(minus, True)) / (2.0 * step)
        error = np.max(np.abs(fd - T[:, column]))
        maximum = max(maximum, error)
        if error > 1e-6:
            print("列", column, "误差", error, "最大误差索引", int(np.argmax(np.abs(fd - T[:, column]))))
    print("映射雅可比最大绝对误差:", maximum)


KEEP = np.array([index for index in range(35) if index not in FIXED_BETA])


def project(reduced: np.ndarray) -> np.ndarray:
    full = expand_beta(reduced)
    full[3] = max(full[3], 0.05)
    full[9] = max(full[9], 0.05)
    full[15] = max(full[15], 0.05)
    m1 = params.MASSES[0]
    cx, cy = full[1] / m1, full[2] / m1
    full[0] = max(full[0], m1 * (cx * cx + cy * cy) + 1.0e-4)
    full[31:35] = np.maximum(full[31:35], 0.0)
    for body in range(1, 4):
        position = 3 + 6 * (body - 1)
        m = full[position]
        c = full[position + 1:position + 4] / m
        a = ip.AXES[body]
        p_axis = np.array(ip.PLANES[body][0])
        q_axis = np.array(ip.PLANES[body][1])
        longitudinal_min = m * ((c @ c) - (a @ c) ** 2) + 1.0e-4
        full[position + 4] = max(full[position + 4], longitudinal_min)
        B = (q_axis @ c) ** 2 - (p_axis @ c) ** 2
        gauge_trace = float(p_axis @ ip.inertia_matrix(body) @ p_axis
                            + q_axis @ ip.inertia_matrix(body) @ q_axis)
        d_value = full[position + 5] - m * B
        limit = max(gauge_trace - 1.0e-4, 1.0e-4)
        if d_value > limit:
            full[position + 5] = m * B + limit
        elif d_value < -limit:
            full[position + 5] = m * B - limit
    for _ in range(6):
        x = ip.map_beta(full, True)
        changed = False
        for body in range(4):
            matrix = np.array([[x[16 + 6 * body], x[16 + 6 * body + 3], x[16 + 6 * body + 4]],
                               [x[16 + 6 * body + 3], x[17 + 6 * body], x[16 + 6 * body + 5]],
                               [x[16 + 6 * body + 4], x[16 + 6 * body + 5], x[18 + 6 * body]]])
            minimum = float(np.linalg.eigvalsh(matrix)[0])
            if minimum <= 1.0e-6:
                if body == 0:
                    full[0] += 2.0 * (1.0e-6 - minimum) + 1.0e-6
                else:
                    full[3 + 6 * (body - 1) + 4] += 2.0 * (1.0e-6 - minimum) + 1.0e-6
                changed = True
        if not changed:
            break
    return full[KEEP]


def curriculum(data: Data, arguments) -> None:
    beta0 = ip.beta_of_prior()
    reference = beta0[KEEP].copy()
    reduced = project(reference + 0.02 * ip.beta_scales()[KEEP])
    scale = ip.beta_scales()[KEEP]
    steps_list = [int(value) for value in arguments.stages.split(",") if value]
    moment = np.zeros_like(reduced)
    velocity = np.zeros_like(reduced)
    beta1 = 0.9
    beta2 = 0.999
    regularization = arguments.regularization
    learning_rate = arguments.learning_rate
    segment_count = min(arguments.segments, data.q.shape[0])

    def regularization_loss(beta: np.ndarray) -> float:
        return regularization * float(np.sum(((beta - reference) / scale) ** 2))

    def data_loss(beta: np.ndarray, steps: int) -> float:
        total = 0.0
        for segment in range(segment_count):
            segment_context = gd.Context(1)
            try:
                loss, _ = loss_gradient(segment_context, data, beta, segment, steps)
            finally:
                segment_context.close()
            total += loss
        return total / segment_count

    for stage, steps in enumerate(steps_list):
        moment[:] = 0.0
        velocity[:] = 0.0
        for iteration in range(arguments.iterations):
            total_loss = 0.0
            total_gradient = np.zeros_like(reduced)
            for segment in range(segment_count):
                segment_context = gd.Context(1)
                try:
                    loss, gradient = loss_gradient(segment_context, data, reduced, segment, steps)
                finally:
                    segment_context.close()
                total_loss += loss
                total_gradient += gradient
            total_loss /= segment_count
            total_gradient /= segment_count
            total_loss += regularization_loss(reduced)
            total_gradient += 2.0 * regularization * (reduced - reference) / (scale * scale)
            gradient_u = scale * total_gradient
            norm = np.linalg.norm(gradient_u)
            if norm > 1.0:
                gradient_u = gradient_u / norm
            moment = beta1 * moment + (1.0 - beta1) * gradient_u
            velocity = beta2 * velocity + (1.0 - beta2) * gradient_u * gradient_u
            bias1 = 1.0 - beta1 ** (iteration + 1)
            bias2 = 1.0 - beta2 ** (iteration + 1)
            candidate = project(reduced - learning_rate * (moment / bias1) / (
                np.sqrt(velocity / bias2) + 1.0e-8))
            try:
                candidate_loss = data_loss(candidate, steps) + regularization_loss(candidate)
            except RuntimeError:
                candidate_loss = float("inf")
            if candidate_loss < total_loss:
                reduced = candidate
                learning_rate = min(learning_rate * 1.05, 0.01)
                print("课程段 %d/%d，迭代 %d，步数 %d，损失 %.6e -> %.6e" % (
                    stage + 1, len(steps_list), iteration + 1, steps,
                    total_loss, candidate_loss))
            else:
                learning_rate *= 0.5
                print("课程段 %d/%d，迭代 %d，步数 %d，拒绝更新，损失 %.6e，步长 %.2e" % (
                    stage + 1, len(steps_list), iteration + 1, steps,
                    total_loss, learning_rate))
    full_steps = min(steps_list[-1], data.q.shape[1])
    total_loss = 0.0
    for segment in range(segment_count):
        segment_context = gd.Context(1)
        try:
            loss, _ = loss_gradient(segment_context, data, reduced, segment, full_steps)
        finally:
            segment_context.close()
        total_loss += loss
    print("最终平均损失（%d 步）: %.6e" % (full_steps, total_loss / segment_count))
    estimated = ip.map_beta(expand_beta(reduced), True)
    truth = ip.map_beta(expand_beta(beta0[KEEP]), True)
    names = ip.identify.parameter_names()
    error = estimated - truth
    print("相对初始基参数映射的原始参数变化前 10：")
    for index in np.argsort(np.abs(error))[::-1][:10]:
        print("  %-18s %+.6e" % (names[index], error[index]))
    print("各基参数误差（真值 / 估计 / 绝对误差 / 尺度归一化误差）：")
    names = base_names()
    for index in range(reduced.size):
        print("  %-14s 真值 % .6e  估计 % .6e  误差 % .6e  归一化 % .4f" % (
            names[index], reference[index], reduced[index],
            reduced[index] - reference[index],
            (reduced[index] - reference[index]) / scale[index]))
    arguments.save.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arguments.save,
        reduced=reduced,
        reference=reference,
        scale=scale,
        keep=KEEP,
        full_beta=expand_beta(reduced),
        model=ip.map_beta(expand_beta(reduced), True),
        stages=np.array(steps_list),
        final_loss=total_loss / segment_count,
    )
    print("辨识参数已保存到", arguments.save)



def base_names() -> list[str]:
    names = ["B1.beta1", "B1.beta2", "B1.beta3"]
    for body in range(2, 5):
        names += [f"B{body}.m", f"B{body}.hx", f"B{body}.hy", f"B{body}.hz",
                  f"B{body}.a_I_a", f"B{body}.diff"]
    names += ["o1.x", "o1.y", "o2.x", "o2.y",
              "o3.x", "o3.y", "o3.z", "o4.x", "o4.y", "o4.z"]
    names += ["d1", "d2", "d3", "d4"]
    return [name for index, name in enumerate(names) if index not in FIXED_BETA]


def expand_beta(reduced: np.ndarray) -> np.ndarray:
    full = np.zeros(35)
    full[KEEP] = reduced
    return full


class Data:
    def __init__(self, path: Path):
        self.q, self.qdot, self.qdd, self.tau, self.terrain, self.base_yaw = ip.load_data(path)

    def segment(self, index: int, steps: int):
        terrain = gd.make_terrain(float(self.terrain[index, 0]), float(self.terrain[index, 1]))
        base = gd.make_base(yaw=float(self.base_yaw[index]))
        initial = gd.make_state(self.q[index, 0], self.qdot[index, 0])
        return (terrain, base, initial, self.tau[index, :steps],
                self.q[index, :steps], self.qdot[index, :steps])


def loss_gradient(context: gd.Context, data: Data, beta: np.ndarray,
                  segment: int, steps: int):
    x = ip.map_beta(expand_beta(beta), True)
    model = ip.build_model(x)
    terrain, base, initial, torques, target_angle, target_velocity = data.segment(segment, steps)
    loss, gradient_library, _ = context.parameter_gradient(
        model, terrain, base, initial, torques, target_angle, target_velocity,
        [1.0] * 4, [0.1] * 4, [0.0] * 4, 0.01, 2,
    )
    gradient_grouped = np.zeros(56)
    for grouped in range(56):
        gradient_grouped[grouped] = gradient_library[PERM[grouped]]
    T = mapping_jacobian(expand_beta(beta))[:, KEEP]
    return loss, T.T @ gradient_grouped


def check_loss_gradient(context: gd.Context, data: Data, beta: np.ndarray,
                        segment: int, steps: int) -> None:
    loss, gradient = loss_gradient(context, data, beta, segment, steps)
    maximum = 0.0
    for column in range(beta.size):
        step = 1e-6 * max(1.0, abs(beta[column]))
        plus = beta.copy()
        minus = beta.copy()
        plus[column] += step
        minus[column] -= step
        loss_plus, _ = loss_gradient(context, data, plus, segment, steps)
        loss_minus, _ = loss_gradient(context, data, minus, segment, steps)
        fd = (loss_plus - loss_minus) / (2.0 * step)
        error = abs(fd - gradient[column])
        maximum = max(maximum, error)
        if error > 1e-5:
            print("基参数", column, "梯度", gradient[column], "差分", fd, "误差", error)
    print("损失梯度最大绝对误差:", maximum)


def main() -> None:
    parser = argparse.ArgumentParser(description="物理基参数梯度下降")
    parser.add_argument("--check-jacobian", action="store_true")
    parser.add_argument("--check-loss-gradient", action="store_true")
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--segments", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--stages", type=str, default="40,80,160,300")
    parser.add_argument("--save", type=Path, default=Path("data/identified_beta.npz"))
    parser.add_argument("--data", type=Path, default=Path("data/identification_data.npz"))
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--segment", type=int, default=0)
    arguments = parser.parse_args()
    beta_full = ip.beta_of_prior()
    if arguments.check_jacobian:
        check_mapping_jacobian(beta_full)
        return
    if arguments.check_loss_gradient:
        data = Data(arguments.data)
        beta = beta_full[KEEP].copy()
        context = gd.Context(1)
        try:
            check_loss_gradient(context, data, beta, arguments.segment, arguments.steps)
        finally:
            context.close()
        return
    if arguments.optimize:
        data = Data(arguments.data)
        curriculum(data, arguments)
        return
    print("缺少优化实现")


if __name__ == "__main__":
    main()
