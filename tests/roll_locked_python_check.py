"""锁定 roll 轴三自由度模型的 Python 绑定校验。

运行：python3 tests/roll_locked_python_check.py
"""

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import gimbal_dynamics as gd
import params


def ry(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def reference_merged(locked_angle):
    masses = np.array(params.MASSES)
    com = np.array(params.COM)
    inertia = []
    for row in params.INERTIA:
        xx, yy, zz, xy, xz, yz = row
        inertia.append(np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]))
    inertia = np.array(inertia)
    offsets = np.array(params.JOINT_OFFSETS)
    damping = np.array(params.DAMPING)
    F = ry(locked_angle)
    m1, m2 = masses[0], masses[1]
    m = m1 + m2
    c1, c2 = com[0], com[1]
    c = (m1 * c1 + m2 * (offsets[1] + F @ c2)) / m
    d1 = c1 - c
    d2 = offsets[1] + F @ c2 - c
    inertia12 = (inertia[0] + m1 * (d1 @ d1 * np.eye(3) - np.outer(d1, d1))
                 + F @ inertia[1] @ F.T
                 + m2 * (d2 @ d2 * np.eye(3) - np.outer(d2, d2)))
    return {
        "mass": np.array([m, masses[2], masses[3]]),
        "com": np.array([c, com[2], com[3]]),
        "inertia": np.array([inertia12, inertia[2], inertia[3]]),
        "joint_offsets": np.array([offsets[0], offsets[1] + F @ offsets[2], offsets[3]]),
        "damping": np.array([damping[0], damping[2], damping[3]]),
        "gravity": params.GRAVITY,
        "locked_angle": locked_angle,
    }


def inertia_to_matrix(row):
    xx, yy, zz, xy, xz, yz = row
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def check_parameters(context, model):
    max_error = 0.0
    for locked_angle in (0.0, 0.4, -1.1, math.pi / 2.0):
        actual = context.roll_locked_parameters(model, locked_angle)
        expected = reference_merged(locked_angle)
        for name in ("mass", "com", "joint_offsets", "damping"):
            actual_array = np.asarray(actual[name], dtype=float)
            expected_array = np.asarray(expected[name], dtype=float)
            max_error = max(max_error,
                            np.max(np.abs(actual_array - expected_array)) /
                            max(1.0, np.max(np.abs(expected_array))))
        for body in range(3):
            actual_inertia = inertia_to_matrix(actual["inertia"][body])
            expected_inertia = expected["inertia"][body]
            max_error = max(max_error,
                            np.max(np.abs(actual_inertia - expected_inertia)) /
                            max(1.0, np.max(np.abs(expected_inertia))))
        max_error = max(max_error,
                        abs(actual["gravity"] - expected["gravity"]) /
                        max(1.0, expected["gravity"]))
    print(f"Python 合并参数与参考公式最大相对偏差 = {max_error:.3e}")
    return max_error


def limit_hinge(angle, lower, upper, margin):
    if angle > upper - margin:
        return angle - (upper - margin)
    if angle < lower + margin:
        return angle - (lower + margin)
    return 0.0


def forward_loss(context, model, locked_angle, terrain, base, initial, torques,
                 targets_angle, targets_velocity, weights, previous_torque, dt, refinement):
    state = gd.make_roll_locked_state(
        [initial.angle[i] for i in range(3)],
        [initial.velocity[i] for i in range(3)],
    )
    states = []
    for step in range(len(torques)):
        state = context.roll_locked_forward_step(
            model, locked_angle, terrain, base[step], state, torques[step], dt, refinement)
        states.append(gd.make_roll_locked_state(
            [state.angle[i] for i in range(3)],
            [state.velocity[i] for i in range(3)],
        ))
    total = 0.0
    for step, state in enumerate(states):
        for joint in range(3):
            angle_error = state.angle[joint] - targets_angle[step][joint]
            velocity_error = state.velocity[joint] - targets_velocity[step][joint]
            hinge = limit_hinge(state.angle[joint], weights["limit_lower"][joint],
                                weights["limit_upper"][joint], weights["limit_margin"][joint])
            total += (weights["angle_weights"][joint] * angle_error * angle_error
                      + weights["velocity_weights"][joint] * velocity_error * velocity_error
                      + weights["torque_weights"][joint] * torques[step][joint] ** 2
                      + weights["limit_weights"][joint] * hinge * hinge)
            if step == 0:
                reference = previous_torque
            else:
                reference = torques[step - 1]
            total += weights["torque_rate_weights"][joint] * (
                torques[step][joint] - reference[joint]) ** 2
    return total / len(torques), states


def main():
    rng = np.random.default_rng(20261125)
    model = gd.make_model()
    terrain = gd.make_terrain(0.05, 0.7)
    with gd.Context(1) as context:
        max_parameter_error = check_parameters(context, model)
        locked_angle = 0.4
        steps = 5
        dt = 0.01
        refinement = 2
        initial = gd.make_roll_locked_state(
            rng.uniform(-0.2, 0.2, 3), rng.uniform(-0.3, 0.3, 3))
        base = [gd.make_base(rng.uniform(-0.3, 0.3, 3),
                             rng.uniform(-3.0, 3.0),
                             rng.uniform(-2.0, 2.0),
                             rng.uniform(-5.0, 5.0)) for _ in range(steps)]
        torques = rng.uniform(-0.5, 0.5, (steps, 3))
        target_angle = rng.uniform(-0.2, 0.2, (steps, 3))
        target_velocity = rng.uniform(-0.3, 0.3, (steps, 3))
        previous_torque = np.array([0.1, -0.08, 0.02])
        weights = {
            "angle_weights": [1.0, 2.0, 0.8],
            "velocity_weights": [0.3, 0.5, 0.2],
            "torque_weights": [0.05, 0.04, 0.03],
            "torque_rate_weights": [0.7, 0.4, 0.6],
            "limit_weights": [2.0, 2.5, 1.5],
            "limit_lower": [-0.1, -0.08, -0.2],
            "limit_upper": [0.1, 0.08, 0.2],
            "limit_margin": [0.01, 0.01, 0.01],
        }
        loss, gradient, final_state = context.roll_locked_control_gradient(
            model, locked_angle, terrain, base, initial, torques,
            target_angle=target_angle,
            target_velocity=target_velocity,
            previous_torque=previous_torque,
            step=dt,
            refinement=refinement,
            **weights,
        )
        python_loss, states = forward_loss(
            context, model, locked_angle, terrain, base, initial, torques,
            target_angle, target_velocity, weights, previous_torque, dt, refinement)
        final_from_steps = states[-1]
        final_error = max(abs(final_state.angle[i] - final_from_steps.angle[i])
                          for i in range(3))
        final_error = max(final_error,
                          max(abs(final_state.velocity[i] - final_from_steps.velocity[i])
                              for i in range(3)))
        loss_error = abs(python_loss - loss) / max(1.0, abs(python_loss))
        print(f"Python 前向损失与 C++ 梯度损失相对偏差 = {loss_error:.3e}")
        print(f"Python 逐步前向末状态与 C++ 梯度末状态最大偏差 = {final_error:.3e}")

        epsilon = 1e-6
        max_gradient_error = 0.0
        for step in range(steps):
            for joint in range(3):
                plus = np.array(torques, copy=True)
                minus = np.array(torques, copy=True)
                plus[step, joint] += epsilon
                minus[step, joint] -= epsilon
                plus_loss, _ = forward_loss(
                    context, model, locked_angle, terrain, base, initial, plus,
                    target_angle, target_velocity, weights, previous_torque, dt, refinement)
                minus_loss, _ = forward_loss(
                    context, model, locked_angle, terrain, base, initial, minus,
                    target_angle, target_velocity, weights, previous_torque, dt, refinement)
                difference = (plus_loss - minus_loss) / (2.0 * epsilon)
                analytic = gradient[step, joint]
                max_gradient_error = max(max_gradient_error,
                                         abs(difference - analytic) /
                                         max(1.0, abs(analytic)))
        print(f"Python 控制量梯度与中心差分最大相对偏差 = {max_gradient_error:.3e}")

        forward_final = context.roll_locked_forward_simulate(
            model, locked_angle, terrain, base[0], initial, torques, dt, refinement)
        constant_base_state = gd.make_roll_locked_state(
            [initial.angle[i] for i in range(3)],
            [initial.velocity[i] for i in range(3)],
        )
        for step in range(steps):
            constant_base_state = context.roll_locked_forward_step(
                model, locked_angle, terrain, base[0], constant_base_state,
                torques[step], dt, refinement)
        simulate_error = max(abs(forward_final.angle[i] - constant_base_state.angle[i])
                             for i in range(3))
        simulate_error = max(simulate_error,
                             max(abs(forward_final.velocity[i] - constant_base_state.velocity[i])
                                 for i in range(3)))
        print(f"Python 一体化前向与逐步前向末状态最大偏差 = {simulate_error:.3e}")

        if (max_parameter_error > 1e-12 or loss_error > 1e-9 or final_error > 1e-9
                or max_gradient_error > 2e-6 or simulate_error > 1e-9):
            raise SystemExit("Python 绑定校验失败")
        print("Python 绑定校验通过")


if __name__ == "__main__":
    main()
