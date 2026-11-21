"""运动方向一致性检查：在状态网格上比较辨识模型与真实模型的初始加速度与短时位移方向。

真实参数只用于评估。检查覆盖：
1. 零力矩与小幅常力矩下的初始加速度方向；
2. 15 步开环净位移方向；
3. 逐关节的相对加速度误差。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import gimbal_dynamics as gd
import identify_physical as ip

ROLL = (-0.7853981634, 0.0872664626)
SMALL_YAW = (-0.8726646260, 0.8726646260)
PITCH = (-0.7853981634, 0.7853981634)


def grid_states():
    states = []
    for slope in (0.0, 0.12, 0.25, 0.349):
        for direction in (0.0, 1.5707963, 3.1415927, 4.7123890):
            for yaw in (-3.1415927, -1.5707963, 0.0, 1.5707963):
                for big_yaw in (-1.2, 0.0, 1.2):
                    for roll in (-0.75, -0.45, -0.15, 0.05):
                        for small_yaw in (-0.8, 0.0, 0.8):
                            for pitch in (-0.75, 0.0, 0.75):
                                states.append((np.array([big_yaw, roll, small_yaw, pitch]),
                                               slope, direction, yaw))
    return states


def dynamics_acceleration(context, model, terrain, base, state, torque):
    zero = np.array(context.inverse_dynamics(model, terrain, base, state, np.zeros(4)))
    matrix = np.column_stack([
        np.array(context.inverse_dynamics(model, terrain, base, state, np.eye(4)[j])) - zero
        for j in range(4)])
    velocity = np.array([state.velocity[i] for i in range(4)])
    command = np.asarray(torque, dtype=float) - model.damping[0] * velocity * 0.0
    damping = np.array([model.damping[i] for i in range(4)])
    return np.linalg.solve(matrix, command - damping * velocity - zero)


def simulate(context, model, terrain, base, initial_q, initial_v, torques):
    state = gd.make_state(initial_q, initial_v)
    for torque in torques:
        state = context.forward_step(model, terrain, base, state, torque, 0.01, 2)
    return np.array([state.angle[i] for i in range(4)])


def main() -> None:
    parser = argparse.ArgumentParser(description="运动方向一致性检查")
    parser.add_argument("--identified", type=Path, default=Path("data/identified_regressor.npz"))
    parser.add_argument("--stride", type=int, default=1)
    arguments = parser.parse_args()
    identified = ip.build_model(np.load(arguments.identified)["model"])
    truth = gd.make_model()
    states = grid_states()[::arguments.stride]
    print("状态网格数量", len(states))

    context = gd.Context(1)
    cases = (("零力矩", np.zeros(4)), ("正力矩", np.full(4, 0.3)), ("负力矩", np.full(4, -0.3)))
    try:
        for name, torque in cases:
            sign_total = 0
            sign_bad = 0
            relative = []
            worst = None
            for q, slope, direction, yaw in states:
                terrain = gd.make_terrain(slope, direction)
                base = gd.make_base(yaw=yaw)
                state = gd.make_state(q, np.zeros(4))
                truth_acceleration = dynamics_acceleration(context, truth, terrain, base, state,
                                                           torque)
                ident_acceleration = dynamics_acceleration(context, identified, terrain, base,
                                                           state, torque)
                scale = max(np.max(np.abs(truth_acceleration)), 1.0e-3)
                relative.append(np.max(np.abs(ident_acceleration - truth_acceleration)) / scale)
                dominant = np.max(np.abs(truth_acceleration))
                for joint in range(4):
                    if abs(truth_acceleration[joint]) > 0.05 * dominant:
                        sign_total += 1
                        if np.sign(truth_acceleration[joint]) != np.sign(ident_acceleration[joint]):
                            sign_bad += 1
                            if worst is None:
                                worst = (q.copy(), slope, direction, yaw, joint,
                                         truth_acceleration.copy(), ident_acceleration.copy())
            print("%s：符号比较 %d 次，反向 %d 次，相对误差中位数 %.3f，最大 %.3f" % (
                name, sign_total, sign_bad, np.median(relative), np.max(relative)))
            if worst is not None:
                print("  首个反向例子：q=%s 坡度=%.2f 坡向=%.2f 偏航=%.2f 关节=%d" % (
                    np.array2string(worst[0], precision=2), worst[1], worst[2], worst[3], worst[4]))
                print("    真值加速度 %s，辨识加速度 %s" % (
                    np.array2string(worst[5], precision=4),
                    np.array2string(worst[6], precision=4)))

        # 短时净位移方向（15 步）
        for name, torque in cases:
            total = 0
            bad = 0
            severe = []
            for q, slope, direction, yaw in states[::4]:
                terrain = gd.make_terrain(slope, direction)
                base = gd.make_base(yaw=yaw)
                initial = np.zeros(4)
                sequence = [torque] * 15
                truth_angles = simulate(context, truth, terrain, base, q, initial, sequence)
                ident_angles = simulate(context, identified, terrain, base, q, initial, sequence)
                movement_truth = truth_angles - q
                movement_ident = ident_angles - q
                for joint in range(4):
                    if abs(movement_truth[joint]) > 2.0e-3:
                        total += 1
                        if np.sign(movement_truth[joint]) != np.sign(movement_ident[joint]):
                            bad += 1
                            if abs(movement_truth[joint]) > 1.0e-2:
                                severe.append((q.copy(), slope, direction, yaw, joint,
                                               float(movement_truth[joint]),
                                               float(movement_ident[joint])))
            print("15 步净位移方向（%s）：比较 %d 次，反向 %d 次，其中真值位移>1e-2 的反向 %d 次"
                  % (name, total, bad, len(severe)))
            for item in severe[:3]:
                print("    q=%s 坡度=%.2f 坡向=%.2f 偏航=%.2f 关节=%d 真值位移 %.4f 辨识位移 %.4f"
                      % (np.array2string(item[0], precision=2), item[1], item[2], item[3],
                         item[4], item[5], item[6]))
    finally:
        context.close()


if __name__ == "__main__":
    main()