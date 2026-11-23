"""控制量梯度的中心差分校验。

覆盖软限位二次铰链、力矩变化率（含 previous_torque）与逐步跟踪全部激活的情形：
先用 tcrg_forward_step 逐步重建轨迹，在 Python 里按 C 接口的损失定义算出参考损失，
确认损失构成与字段映射正确；再对每个力矩分量做中心差分，与 tcrg_control_gradient
返回的解析梯度比较，要求双精度相对偏差小于 1e-6。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gimbal_dynamics as gd
from mpc import scenarios

TOLERANCE = 1e-6
LOSS_TOLERANCE = 1e-10
FD_STEP = 1e-6

LIMIT_MARGIN = 0.07


def build_case():
    """构造所有损失项都激活的测试用例。"""
    model = scenarios.identified_model()
    horizon = 10
    terrain = gd.make_terrain(0.12, 0.9)
    base = gd.make_base(acceleration=np.array([0.2, -0.1, 0.05]),
                        yaw=0.4, yaw_rate=0.25, yaw_acceleration=0.2)
    # 关节 2 起始角高于上软限，关节 4 起始角低于下软限，限位铰链在轨迹开头即激活
    initial = gd.make_state([0.15, 0.05, -0.10, -0.72],
                            [0.4, -0.3, 0.2, 0.1])
    index = np.arange(horizon)[:, None]
    joint = np.arange(4)[None, :]
    torques = 0.6 * np.sin(0.6 * index + 0.9 * joint) - 0.2
    previous_torque = np.array([0.9, -0.7, 0.4, -0.2])
    target_angle = 0.2 * np.sin(0.5 * index + 0.3 * joint) + 0.05 * joint
    target_velocity = 0.3 * np.cos(0.4 * index + 0.2 * joint)
    weights = {
        "angle_weights": np.array([3.0, 2.0, 4.0, 5.0]),
        "velocity_weights": np.array([0.3, 0.2, 0.5, 0.4]),
        "torque_weights": np.array([0.01, 0.02, 0.03, 0.04]),
        "torque_rate_weights": np.array([0.5, 0.4, 0.3, 0.2]),
        "limit_weights": np.array([0.0, 800.0, 600.0, 900.0]),
        "limit_lower": np.array([-np.inf, -math.radians(45.0),
                                 -math.radians(50.0), -math.radians(45.0)]),
        "limit_upper": np.array([np.inf, math.radians(5.0),
                                 math.radians(50.0), math.radians(45.0)]),
        "limit_margin": np.full(4, LIMIT_MARGIN),
    }
    return {"model": model, "terrain": terrain, "base": base, "initial": initial,
            "torques": torques, "previous_torque": previous_torque,
            "target_angle": target_angle, "target_velocity": target_velocity,
            "weights": weights, "horizon": horizon}


def step_states(context, case) -> list:
    """用 tcrg_forward_step 逐步重建轨迹，与梯度接口内部的前向积分使用同一算法。"""
    state = case["initial"]
    states = []
    for torque in case["torques"]:
        state = context.forward_step(case["model"], case["terrain"], case["base"],
                                     state, torque, 0.01, 2)
        states.append(state)
    return states


def hinge(angle: float, lower: float, upper: float, margin: float) -> float:
    if angle > upper - margin:
        return angle - (upper - margin)
    if angle < lower + margin:
        return (lower + margin) - angle
    return 0.0


def reference_loss(states, case) -> float:
    """按 C 接口的损失定义在 Python 里逐项求和。"""
    torques = case["torques"]
    weights = case["weights"]
    count = len(torques)
    total = 0.0
    for k in range(count):
        angle = np.array([states[k].angle[i] for i in range(4)])
        velocity = np.array([states[k].velocity[i] for i in range(4)])
        previous = (case["previous_torque"] if k == 0 else torques[k - 1])
        for j in range(4):
            angle_error = angle[j] - case["target_angle"][k, j]
            velocity_error = velocity[j] - case["target_velocity"][k, j]
            total += (weights["angle_weights"][j] * angle_error ** 2
                      + weights["velocity_weights"][j] * velocity_error ** 2
                      + weights["torque_weights"][j] * torques[k, j] ** 2
                      + weights["torque_rate_weights"][j]
                      * (torques[k, j] - previous[j]) ** 2
                      + weights["limit_weights"][j]
                      * hinge(angle[j], weights["limit_lower"][j],
                             weights["limit_upper"][j],
                             weights["limit_margin"][j]) ** 2)
    return total / count


def term_values(states, case) -> dict:
    """各项损失贡献（因子 1/K 已计入），用于确认全部损失项激活。"""
    torques = case["torques"]
    weights = case["weights"]
    count = len(torques)
    terms = {"tracking": 0.0, "torque": 0.0, "rate": 0.0, "limit": 0.0}
    for k in range(count):
        angle = np.array([states[k].angle[i] for i in range(4)])
        velocity = np.array([states[k].velocity[i] for i in range(4)])
        previous = (case["previous_torque"] if k == 0 else torques[k - 1])
        for j in range(4):
            terms["tracking"] += (
                weights["angle_weights"][j]
                * (angle[j] - case["target_angle"][k, j]) ** 2
                + weights["velocity_weights"][j]
                * (velocity[j] - case["target_velocity"][k, j]) ** 2)
            terms["torque"] += weights["torque_weights"][j] * torques[k, j] ** 2
            terms["rate"] += (weights["torque_rate_weights"][j]
                              * (torques[k, j] - previous[j]) ** 2)
            terms["limit"] += (weights["limit_weights"][j]
                               * hinge(angle[j], weights["limit_lower"][j],
                                      weights["limit_upper"][j],
                                      weights["limit_margin"][j]) ** 2)
    return {name: value / count for name, value in terms.items()}


def loss_of(context, case, torques: np.ndarray) -> float:
    loss, _, _ = context.control_gradient(
        case["model"], case["terrain"], case["base"], case["initial"], torques,
        target_angle=case["target_angle"],
        target_velocity=case["target_velocity"],
        previous_torque=case["previous_torque"],
        step=0.01, refinement=2, **case["weights"])
    return loss


def main() -> None:
    parser = argparse.ArgumentParser(description="控制量梯度中心差分校验")
    parser.add_argument("--identified", type=Path,
                        default=Path("data/identified_final.npz"))
    arguments = parser.parse_args()
    case = build_case()
    if arguments.identified != Path("data/identified_final.npz"):
        case["model"] = scenarios.build_model(np.load(arguments.identified)["model"])
    torques = case["torques"]
    horizon = torques.shape[0]
    passed = True
    with gd.Context(1) as context:
        states = step_states(context, case)
        expected_loss = reference_loss(states, case)
        loss, gradient, final_state = context.control_gradient(
            case["model"], case["terrain"], case["base"], case["initial"], torques,
            target_angle=case["target_angle"],
            target_velocity=case["target_velocity"],
            previous_torque=case["previous_torque"],
            step=0.01, refinement=2, **case["weights"])
        loss_deviation = abs(loss - expected_loss) / max(1.0, abs(expected_loss))
        print("损失构成校验: C 接口 %.12e  参考 %.12e  相对偏差 %.3e"
              % (loss, expected_loss, loss_deviation))
        if not loss_deviation < LOSS_TOLERANCE:
            passed = False
            print("  [失败] 损失与损失构成定义不一致")
        terms = term_values(states, case)
        print("各项损失贡献:",
              "  ".join("%s=%.6e" % (name, value)
                        for name, value in terms.items()))
        for name, value in terms.items():
            if not value > 1e-9:
                passed = False
                print(f"  [失败] 损失项 {name} 未激活（贡献 {value:.3e}）")

        difference = np.zeros((horizon, 4))
        for k in range(horizon):
            for j in range(4):
                perturbation = np.zeros((horizon, 4))
                perturbation[k, j] = FD_STEP
                forward = loss_of(context, case, torques + perturbation)
                backward = loss_of(context, case, torques - perturbation)
                difference[k, j] = (forward - backward) / (2.0 * FD_STEP)
        analytic = np.asarray(gradient, dtype=float)
        absolute = np.abs(analytic - difference)
        norm_relative = (np.linalg.norm(analytic - difference)
                         / np.linalg.norm(difference))
        print("梯度校验: 分量数 %d  中心差分最大绝对偏差 %.3e  范数相对偏差 %.3e"
              % (analytic.size, absolute.max(), norm_relative))
        print("  解析梯度范围 [%.3e, %.3e]  差分梯度范围 [%.3e, %.3e]"
              % (analytic.min(), analytic.max(), difference.min(), difference.max()))
        if not norm_relative < TOLERANCE:
            passed = False
            print("  [失败] 范数相对偏差超过 %.1e" % TOLERANCE)
        worst = int(np.argmax(absolute))
        print("  最大偏差分量 k=%d joint=%d 解析 %.6e 差分 %.6e"
              % (worst // 4, worst % 4, analytic.reshape(-1)[worst],
                 difference.reshape(-1)[worst]))

        simulated = context.forward_simulate(
            case["model"], case["terrain"], case["base"], case["initial"],
            torques, 0.01, 2)
        state_deviation = max(
            max(abs(final_state.angle[i] - simulated.angle[i]) for i in range(4)),
            max(abs(final_state.velocity[i] - simulated.velocity[i]) for i in range(4)))
        print("末端状态校验: 与 tcrg_forward_simulate 最大偏差 %.3e" % state_deviation)
        if not state_deviation < 1e-10:
            passed = False
            print("  [失败] 梯度接口末端状态与前向仿真不一致")
    print("结果:", "通过" if passed else "未通过")
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
