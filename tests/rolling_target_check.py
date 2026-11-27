"""滚动目标序列校验：目标点只按时间滚动，MPC 的预测目标与 RL 的预瞄偏移都由它给出。

运行：python3 tests/rolling_target_check.py
"""

import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python" / "visualization"))

from controllers import MpcController
from target_sequence import TargetSequence

DT = 0.01


def observation() -> dict:
    return {
        "angle": [0.0] * 4,
        "velocity": [0.0] * 4,
        "slope_angle": 0.0,
        "slope_direction": 0.0,
        "base_acceleration": [0.0, 0.0, 0.0],
        "base_yaw": 0.0,
        "base_yaw_rate": 0.0,
        "base_yaw_acceleration": 0.0,
    }


def check_sequence_rolls():
    initial = np.zeros(4)
    sequence = TargetSequence(3, initial)
    if sequence.points.shape != (4, 4):
        raise RuntimeError(f"目标序列形状错误：{sequence.points.shape}")
    if not np.array_equal(sequence.points, np.zeros((4, 4))):
        raise RuntimeError("复位没有用当前目标填满目标序列")

    expected = deque([initial.copy() for _ in range(4)], maxlen=4)
    previous = sequence.points
    for step in range(1, 6):
        target = np.array([0.01 * step, -0.02 * step, 0.0, 0.0])
        sequence.push(target)
        expected.append(target)
        points = sequence.points
        if not np.array_equal(points, np.stack(expected)):
            raise RuntimeError("目标序列与按时间滚动的期望内容不一致")
        if not np.array_equal(points[-1], target):
            raise RuntimeError("新压入的当前目标没有出现在序列尾端")
        if not np.array_equal(points[:-1], previous[1:]):
            raise RuntimeError("除尾端外已有目标点被修改")
        if not np.array_equal(sequence.reference, points[0]):
            raise RuntimeError("参考目标不是序列最旧的元素")
        if not np.array_equal(sequence.predictions, points[1:]):
            raise RuntimeError("预测目标不是序列尾端的 steps 个元素")
        previous = points
    print("滚动目标序列校验通过：每拍只新增尾端，其余目标点数值不变")


def check_relative_preview():
    sequence = TargetSequence(2, np.zeros(4))
    for step in range(1, 6):
        sequence.push(np.array([0.01 * step, 0.0, 0.0, 0.0]))
    reference, rate, offsets = sequence.relative_preview(DT)
    points = sequence.points
    if not np.array_equal(reference, points[0]):
        raise RuntimeError("预瞄的参考目标不是序列最旧的元素")
    if not np.allclose(rate, (points[1] - points[0]) / DT):
        raise RuntimeError("参考目标速度不是相邻目标点之差除以控制周期")
    if not np.allclose(offsets.reshape(-1, 4), points[1:] - points[0]):
        raise RuntimeError("预瞄偏移不是预测目标相对参考目标的差值")
    if not np.allclose(rate, [1.0, 0.0, 0.0, 0.0]):
        raise RuntimeError(f"匀速移动时参考目标速度不正确：{rate}")
    print("预瞄校验通过：参考目标、参考目标速度与相对偏移都由滚动目标点给出")


def check_mpc_targets():
    controller = MpcController(target_rate=1.0)
    captured = []

    def record(angle, velocity, target_angle, terrain, base):
        captured.append((np.array(target_angle, dtype=float),
                         controller.targets.predictions.copy()))
        return np.zeros(4)

    try:
        controller.mpc.command = record
        controller.reset(observation())
        for step in range(3):
            controller.command(observation(), [(0, 1.0)], DT)
        if len(captured) != 3:
            raise RuntimeError(f"MPC 求解调用次数错误：{len(captured)}")
        horizon = controller.mpc.horizon
        for index, (target_angle, predictions) in enumerate(captured):
            if target_angle.shape != (horizon, 4):
                raise RuntimeError(f"MPC 预测目标形状错误：{target_angle.shape}")
            if not np.array_equal(target_angle, predictions):
                raise RuntimeError("MPC 求解使用的预测目标与滚动目标序列不一致")
            if not np.allclose(target_angle[-1], [0.01 * (index + 1), 0.0, 0.0, 0.0]):
                raise RuntimeError("预测目标尾端不是当前目标")
        for previous, current in zip(captured[:-1], captured[1:]):
            if not np.allclose(current[0][:-1], previous[0][1:]):
                raise RuntimeError("相邻两拍的预测目标没有按时间滚动")

        for _ in range(horizon + 1):
            controller.command(observation(), [], DT)
        final = captured[-1][0]
        if not np.allclose(final, np.array([0.03, 0.0, 0.0, 0.0])):
            raise RuntimeError("松开按键后预测目标仍在变化")
        if np.max(np.abs(final)) > 0.03 + 1.0e-12:
            raise RuntimeError("预测目标超出了已压入的目标点")
        print("MPC 目标校验通过：预测目标由滚动目标点给出，松开按键后保持常值")
    finally:
        controller.close()


def check_rl_references():
    import deploy

    controller = deploy.PolicyController.__new__(deploy.PolicyController)
    controller.targets = TargetSequence(deploy.PREVIEW_COUNT, np.zeros(4))
    reference, rate, offsets = deploy.PolicyController.references(
        controller, np.array([0.05, 0.0, 0.0, 0.0]))
    points = controller.targets.points
    if not np.array_equal(reference, points[0]):
        raise RuntimeError("RL 参考目标不是滚动目标序列最旧的元素")
    if not np.array_equal(offsets.reshape(-1, 4), points[1:] - points[0]):
        raise RuntimeError("RL 预瞄偏移不是预测目标相对参考目标的差值")
    if offsets.size != deploy.PREVIEW_COUNT * 4:
        raise RuntimeError(f"RL 预瞄维度错误：{offsets.size}")

    for step in range(2, deploy.PREVIEW_COUNT + 3):
        reference, rate, offsets = deploy.PolicyController.references(
            controller, np.array([0.05 * step, 0.0, 0.0, 0.0]))
    if not np.allclose(rate, [deploy.VELOCITY_LIMIT, 0.0, 0.0, 0.0]):
        raise RuntimeError(f"RL 参考目标速度没有按上限裁剪：{rate}")
    if not np.allclose(offsets.reshape(-1, 4)[:, 0],
                       np.arange(1, deploy.PREVIEW_COUNT + 1) * 0.05):
        raise RuntimeError("RL 预瞄偏移没有随滚动目标序列推进")
    print("RL 参考校验通过：参考目标、速度与预瞄偏移都由滚动目标序列给出")


def main():
    check_sequence_rolls()
    check_relative_preview()
    check_mpc_targets()
    check_rl_references()
    print("滚动目标序列校验通过")


if __name__ == "__main__":
    main()
