"""roll 轴锁定：完全非弹性碰撞与 UI 路径校验。

运行：python3 tests/roll_lock_ui_check.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import gimbal_dynamics as gd
import gimbal_env
from gimbal_env import GimbalEnv


def mass_matrix(context, model, terrain, base, state):
    zero = [0.0, 0.0, 0.0, 0.0]
    base_torque = np.asarray(context.inverse_dynamics(model, terrain, base, state, zero),
                             dtype=float)
    mass = np.zeros((4, 4))
    for joint in range(4):
        qdd = [0.0, 0.0, 0.0, 0.0]
        qdd[joint] = 1.0
        mass[:, joint] = np.asarray(
            context.inverse_dynamics(model, terrain, base, state, qdd), dtype=float) - base_torque
    return mass


def check_environment_physics():
    if gimbal_env.Environment.supports_roll_lock:
        raise RuntimeError("Environment 默认不应支持 roll 锁定")
    if not GimbalEnv.supports_roll_lock:
        raise RuntimeError("GimbalEnv 应支持 roll 锁定")

    env = GimbalEnv(dt=0.01, refinement=2)
    try:
        angle = np.array([0.20, 0.05, -0.30, 0.10])
        velocity = np.array([0.50, 1.20, -0.80, 0.60])
        env.reset({"angle": angle, "velocity": velocity})
        actual_angle = np.array([env._state.angle[i] for i in range(4)])
        actual_velocity = np.array([env._state.velocity[i] for i in range(4)])
        pre_state = gd.make_state(actual_angle, actual_velocity)
        mass = mass_matrix(env.context, env.model, env.terrain, env.base, pre_state)

        env.lock_roll()
        if not env._locked:
            raise RuntimeError("lock_roll 后没有进入锁定状态")
        post_angle = np.array([env._state.angle[i] for i in range(4)])
        post_velocity = np.array([env._state.velocity[i] for i in range(4)])
        locked_angle = float(actual_angle[1])
        if abs(post_angle[1] - locked_angle) > 1e-15:
            raise RuntimeError("锁定后 roll 角发生变化")
        if abs(post_velocity[1]) > 1e-15:
            raise RuntimeError("锁定后 roll 速度没有归零")

        impulse = mass @ (post_velocity - actual_velocity)
        norm = float(np.linalg.norm(impulse))
        if norm > 1e-15:
            direction = impulse / norm
            direction_error = max(abs(direction[0]), abs(direction[2]), abs(direction[3]),
                                  abs(abs(direction[1]) - 1.0))
            if direction_error > 1e-10:
                raise RuntimeError(f"冲量方向不沿 roll 轴：{direction_error}")
        ke_before = 0.5 * actual_velocity @ mass @ actual_velocity
        ke_after = 0.5 * post_velocity @ mass @ post_velocity
        if ke_after > ke_before + 1e-12:
            raise RuntimeError("锁定后关节动能增加")

        locked_state = gd.make_roll_locked_state(
            [env._locked_state.angle[i] for i in range(3)],
            [env._locked_state.velocity[i] for i in range(3)])
        torque = np.array([0.10, -0.20, 0.30, -0.10])
        expected = env.context.roll_locked_forward_step(
            env.model, env._locked_angle, env.terrain, env.base, locked_state,
            [torque[0], torque[2], torque[3]], env.dt, env.refinement)
        observation = env.step({"torque": torque})
        expected_angle = np.array([expected.angle[0], env._locked_angle,
                                   expected.angle[1], expected.angle[2]])
        expected_velocity = np.array([expected.velocity[0], 0.0,
                                      expected.velocity[1], expected.velocity[2]])
        if np.max(np.abs(np.asarray(observation["angle"]) - expected_angle)) > 1e-14:
            raise RuntimeError("锁定 step 的角度与直接调用三自由度模型不一致")
        if np.max(np.abs(np.asarray(observation["velocity"]) - expected_velocity)) > 1e-14:
            raise RuntimeError("锁定 step 的速度与直接调用三自由度模型不一致")

        env.unlock_roll()
        if env._locked:
            raise RuntimeError("unlock_roll 后仍处于锁定状态")
        unlocked_angle = np.array([env._state.angle[i] for i in range(4)])
        unlocked_velocity = np.array([env._state.velocity[i] for i in range(4)])
        if abs(unlocked_angle[1] - locked_angle) > 1e-15 or abs(unlocked_velocity[1]) > 1e-15:
            raise RuntimeError("解锁后的 roll 状态不正确")
        env.step({"torque": [0.0, 0.0, 0.0, 0.0]})
        print("环境物理校验通过：roll 速度归零、冲量沿 roll 轴、动能不增加、锁定/解锁状态正确")
    finally:
        env.close()


def check_real_environment_flag():
    try:
        from robot_env import RealGimbalEnv
    except Exception as error:
        print("跳过实机环境属性检查：", error)
        return
    if RealGimbalEnv.supports_roll_lock:
        raise RuntimeError("实机环境不应支持 roll 锁定")
    print("实机环境 roll 锁定不可用标记检查通过")


def check_viewer_direct():
    try:
        from visualization.ui import Viewer
    except ImportError as error:
        print("跳过 Viewer 检查：", error)
        return
    viewer = Viewer(direct=True, mode="pid")
    try:
        if not viewer.lock_available:
            raise RuntimeError("仿真 Viewer 未启用 roll 锁定")
        angle = [0.20, 0.05, -0.30, 0.10]
        velocity = [0.10, 0.80, -0.20, 0.15]
        viewer.env.reset({"angle": angle, "velocity": velocity})
        viewer.observation = viewer.env.observation()
        viewer.controller().reset(viewer.observation)
        viewer.roll_locked = False
        viewer.locked_roll_angle = None

        locked_angle = float(viewer.observation["angle"][1])
        viewer.toggle_roll_lock()
        if not viewer.roll_locked:
            raise RuntimeError("Viewer 锁定失败")
        if abs(float(viewer.observation["velocity"][1])) > 1e-15:
            raise RuntimeError("Viewer 锁定后 roll 速度没有归零")
        target = viewer.controllers["pid"].target
        if target is None or abs(float(target[1]) - locked_angle) > 1e-15:
            raise RuntimeError("Viewer 锁定后 roll 目标没有固定")

        viewer.pressed = {ord("3")}
        if viewer.inputs():
            raise RuntimeError("Viewer 锁定后 roll 按键仍然生效")
        viewer.pressed = {ord("1")}
        if viewer.inputs() != [(0, 1.0)]:
            raise RuntimeError("Viewer 锁定后其他关节按键失效")
        viewer.pressed.clear()
        for _ in range(3):
            viewer.advance()
        if abs(float(viewer.controllers["pid"].target[1]) - locked_angle) > 1e-15:
            raise RuntimeError("Viewer 推进过程中 roll 目标发生变化")

        viewer.toggle_roll_lock()
        if viewer.roll_locked:
            raise RuntimeError("Viewer 解锁失败")
        if abs(float(viewer.controllers["pid"].target[1]) - locked_angle) > 1e-15:
            raise RuntimeError("Viewer 解锁后 roll 目标没有保持锁定值")
        viewer.pressed = {ord("3")}
        if viewer.inputs() != [(1, 1.0)]:
            raise RuntimeError("Viewer 解锁后 roll 按键没有恢复")
        print("Viewer 直接模式校验通过：锁定目标固定、roll 按键过滤、解锁后目标保持")
    finally:
        viewer.close()


def main():
    check_environment_physics()
    check_real_environment_flag()
    check_viewer_direct()
    print("roll 锁定 UI 校验通过")


if __name__ == "__main__":
    main()
