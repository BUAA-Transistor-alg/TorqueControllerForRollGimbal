"""四自由度串链动力学可视化。

PyBullet 只用于显示与键盘交互，不参与动力学计算；每个渲染帧调用 C 接口中的
双精度前向仿真若干步，再把各连杆的位姿同步到 PyBullet 的可视化物体上。

键盘：
  1 / 2   关节 1  +1 / -1 N·m
  3 / 4   关节 2  +1 / -1 N·m
  5 / 6   关节 3  +1 / -1 N·m
  7 / 8   关节 4  +1 / -1 N·m
  r       回到零位
  q/ESC   退出

用法：
  python ui.py                 # GUI
  python ui.py --direct --steps 100   # 无界面自检

依赖：pybullet、本仓库的 python 目录（gimbal_dynamics.py、params.py）。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import math
import sys
import time

import numpy as np

try:
    import pybullet as p
except ImportError as error:
    raise SystemExit("需要 pybullet：pip install pybullet") from error

import gimbal_dynamics as gd
import params

STEP = 0.004          # 每个主步的时长
REFINEMENT = 2       # 每个主步内的 RK4 子步数
STEPS_PER_FRAME = 4  # 每渲染帧推进的主步数，约 62.5 Hz
FRAME_DT = STEP * STEPS_PER_FRAME

KEY_TORQUE = {
    ord("1"): (0, +1.0),
    ord("2"): (0, -1.0),
    ord("3"): (1, +1.0),
    ord("4"): (1, -1.0),
    ord("5"): (2, +1.0),
    ord("6"): (2, -1.0),
    ord("7"): (3, +1.0),
    ord("8"): (3, -1.0),
}


def rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quaternion_from_matrix(matrix: np.ndarray) -> list[float]:
    trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
    if trace > 0.0:
        scale = 2.0 * math.sqrt(trace + 1.0)
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = 2.0 * math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = 2.0 * math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = 2.0 * math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale
    return [x, y, z, w]


class PidController:
    def __init__(self) -> None:
        self.integral = [0.0, 0.0, 0.0, 0.0]

    def reset(self) -> None:
        self.integral = [0.0, 0.0, 0.0, 0.0]

    def torque(self, angles: list[float], velocities: list[float],
               target: list[float], dt: float) -> list[float]:
        command = [0.0, 0.0, 0.0, 0.0]
        for i in range(4):
            error = target[i] - angles[i]
            self.integral[i] = max(
                -params.PID_INTEGRAL_LIMIT,
                min(params.PID_INTEGRAL_LIMIT, self.integral[i] + error * dt),
            )
            value = (
                params.PID_KP[i] * error
                + params.PID_KI[i] * self.integral[i]
                - params.PID_KD[i] * velocities[i]
            )
            command[i] = max(-params.PID_TORQUE_LIMIT, min(params.PID_TORQUE_LIMIT, value))
        return command


def link_poses(angles: list[float]):
    q1, q2, q3, q4 = angles
    r1 = rot_z(q1)
    r2 = r1 @ rot_y(q2)
    r3 = r2 @ rot_z(q3)
    r4 = r3 @ rot_x(q4)
    o1 = np.array(params.JOINT_OFFSETS[0])
    o2 = np.array(params.JOINT_OFFSETS[1])
    o3 = np.array(params.JOINT_OFFSETS[2])
    o4 = np.array(params.JOINT_OFFSETS[3])
    p1 = o1
    p2 = p1 + r1 @ o2
    p3 = p2 + r2 @ o3
    p4 = p3 + r3 @ o4
    return [(p1, r1), (p2, r2), (p3, r3), (p4, r4)]


def create_visual_body(shape_id: int, position, color, mass: float = 0.0) -> int:
    visual = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=shape_id,
        rgbaColor=color,
    )
    return p.createMultiBody(
        baseMass=mass,
        baseVisualShapeIndex=visual,
        basePosition=position,
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
    )


class Viewer:
    def __init__(self, direct: bool = False, mode: str = "torque",
                 target_rate: float | None = None):
        self.direct = direct
        if direct:
            self.client = p.connect(p.DIRECT)
        else:
            self.client = p.connect(p.GUI)
        if self.client < 0:
            raise SystemExit("无法连接 PyBullet")
        p.setGravity(0.0, 0.0, 0.0)
        p.setTimeStep(FRAME_DT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        if not direct:
            p.resetDebugVisualizerCamera(1.2, 45.0, -18.0, [0.0, 0.12, 0.28])
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

        # 地面参考
        ground = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.8, 0.8, 0.005],
            rgbaColor=[0.55, 0.55, 0.55, 1.0],
        )
        self.ground = p.createMultiBody(0.0, ground, basePosition=[0.0, 0.0, -0.005])

        self.bodies = [
            create_visual_body([0.08, 0.08, 0.055], [0, 0, 0], [0.20, 0.45, 0.85, 1.0]),
            create_visual_body([0.05, 0.10, 0.04], [0, 0.09, 0.15], [0.20, 0.75, 0.35, 1.0]),
            create_visual_body([0.05, 0.05, 0.07], [0, 0.09, 0.20], [0.90, 0.55, 0.15, 1.0]),
            create_visual_body([0.05, 0.12, 0.04], [0, 0.09, 0.36], [0.85, 0.25, 0.25, 1.0]),
        ]
        self.base_body = create_visual_body([0.10, 0.10, 0.04], [0, 0, 0], [0.35, 0.35, 0.35, 1.0])
        self.axis_ids = [-1, -1, -1, -1]

        self.model = gd.make_model()
        self.terrain = gd.make_terrain()
        self.base = gd.make_base()
        self.state = gd.make_state()
        self.context = gd.Context(1)
        self.pressed: set[int] = set()
        self.mode = mode
        self.target_rate = params.PID_TARGET_RATE if target_rate is None else target_rate
        self.target = [0.0, 0.0, 0.0, 0.0]
        self.pid = PidController()
        self.mode_text_id = -1
        self.reset()

    def reset(self) -> None:
        self.state = gd.make_state()
        self.pressed.clear()
        self.target = [0.0, 0.0, 0.0, 0.0]
        self.pid.reset()
        self.update_visuals()

    def torque_command(self) -> list[float]:
        command = [0.0, 0.0, 0.0, 0.0]
        for key in self.pressed:
            joint, sign = KEY_TORQUE[key]
            command[joint] += sign
        return [max(-1.0, min(1.0, value)) for value in command]

    def handle_keyboard(self) -> bool:
        for key, event in p.getKeyboardEvents().items():
            if key in (27, ord("q")):
                return False
            if key == ord("r") and (event & p.KEY_WAS_TRIGGERED):
                self.reset()
                continue
            if key == ord("m") and (event & p.KEY_WAS_TRIGGERED):
                self.pressed.clear()
                self.mode = "pid" if self.mode == "torque" else "torque"
                if self.mode == "pid":
                    self.target = [self.state.angle[i] for i in range(4)]
                    self.pid.reset()
                continue
            if key in KEY_TORQUE:
                if event & p.KEY_WAS_RELEASED:
                    self.pressed.discard(key)
                elif event & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
                    self.pressed.add(key)
        return True

    def update_targets(self, dt: float) -> None:
        for key in self.pressed:
            joint, sign = KEY_TORQUE[key]
            value = self.target[joint] + sign * self.target_rate * dt
            self.target[joint] = max(
                -params.PID_TARGET_LIMIT,
                min(params.PID_TARGET_LIMIT, value),
            )

    def pid_torque(self, dt: float) -> list[float]:
        angles = [self.state.angle[i] for i in range(4)]
        velocities = [self.state.velocity[i] for i in range(4)]
        feedback = self.pid.torque(angles, velocities, self.target, dt)
        gravity = self.context.bias_torque(
            self.model, self.terrain, self.base, self.state
        )
        return [
            max(-params.PID_TORQUE_LIMIT, min(params.PID_TORQUE_LIMIT, feedback[i] + gravity[i]))
            for i in range(4)
        ]

    def advance(self) -> None:
        if self.mode == "torque":
            torque = self.torque_command()
            torques = [torque for _ in range(STEPS_PER_FRAME)]
            self.state = self.context.forward_simulate(
                self.model,
                self.terrain,
                self.base,
                self.state,
                torques,
                step=STEP,
                refinement=REFINEMENT,
            )
        else:
            for _ in range(STEPS_PER_FRAME):
                self.update_targets(STEP)
                torque = self.pid_torque(STEP)
                self.state = self.context.forward_step(
                    self.model,
                    self.terrain,
                    self.base,
                    self.state,
                    torque,
                    step=STEP,
                    refinement=REFINEMENT,
                )
        self.update_visuals()

    def update_visuals(self) -> None:
        angles = [self.state.angle[i] for i in range(4)]
        velocities = [self.state.velocity[i] for i in range(4)]
        poses = link_poses(angles)
        for body, (position, rotation) in zip(self.bodies, poses):
            p.resetBasePositionAndOrientation(
                body,
                position.tolist(),
                quaternion_from_matrix(rotation),
            )
        axis_positions = []
        parent_rotations = [np.eye(3)]
        parent_rotations.append(poses[0][1])
        parent_rotations.append(poses[1][1])
        parent_rotations.append(poses[2][1])
        for joint in range(4):
            origin = poses[joint][0]
            axis = parent_rotations[joint] @ np.array(params.JOINT_AXES[joint])
            end = origin + 0.10 * axis
            self.axis_ids[joint] = p.addUserDebugLine(
                origin.tolist(),
                end.tolist(),
                lineColorRGB=[0.1, 0.9, 0.9],
                lineWidth=2.0,
                replaceItemUniqueId=self.axis_ids[joint],
            )
        if self.mode == "pid":
            text = "mode: PID target"
            for i in range(4):
                text += f"  T{i + 1}={self.target[i]:+.2f}"
                text += f" Q{i + 1}={angles[i]:+.2f}"
        else:
            text = "mode: direct torque"
        self.mode_text_id = p.addUserDebugText(
            text,
            [0.0, 0.0, 0.62],
            textColorRGB=[0.95, 0.95, 0.95],
            textSize=1.0,
            replaceItemUniqueId=self.mode_text_id,
        )

    def step(self) -> None:
        if not self.direct:
            if not self.handle_keyboard():
                raise KeyboardInterrupt
        self.advance()
        p.stepSimulation()

    def close(self) -> None:
        try:
            self.context.close()
        finally:
            p.disconnect(self.client)


def main() -> int:
    parser = argparse.ArgumentParser(description="四自由度串链动力学 PyBullet 可视化")
    parser.add_argument("--direct", action="store_true", help="无界面模式，用于自检")
    parser.add_argument("--steps", type=int, default=0, help="无界面模式下推进的帧数")
    parser.add_argument("--torque", type=float, default=0.0,
                        help="无界面模式下恒定的力矩（力矩模式）或目标移动方向（PID 模式）")
    parser.add_argument("--mode", choices=("torque", "pid"), default="torque",
                        help="初始模式：直接力矩或 PID 目标位置")
    parser.add_argument("--target-rate", type=float, default=None,
                        help="PID 模式下按住按键时目标位置速度，rad/s")
    arguments = parser.parse_args()

    viewer = Viewer(direct=arguments.direct, mode=arguments.mode,
                    target_rate=arguments.target_rate)
    try:
        if arguments.direct:
            steps = arguments.steps if arguments.steps > 0 else 60
            if arguments.torque != 0.0:
                viewer.pressed = {ord("1") if arguments.torque > 0 else ord("2")}
            for _ in range(steps):
                viewer.step()
            angles, velocities = gd.state_to_lists(viewer.state)
            print("mode:", viewer.mode)
            print("angles:", [round(value, 6) for value in angles])
            print("velocities:", [round(value, 6) for value in velocities])
            print("targets:", [round(value, 6) for value in viewer.target])
            return 0
        print("键盘：1/2 关节1，3/4 关节2，5/6 关节3，7/8 关节4；"
              "力矩模式输出 ±1 N·m，PID 模式移动目标位置；m 切换模式；r 复位；q/ESC 退出")
        next_frame = time.perf_counter()
        while True:
            viewer.step()
            next_frame += FRAME_DT
            delay = next_frame - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_frame = time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
