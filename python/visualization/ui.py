"""四自由度串链动力学可视化。

PyBullet 只用于显示与键盘交互，不参与动力学计算。动力学由仿真后端承担：后端实现
`gimbal_env.Environment` 的 reset / step / close 接口，默认是 100 Hz（dt = 0.01 s）的
GimbalEnv；换成真实世界控制时实现同一个接口并在创建 Viewer 时传入即可，控制器与界面不需要
改动。控制器只读取观测字典并输出四维力矩，四种模式共用同一个 100 Hz 循环。

键盘：
  1 / 2   关节 1  +1 / -1 N·m（力矩模式）或目标位置 +/−（目标位置模式）
  3 / 4   关节 2  同上
  5 / 6   关节 3  同上
  7 / 8   关节 4  同上
  m       在直接力矩、PID 目标位置、策略网络目标位置、MPC 目标位置四种模式间切换
  l       锁定/解锁 roll 轴（仅仿真后端；锁定在当前实际角度）
  r       回到零位
  q/ESC   退出

锁定 roll 轴后动力学切换到三自由度合并模型，roll 目标固定为锁定角且 roll 按键失效；
解锁后 roll 目标保持锁定角，按键恢复控制。实机后端没有这个功能。

策略模式使用训练好的循环残差 SAC 控制器（默认 data/rl_final.pt）：目标位置与 PID 模式同样由
按键以固定速度移动，策略模式下大 yaw 的目标位置不限幅，其余三轴的目标位置限制在硬限位内侧
0.07 rad（与控制器内部的目标裁剪一致）。控制器每拍把当前目标压入滚动目标序列，序列最旧的元素
作为当前参考目标，其余 20 个元素作为未来 0.2 s 的预瞄目标，已压入的目标点数值不再变化；按训练时
的 142 维观测布局输出归一化动作，再乘以标称电机力矩系数得到力矩。

MPC 模式用标称模型求解固定时域最优控制问题，代价包含角度跟踪、零目标的速度正则化、力矩幅值、
力矩变化率与关节软限位，求解器在 C++ 端用反向伴随梯度完成优化；目标位置同样由按键以固定速度移动，
未来目标由滚动目标序列给出，每拍只新增尾端的当前目标。

界面按观测中的坡度、坡向与底盘 yaw 计算底盘姿态 R0 = R_G R_z(ψ)，地面、基座与四个刚体都画在
该姿态下。仿真后端可在启动时设置坡度、坡向、底盘 yaw 与参考点加速度；实机后端每个控制拍用头部
IMU 反解的坡度、坡向与 yaw 更新显示，底盘加速度暂未解算，显示留空。

用法：
  python ui.py                        # GUI
  python ui.py --mode rl              # 启动即进入策略模式
  python ui.py --mode mpc             # 启动即进入 MPC 模式
  python ui.py --mode mpc --slope-angle 12 --slope-direction -35 --base-yaw 25 --base-acceleration 0.1 -0.05 0.2
  python ui.py --robot --mode mpc     # 实机后端，坡度、坡向、底盘 yaw 来自 IMU 反解
  python ui.py --direct --steps 100 --mode rl --torque 1   # 无界面自检
  python ui.py --direct --steps 100 --mode mpc --torque 1  # MPC 无界面自检

依赖：pybullet、torch（仅策略模式）、本仓库的 python 目录。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import math
import time

import numpy as np
import pybullet as p

import params
from backend import make_simulation_env
from controllers import MpcController, PidController, RlController, TorqueController

DEFAULT_POLICY = Path(__file__).resolve().parents[2] / "data" / "rl_final.pt"
STEP = 0.01  # 仿真步长（100 Hz），与仿真后端一致
FRAME_DT = STEP

KEY_JOINT = {
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


def terrain_rotation(slope_angle: float, slope_direction: float) -> np.ndarray:
    """地面平面系相对世界系的旋转 R_G = R_z(γ) R_y(λ) R_z(−γ)。"""
    return (rot_z(slope_direction) @ rot_y(slope_angle)
            @ rot_z(-slope_direction))


def base_rotation(slope_angle: float, slope_direction: float,
                  yaw: float) -> np.ndarray:
    """底盘姿态 R_0 = R_G R_z(ψ)，与动力学模型的坐标定义一致。"""
    return terrain_rotation(slope_angle, slope_direction) @ rot_z(yaw)


def link_poses(angle, rotation=None, position=None):
    """由关节角与底盘位姿计算四个刚体在世界系中的位置与姿态。"""
    q1, q2, q3, q4 = angle
    r0 = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
    p0 = np.zeros(3) if position is None else np.asarray(position, dtype=float)
    r1 = r0 @ rot_z(q1)
    r2 = r1 @ rot_y(q2)
    r3 = r2 @ rot_z(q3)
    r4 = r3 @ rot_x(q4)
    o1 = np.array(params.JOINT_OFFSETS[0])
    o2 = np.array(params.JOINT_OFFSETS[1])
    o3 = np.array(params.JOINT_OFFSETS[2])
    o4 = np.array(params.JOINT_OFFSETS[3])
    p1 = p0 + r0 @ o1
    p2 = p1 + r1 @ o2
    p3 = p2 + r2 @ o3
    p4 = p3 + r3 @ o4
    return [(p1, r1), (p2, r2), (p3, r3), (p4, r4)]


def create_visual_body(shape_id, position, color, mass: float = 0.0) -> int:
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
                 target_rate: float | None = None,
                 policy_path: str | None = None, environment=None,
                 terrain: dict | None = None, base: dict | None = None):
        if mode not in ("torque", "pid", "rl", "mpc"):
            raise ValueError("未知模式 %s" % mode)
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
        self.ground = p.createMultiBody(0.0, baseVisualShapeIndex=ground,
                                        basePosition=[0.0, 0.0, -0.005])

        self.bodies = [
            create_visual_body([0.08, 0.08, 0.055], [0, 0, 0], [0.20, 0.45, 0.85, 1.0]),
            create_visual_body([0.05, 0.10, 0.04], [0, 0.09, 0.15], [0.20, 0.75, 0.35, 1.0]),
            create_visual_body([0.05, 0.05, 0.07], [0, 0.09, 0.20], [0.90, 0.55, 0.15, 1.0]),
            create_visual_body([0.05, 0.12, 0.04], [0, 0.09, 0.36], [0.85, 0.25, 0.25, 1.0]),
        ]
        self.base_body = create_visual_body([0.10, 0.10, 0.04], [0, 0, 0],
                                            [0.35, 0.35, 0.35, 1.0])
        self.axis_ids = [-1, -1, -1, -1]

        self.env = make_simulation_env() if environment is None else environment
        self.lock_available = bool(getattr(self.env, "supports_roll_lock", False))
        self.base_acceleration_available = bool(
            getattr(self.env, "supports_base_acceleration", True))
        self.terrain_config = (dict(terrain) if terrain is not None
                               else {"slope_angle": 0.0, "slope_direction": 0.0})
        self.base_config = (dict(base) if base is not None
                            else {"acceleration": [0.0, 0.0, 0.0], "yaw": 0.0})
        self.roll_locked = False
        self.locked_roll_angle = None
        self.pressed: set[int] = set()
        self.mode = mode
        self.target_rate = params.PID_TARGET_RATE if target_rate is None else target_rate
        self.policy_path = Path(policy_path) if policy_path else DEFAULT_POLICY
        self.controllers = {}
        self.observation = None
        self.mode_text_id = -1
        self.base_text_id = -1
        self.reset()

    def controller(self):
        if self.mode not in self.controllers:
            if self.mode == "torque":
                self.controllers["torque"] = TorqueController()
            elif self.mode == "pid":
                self.controllers["pid"] = PidController(self.target_rate)
            elif self.mode == "rl":
                self.controllers["rl"] = RlController(self.target_rate, self.policy_path)
            else:
                self.controllers["mpc"] = MpcController(self.target_rate)
        return self.controllers[self.mode]

    def reset(self) -> None:
        self.roll_locked = False
        self.locked_roll_angle = None
        self.pressed.clear()
        config = {
            "angle": [0.0] * 4,
            "velocity": [0.0] * 4,
            "terrain": dict(self.terrain_config),
            "base": dict(self.base_config),
        }
        self.observation = self.env.reset(config)
        self.controller()
        for controller in self.controllers.values():
            controller.reset(self.observation)
        self.update_visuals()

    def inputs(self) -> list[tuple[int, float]]:
        pairs = [KEY_JOINT[key] for key in self.pressed]
        if self.roll_locked:
            pairs = [pair for pair in pairs if pair[0] != 1]
        return pairs

    def _enforce_roll_target(self) -> None:
        if not self.roll_locked or self.locked_roll_angle is None:
            return
        controller = self.controllers.get(self.mode)
        target = getattr(controller, "target", None)
        if target is not None:
            target[1] = float(self.locked_roll_angle)

    def toggle_roll_lock(self) -> None:
        if not self.lock_available:
            return
        if self.roll_locked:
            self.env.unlock_roll()
            self.roll_locked = False
            self.observation = self.env.observation()
            self._enforce_roll_target()
            self.update_visuals()
            return
        self.env.lock_roll()
        self.roll_locked = True
        self.observation = self.env.observation()
        self.locked_roll_angle = float(self.observation["angle"][1])
        self.pressed.discard(ord("3"))
        self.pressed.discard(ord("4"))
        self.controller()
        self._enforce_roll_target()
        self.update_visuals()

    def handle_keyboard(self) -> bool:
        for key, event in p.getKeyboardEvents().items():
            if key in (27, ord("q")):
                return False
            if key == ord("r") and (event & p.KEY_WAS_TRIGGERED):
                self.reset()
                continue
            if key == ord("m") and (event & p.KEY_WAS_TRIGGERED):
                self.pressed.clear()
                modes = ("torque", "pid", "rl", "mpc")
                self.mode = modes[(modes.index(self.mode) + 1) % len(modes)]
                self.controller().reset(self.observation)
                self._enforce_roll_target()
                continue
            if key == ord("l") and (event & p.KEY_WAS_TRIGGERED):
                self.toggle_roll_lock()
                continue
            if key in KEY_JOINT:
                if event & p.KEY_WAS_RELEASED:
                    self.pressed.discard(key)
                elif event & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
                    self.pressed.add(key)
        return True

    def advance(self) -> None:
        controller = self.controller()
        self._enforce_roll_target()
        torque = controller.command(self.observation, self.inputs(), STEP)
        self.observation = self.env.step({"torque": torque})
        self.update_visuals()

    def update_visuals(self) -> None:
        angles = [float(value) for value in self.observation["angle"]]
        slope = float(self.observation.get("slope_angle", 0.0))
        direction = float(self.observation.get("slope_direction", 0.0))
        yaw = float(self.observation.get("base_yaw", 0.0))
        ground_rotation = terrain_rotation(slope, direction)
        ground_normal = ground_rotation @ np.array([0.0, 0.0, 1.0])
        p.resetBasePositionAndOrientation(
            self.ground,
            (-0.005 * ground_normal).tolist(),
            quaternion_from_matrix(ground_rotation),
        )
        rotation = base_rotation(slope, direction, yaw)
        p.resetBasePositionAndOrientation(
            self.base_body,
            [0.0, 0.0, 0.0],
            quaternion_from_matrix(rotation),
        )
        poses = link_poses(angles, rotation)
        for body, (position, pose_rotation) in zip(self.bodies, poses):
            p.resetBasePositionAndOrientation(
                body,
                position.tolist(),
                quaternion_from_matrix(pose_rotation),
            )
        parent_rotations = [rotation]
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
        controller = self.controllers.get(self.mode)
        label = {"torque": "direct torque", "pid": "PID target", "rl": "RL target",
                 "mpc": "MPC target"}[self.mode]
        text = "mode: " + label
        if self.lock_available:
            if self.roll_locked:
                text += f"  roll LOCKED={self.locked_roll_angle:+.2f}"
            else:
                text += "  roll free (l)"
        else:
            text += "  roll lock unavailable"
        for joint in range(4):
            if getattr(controller, "target", None) is not None:
                text += f"  T{joint + 1}={controller.target[joint]:+.2f}"
            text += f"  Q{joint + 1}={angles[joint]:+.2f}"
        load = getattr(controller, "load_estimate", None)
        if load is not None:
            text += "  L=" + "/".join(f"{value:+.2f}" for value in load)
        self.mode_text_id = p.addUserDebugText(
            text,
            [0.0, 0.0, 0.62],
            textColorRGB=[0.95, 0.95, 0.95],
            textSize=1.0,
            replaceItemUniqueId=self.mode_text_id,
        )
        base_text = ("slope=%+.1f deg  dir=%+.1f deg  yaw=%+.1f deg"
                     % (math.degrees(slope), math.degrees(direction),
                        math.degrees(yaw)))
        acceleration = self.observation.get("base_acceleration")
        if self.base_acceleration_available and acceleration is not None:
            base_text += "  acc=(%+.2f, %+.2f, %+.2f) m/s2" % tuple(
                float(value) for value in acceleration)
        else:
            base_text += "  acc=--"
        self.base_text_id = p.addUserDebugText(
            base_text,
            [0.0, 0.0, 0.555],
            textColorRGB=[0.85, 0.85, 0.85],
            textSize=0.9,
            replaceItemUniqueId=self.base_text_id,
        )

    def step(self) -> None:
        if not self.direct:
            if not self.handle_keyboard():
                raise KeyboardInterrupt
        self.advance()
        p.stepSimulation()

    def close(self) -> None:
        for controller in self.controllers.values():
            controller.close()
        self.env.close()
        p.disconnect(self.client)


def main() -> int:
    parser = argparse.ArgumentParser(description="四自由度串链动力学 PyBullet 可视化")
    parser.add_argument("--direct", action="store_true", help="无界面模式，用于自检")
    parser.add_argument("--steps", type=int, default=0, help="无界面模式下推进的帧数")
    parser.add_argument("--torque", type=float, default=0.0,
                        help="无界面模式下按住的方向键（力矩模式给力矩，目标模式移动目标）")
    parser.add_argument("--mode", choices=("torque", "pid", "rl", "mpc"), default="torque",
                        help="初始模式：直接力矩、PID 目标位置、策略网络目标位置或 MPC 目标位置")
    parser.add_argument("--target-rate", type=float, default=None,
                        help="目标位置模式下按住按键时目标位置速度，rad/s")
    parser.add_argument("--policy", default=None,
                        help="策略模式使用的检查点路径，默认 data/rl_final.pt")
    parser.add_argument("--lock-roll", action="store_true",
                        help="无界面模式下在中途锁定 roll 轴，末尾解锁，用于自检")
    parser.add_argument("--slope-angle", type=float, default=0.0,
                        help="仿真模式下底盘所在坡面的坡度，度")
    parser.add_argument("--slope-direction", type=float, default=0.0,
                        help="仿真模式下坡向，度")
    parser.add_argument("--base-yaw", type=float, default=0.0,
                        help="仿真模式下底盘相对坡面坐标系的 yaw，度")
    parser.add_argument("--base-acceleration", type=float, nargs=3,
                        default=(0.0, 0.0, 0.0), metavar=("AX", "AY", "AZ"),
                        help="仿真模式下底盘参考点线加速度，m/s²")
    parser.add_argument("--robot", action="store_true",
                        help="连接实车控制器，坡度、坡向与底盘 yaw 取头部 IMU 反解值")
    arguments = parser.parse_args()

    environment = None
    if arguments.robot:
        from robot_env import RealGimbalEnv
        environment = RealGimbalEnv()
    viewer = Viewer(direct=arguments.direct, mode=arguments.mode,
                    target_rate=arguments.target_rate, policy_path=arguments.policy,
                    environment=environment,
                    terrain={"slope_angle": math.radians(arguments.slope_angle),
                             "slope_direction": math.radians(arguments.slope_direction)},
                    base={"yaw": math.radians(arguments.base_yaw),
                          "acceleration": list(arguments.base_acceleration)})
    try:
        if arguments.direct:
            steps = arguments.steps if arguments.steps > 0 else 60
            if arguments.torque != 0.0:
                viewer.pressed = {ord("1") if arguments.torque > 0 else ord("2")}
            lock_step = min(10, max(1, steps // 2))
            for index in range(steps):
                if arguments.lock_roll and index == lock_step:
                    viewer.toggle_roll_lock()
                viewer.step()
            if arguments.lock_roll:
                print("roll lock available:", viewer.lock_available)
                print("roll locked after run:", viewer.roll_locked)
                if viewer.roll_locked:
                    viewer.toggle_roll_lock()
                print("roll locked after unlock:", viewer.roll_locked)
            angles = viewer.observation["angle"]
            velocities = viewer.observation["velocity"]
            print("mode:", viewer.mode)
            print("angles:", [round(float(value), 6) for value in angles])
            print("velocities:", [round(float(value), 6) for value in velocities])
            print("slope_angle:", round(float(viewer.observation["slope_angle"]), 6))
            print("slope_direction:",
                  round(float(viewer.observation["slope_direction"]), 6))
            print("base_yaw:", round(float(viewer.observation["base_yaw"]), 6))
            if viewer.base_acceleration_available:
                print("base_acceleration:", [
                    round(float(value), 6)
                    for value in viewer.observation["base_acceleration"]])
            else:
                print("base_acceleration: n/a")
            target = getattr(viewer.controllers[viewer.mode], "target", None)
            if target is not None:
                print("target:", [round(float(value), 6) for value in target])
            return 0
        print("键盘：1/2 关节1，3/4 关节2，5/6 关节3，7/8 关节4；"
              "力矩模式输出 ±1 N·m，目标位置模式移动目标位置；"
              "m 循环切换 力矩/PID/策略/MPC 模式；l 锁定/解锁 roll 轴（仅仿真）；"
              "r 复位；q/ESC 退出")
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
