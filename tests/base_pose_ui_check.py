"""底盘不水平时的界面位姿校验。

运行：python3 tests/base_pose_ui_check.py
"""

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python" / "visualization"))

import pybullet as p

from gimbal_env import GimbalEnv
from ui import (Viewer, base_rotation, link_poses, quaternion_from_matrix,
                terrain_rotation)


def quaternion_matrix(quaternion) -> np.ndarray:
    x, y, z, w = quaternion
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])


def check_pose(body, position, rotation, name: str) -> None:
    actual_position, actual_quaternion = p.getBasePositionAndOrientation(body)
    if not np.allclose(actual_position, position, atol=1e-9):
        raise RuntimeError(f"{name} 位置不正确：{actual_position} 与 {position}")
    actual_rotation = quaternion_matrix(actual_quaternion)
    if not np.allclose(actual_rotation, rotation, atol=1e-9):
        raise RuntimeError(f"{name} 姿态不正确")


def check_simulation_config() -> None:
    slope = math.radians(12.0)
    direction = math.radians(-35.0)
    yaw = math.radians(25.0)
    acceleration = [0.1, -0.05, 0.2]
    viewer = Viewer(
        direct=True, mode="pid",
        terrain={"slope_angle": slope, "slope_direction": direction},
        base={"yaw": yaw, "acceleration": acceleration})
    try:
        observation = viewer.observation
        if abs(float(observation["slope_angle"]) - slope) > 1e-12:
            raise RuntimeError("仿真后端没有采用设置的坡度")
        if abs(float(observation["slope_direction"]) - direction) > 1e-12:
            raise RuntimeError("仿真后端没有采用设置的坡向")
        if abs(float(observation["base_yaw"]) - yaw) > 1e-12:
            raise RuntimeError("仿真后端没有采用设置的底盘 yaw")
        if not np.allclose(observation["base_acceleration"], acceleration, atol=1e-12):
            raise RuntimeError("仿真后端没有采用设置的底盘加速度")

        ground_rotation = terrain_rotation(slope, direction)
        normal = ground_rotation @ np.array([0.0, 0.0, 1.0])
        check_pose(viewer.ground, -0.005 * normal, ground_rotation, "地面")
        rotation = base_rotation(slope, direction, yaw)
        check_pose(viewer.base_body, np.zeros(3), rotation, "基座")
        poses = link_poses([0.0] * 4, rotation)
        for index, (position, pose_rotation) in enumerate(poses):
            check_pose(viewer.bodies[index], position, pose_rotation,
                       f"刚体{index + 1}")
        print("仿真坡度、坡向、底盘 yaw 与加速度设置校验通过")
    finally:
        viewer.close()


class RealtimeEnvironment:
    """以变化的坡度与 yaw 模拟实机 IMU 反解值，用于检查界面逐拍更新。"""

    supports_roll_lock = False
    supports_base_acceleration = False

    def __init__(self):
        self.env = GimbalEnv()
        self.steps = 0

    def reset(self, config=None):
        self.env.reset({"angle": [0.0] * 4, "velocity": [0.0] * 4})
        self.steps = 0
        return self.observation()

    def step(self, control):
        self.env.step(control)
        self.steps += 1
        return self.observation()

    def observation(self):
        observation = self.env.observation()
        observation["slope_angle"] = math.radians(3.0 * self.steps)
        observation["slope_direction"] = math.radians(40.0)
        observation["base_yaw"] = math.radians(10.0 + 5.0 * self.steps)
        return observation

    def close(self):
        self.env.close()


def check_realtime_pose() -> None:
    viewer = Viewer(direct=True, mode="torque", environment=RealtimeEnvironment())
    try:
        if viewer.base_acceleration_available:
            raise RuntimeError("实机后端不应把底盘加速度标记为可用")
        for _ in range(3):
            viewer.advance()
            slope = float(viewer.observation["slope_angle"])
            direction = float(viewer.observation["slope_direction"])
            yaw = float(viewer.observation["base_yaw"])
            rotation = base_rotation(slope, direction, yaw)
            check_pose(viewer.base_body, np.zeros(3), rotation, "实机基座")
            poses = link_poses([float(value) for value in viewer.observation["angle"]],
                               rotation)
            check_pose(viewer.bodies[0], poses[0][0], poses[0][1], "实机刚体1")
        print("实机反解坡度与底盘 yaw 的逐拍显示校验通过")
    finally:
        viewer.close()


def main() -> None:
    check_simulation_config()
    check_realtime_pose()
    print("底盘不水平显示校验通过")


if __name__ == "__main__":
    main()
