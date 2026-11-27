"""实车环境：通过 robot_ipc 读写实车控制器，接口与 gimbal_env.Environment 相同。

底盘姿态、坡面参数与角速度来自头部 IMU 的反解，坐标定义与本工程动力学模型一致；
底盘参考点的位置、速度与线加速度暂时为 0，导航数据坐标定义不同，换算之后实现。
reset 返回实测值，实车不能设置位姿。

用法：

    env = RealGimbalEnv()
    observation = env.reset()
    observation = env.step({"torque": [0.0, 0.0, 0.0, 0.0]})
    env.close()
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gimbal_dynamics as gd
import gimbal_env
from robot_ipc import Robot, TcrgRobotObservation

TORQUE_LIMIT = 10.0
SAMPLE_TIMEOUT = 0.5


class RealGimbalEnv(gimbal_env.Environment):
    """实车四自由度云台环境。"""

    supports_base_acceleration = False

    def __init__(self, shm_name: str = "/s2c_serial_comm", dt: float = gimbal_env.DT,
                 torque_limit: float = TORQUE_LIMIT,
                 sample_timeout: float = SAMPLE_TIMEOUT):
        self.dt = float(dt)
        self.torque_limit = float(torque_limit)
        self.robot = Robot(shm_name, sample_timeout)
        self.context = gd.Context(1)
        self.model = gd.make_model()
        # 清空反解的连续量后立即读取一帧，服务端未就绪时在这里抛出 RuntimeError
        self.robot.reset()
        frame = self.robot.update()
        self._cache_frame(frame)
        self.steps = 0
        self._time_origin = float(frame.server_time)
        self._deadline = time.monotonic()

    def reset(self, config=None) -> dict:
        """写入零力矩接管控制权并返回实测初值；config 只为接口兼容而接受。"""
        self.robot.write_torque(np.zeros(4))
        self.robot.reset()
        frame = self.robot.update()
        self._cache_frame(frame)
        self.steps = 0
        self._time_origin = float(frame.server_time)
        self._deadline = time.monotonic()
        return self.observation()

    def step(self, control: dict) -> dict:
        command = np.asarray(control["torque"], dtype=float)
        if command.shape != (4,):
            raise ValueError("torque 长度必须为 4")
        if not np.all(np.isfinite(command)):
            raise ValueError("torque 包含非有限值")
        applied = np.clip(command, -self.torque_limit, self.torque_limit)
        self.robot.write_torque(applied)
        self._pace()
        frame = self.robot.update()
        self._cache_frame(frame)
        self.steps += 1
        return self.observation()

    def observation(self) -> dict:
        return {
            "angle": np.array([self._state.angle[i] for i in range(4)]),
            "velocity": np.array([self._state.velocity[i] for i in range(4)]),
            "torque": self._torque.copy(),
            "slope_angle": float(self.terrain.slope_angle),
            "slope_direction": float(self.terrain.slope_direction),
            "base_yaw": float(self.base.yaw),
            "base_yaw_rate": float(self.base.yaw_rate),
            "base_yaw_acceleration": float(self.base.yaw_acceleration),
            "base_acceleration": np.array([self.base.acceleration[i] for i in range(3)]),
            "time": self._server_time - self._time_origin,
            "steps": self.steps,
        }

    def feedforward_torque(self):
        """当前状态下的偏置力矩（重力、科氏、底盘运动），用于控制器前馈。"""
        return np.asarray(self.context.bias_torque(self.model, self.terrain, self.base,
                                                   self._state))

    def inverse_dynamics(self, acceleration):
        """当前状态下达到给定关节角加速度所需的关节力矩。"""
        return np.asarray(self.context.inverse_dynamics(self.model, self.terrain,
                                                        self.base, self._state,
                                                        acceleration))

    def close(self) -> None:
        """写入零力矩并释放 Robot 与 Context；写入零力矩失败时仍然释放资源。"""
        if self.robot is None:
            return
        robot, self.robot = self.robot, None
        try:
            robot.write_torque(np.zeros(4))
        finally:
            robot.close()
            self.context.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()
        return False

    def _cache_frame(self, frame: TcrgRobotObservation) -> None:
        """把一帧观测换算成 gd 对象缓存，供前馈与逆动力学使用。"""
        self._state = gd.make_state(
            [frame.angle[i] for i in range(4)],
            [frame.velocity[i] for i in range(4)])
        self.terrain = gd.make_terrain(frame.slope_angle, frame.slope_direction)
        self.base = gd.make_base(
            acceleration=[frame.base_acceleration[i] for i in range(3)],
            yaw=frame.base_yaw,
            yaw_rate=frame.base_yaw_rate,
            yaw_acceleration=frame.base_yaw_acceleration)
        self._torque = np.array([frame.torque[i] for i in range(4)])
        self._server_time = float(frame.server_time)

    def _pace(self) -> None:
        """按 dt 的绝对时刻定时；落后超过一个 dt 时以当前时刻重新锚定。"""
        deadline = self._deadline + self.dt
        now = time.monotonic()
        if now - deadline > self.dt:
            deadline = now
        elif deadline > now:
            time.sleep(deadline - now)
        self._deadline = deadline
