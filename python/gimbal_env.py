"""四自由度云台仿真环境：供采集脚本使用的 reset / step 接口。

采集脚本只负责目标序列、控制器与记录，动力学仿真由本环境承担：

    env = GimbalEnv()
    observation = env.reset({"angle": ..., "velocity": ..., "terrain": ...,
                             "base": ...})
    observation = env.step({"torque": command})

reset 接收一个配置字典，返回实际采用的观测字典——仿真模型可能无法完全满足请求
（例如关节角超出限位被夹取），真实环境同样会返回实测值，因此保存数据时一律使用
返回值。step 输入控制字典，返回一步仿真后的观测，其中包含每一步都可能变化的量
（关节角、关节角速度、实际施加力矩、坡度、坡向、底盘状态、时间）。

真实环境接入时实现同样的接口即可：reset 返回实测初值，step 返回实测状态。
"""

from __future__ import annotations

import abc
import math

import numpy as np

import gimbal_dynamics as gd

DT = 0.01
REFINEMENT = 2
TORQUE_LIMIT = 10.0
JOINT_LIMITS = ((None, None),
                (-45.0 * math.pi / 180.0, 5.0 * math.pi / 180.0),
                (-50.0 * math.pi / 180.0, 50.0 * math.pi / 180.0),
                (-45.0 * math.pi / 180.0, 45.0 * math.pi / 180.0))


def make_terrain_from(config) -> gd.TcrgTerrain:
    """从字典、二元组或地形对象得到地形。"""
    if isinstance(config, gd.TcrgTerrain):
        return config
    if isinstance(config, dict):
        return gd.make_terrain(float(config["slope_angle"]),
                               float(config["slope_direction"]))
    slope, direction = config
    return gd.make_terrain(float(slope), float(direction))


def make_base_from(config) -> gd.TcrgBaseState:
    """从字典或底盘状态对象得到底盘状态。"""
    if isinstance(config, gd.TcrgBaseState):
        return config
    return gd.make_base(acceleration=np.asarray(config.get("acceleration",
                                                           np.zeros(3)), dtype=float),
                        yaw=float(config.get("yaw", 0.0)),
                        yaw_rate=float(config.get("yaw_rate", 0.0)),
                        yaw_acceleration=float(config.get("yaw_acceleration", 0.0)))


def clip_angle(angle) -> np.ndarray:
    values = np.array(angle, dtype=float)
    for joint, limits in enumerate(JOINT_LIMITS):
        if limits[0] is None:
            continue
        values[joint] = min(max(values[joint], limits[0]), limits[1])
    return values


class Environment(abc.ABC):
    """采集脚本依赖的环境接口。"""

    @abc.abstractmethod
    def reset(self, config: dict) -> dict:
        """按配置设置初值，返回实际采用的观测。"""

    @abc.abstractmethod
    def step(self, control: dict) -> dict:
        """输入控制量，推进一个时间步，返回推进后的观测。"""

    @abc.abstractmethod
    def close(self) -> None:
        """释放资源。"""


class GimbalEnv(Environment):
    """刚体链动力学环境，状态由 4 个关节角与 4 个关节角速度组成。"""

    def __init__(self, model=None, dt: float = DT, refinement: int = REFINEMENT):
        self.context = gd.Context(1)
        self.model = gd.make_model() if model is None else model
        self.dt = float(dt)
        self.refinement = int(refinement)
        self.terrain = gd.make_terrain()
        self.base = gd.make_base()
        self._base_sequence = None
        self._state = gd.make_state()
        self._applied = np.zeros(4)
        self._torque_limit = TORQUE_LIMIT
        self.steps = 0
        self.time = 0.0

    def reset(self, config: dict) -> dict:
        angle = clip_angle(config.get("angle", np.zeros(4)))
        velocity = np.asarray(config.get("velocity", np.zeros(4)), dtype=float)
        # 未给出场景时沿用当前场景并如实回报，与真实环境的行为一致
        if "terrain" in config:
            self.terrain = make_terrain_from(config["terrain"])
        if "base" in config:
            self.base = make_base_from(config["base"])
        self._base_sequence = config.get("base_sequence")
        self._torque_limit = float(config.get("torque_limit", TORQUE_LIMIT))
        self._state = gd.make_state(angle, velocity)
        self._applied = np.zeros(4)
        self.steps = 0
        self.time = 0.0
        return self.observation()

    def step(self, control: dict) -> dict:
        if self._base_sequence is not None:
            self.base = make_base_from(self._base_sequence[min(
                self.steps, len(self._base_sequence) - 1)])
        command = np.asarray(control["torque"], dtype=float)
        self._applied = np.clip(command, -self._torque_limit, self._torque_limit)
        self._state = self.context.forward_step(self.model, self.terrain, self.base,
                                                self._state, self._applied, self.dt,
                                                self.refinement)
        self.steps += 1
        self.time += self.dt
        # 推进后回报新时刻的底盘状态，与真实环境回报实测值的语义一致
        if self._base_sequence is not None:
            self.base = make_base_from(self._base_sequence[min(
                self.steps, len(self._base_sequence) - 1)])
        return self.observation()

    def observation(self) -> dict:
        return {
            "angle": np.array([self._state.angle[i] for i in range(4)]),
            "velocity": np.array([self._state.velocity[i] for i in range(4)]),
            "torque": self._applied.copy(),
            "slope_angle": float(self.terrain.slope_angle),
            "slope_direction": float(self.terrain.slope_direction),
            "base_yaw": float(self.base.yaw),
            "base_yaw_rate": float(self.base.yaw_rate),
            "base_yaw_acceleration": float(self.base.yaw_acceleration),
            "base_acceleration": np.array([self.base.acceleration[i] for i in range(3)]),
            "time": self.time,
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
        self.context.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()
        return False
