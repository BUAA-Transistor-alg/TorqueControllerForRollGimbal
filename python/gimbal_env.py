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

真实环境接入时实现同样的接口即可：reset 返回实测初值，step 返回实测状态；
实车环境的实现见 robot_env.RealGimbalEnv。
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

    supports_roll_lock = False

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

    supports_roll_lock = True

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
        self._locked = False
        self._locked_angle = None
        self._locked_state = None

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
        self._locked = False
        self._locked_angle = None
        self._locked_state = None
        return self.observation()

    def _update_base_sequence(self) -> None:
        if self._base_sequence is not None:
            self.base = make_base_from(self._base_sequence[min(
                self.steps, len(self._base_sequence) - 1)])

    def _locked_observation_state(self) -> gd.TcrgState:
        return gd.make_state(
            [float(self._locked_state.angle[0]), float(self._locked_angle),
             float(self._locked_state.angle[1]), float(self._locked_state.angle[2])],
            [float(self._locked_state.velocity[0]), 0.0,
             float(self._locked_state.velocity[1]), float(self._locked_state.velocity[2])])

    def _mass_matrix(self) -> np.ndarray:
        zero = [0.0, 0.0, 0.0, 0.0]
        base_torque = np.asarray(self.context.inverse_dynamics(
            self.model, self.terrain, self.base, self._state, zero), dtype=float)
        mass = np.zeros((4, 4))
        for joint in range(4):
            qdd = [0.0, 0.0, 0.0, 0.0]
            qdd[joint] = 1.0
            mass[:, joint] = (
                np.asarray(self.context.inverse_dynamics(
                    self.model, self.terrain, self.base, self._state, qdd), dtype=float)
                - base_torque)
        return mass

    def lock_roll(self) -> None:
        """把 roll 轴锁定在当前实际角度，并按完全非弹性碰撞重分配其余关节速度。"""
        if not self.supports_roll_lock:
            raise RuntimeError("当前环境不支持 roll 轴锁定")
        if self._locked:
            return
        angle = np.array([self._state.angle[i] for i in range(4)], dtype=float)
        velocity = np.array([self._state.velocity[i] for i in range(4)], dtype=float)
        mass = self._mass_matrix()
        e2 = np.array([0.0, 1.0, 0.0, 0.0])
        try:
            column = np.linalg.solve(mass, e2)
        except np.linalg.LinAlgError as error:
            raise RuntimeError("锁定 roll 轴时质量矩阵求解失败") from error
        if abs(float(column[1])) < 1.0e-12:
            raise RuntimeError("锁定 roll 轴时质量矩阵的 roll 对角元过小")
        # 冲量沿 roll 轴：M (qdot_plus - qdot) = e2 * impulse，
        # 约束 qdot_plus[1] = 0，由此解出 impulse 并重分配其余速度。
        velocity_plus = velocity - column * (velocity[1] / column[1])
        velocity_plus[1] = 0.0
        self._locked_angle = float(angle[1])
        self._locked_state = gd.make_roll_locked_state(
            [float(angle[0]), float(angle[2]), float(angle[3])],
            [float(velocity_plus[0]), float(velocity_plus[2]), float(velocity_plus[3])])
        self._locked = True
        self._state = self._locked_observation_state()

    def unlock_roll(self) -> None:
        if not self._locked:
            return
        self._state = self._locked_observation_state()
        self._locked = False
        self._locked_state = None
        self._locked_angle = None

    def step(self, control: dict) -> dict:
        self._update_base_sequence()
        command = np.asarray(control["torque"], dtype=float)
        self._applied = np.clip(command, -self._torque_limit, self._torque_limit)
        if self._locked:
            self._locked_state = self.context.roll_locked_forward_step(
                self.model, self._locked_angle, self.terrain, self.base,
                self._locked_state,
                [float(self._applied[0]), float(self._applied[2]), float(self._applied[3])],
                self.dt, self.refinement)
            self._state = self._locked_observation_state()
        else:
            self._state = self.context.forward_step(self.model, self.terrain, self.base,
                                                    self._state, self._applied, self.dt,
                                                    self.refinement)
        self.steps += 1
        self.time += self.dt
        # 推进后回报新时刻的底盘状态，与真实环境回报实测值的语义一致
        self._update_base_sequence()
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
