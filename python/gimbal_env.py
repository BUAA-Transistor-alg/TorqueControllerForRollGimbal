"""四自由度云台仿真环境：供采集脚本使用的 step / reset 接口。

采集脚本只负责目标序列、PID 与记录，动力学仿真由本环境承担：

    env = GimbalEnv()                       # 默认使用真实模型
    initial = env.reset(angle, velocity, terrain=terrain, base=base)
    command = ...                           # 控制器
    state = env.step(command)               # 推进一步，返回新状态

reset 返回初值（角度、速度），保存数据时直接使用该返回值；step 输入控制力矩，
返回一步仿真后的状态。底盘状态可以每步传入，用于带底盘晃动的采集。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import gimbal_dynamics as gd

DT = 0.01
REFINEMENT = 2


@dataclass
class Observation:
    angle: np.ndarray
    velocity: np.ndarray


class GimbalEnv:
    """刚体链动力学环境，状态由 4 个关节角与 4 个关节角速度组成。"""

    def __init__(self, model=None, dt: float = DT, refinement: int = REFINEMENT):
        self.context = gd.Context(1)
        self.model = gd.make_model() if model is None else model
        self.dt = float(dt)
        self.refinement = int(refinement)
        self.terrain = gd.make_terrain()
        self.base = gd.make_base()
        self._state = gd.make_state()
        self.steps = 0

    def reset(self, angle, velocity, terrain=None, base=None) -> Observation:
        """设置初值状态与场景，返回初值。"""
        if terrain is not None:
            self.terrain = terrain
        if base is not None:
            self.base = base
        self._state = gd.make_state(angle, velocity)
        self.steps = 0
        return self.observation()

    def step(self, torque, base=None) -> Observation:
        """输入控制力矩，推进一个时间步，返回推进后的状态。"""
        if base is not None:
            self.base = base
        self._state = self.context.forward_step(self.model, self.terrain, self.base,
                                               self._state, torque, self.dt,
                                               self.refinement)
        self.steps += 1
        return self.observation()

    def observation(self) -> Observation:
        return Observation(angle=np.array([self._state.angle[i] for i in range(4)]),
                           velocity=np.array([self._state.velocity[i] for i in range(4)]))

    def bias_torque(self):
        """当前状态下的偏置力矩（重力、科氏、底盘运动），用于控制器前馈。"""
        return np.asarray(self.context.bias_torque(self.model, self.terrain, self.base,
                                                   self._state))

    def inverse_dynamics(self, acceleration):
        """当前状态下达到给定关节角加速度所需的关节力矩。"""
        return np.asarray(self.context.inverse_dynamics(self.model, self.terrain,
                                                        self.base, self._state,
                                                        acceleration))

    def set_scene(self, terrain=None, base=None) -> None:
        if terrain is not None:
            self.terrain = terrain
        if base is not None:
            self.base = base

    def close(self) -> None:
        self.context.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()
        return False