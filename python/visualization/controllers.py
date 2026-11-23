"""界面控制器：直接力矩、PID 目标位置、策略网络目标位置。

控制器只读取观测字典（关节角、角速度、坡度、坡向、底盘状态）、按键输入与时间步，输出四维力矩；
不依赖仿真后端的具体实现，接真实环境时同一套控制器可以直接使用。
"""

from __future__ import annotations

import abc
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rl"))

import gimbal_dynamics as gd
import params
from mpc.controller import LIMITS as MPC_LIMITS
from mpc.controller import LIMIT_MARGIN as MPC_LIMIT_MARGIN
from mpc.controller import TORQUE_LIMIT as MPC_TORQUE_LIMIT
from mpc.controller import MpcController as HorizonMpcController

TARGET_LIMIT = params.PID_TARGET_LIMIT


def state_from(observation: dict):
    return gd.make_state(np.asarray(observation["angle"], dtype=float),
                         np.asarray(observation["velocity"], dtype=float))


def scenario_from(observation: dict):
    terrain = gd.make_terrain(float(observation["slope_angle"]),
                              float(observation["slope_direction"]))
    base = gd.make_base(
        acceleration=np.asarray(observation["base_acceleration"], dtype=float),
        yaw=float(observation["base_yaw"]),
        yaw_rate=float(observation["base_yaw_rate"]),
        yaw_acceleration=float(observation["base_yaw_acceleration"]),
    )
    return terrain, base


class Controller(abc.ABC):
    """控制器接口。target 只在带目标位置的控制器里存在。"""

    target: list[float] | None = None

    def reset(self, observation: dict) -> None:
        """按初始观测复位内部状态。"""

    @abc.abstractmethod
    def command(self, observation: dict, inputs: list[tuple[int, float]],
                dt: float) -> list[float]:
        """输入观测、按键（关节号与方向）与时间步，输出四维力矩。"""

    def close(self) -> None:
        """释放控制器占用的资源。"""


class TorqueController(Controller):
    """直接力矩：按住按键给对应关节 ±1 N·m。"""

    def command(self, observation: dict, inputs: list[tuple[int, float]],
                dt: float) -> list[float]:
        torque = [0.0, 0.0, 0.0, 0.0]
        for joint, sign in inputs:
            torque[joint] += sign
        return [max(-1.0, min(1.0, value)) for value in torque]


class TargetController(Controller):
    """带目标位置的控制器：按住按键以固定速度移动目标，目标按 bounds 限幅。"""

    def __init__(self, target_rate: float):
        self.target_rate = float(target_rate)
        self.target = [0.0, 0.0, 0.0, 0.0]
        self.bounds = [(-TARGET_LIMIT, TARGET_LIMIT)] * 4

    def reset(self, observation: dict) -> None:
        self.target = self.clamp([float(value) for value in observation["angle"]])

    def clamp(self, target: list[float]) -> list[float]:
        return [max(self.bounds[joint][0], min(self.bounds[joint][1], value))
                for joint, value in enumerate(target)]

    def move_target(self, inputs: list[tuple[int, float]], dt: float) -> None:
        for joint, sign in inputs:
            self.target[joint] += sign * self.target_rate * dt
        self.target = self.clamp(self.target)


class PidController(TargetController):
    """PID 目标位置：比例-积分-微分反馈加重力前馈，目标限幅 ±π。"""

    def __init__(self, target_rate: float):
        super().__init__(target_rate)
        self.integral = [0.0, 0.0, 0.0, 0.0]
        self.context = gd.Context(1)
        self.model = gd.make_model()

    def reset(self, observation: dict) -> None:
        super().reset(observation)
        self.integral = [0.0, 0.0, 0.0, 0.0]

    def command(self, observation: dict, inputs: list[tuple[int, float]],
                dt: float) -> list[float]:
        self.move_target(inputs, dt)
        angle = np.asarray(observation["angle"], dtype=float)
        velocity = np.asarray(observation["velocity"], dtype=float)
        feedback = [0.0, 0.0, 0.0, 0.0]
        for joint in range(4):
            error = self.target[joint] - angle[joint]
            self.integral[joint] = max(
                -params.PID_INTEGRAL_LIMIT,
                min(params.PID_INTEGRAL_LIMIT, self.integral[joint] + error * dt),
            )
            value = (params.PID_KP[joint] * error
                     + params.PID_KI[joint] * self.integral[joint]
                     - params.PID_KD[joint] * velocity[joint])
            feedback[joint] = max(-params.PID_TORQUE_LIMIT,
                                  min(params.PID_TORQUE_LIMIT, value))
        terrain, base = scenario_from(observation)
        gravity = self.context.bias_torque(self.model, terrain, base,
                                           state_from(observation))
        return [max(-params.PID_TORQUE_LIMIT,
                    min(params.PID_TORQUE_LIMIT, feedback[joint] + gravity[joint]))
                for joint in range(4)]

    def close(self) -> None:
        self.context.close()


class RlController(TargetController):
    """策略网络目标位置：循环残差 SAC 控制器，目标范围与控制器内部的目标裁剪一致。"""

    def __init__(self, target_rate: float, policy_path):
        from deploy import PolicyController, target_limits

        super().__init__(target_rate)
        self.policy = PolicyController(policy_path)
        self.bounds = target_limits()

    def reset(self, observation: dict) -> None:
        super().reset(observation)
        self.policy.reset(self.target)

    def command(self, observation: dict, inputs: list[tuple[int, float]],
                dt: float) -> list[float]:
        self.move_target(inputs, dt)
        terrain, base = scenario_from(observation)
        return self.policy.torque(state_from(observation), self.target, terrain, base)

    @property
    def load_estimate(self):
        return self.policy.load_estimate

    def close(self) -> None:
        self.policy.close()


class MpcController(TargetController):
    """滚动时域 MPC：按住按键移动目标，用目标历史斜率外推未来预测区间的参考。

    每拍求解一个固定时域最优控制问题，代价由角度跟踪、零目标的速度正则化、力矩幅值、
    力矩变化率与关节软限位二次铰链组成，求解器在 C++ 端用反向伴随梯度完成优化。
    目标位置的产生方式与 PID 模式相同；大 yaw 的目标限幅 ±π，其余三轴的目标限制在
    硬限位内侧 0.07 rad，与代价里的软限位一致。
    """

    HISTORY_LENGTH = 21
    HORIZON = 15
    MAX_ITER = 8

    def __init__(self, target_rate: float, model=None,
                 horizon: int = HORIZON, max_iter: int = MAX_ITER):
        super().__init__(target_rate)
        self.bounds = [
            (float(MPC_LIMITS[joint][0]) + MPC_LIMIT_MARGIN,
             float(MPC_LIMITS[joint][1]) - MPC_LIMIT_MARGIN)
            if MPC_LIMITS[joint][0] is not None else (-TARGET_LIMIT, TARGET_LIMIT)
            for joint in range(4)
        ]
        self.model = gd.make_model() if model is None else model
        self.mpc = HorizonMpcController(self.model, horizon=horizon,
                                        torque_limit=MPC_TORQUE_LIMIT,
                                        max_iter=max_iter, thread_count=4)
        self.target_history = deque(
            [np.zeros(4) for _ in range(self.HISTORY_LENGTH)],
            maxlen=self.HISTORY_LENGTH)

    def reset(self, observation: dict) -> None:
        super().reset(observation)
        target = np.array(self.target, dtype=float)
        self.target_history = deque([target.copy() for _ in range(self.HISTORY_LENGTH)],
                                    maxlen=self.HISTORY_LENGTH)
        self.mpc.reset()

    def references(self, dt: float):
        """由目标历史的最小二乘斜率外推未来 HORIZON 步的目标位置。"""
        history = np.stack(self.target_history)
        times = np.arange(len(history)) * float(dt)
        centered = times - times.mean()
        denominator = max(float(np.sum(centered ** 2)), 1.0e-9)
        slope = ((centered[:, None] * (history - history.mean(axis=0))).sum(axis=0)
                 / denominator)
        steps = np.arange(1, self.mpc.horizon + 1) * float(dt)
        return np.stack([self.clamp(list(target))
                         for target in history[-1][None, :] + slope[None, :] * steps[:, None]])

    def command(self, observation: dict, inputs: list[tuple[int, float]],
                dt: float) -> list[float]:
        self.move_target(inputs, dt)
        self.target_history.append(np.array(self.target, dtype=float))
        angle = np.asarray(observation["angle"], dtype=float)
        velocity = np.asarray(observation["velocity"], dtype=float)
        target_angle = self.references(dt)
        terrain, base = scenario_from(observation)
        torque = self.mpc.command(angle, velocity, target_angle, terrain, base)
        return [float(value) for value in torque]

    @property
    def last_info(self):
        return self.mpc.last_info

    def close(self) -> None:
        self.mpc.close()
