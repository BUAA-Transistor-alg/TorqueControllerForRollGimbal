"""用于强化学习的四自由度云台跟踪环境。

每个回合重置时采样一组动力学参数（可直接固定为 params.py 的真值），生成一条
参考轨迹，并让关节角从参考初值附近出发。观测包含关节角、角速度、参考轨迹、
跟踪误差、上一步力矩与由角速度差分得到的角加速度；这些量都对应真实系统里
可测的信号，控制器需要借助它们推断当前参数。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "identification"))

import gimbal_dynamics as gd
import identify_physical as ip
import param_basis as pb
import params

DT = 0.01
TORQUE_LIMIT = 4.0
BIG_YAW_LIMIT = (-math.pi, math.pi)
ROLL_LIMIT = (-45.0 * math.pi / 180.0, 5.0 * math.pi / 180.0)
SMALL_YAW_LIMIT = (-50.0 * math.pi / 180.0, 50.0 * math.pi / 180.0)
PITCH_LIMIT = (-45.0 * math.pi / 180.0, 45.0 * math.pi / 180.0)
SLOPE_LIMIT = 10.0 * math.pi / 180.0

REFERENCE_CENTERS = [0.0, -0.20, 0.0, 0.0]
REFERENCE_AMPLITUDES = [0.60, 0.20, 0.25, 0.25]
POSITION_SCALE = 0.5
VELOCITY_SCALE = 2.0
ERROR_SCALE = 0.20
POSITION_COST_SCALE = 0.02
VELOCITY_COST_SCALE = 0.5
ACCELERATION_SCALE = 20.0
OBSERVATION_DIM = 28


def truth_parameters() -> np.ndarray:
    return np.concatenate([params.MASSES, np.array(params.COM).reshape(-1),
                           np.array(params.INERTIA).reshape(-1),
                           np.array(params.JOINT_OFFSETS).reshape(-1), params.DAMPING])


def parameterize_truth() -> np.ndarray:
    """真值在可辨识坐标下的表示（解码任务的回归目标）。"""
    basis = pb.compute_basis()[0]
    return basis.T @ pb.inertial_from_raw(truth_parameters())


def validate_parameters(candidate: np.ndarray, context) -> bool:
    """在若干姿态上检查复合惯量矩阵正定，避免采样到仿真会拒绝的参数。"""
    model = ip.build_model(candidate)
    terrain = gd.make_terrain(0.1, 0.3)
    base = gd.make_base(yaw=0.2)
    for angle in ((0.0, 0.0, 0.0, 0.0), (0.5, -0.5, 0.4, 0.3), (-0.4, 0.3, -0.5, -0.3)):
        state = gd.make_state(np.array(angle), np.zeros(4))
        context.inverse_dynamics(model, terrain, base, state, np.zeros(4))
    return True


def sample_parameters(rng: np.random.Generator, scale: float, context=None) -> np.ndarray:
    """在物理合法范围内采样一组参数；scale 为 0 时返回真值。"""
    if scale <= 0.0:
        return truth_parameters()
    masses = np.array(params.MASSES) * np.exp(rng.uniform(-scale, scale, 4))
    com = np.array(params.COM) + rng.uniform(-0.02, 0.02, (4, 3)) * scale
    inertia = np.array(params.INERTIA) * np.exp(rng.uniform(-scale, scale, (4, 6)))
    offsets = np.array(params.JOINT_OFFSETS) + rng.normal(0.0, 0.02 * scale, (4, 3))
    damping = np.array(params.DAMPING) * np.exp(rng.uniform(-scale, scale, 4))
    for _attempt in range(40):
        candidate = np.concatenate([masses, com.reshape(-1), inertia.reshape(-1),
                                    offsets.reshape(-1), damping])
        try:
            ip.build_model(candidate)
            if context is None or validate_parameters(candidate, context):
                return candidate
            raise RuntimeError("姿态检查未通过")
        except RuntimeError:
            masses = np.array(params.MASSES) * np.exp(rng.uniform(-scale, scale, 4))
            com = np.array(params.COM) + rng.uniform(-0.02, 0.02, (4, 3)) * scale
            inertia = np.array(params.INERTIA) * np.exp(rng.uniform(-scale, scale, (4, 6)))
            offsets = np.array(params.JOINT_OFFSETS) + rng.normal(0.0, 0.02 * scale, (4, 3))
            damping = np.array(params.DAMPING) * np.exp(rng.uniform(-scale, scale, 4))
    raise RuntimeError("参数采样连续 40 次不可行")


def reference_trajectory(rng: np.random.Generator, steps: int):
    """参考轨迹：每个关节两个正弦分量，初相位随机。"""
    time = np.arange(steps) * DT
    angle = np.zeros((steps, 4))
    velocity = np.zeros((steps, 4))
    acceleration = np.zeros((steps, 4))
    for joint in range(4):
        signal = np.full(steps, REFERENCE_CENTERS[joint])
        derivative = np.zeros(steps)
        second = np.zeros(steps)
        for _ in range(2):
            frequency = rng.uniform(0.2, 1.2)
            amplitude = REFERENCE_AMPLITUDES[joint] * rng.uniform(0.4, 0.7)
            phase = rng.uniform(-math.pi, math.pi)
            signal += amplitude * np.sin(frequency * time + phase)
            derivative += amplitude * frequency * np.cos(frequency * time + phase)
            second -= amplitude * frequency ** 2 * np.sin(frequency * time + phase)
        angle[:, joint] = signal
        velocity[:, joint] = derivative
        acceleration[:, joint] = second
    angle[:, 1] = np.clip(angle[:, 1], ROLL_LIMIT[0] + 0.02, ROLL_LIMIT[1] - 0.02)
    angle[:, 2] = np.clip(angle[:, 2], SMALL_YAW_LIMIT[0] + 0.02, SMALL_YAW_LIMIT[1] - 0.02)
    angle[:, 3] = np.clip(angle[:, 3], PITCH_LIMIT[0] + 0.02, PITCH_LIMIT[1] - 0.02)
    return angle, velocity, acceleration


class TrackingEnv:
    """单环境：跟踪参考轨迹，回合长度固定，无提前终止。"""

    def __init__(self, seed: int, episode_steps: int = 1500, parameter_scale: float = 0.0,
                 torque_limit: float = TORQUE_LIMIT):
        self.rng = np.random.default_rng(seed)
        self.episode_steps = int(episode_steps)
        self.parameter_scale = float(parameter_scale)
        self.torque_limit = float(torque_limit)
        self.context = gd.Context(1)
        self.raw = None
        self.model = None
        self.terrain = None
        self.base = None
        self.state = None
        self.angle_ref = None
        self.velocity_ref = None
        self.step_index = 0
        self.previous_torque = np.zeros(4)
        self.previous_velocity = np.zeros(4)

    def reset(self):
        self.raw = sample_parameters(self.rng, self.parameter_scale, self.context)
        self.model = ip.build_model(self.raw)
        slope = self.rng.uniform(0.0, SLOPE_LIMIT)
        direction = self.rng.uniform(0.0, 2.0 * math.pi)
        self.terrain = gd.make_terrain(slope, direction)
        self.base = gd.make_base(yaw=self.rng.uniform(-math.pi, math.pi))
        self.angle_ref, self.velocity_ref, _ = reference_trajectory(
            self.rng, self.episode_steps)
        initial_angle = self.angle_ref[0] + self.rng.uniform(-0.05, 0.05, 4)
        self.state = gd.make_state(initial_angle, self.velocity_ref[0])
        self.step_index = 0
        self.previous_torque = np.zeros(4)
        self.previous_velocity = np.array([self.state.velocity[i] for i in range(4)])
        return self.observation()

    def observation(self) -> np.ndarray:
        index = min(self.step_index, self.episode_steps - 1)
        angle = np.array([self.state.angle[i] for i in range(4)])
        velocity = np.array([self.state.velocity[i] for i in range(4)])
        angle_ref = self.angle_ref[index]
        velocity_ref = self.velocity_ref[index]
        error = angle_ref - angle
        acceleration = (velocity - self.previous_velocity) / DT
        observation = np.concatenate([
            angle / POSITION_SCALE, velocity / VELOCITY_SCALE,
            angle_ref / POSITION_SCALE, velocity_ref / VELOCITY_SCALE,
            error / ERROR_SCALE, self.previous_torque / self.torque_limit,
            acceleration / ACCELERATION_SCALE,
        ])
        return np.clip(observation, -5.0, 5.0).astype(np.float32)

    def step(self, action):
        torque = np.clip(np.asarray(action, dtype=float), -1.0, 1.0) * self.torque_limit
        angle = np.array([self.state.angle[i] for i in range(4)])
        velocity = np.array([self.state.velocity[i] for i in range(4)])
        angle_ref = self.angle_ref[self.step_index]
        velocity_ref = self.velocity_ref[self.step_index]
        error = angle_ref - angle
        position_ratio = (error / POSITION_COST_SCALE) ** 2
        velocity_ratio = ((velocity - velocity_ref) / VELOCITY_COST_SCALE) ** 2
        position_cost = (position_ratio / (1.0 + position_ratio / 25.0)).mean()
        velocity_cost = (velocity_ratio / (1.0 + velocity_ratio / 25.0)).mean()
        effort_cost = ((torque / 2.0) ** 2).mean()
        smooth_cost = (((torque - self.previous_torque) / 2.0) ** 2).mean()
        cost = position_cost + 0.05 * velocity_cost + 0.02 * effort_cost + 0.02 * smooth_cost
        self.previous_torque = torque.copy()
        self.previous_velocity = velocity
        self.state = self.context.forward_step(self.model, self.terrain, self.base,
                                               self.state, torque, DT, 2)
        self.step_index += 1
        self._enforce_limits()
        done = self.step_index >= self.episode_steps - 1
        reward = float(-cost)
        return self.observation(), reward, done, self.info()

    def _enforce_limits(self):
        """关节限位按刚性挡块处理：超出限位时把角度夹回并把该关节速度清零。"""
        angle = np.array([self.state.angle[i] for i in range(4)])
        velocity = np.array([self.state.velocity[i] for i in range(4)])
        clamped = angle.copy()
        for joint, limits in ((0, BIG_YAW_LIMIT), (1, ROLL_LIMIT), (2, SMALL_YAW_LIMIT),
                              (3, PITCH_LIMIT)):
            if clamped[joint] < limits[0]:
                clamped[joint] = limits[0]
                velocity[joint] = 0.0
            elif clamped[joint] > limits[1]:
                clamped[joint] = limits[1]
                velocity[joint] = 0.0
        if not np.array_equal(clamped, angle):
            self.state = gd.make_state(clamped, velocity)

    def info(self) -> dict:
        angle = np.array([self.state.angle[i] for i in range(4)])
        error = self.angle_ref[min(self.step_index, self.episode_steps - 1)] - angle
        return {"raw": self.raw.copy(), "error": error.astype(np.float32)}

    def close(self):
        self.context.close()
