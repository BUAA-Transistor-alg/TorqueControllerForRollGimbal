"""用于强化学习的四自由度云台跟踪环境。

每个回合重置时采样一组动力学参数、一组观测偏差与噪声实现、以及一条带阶跃的参考轨迹。
参考轨迹可能把目标推到关节限位之外，控制器需要在限位内尽量贴近目标且不碰撞限位。
观测包含带噪声与偏差的关节角、角速度、参考轨迹、跟踪误差、上一步力矩与滤波后的角加速度，
对应真实系统可测的信号；控制器需要借助历史观测推断当前参数与观测偏差。
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
TORQUE_LIMIT = 8.0
BIG_YAW_LIMIT = (-math.pi, math.pi)
LIMIT_MARGIN = 2.0 * math.pi / 180.0
ROLL_LIMIT = (-45.0 * math.pi / 180.0, 5.0 * math.pi / 180.0)
SMALL_YAW_LIMIT = (-50.0 * math.pi / 180.0, 50.0 * math.pi / 180.0)
PITCH_LIMIT = (-45.0 * math.pi / 180.0, 45.0 * math.pi / 180.0)
SLOPE_LIMIT = 10.0 * math.pi / 180.0

REFERENCE_CENTERS = [0.0, -0.20, 0.0, 0.0]
REFERENCE_AMPLITUDES = [0.60, 0.20, 0.25, 0.25]

ANGLE_NOISE = 0.003
VELOCITY_NOISE = 0.05
ANGLE_BIAS = 0.010
VELOCITY_BIAS = 0.020
VELOCITY_SCALE_ERROR = 0.03
ACCELERATION_FILTER = 0.2
LIMIT_WEIGHT = 100.0
HIT_PENALTY = 50.0

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
    basis = pb.compute_basis()[0]
    return basis.T @ pb.inertial_from_raw(truth_parameters())


def validate_parameters(candidate: np.ndarray, context) -> bool:
    """在若干姿态上检查复合惯量矩阵正定，避免采样到仿真会拒绝的参数。"""
    model = ip.build_model(candidate)
    terrain = gd.make_terrain(0.1, 0.3)
    base = gd.make_base(yaw=0.2)
    for angle in ((0.0, 0.0, 0.0, 0.0), (0.5, -0.5, 0.4, 0.3), (-0.4, 0.3, -0.5, -0.3),
                  (0.8, -0.7, 0.7, 0.6), (-0.9, 0.6, -0.8, -0.6)):
        state = gd.make_state(np.array(angle), np.zeros(4))
        context.inverse_dynamics(model, terrain, base, state, np.zeros(4))
    return True


def sample_parameters(rng: np.random.Generator, scale: float, context=None) -> np.ndarray:
    """在物理合法范围内采样一组参数；scale 为 0 时返回真值。"""
    if scale <= 0.0:
        return truth_parameters()
    for _attempt in range(60):
        masses = np.array(params.MASSES) * np.exp(rng.uniform(-scale, scale, 4))
        com = np.array(params.COM) + rng.uniform(-0.03, 0.03, (4, 3)) * min(scale, 1.0)
        inertia = np.array(params.INERTIA) * np.exp(rng.uniform(-scale, scale, (4, 6)))
        offsets = np.array(params.JOINT_OFFSETS) + rng.normal(
            0.0, 0.03 * min(scale, 1.0), (4, 3))
        damping = np.array(params.DAMPING) * np.exp(rng.uniform(-scale, scale, 4))
        candidate = np.concatenate([masses, com.reshape(-1), inertia.reshape(-1),
                                    offsets.reshape(-1), damping])
        try:
            ip.build_model(candidate)
            if context is None or validate_parameters(candidate, context):
                return candidate
        except RuntimeError:
            continue
    raise RuntimeError("参数采样连续 60 次不可行")


def generate_reference(rng: np.random.Generator, steps: int):
    """多正弦加阶跃的参考轨迹；阶跃允许把目标推到关节限位之外。"""
    time = np.arange(steps) * DT
    angle = np.zeros((steps, 4))
    for joint in range(4):
        signal = np.full(steps, REFERENCE_CENTERS[joint])
        for _ in range(2):
            frequency = rng.uniform(0.2, 1.2)
            amplitude = REFERENCE_AMPLITUDES[joint] * rng.uniform(0.4, 0.7)
            phase = rng.uniform(-math.pi, math.pi)
            signal += amplitude * np.sin(frequency * time + phase)
        if rng.uniform() < 0.8:
            positions = []
            for _ in range(int(rng.integers(1, 3))):
                index = int(rng.integers(steps // 5, steps))
                if any(abs(index - other) < 200 for other in positions):
                    continue
                positions.append(index)
                amplitude = 0.50
                signal[index:] += rng.uniform(-amplitude, amplitude)
        angle[:, joint] = signal
    velocity = np.clip(np.gradient(angle, DT, axis=0), -3.0, 3.0)
    acceleration = np.gradient(velocity, DT, axis=0)
    return angle, velocity, acceleration


class TrackingEnv:
    """单环境：跟踪带阶跃的参考轨迹，回合长度固定，无提前终止。"""

    def __init__(self, seed: int, episode_steps: int = 1500, parameter_scale: float = 0.0,
                 torque_limit: float = TORQUE_LIMIT, noise_scale: float = 1.0,
                 bias_scale: float = 1.0):
        self.rng = np.random.default_rng(seed)
        self.episode_steps = int(episode_steps)
        self.parameter_scale = float(parameter_scale)
        self.torque_limit = float(torque_limit)
        self.noise_scale = float(noise_scale)
        self.bias_scale = float(bias_scale)
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
        self.filtered_velocity = np.zeros(4)
        self.previous_filtered_velocity = np.zeros(4)
        self.angle_bias = np.zeros(4)
        self.velocity_bias = np.zeros(4)
        self.velocity_scale = np.ones(4)
        self.hits = 0
        self.previous_measurement = np.zeros(4)

    def reset(self):
        self.raw = sample_parameters(self.rng, self.parameter_scale, self.context)
        self.model = ip.build_model(self.raw)
        slope = self.rng.uniform(0.0, SLOPE_LIMIT)
        direction = self.rng.uniform(0.0, 2.0 * math.pi)
        self.terrain = gd.make_terrain(slope, direction)
        self.base = gd.make_base(yaw=self.rng.uniform(-math.pi, math.pi))
        self.angle_ref, self.velocity_ref, _ = generate_reference(self.rng,
                                                                 self.episode_steps)
        self.angle_bias = self.rng.uniform(-1.0, 1.0, 4) * ANGLE_BIAS * self.bias_scale
        self.velocity_bias = self.rng.uniform(-1.0, 1.0, 4) * VELOCITY_BIAS * self.bias_scale
        self.velocity_scale = 1.0 + self.rng.uniform(-1.0, 1.0, 4) * VELOCITY_SCALE_ERROR * self.bias_scale
        initial_angle = self.angle_ref[0] + self.rng.uniform(-0.05, 0.05, 4)
        self.state = gd.make_state(initial_angle, self.velocity_ref[0])
        self.step_index = 0
        self.previous_torque = np.zeros(4)
        self.hits = 0
        self.filtered_velocity = np.zeros(4)
        self.previous_filtered_velocity = np.zeros(4)
        self.previous_measurement = np.zeros(4)
        return self.observation()

    def measurement(self):
        angle = np.array([self.state.angle[i] for i in range(4)])
        velocity = np.array([self.state.velocity[i] for i in range(4)])
        angle_measured = (angle + self.angle_bias
                          + self.rng.normal(0.0, ANGLE_NOISE * self.noise_scale, 4))
        velocity_measured = (self.velocity_scale * velocity + self.velocity_bias
                             + self.rng.normal(0.0, VELOCITY_NOISE * self.noise_scale, 4))
        return angle_measured, velocity_measured

    def observation(self) -> np.ndarray:
        index = min(self.step_index, self.episode_steps - 1)
        angle_measured, velocity_measured = self.measurement()
        self.filtered_velocity = ((1.0 - ACCELERATION_FILTER) * self.filtered_velocity
                                  + ACCELERATION_FILTER * velocity_measured)
        acceleration = ((self.filtered_velocity - self.previous_filtered_velocity) / DT)
        self.previous_filtered_velocity = self.filtered_velocity.copy()
        angle_ref = self.angle_ref[index]
        velocity_ref = self.velocity_ref[index]
        error = angle_ref - angle_measured
        observation = np.concatenate([
            angle_measured / POSITION_SCALE, velocity_measured / VELOCITY_SCALE,
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
        effort_cost = ((torque / (self.torque_limit / 2.0)) ** 2).mean()
        smooth_cost = (((torque - self.previous_torque) / (self.torque_limit / 2.0)) ** 2).mean()
        limit_cost = self._limit_cost(angle)
        self.previous_torque = torque.copy()
        self.state = self.context.forward_step(self.model, self.terrain, self.base,
                                               self.state, torque, DT, 2)
        self.step_index += 1
        hit = self._enforce_limits()
        self.hits += int(hit)
        cost = (position_cost + 0.05 * velocity_cost + 0.02 * effort_cost
                + 0.02 * smooth_cost + LIMIT_WEIGHT * limit_cost
                + HIT_PENALTY * float(hit))
        done = self.step_index >= self.episode_steps - 1
        return self.observation(), float(-cost), done, self.info()

    def _limit_cost(self, angle) -> float:
        total = 0.0
        for joint, limits in ((0, BIG_YAW_LIMIT), (1, ROLL_LIMIT), (2, SMALL_YAW_LIMIT),
                              (3, PITCH_LIMIT)):
            distance = min(angle[joint] - limits[0], limits[1] - angle[joint])
            penetration = max(0.0, LIMIT_MARGIN - distance) / LIMIT_MARGIN
            total += penetration ** 2
        return total / 4.0

    def _enforce_limits(self) -> bool:
        """关节限位按刚性挡块处理；返回这一步是否撞到限位。"""
        angle = np.array([self.state.angle[i] for i in range(4)])
        velocity = np.array([self.state.velocity[i] for i in range(4)])
        clamped = angle.copy()
        hit = False
        for joint, limits in ((0, BIG_YAW_LIMIT), (1, ROLL_LIMIT), (2, SMALL_YAW_LIMIT),
                              (3, PITCH_LIMIT)):
            if clamped[joint] < limits[0]:
                clamped[joint] = limits[0]
                velocity[joint] = 0.0
                hit = True
            elif clamped[joint] > limits[1]:
                clamped[joint] = limits[1]
                velocity[joint] = 0.0
                hit = True
        if hit:
            self.state = gd.make_state(clamped, velocity)
        return hit

    def info(self) -> dict:
        angle = np.array([self.state.angle[i] for i in range(4)])
        reference = self.angle_ref[min(self.step_index, self.episode_steps - 1)]
        return {"raw": self.raw.copy(), "error": (reference - angle).astype(np.float32),
                "hits": self.hits}

    def close(self):
        self.context.close()
