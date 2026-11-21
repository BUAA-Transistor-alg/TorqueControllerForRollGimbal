"""用于强化学习的四自由度云台跟踪环境（含底盘运动、目标预瞄与全状态噪声）。

每个回合重置时采样一组动力学参数、一组电机力矩系数、一段底盘运动（偏航摆动与线加速度）、
一组观测偏差与噪声实现，以及一条带随机阶跃与漂移的参考轨迹。控制器除了当前状态与目标，
还能看到未来 0.2 s（20 个点）的目标预瞄（带随时间增长的偏移与噪声），用于提前调整动作；
目标可能落在关节限位之外，控制器需要在限位内尽量贴近目标且不碰撞限位。大 yaw 可以多圈连续转动，
所有角度输入都取 sin/cos 或 wrap 到 -pi 到 pi 的相对量。
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
VELOCITY_LIMIT = 15.0
PREVIEW_STEPS = 20
LIMIT_MARGIN = 2.0 * math.pi / 180.0
ROLL_LIMIT = (-45.0 * math.pi / 180.0, 5.0 * math.pi / 180.0)
SMALL_YAW_LIMIT = (-50.0 * math.pi / 180.0, 50.0 * math.pi / 180.0)
PITCH_LIMIT = (-45.0 * math.pi / 180.0, 45.0 * math.pi / 180.0)
LIMITS = (None, ROLL_LIMIT, SMALL_YAW_LIMIT, PITCH_LIMIT)
SLOPE_LIMIT = 10.0 * math.pi / 180.0

REFERENCE_CENTERS = [0.0, -0.20, 0.0, 0.0]
REFERENCE_AMPLITUDES = [0.60, 0.30, 0.35, 0.35]
STEP_AMPLITUDE = 0.35
DRIFT_RATE = 0.4
OUT_OF_RANGE_MARGIN = 0.35

BASE_YAW_AMPLITUDE = 6.0 * math.pi / 180.0
BASE_YAW_FREQUENCY = (0.5, 3.0)
BASE_ACCELERATION = 0.3

ANGLE_NOISE = 0.003
VELOCITY_NOISE = 0.05
SLOPE_NOISE = 0.5 * math.pi / 180.0
DIRECTION_NOISE = 2.0 * math.pi / 180.0
YAW_NOISE = 2.0 * math.pi / 180.0
YAW_RATE_NOISE = 0.05
BASE_ACCEL_NOISE = 0.02
ANGLE_BIAS = 0.010
VELOCITY_BIAS = 0.020
VELOCITY_SCALE_ERROR = 0.03
SLOPE_BIAS = 1.0 * math.pi / 180.0
DIRECTION_BIAS = 5.0 * math.pi / 180.0
YAW_BIAS = 5.0 * math.pi / 180.0
YAW_RATE_BIAS = 0.1
BASE_ACCEL_BIAS = 0.05
PREVIEW_OFFSET = 0.03
PREVIEW_NOISE = 0.015
ACCELERATION_FILTER = 0.2
LIMIT_WEIGHT = 30.0
HIT_PENALTY = 20.0

VELOCITY_SCALE = 2.0
ERROR_SCALE = 0.2
PREVIEW_SCALE = 0.2
ACCELERATION_SCALE = 20.0
BASE_ACCEL_SCALE = 0.3
YAW_RATE_SCALE = 2.0
OBSERVATION_DIM = 126


def truth_parameters() -> np.ndarray:
    return np.concatenate([params.MASSES, np.array(params.COM).reshape(-1),
                           np.array(params.INERTIA).reshape(-1),
                           np.array(params.JOINT_OFFSETS).reshape(-1), params.DAMPING])


def parameterize_truth() -> np.ndarray:
    basis = pb.compute_basis()[0]
    return basis.T @ pb.inertial_from_raw(truth_parameters())


def max_gravity_torque(model, context) -> np.ndarray:
    rng = np.random.default_rng(0)
    terrain = gd.make_terrain(0.0, 0.0)
    base = gd.make_base(yaw=0.0)
    worst = np.zeros(4)
    for _ in range(24):
        angle = np.array([rng.uniform(-math.pi, math.pi), rng.uniform(*ROLL_LIMIT),
                          rng.uniform(*SMALL_YAW_LIMIT), rng.uniform(*PITCH_LIMIT)])
        state = gd.make_state(angle, np.zeros(4))
        torque = np.abs(np.asarray(context.bias_torque(model, terrain, base, state)))
        worst = np.maximum(worst, torque)
    return worst


def validate_parameters(candidate: np.ndarray, context) -> bool:
    model = ip.build_model(candidate)
    terrain = gd.make_terrain(0.1, 0.3)
    base = gd.make_base(yaw=0.2)
    rng = np.random.default_rng(0)
    for _ in range(8):
        angle = np.array([rng.uniform(-3.0, 3.0), rng.uniform(*ROLL_LIMIT),
                          rng.uniform(*SMALL_YAW_LIMIT), rng.uniform(*PITCH_LIMIT)])
        state = gd.make_state(angle, rng.uniform(-1.0, 1.0, 4))
        context.inverse_dynamics(model, terrain, base, state, np.zeros(4))
        context.forward_step(model, terrain, base, state, np.zeros(4), DT, 2)
    state = gd.make_state(np.zeros(4), np.zeros(4))
    for _ in range(120):
        torque = rng.uniform(-4.0, 4.0, 4)
        state = context.forward_step(model, terrain, base, state, torque, DT, 2)
    return True


def sample_parameters(rng: np.random.Generator, scale: float, context=None):
    """采样参数与电机力矩系数；电机系数需保证至少能克服该参数下的自身重量。"""
    if scale <= 0.0:
        candidate = truth_parameters()
    else:
        candidate = None
        for _attempt in range(80):
            masses = np.array(params.MASSES) * np.exp(rng.uniform(-scale, scale, 4))
            com = np.array(params.COM) + rng.uniform(-0.03, 0.03, (4, 3)) * min(scale, 1.0)
            body_scale = np.exp(rng.uniform(-min(scale, 1.2), min(scale, 1.2), 4))
            inertia = np.array(params.INERTIA) * body_scale[:, None]
            offsets = np.array(params.JOINT_OFFSETS) + rng.normal(
                0.0, 0.03 * min(scale, 1.0), (4, 3))
            damping = np.array(params.DAMPING) * np.exp(rng.uniform(-scale, scale, 4))
            trial = np.concatenate([masses, com.reshape(-1), inertia.reshape(-1),
                                    offsets.reshape(-1), damping])
            try:
                ip.build_model(trial)
                if context is None or validate_parameters(trial, context):
                    candidate = trial
                    break
            except RuntimeError:
                continue
        if candidate is None:
            raise RuntimeError("参数采样连续 80 次不可行")
    model = ip.build_model(candidate)
    gravity = max_gravity_torque(model, context) if context is not None else np.zeros(4)
    if np.any(gravity * 1.25 > TORQUE_LIMIT):
        if scale > 0.0:
            return sample_parameters(rng, scale, context)
        raise RuntimeError("真值参数下的重力力矩超出电机能力")
    motor_scale = np.minimum(1.25 * gravity + rng.uniform(0.0, 2.0, 4), TORQUE_LIMIT)
    return candidate, motor_scale


def wrap_angle(values):
    return (np.asarray(values) + np.pi) % (2.0 * np.pi) - np.pi


def generate_reference(rng: np.random.Generator, steps: int, difficulty: float = 1.0):
    """多正弦加 0 到 3 个随机阶跃与可能漂移的参考轨迹；允许目标落在关节限位之外。"""
    time = np.arange(steps) * DT
    angle = np.zeros((steps, 4))
    for joint in range(4):
        signal = np.full(steps, REFERENCE_CENTERS[joint])
        for _ in range(2):
            frequency = rng.uniform(0.2, 1.2)
            amplitude = REFERENCE_AMPLITUDES[joint] * rng.uniform(0.5, 0.8) * difficulty
            phase = rng.uniform(-math.pi, math.pi)
            signal += amplitude * np.sin(frequency * time + phase)
        count = int(rng.integers(0, 4))
        positions = []
        for _ in range(count):
            index = int(rng.integers(steps // 6, steps))
            if any(abs(index - other) < 100 for other in positions):
                continue
            positions.append(index)
            signal[index:] += rng.uniform(-STEP_AMPLITUDE, STEP_AMPLITUDE) * difficulty
        if joint == 0 and rng.uniform() < 0.6:
            start = int(rng.integers(0, steps // 3))
            drift = rng.uniform(0.1, DRIFT_RATE) * rng.choice([-1.0, 1.0]) * difficulty
            signal[start:] += drift * (time[start:] - time[start])
        limits = LIMITS[joint]
        if limits is not None:
            signal = np.clip(signal, limits[0] - OUT_OF_RANGE_MARGIN,
                             limits[1] + OUT_OF_RANGE_MARGIN)
        angle[:, joint] = signal
    velocity = np.clip(np.gradient(angle, DT, axis=0), -3.0, 3.0)
    acceleration = np.gradient(velocity, DT, axis=0)
    return angle, velocity, acceleration


def clip_to_limits(angle):
    values = np.array(angle, dtype=float)
    for joint in range(4):
        if LIMITS[joint] is None:
            continue
        values[joint] = min(max(values[joint], LIMITS[joint][0]), LIMITS[joint][1])
    return values


class TrackingEnv:
    """单环境：跟踪带阶跃与漂移的参考轨迹，回合长度固定，无提前终止。"""

    def __init__(self, seed: int, episode_steps: int = 1500, parameter_scale: float = 0.0,
                 torque_limit: float = TORQUE_LIMIT, noise_scale: float = 1.0,
                 bias_scale: float = 1.0, difficulty: float = 1.0):
        self.rng = np.random.default_rng(seed)
        self.difficulty = float(difficulty)
        self.episode_steps = int(episode_steps)
        self.parameter_scale = float(parameter_scale)
        self.torque_limit = float(torque_limit)
        self.noise_scale = float(noise_scale)
        self.bias_scale = float(bias_scale)
        self.context = gd.Context(1)
        self.raw = None
        self.model = None
        self.motor_scale = np.ones(4)
        self.terrain = None
        self.base = gd.make_base()
        self.state = None
        self.angle_ref = None
        self.velocity_ref = None
        self.step_index = 0
        self.previous_action = np.zeros(4)
        self.filtered_velocity = np.zeros(4)
        self.previous_filtered_velocity = np.zeros(4)
        self.angle_bias = np.zeros(4)
        self.velocity_bias = np.zeros(4)
        self.velocity_scale = np.ones(4)
        self.slope_measured = 0.0
        self.direction_measured = 0.0
        self.yaw_bias = 0.0
        self.yaw_rate_bias = 0.0
        self.base_accel_bias = np.zeros(3)
        self.preview_offset = np.zeros((PREVIEW_STEPS, 4))
        self.hits = 0

    def _base_motion(self, index: int):
        time = index * DT
        yaw = self.yaw_center + self.yaw_amplitude * math.sin(
            self.yaw_frequency * time + self.yaw_phase)
        yaw_rate = self.yaw_amplitude * self.yaw_frequency * math.cos(
            self.yaw_frequency * time + self.yaw_phase)
        yaw_acceleration = -self.yaw_amplitude * self.yaw_frequency ** 2 * math.sin(
            self.yaw_frequency * time + self.yaw_phase)
        phase = self.acceleration_frequency * time + self.acceleration_phase
        acceleration = np.array([math.cos(self.acceleration_direction),
                                 math.sin(self.acceleration_direction), 0.0]) * (
            self.acceleration_amplitude * math.sin(phase))
        base = gd.make_base(acceleration=acceleration, yaw=yaw, yaw_rate=yaw_rate,
                            yaw_acceleration=yaw_acceleration)
        return base, acceleration

    def reset(self):
        self.raw, self.motor_scale = sample_parameters(self.rng, self.parameter_scale,
                                                        self.context)
        self.model = ip.build_model(self.raw)
        slope = self.rng.uniform(0.0, SLOPE_LIMIT)
        direction = self.rng.uniform(0.0, 2.0 * math.pi)
        self.terrain = gd.make_terrain(slope, direction)
        self.yaw_center = self.rng.uniform(-math.pi, math.pi)
        self.yaw_amplitude = self.rng.uniform(0.0, 1.0) * BASE_YAW_AMPLITUDE
        self.yaw_frequency = self.rng.uniform(*BASE_YAW_FREQUENCY)
        self.yaw_phase = self.rng.uniform(-math.pi, math.pi)
        self.acceleration_amplitude = self.rng.uniform(0.0, 1.0) * BASE_ACCELERATION
        self.acceleration_frequency = self.rng.uniform(0.3, 2.0)
        self.acceleration_phase = self.rng.uniform(-math.pi, math.pi)
        self.acceleration_direction = self.rng.uniform(0.0, 2.0 * math.pi)
        self.base, _ = self._base_motion(0)
        self.slope_measured = slope + self.rng.uniform(-1.0, 1.0) * SLOPE_BIAS * self.bias_scale
        self.direction_measured = direction + self.rng.uniform(
            -1.0, 1.0) * DIRECTION_BIAS * self.bias_scale
        self.yaw_bias = self.rng.uniform(-1.0, 1.0) * YAW_BIAS * self.bias_scale
        self.yaw_rate_bias = self.rng.uniform(-1.0, 1.0) * YAW_RATE_BIAS * self.bias_scale
        self.base_accel_bias = self.rng.uniform(
            -1.0, 1.0, 3) * BASE_ACCEL_BIAS * self.bias_scale
        self.preview_offset = self.rng.uniform(-1.0, 1.0, (PREVIEW_STEPS, 4)) * (
            np.linspace(0.15, 1.0, PREVIEW_STEPS) ** 1.5)[:, None] * PREVIEW_OFFSET * self.bias_scale
        self.angle_ref, self.velocity_ref, _ = generate_reference(
            self.rng, self.episode_steps, self.difficulty)
        self.angle_bias = self.rng.uniform(-1.0, 1.0, 4) * ANGLE_BIAS * self.bias_scale
        self.velocity_bias = self.rng.uniform(-1.0, 1.0, 4) * VELOCITY_BIAS * self.bias_scale
        self.velocity_scale = 1.0 + self.rng.uniform(
            -1.0, 1.0, 4) * VELOCITY_SCALE_ERROR * self.bias_scale
        initial_angle = clip_to_limits(self.angle_ref[0] + self.rng.uniform(-0.05, 0.05, 4))
        self.state = gd.make_state(initial_angle, self.velocity_ref[0])
        self.step_index = 0
        self.previous_action = np.zeros(4)
        self.hits = 0
        self.filtered_velocity = np.zeros(4)
        self.previous_filtered_velocity = np.zeros(4)
        return self.observation()

    def measurement(self):
        angle = np.array([self.state.angle[i] for i in range(4)])
        velocity = np.array([self.state.velocity[i] for i in range(4)])
        angle_measured = (angle + self.angle_bias
                          + self.rng.normal(0.0, ANGLE_NOISE * self.noise_scale, 4))
        velocity_measured = (self.velocity_scale * velocity + self.velocity_bias
                             + self.rng.normal(0.0, VELOCITY_NOISE * self.noise_scale, 4))
        slope = self.slope_measured + self.rng.normal(0.0, SLOPE_NOISE * self.noise_scale)
        direction = self.direction_measured + self.rng.normal(
            0.0, DIRECTION_NOISE * self.noise_scale)
        yaw_rate = (self.base.yaw_rate + self.yaw_rate_bias
                    + self.rng.normal(0.0, YAW_RATE_NOISE * self.noise_scale))
        acceleration = np.array([self.base.acceleration[i] for i in range(3)]) + (
            self.base_accel_bias + self.rng.normal(0.0, BASE_ACCEL_NOISE * self.noise_scale, 3))
        return angle_measured, velocity_measured, slope, direction, yaw_rate, acceleration

    def _preview(self, index: int):
        """未来 0.2 s 的目标相对位置（相对当前目标），带随时间增长的偏移与噪声。"""
        indices = np.clip(index + np.arange(1, PREVIEW_STEPS + 1), 0, self.episode_steps - 1)
        relative = wrap_angle(self.angle_ref[indices] - self.angle_ref[index])
        growth = (np.arange(1, PREVIEW_STEPS + 1) / PREVIEW_STEPS) ** 1.5
        noise = self.rng.normal(0.0, 1.0, (PREVIEW_STEPS, 4)) * growth[:, None] * (
            PREVIEW_NOISE * self.noise_scale)
        return (relative + self.preview_offset + noise).reshape(-1)

    def observation(self) -> np.ndarray:
        index = min(self.step_index, self.episode_steps - 1)
        angle_measured, velocity_measured, slope, direction, yaw_rate, acceleration = (
            self.measurement())
        self.filtered_velocity = ((1.0 - ACCELERATION_FILTER) * self.filtered_velocity
                                  + ACCELERATION_FILTER * velocity_measured)
        filtered_acceleration = (self.filtered_velocity - self.previous_filtered_velocity) / DT
        self.previous_filtered_velocity = self.filtered_velocity.copy()
        angle_ref = self.angle_ref[index]
        velocity_ref = self.velocity_ref[index]
        error = wrap_angle(angle_ref - angle_measured)
        yaw = self.base.yaw + self.yaw_bias
        observation = np.concatenate([
            np.sin(angle_measured), np.cos(angle_measured),
            velocity_measured / VELOCITY_SCALE,
            np.sin(angle_ref), np.cos(angle_ref),
            error / ERROR_SCALE, velocity_ref / VELOCITY_SCALE,
            self._preview(index) / PREVIEW_SCALE,
            self.previous_action,
            filtered_acceleration / ACCELERATION_SCALE,
            np.array([math.sin(slope), math.cos(slope), math.sin(direction),
                      math.cos(direction), math.sin(yaw), math.cos(yaw)]),
            np.array([yaw_rate / YAW_RATE_SCALE]),
            acceleration / BASE_ACCEL_SCALE,
        ])
        return np.clip(observation, -5.0, 5.0).astype(np.float32)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        torque = action * self.motor_scale
        angle = np.array([self.state.angle[i] for i in range(4)])
        velocity = np.array([self.state.velocity[i] for i in range(4)])
        angle_ref = self.angle_ref[self.step_index]
        velocity_ref = self.velocity_ref[self.step_index]
        error = wrap_angle(angle_ref - angle)
        position_ratio = (error / 0.02) ** 2
        velocity_ratio = ((velocity - velocity_ref) / 0.5) ** 2
        position_cost = (position_ratio / (1.0 + position_ratio / 25.0)).mean()
        velocity_cost = (velocity_ratio / (1.0 + velocity_ratio / 25.0)).mean()
        effort_cost = (action ** 2 / 4.0).mean()
        smooth_cost = ((action - self.previous_action) ** 2 / 4.0).mean()
        limit_cost = self._limit_cost(angle)
        self.previous_action = action.copy()
        self.state = self.context.forward_step(self.model, self.terrain, self.base,
                                               self.state, torque, DT, 2)
        velocity = np.array([self.state.velocity[i] for i in range(4)])
        if np.any(np.abs(velocity) > VELOCITY_LIMIT) or not np.all(np.isfinite(velocity)):
            clamped = np.clip(velocity, -VELOCITY_LIMIT, VELOCITY_LIMIT)
            angle = np.array([self.state.angle[i] for i in range(4)])
            self.state = gd.make_state(angle, clamped)
        self.step_index += 1
        self.base, _ = self._base_motion(self.step_index)
        hit = self._enforce_limits()
        self.hits += int(hit)
        cost = (position_cost + 0.05 * velocity_cost + 0.02 * effort_cost
                + 0.02 * smooth_cost + LIMIT_WEIGHT * limit_cost
                + HIT_PENALTY * float(hit))
        done = self.step_index >= self.episode_steps - 1
        return self.observation(), float(-cost), done, self.info()

    def _limit_cost(self, angle) -> float:
        total = 0.0
        for joint in range(4):
            limits = LIMITS[joint]
            if limits is None:
                continue
            distance = min(angle[joint] - limits[0], limits[1] - angle[joint])
            penetration = max(0.0, LIMIT_MARGIN - distance) / LIMIT_MARGIN
            total += penetration ** 2
        return total / 3.0

    def _enforce_limits(self) -> bool:
        angle = np.array([self.state.angle[i] for i in range(4)])
        velocity = np.array([self.state.velocity[i] for i in range(4)])
        clamped = angle.copy()
        hit = False
        for joint in range(4):
            limits = LIMITS[joint]
            if limits is None:
                continue
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
        return {"raw": self.raw.copy(),
                "error": wrap_angle(reference - angle).astype(np.float32),
                "hits": self.hits, "motor_scale": self.motor_scale.copy()}

    def close(self):
        self.context.close()
