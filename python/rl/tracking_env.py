"""用于强化学习的四自由度云台跟踪环境（含底盘运动、目标预瞄与全状态噪声）。

每个回合重置时采样一组动力学参数、一组电机力矩系数、一段底盘运动（偏航摆动与线加速度）、
一组观测偏差与噪声实现，以及一条带随机阶跃与漂移的参考轨迹。控制器除了当前状态与目标，
还能看到未来 0.2 s（20 个点）的目标预瞄（带随时间增长的偏移与噪声），用于提前调整动作；
目标可能落在关节限位之外，控制器需要在限位内尽量贴近目标且不碰撞限位。大 yaw 可以多圈连续转动，
所有角度输入都取 sin/cos 或 wrap 到 -pi 到 pi 的相对量。

参数漂移：`drift_scale` 与 `drift_motor` 大于零时，每个回合在重置时采样参数与电机系数的终点，
回合内按几何插值从起点缓慢变到终点；动力学模型每 `DRIFT_REFRESH` 步重建一次。

辅助目标：每步的转移信息（真实关节角、角速度、下一拍真实角加速度、当前可辨识坐标）记录在
`info()["transition"]` 里，供动力学预测头与参数回归头使用。
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
LIMIT_LOOKAHEAD = 0.04
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

# 奖励各项权重（全部在未缩放的代价空间里，最后统一乘 reward_scale）
POSITION_NEAR_SCALE = 0.02
POSITION_FAR_WEIGHT = 1.5
VELOCITY_COST_WEIGHT = 0.05
EFFORT_COST_WEIGHT = 0.02
SMOOTH_COST_WEIGHT = 0.02
LIMIT_WEIGHT = 30.0
HIT_PENALTY = 30.0
OVERSHOOT_WEIGHT = 500.0
REWARD_SCALE = 0.05
DRIFT_REFRESH = 10
GRAVITY_MARGIN = 1.5
INERTIA_MARGIN = 20.0
IDENTIFIABLE_DIMENSION = 28

VELOCITY_SCALE = 2.0
ERROR_SCALE = 0.2
PREVIEW_SCALE = 0.2
ACCELERATION_SCALE = 20.0
BASE_ACCEL_SCALE = 0.3
YAW_RATE_SCALE = 2.0
OBSERVATION_BASE_DIM = 126
# 标称模型物理特征：用设计参数（不是被控对象的随机参数）计算的模型基前馈、偏置力矩，
# 以及对测得角加速度做模型反演得到的"本该施加的动作"。三者全部由当前与历史可测量构造，
# 不含任何被控对象的真实随机参数，是控制器做前馈与在线辨识的物理先验。
NOMINAL_KP = 196.0
NOMINAL_KD = 28.0
NOMINAL_MARGIN = 0.07
NOMINAL_BRAKE = 14.0
NOMINAL_ACCEL_LIMIT = 55.0
NOMINAL_FEEDFORWARD_LIMIT = 12.0
NOMINAL_MISMATCH_LIMIT = 1.5
ACTION_EMA_RATE = 0.2
LOAD_RATE = 0.1
NOMINAL_OBSERVATION_SCALE = 5.0
NOMINAL_DIM = 16
OBSERVATION_DIM = OBSERVATION_BASE_DIM + NOMINAL_DIM
# 布局：基础观测 126；模型基前馈需求 4（放大 NOMINAL_OBSERVATION_SCALE 倍）；偏置力矩 4（同）；
# 动作失配 4；动作失配滑动平均（等效扰动估计）4。
# 快路吃基础观测与当前标称物理特征，慢路吃动作失配与其滑动平均，负责把失配积分成扰动估计。
DEMAND_SLICE = slice(OBSERVATION_BASE_DIM, OBSERVATION_BASE_DIM + 4)
BIAS_SLICE = slice(OBSERVATION_BASE_DIM + 4, OBSERVATION_BASE_DIM + 8)
MISMATCH_SLICE = slice(OBSERVATION_BASE_DIM + 8, OBSERVATION_BASE_DIM + 12)
LOAD_SLICE = slice(OBSERVATION_BASE_DIM + 12, OBSERVATION_BASE_DIM + 16)
FAST_INPUT_DIM = OBSERVATION_BASE_DIM + 8
SLOW_INPUT_DIM = 8
TRANSITION_STATE_DIM = 18
RESPONSE_DIM = 12

SOFT_LOW = np.array([-np.inf] + [LIMITS[index][0] + NOMINAL_MARGIN for index in (1, 2, 3)])
SOFT_HIGH = np.array([np.inf] + [LIMITS[index][1] - NOMINAL_MARGIN for index in (1, 2, 3)])

_NOMINAL = {}


def nominal_model(context):
    """设计标称模型与按标称模型定的执行器力矩尺度（均为常数，与回合参数无关）。"""
    if "model" not in _NOMINAL:
        model = ip.build_model(truth_parameters())
        _NOMINAL["model"] = model
        _NOMINAL["scale"] = np.maximum(
            INERTIA_MARGIN * joint_inertia(model, context),
            GRAVITY_MARGIN * max_gravity_torque(model, context)) + 1.0
        _NOMINAL["terrain"] = gd.make_terrain()
        _NOMINAL["base"] = gd.make_base()
    return _NOMINAL["model"], _NOMINAL["scale"]


def nominal_features(context, angle, velocity, filtered_acceleration, reference,
                     velocity_reference, preview) -> np.ndarray:
    """返回 12 维标称特征：模型基前馈需求（经有界分配收缩）、偏置力矩、模型反演动作。"""
    model, scale = nominal_model(context)
    terrain = _NOMINAL["terrain"]
    base = _NOMINAL["base"]
    target = np.clip(reference, SOFT_LOW, SOFT_HIGH)
    target_velocity = np.array(velocity_reference, dtype=float)
    target_velocity[target != reference] = 0.0
    future = np.clip(reference + np.reshape(preview, (-1, 4)), SOFT_LOW, SOFT_HIGH)
    times = DT * np.arange(1, future.shape[0] + 1)
    weights = 2.0 * times ** 2 / np.sum(times ** 4)
    feedforward = np.clip(weights @ (future - target - times[:, None] * target_velocity),
                          -NOMINAL_FEEDFORWARD_LIMIT, NOMINAL_FEEDFORWARD_LIMIT)
    desired = (feedforward + NOMINAL_KP * wrap_angle(target - angle)
               + NOMINAL_KD * (target_velocity - velocity))
    desired = np.clip(desired, -NOMINAL_ACCEL_LIMIT, NOMINAL_ACCEL_LIMIT)
    lag = NOMINAL_BRAKE * DT
    distances = np.stack((angle - SOFT_LOW, SOFT_HIGH - angle))
    speeds = np.sqrt(lag ** 2 + 2.0 * NOMINAL_BRAKE * np.maximum(distances, 0.0)) - lag
    speeds = speeds + (2.0 * NOMINAL_KP / NOMINAL_KD) * np.minimum(distances, 0.0)
    desired = np.clip(desired, NOMINAL_KD * (-speeds[0] - velocity),
                      NOMINAL_KD * (speeds[1] - velocity))
    desired = np.clip(desired, -NOMINAL_ACCEL_LIMIT, NOMINAL_ACCEL_LIMIT)
    state = gd.make_state(angle, velocity)
    bias = np.asarray(context.inverse_dynamics(model, terrain, base, state, np.zeros(4))) / scale
    demand = np.asarray(context.inverse_dynamics(model, terrain, base, state,
                                                 desired)) / scale - bias
    alpha = float(np.clip(np.min((1.0 - np.sign(demand) * bias)
                                 / np.maximum(np.abs(demand), 1.0e-10)), 0.0, 1.0))
    estimate = np.asarray(context.inverse_dynamics(
        model, terrain, base, state, filtered_acceleration)) / scale
    return np.concatenate([demand * alpha, bias, estimate])


def truth_parameters() -> np.ndarray:
    return np.concatenate([params.MASSES, np.array(params.COM).reshape(-1),
                           np.array(params.INERTIA).reshape(-1),
                           np.array(params.JOINT_OFFSETS).reshape(-1), params.DAMPING])


_BASIS = None


def identifiable_basis():
    global _BASIS
    if _BASIS is None:
        _BASIS = pb.compute_basis()[0]
    return _BASIS


def parameterize_truth() -> np.ndarray:
    return identifiable_basis().T @ pb.inertial_from_raw(truth_parameters())


def identifiable_coordinates(raw: np.ndarray) -> np.ndarray:
    return identifiable_basis().T @ pb.inertial_from_raw(raw)


def gauge_scale(motor_scale) -> float:
    """整体尺度的规范选取：四个电机系数的几何平均。

    动力学只依赖惯量、阻尼与电机系数三者的整体比例（同时乘任意正数，关节加速度不变），
    因此绝对尺度不可观测。用这个规范把参数标签变成可辨识的量（比例），供提取器学习。
    """
    return float(np.exp(np.log(np.maximum(np.asarray(motor_scale, dtype=float),
                                          1.0e-6)).mean()))


def gauge_fixed_coordinates(raw: np.ndarray, motor_scale) -> np.ndarray:
    """把惯量、质量、阻尼除以规范尺度后投影到可辨识基：整体尺度变化不影响该标签。"""
    scale = gauge_scale(motor_scale)
    scaled = raw.copy()
    scaled[0:4] = scaled[0:4] / scale
    scaled[16:40] = scaled[16:40] / scale
    scaled[52:56] = scaled[52:56] / scale
    return identifiable_coordinates(scaled)


def relative_motor_scale(motor_scale) -> np.ndarray:
    """电机系数相对规范尺度的大小，反映各关节执行器的相对增益。"""
    values = np.asarray(motor_scale, dtype=float)
    return values / gauge_scale(values)


def joint_inertia(model, context, angle=None) -> np.ndarray:
    """关节空间惯量矩阵的对角（零速度），用于给电机系数定下限与构造响应标签。"""
    terrain = gd.make_terrain(0.0, 0.0)
    base = gd.make_base()
    values = np.zeros(4) if angle is None else np.asarray(angle, dtype=float)
    state = gd.make_state(values, np.zeros(4))
    reference = np.asarray(context.inverse_dynamics(model, terrain, base, state,
                                                    np.zeros(4)))
    diagonal = np.zeros(4)
    for joint in range(4):
        unit = np.zeros(4)
        unit[joint] = 1.0
        torque = np.asarray(context.inverse_dynamics(model, terrain, base, state, unit))
        diagonal[joint] = torque[joint] - reference[joint]
    return np.maximum(diagonal, 0.0)


def max_gravity_torque(model, context, count: int = 48) -> np.ndarray:
    """限位内随机姿态（含限位角点）上的最大重力力矩，用于保证电机力矩系数留有裕度。"""
    rng = np.random.default_rng(0)
    worst = np.zeros(4)
    terrains = [gd.make_terrain(0.0, 0.0), gd.make_terrain(SLOPE_LIMIT, 0.7),
                gd.make_terrain(SLOPE_LIMIT, 3.9)]
    for terrain in terrains:
        base = gd.make_base(yaw=0.0)
        for _ in range(count):
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
        velocity = np.array([state.velocity[i] for i in range(4)])
        if np.max(np.abs(velocity)) > VELOCITY_LIMIT or not np.all(np.isfinite(velocity)):
            angle = np.array([state.angle[i] for i in range(4)])
            state = gd.make_state(angle, np.clip(velocity, -VELOCITY_LIMIT, VELOCITY_LIMIT))
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
            ip.build_model(trial)
            if context is None or validate_parameters(trial, context):
                candidate = trial
                break
        if candidate is None:
            raise RuntimeError("参数采样连续 80 次不可行")
    model = ip.build_model(candidate)
    if context is not None:
        gravity = max_gravity_torque(model, context)
        inertia = joint_inertia(model, context)
    else:
        gravity = np.zeros(4)
        inertia = np.zeros(4)
    if np.any(gravity * GRAVITY_MARGIN > TORQUE_LIMIT
              ) or np.any(inertia * INERTIA_MARGIN > TORQUE_LIMIT):
        if scale > 0.0:
            return sample_parameters(rng, scale, context)
        raise RuntimeError("真值参数下的重力力矩超出电机能力")
    # 电机系数同时覆盖静态重力力矩与参考角加速度所需的惯量力矩
    motor_scale = np.minimum(
        np.maximum(GRAVITY_MARGIN * gravity, INERTIA_MARGIN * inertia)
        + rng.uniform(0.5, 1.5, 4), TORQUE_LIMIT)
    return candidate, motor_scale


def sample_drift(rng: np.random.Generator, raw_start: np.ndarray, motor_start: np.ndarray,
                 drift_scale: float, drift_motor: float, context):
    """采样参数漂移的终点；终点必须仍可被电机力矩控制。"""
    for _attempt in range(40):
        raw_end = raw_start.copy()
        raw_end[0:4] = raw_start[0:4] * np.exp(rng.uniform(-drift_scale, drift_scale, 4))
        body_scale = np.exp(rng.uniform(-drift_scale, drift_scale, 4))
        raw_end[16:40] = raw_start[16:40] * np.tile(body_scale, 6)
        raw_end[52:56] = raw_start[52:56] * np.exp(rng.uniform(-drift_scale, drift_scale, 4))
        model = ip.build_model(raw_end)
        gravity = max_gravity_torque(model, context)
        inertia = joint_inertia(model, context)
        motor_end = motor_start * np.exp(rng.uniform(-drift_motor, drift_motor, 4))
        motor_end = np.maximum(motor_end, INERTIA_MARGIN * inertia * 0.9)
        motor_end = np.maximum(motor_end, 1.05 * gravity)
        if (np.all(INERTIA_MARGIN * inertia <= TORQUE_LIMIT)
                and np.all(motor_end <= TORQUE_LIMIT)
                and validate_parameters(raw_end, context)):
            return raw_end, np.minimum(motor_end, TORQUE_LIMIT)
    raise RuntimeError("参数漂移终点连续 40 次不可行")


def wrap_angle(values):
    return (np.asarray(values) + np.pi) % (2.0 * np.pi) - np.pi


def geometric_interpolate(start, end, fraction: float):
    """按幅值做几何插值并保持原符号；负的惯量非对角元不会翻号。"""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    return np.sign(start) * np.abs(start) ** (1.0 - fraction) * np.abs(end) ** fraction


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


def observation_from_parts(angle_measured, velocity_measured, slope, direction, yaw, yaw_rate,
                           acceleration, angle_ref, velocity_ref, preview, previous_action,
                           filtered_acceleration, nominal, mismatch, load):
    """把各段观测拼成 142 维向量；仿真环境与部署控制器共用同一份布局。"""
    error = wrap_angle(angle_ref - angle_measured)
    observation = np.concatenate([
        np.sin(angle_measured), np.cos(angle_measured),
        velocity_measured / VELOCITY_SCALE,
        np.sin(angle_ref), np.cos(angle_ref),
        error / ERROR_SCALE, velocity_ref / VELOCITY_SCALE,
        preview / PREVIEW_SCALE,
        previous_action,
        filtered_acceleration / ACCELERATION_SCALE,
        np.array([math.sin(slope), math.cos(slope), math.sin(direction),
                  math.cos(direction), math.sin(yaw), math.cos(yaw)]),
        np.array([yaw_rate / YAW_RATE_SCALE]),
        acceleration / BASE_ACCEL_SCALE,
        nominal[:8] * NOMINAL_OBSERVATION_SCALE,
        mismatch,
        load,
    ])
    return np.clip(observation, -5.0, 5.0).astype(np.float32)


class TrackingEnv:
    """单环境：跟踪带阶跃与漂移的参考轨迹，回合长度固定，无提前终止。"""

    def __init__(self, seed: int, episode_steps: int = 1500, parameter_scale: float = 0.0,
                 torque_limit: float = TORQUE_LIMIT, noise_scale: float = 1.0,
                 bias_scale: float = 1.0, difficulty: float = 1.0, drift_scale: float = 0.0,
                 drift_motor: float = 0.0, limit_weight: float = LIMIT_WEIGHT,
                 hit_penalty: float = HIT_PENALTY, overshoot_weight: float = OVERSHOOT_WEIGHT,
                 reward_scale: float = REWARD_SCALE, load_in_observation: bool = True,
                 limit_margin: float = LIMIT_MARGIN, limit_lookahead: float = LIMIT_LOOKAHEAD,
                 freeze_drift: bool = False):
        self.rng = np.random.default_rng(seed)
        self.difficulty = float(difficulty)
        self.episode_steps = int(episode_steps)
        self.parameter_scale = float(parameter_scale)
        self.drift_scale = float(drift_scale)
        self.drift_motor = float(drift_motor)
        self.torque_limit = float(torque_limit)
        self.noise_scale = float(noise_scale)
        self.bias_scale = float(bias_scale)
        self.limit_weight = float(limit_weight)
        self.hit_penalty = float(hit_penalty)
        self.overshoot_weight = float(overshoot_weight)
        self.reward_scale = float(reward_scale)
        self.load_in_observation = bool(load_in_observation)
        self.limit_margin = float(limit_margin)
        self.limit_lookahead = float(limit_lookahead)
        self.freeze_drift = bool(freeze_drift)
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
        self.coordinates = None
        self.transition = None
        self.drift_start_raw = None
        self.drift_end_raw = None
        self.drift_start_motor = None
        self.drift_end_motor = None

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

    def _set_parameters(self, fraction: float):
        """按回合进度把参数与电机系数从起点几何插值到终点，并重建模型与坐标。"""
        if self.freeze_drift or (self.drift_scale <= 0.0 and self.drift_motor <= 0.0):
            return
        raw = self.raw.copy()
        raw[0:4] = geometric_interpolate(self.drift_start_raw[0:4],
                                         self.drift_end_raw[0:4], fraction)
        raw[4:16] = ((1.0 - fraction) * self.drift_start_raw[4:16]
                     + fraction * self.drift_end_raw[4:16])
        raw[16:40] = geometric_interpolate(self.drift_start_raw[16:40],
                                           self.drift_end_raw[16:40], fraction)
        raw[40:52] = ((1.0 - fraction) * self.drift_start_raw[40:52]
                      + fraction * self.drift_end_raw[40:52])
        raw[52:56] = geometric_interpolate(self.drift_start_raw[52:56],
                                           self.drift_end_raw[52:56], fraction)
        self.raw = raw
        self.model = ip.build_model(raw)
        self.coordinates = identifiable_coordinates(raw).astype(np.float32)
        self.motor_scale = np.maximum(
            self.drift_start_motor ** (1.0 - fraction) * self.drift_end_motor ** fraction,
            1.0e-3)

    def reset(self):
        self.raw, self.motor_scale = sample_parameters(self.rng, self.parameter_scale,
                                                      self.context)
        self.model = ip.build_model(self.raw)
        if self.drift_scale > 0.0 or self.drift_motor > 0.0:
            self.drift_start_raw = self.raw.copy()
            self.drift_start_motor = self.motor_scale.copy()
            self.drift_end_raw, self.drift_end_motor = sample_drift(
                self.rng, self.drift_start_raw, self.drift_start_motor,
                self.drift_scale, self.drift_motor, self.context)
        else:
            self.drift_start_raw = self.raw.copy()
            self.drift_end_raw = self.raw.copy()
            self.drift_start_motor = self.motor_scale.copy()
            self.drift_end_motor = self.motor_scale.copy()
        self.coordinates = identifiable_coordinates(self.raw).astype(np.float32)
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
        # 初始状态放在软限位以内：参考初值可能正好落在硬限位上，直接从硬限位起步会让
        # 每个回合一开始就处在"擦边"状态，把初始化误差算成碰撞。
        initial_angle = self.angle_ref[0] + self.rng.uniform(-0.05, 0.05, 4)
        for joint in range(4):
            limits = LIMITS[joint]
            if limits is None:
                continue
            low = limits[0] + NOMINAL_MARGIN
            high = limits[1] - NOMINAL_MARGIN
            initial_angle[joint] = min(max(initial_angle[joint], low), high)
        self.state = gd.make_state(initial_angle, self.velocity_ref[0])
        self.step_index = 0
        self.previous_action = np.zeros(4)
        self.hits = 0
        self.transition = None
        self.filtered_velocity = np.zeros(4)
        self.previous_filtered_velocity = np.zeros(4)
        self.previous_action_ema = np.zeros(4)
        self.load_ema = np.zeros(4)
        self.inertia_diagonal = joint_inertia(self.model, self.context)
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
        yaw = self.base.yaw + self.yaw_bias
        preview = self._preview(index)
        nominal = nominal_features(self.context, angle_measured, velocity_measured,
                                   filtered_acceleration, angle_ref, velocity_ref, preview)
        # 动作残差：指令动作（低通）与"按标称模型产生同样角加速度本该施加的动作"之差，
        # 直接反映当前模型失配与执行器增益偏差，由慢路上下文积分成等效扰动估计。
        self.previous_action_ema = (1.0 - ACTION_EMA_RATE) * self.previous_action_ema + (
            ACTION_EMA_RATE * self.previous_action)
        mismatch = np.clip(self.previous_action_ema - nominal[8:12],
                           -NOMINAL_MISMATCH_LIMIT, NOMINAL_MISMATCH_LIMIT)
        self.load_ema = (1.0 - LOAD_RATE) * self.load_ema + LOAD_RATE * mismatch
        load = self.load_ema if self.load_in_observation else np.zeros(4)
        return observation_from_parts(
            angle_measured, velocity_measured, slope, direction, yaw, yaw_rate,
            acceleration, angle_ref, velocity_ref, preview, self.previous_action,
            filtered_acceleration, nominal, mismatch, load)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        torque = action * self.motor_scale
        angle = np.array([self.state.angle[i] for i in range(4)])
        velocity = np.array([self.state.velocity[i] for i in range(4)])
        angle_ref = self.angle_ref[self.step_index]
        velocity_ref = self.velocity_ref[self.step_index]
        error = wrap_angle(angle_ref - angle)
        position_ratio = (error / POSITION_NEAR_SCALE) ** 2
        velocity_ratio = ((velocity - velocity_ref) / 0.5) ** 2
        # 小误差用饱和代价鼓励精确跟踪，大误差用不饱和的绝对值代价保证梯度不消失
        position_cost = (position_ratio / (1.0 + position_ratio / 9.0)
                         + POSITION_FAR_WEIGHT * np.abs(error)).mean()
        velocity_cost = (velocity_ratio / (1.0 + velocity_ratio / 25.0)).mean()
        effort_cost = (action ** 2 / 4.0).mean()
        smooth_cost = ((action - self.previous_action) ** 2 / 4.0).mean()
        limit_cost = self._limit_cost(angle, velocity)
        self.previous_action = action.copy()
        # 转移信息：动力学预测头与参数回归头的监督目标
        transition_state = np.concatenate([
            angle, velocity,
            np.array([math.sin(self.terrain.slope_angle),
                      math.cos(self.terrain.slope_angle),
                      math.sin(self.terrain.slope_direction),
                      math.cos(self.terrain.slope_direction),
                      math.sin(self.base.yaw), math.cos(self.base.yaw),
                      self.base.yaw_rate]),
            np.array([self.base.acceleration[i] for i in range(3)]),
        ])
        fixed_coordinates = gauge_fixed_coordinates(self.raw, self.motor_scale)
        relative_motor = relative_motor_scale(self.motor_scale)
        bias_torque = np.asarray(self.context.inverse_dynamics(
            self.model, self.terrain, self.base, gd.make_state(angle, velocity), np.zeros(4)))
        if self.step_index % DRIFT_REFRESH == 0:
            self.inertia_diagonal = joint_inertia(self.model, self.context, angle)
        response = np.concatenate([bias_torque / self.motor_scale,
                                   self.inertia_diagonal / self.motor_scale,
                                   self.load_ema])
        self.state = self.context.forward_step(self.model, self.terrain, self.base,
                                               self.state, torque, DT, 2)
        next_velocity = np.array([self.state.velocity[i] for i in range(4)])
        # 动力学监督目标取未受限位夹紧影响的纯动力学加速度
        acceleration = (next_velocity - velocity) / DT
        if np.any(np.abs(next_velocity) > VELOCITY_LIMIT) or not np.all(np.isfinite(next_velocity)):
            clamped = np.clip(next_velocity, -VELOCITY_LIMIT, VELOCITY_LIMIT)
            angle = np.array([self.state.angle[i] for i in range(4)])
            self.state = gd.make_state(angle, clamped)
        self.step_index += 1
        if self.step_index % DRIFT_REFRESH == 0:
            self._set_parameters(min(1.0, self.step_index / max(1, self.episode_steps - 1)))
        self.base, _ = self._base_motion(self.step_index)
        hit, overshoot = self._enforce_limits()
        self.hits += int(hit)
        cost = (position_cost + VELOCITY_COST_WEIGHT * velocity_cost
                + EFFORT_COST_WEIGHT * effort_cost + SMOOTH_COST_WEIGHT * smooth_cost
                + self.limit_weight * limit_cost + self.hit_penalty * float(hit)
                + self.overshoot_weight * overshoot ** 2)
        self.transition = {"state": transition_state.astype(np.float32),
                           "acceleration": acceleration.astype(np.float32),
                           "coordinates": fixed_coordinates.astype(np.float32),
                           "relative_motor": relative_motor.astype(np.float32),
                           "response": response.astype(np.float32)}
        done = self.step_index >= self.episode_steps - 1
        return self.observation(), float(-self.reward_scale * cost), done, self.info()

    def _limit_cost(self, angle, velocity) -> float:
        """限位裕度代价，用角速度做短时预判，避免高速冲向限位时惩罚来得太晚。"""
        total = 0.0
        for joint in range(4):
            limits = LIMITS[joint]
            if limits is None:
                continue
            lower = angle[joint] - limits[0]
            upper = limits[1] - angle[joint]
            distance = min(lower, upper)
            inward = (max(0.0, -velocity[joint]) if lower < upper
                      else max(0.0, velocity[joint]))
            predicted = distance - inward * self.limit_lookahead
            penetration = max(0.0, self.limit_margin - predicted) / self.limit_margin
            total += min(1.0, penetration) ** 2
        return total / 3.0

    def _enforce_limits(self):
        """关节限位按刚性挡块处理；返回是否撞到限位与超出限位的最大幅度。"""
        angle = np.array([self.state.angle[i] for i in range(4)])
        velocity = np.array([self.state.velocity[i] for i in range(4)])
        clamped = angle.copy()
        hit = False
        overshoot = 0.0
        for joint in range(4):
            limits = LIMITS[joint]
            if limits is None:
                continue
            if clamped[joint] < limits[0]:
                overshoot = max(overshoot, limits[0] - clamped[joint])
                clamped[joint] = limits[0]
                velocity[joint] = 0.0
                hit = True
            elif clamped[joint] > limits[1]:
                overshoot = max(overshoot, clamped[joint] - limits[1])
                clamped[joint] = limits[1]
                velocity[joint] = 0.0
                hit = True
        if hit:
            self.state = gd.make_state(clamped, velocity)
        return hit, overshoot

    def info(self) -> dict:
        angle = np.array([self.state.angle[i] for i in range(4)])
        reference = self.angle_ref[min(self.step_index, self.episode_steps - 1)]
        return {"raw": self.raw.copy(),
                "error": wrap_angle(reference - angle).astype(np.float32),
                "hits": self.hits, "motor_scale": self.motor_scale.copy(),
                "coordinates": self.coordinates,
                "transition": self.transition,
                "parameters_drifted": (float(self.drift_scale) > 0.0
                                       or float(self.drift_motor) > 0.0)}

    def close(self):
        self.context.close()
