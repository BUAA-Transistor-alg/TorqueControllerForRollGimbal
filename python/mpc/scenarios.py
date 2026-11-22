"""闭环评估场景：参考轨迹、底盘运动、地形、量测噪声与被控对象。

被控对象用真实参数或按比例失配后的参数，限定 8 N·m 力矩上限，关节硬限位按
刚性挡块处理并统计撞限位步数；控制器只能看到带噪声的量测和给定模型。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import gimbal_dynamics as gd
import gimbal_env
import params

DT = gimbal_env.DT
REFINEMENT = gimbal_env.REFINEMENT
TORQUE_LIMIT = 8.0
VELOCITY_LIMIT = 15.0
LIMIT_MARGIN = 0.07
OUT_OF_RANGE_MARGIN = 0.30

LIMITS = gimbal_env.JOINT_LIMITS
LOWER = np.array([-np.inf if limits[0] is None else limits[0] for limits in LIMITS])
UPPER = np.array([np.inf if limits[1] is None else limits[1] for limits in LIMITS])
LIMITED = np.array([limits[0] is not None for limits in LIMITS])

# 参考轨迹与跟踪环境一致的关节中心与幅值
REFERENCE_CENTERS = np.array([0.0, -0.20, 0.0, 0.0])
REFERENCE_AMPLITUDES = np.array([0.60, 0.30, 0.35, 0.35])
STEP_AMPLITUDE = 0.35
DRIFT_RATE = 0.40

BASE_YAW_AMPLITUDE = 6.0 * math.pi / 180.0
BASE_YAW_FREQUENCY = (0.5, 3.0)
BASE_ACCELERATION = 0.3

ANGLE_NOISE = 0.003
VELOCITY_NOISE = 0.05
SLOPE_NOISE = 0.5 * math.pi / 180.0
DIRECTION_NOISE = 2.0 * math.pi / 180.0
YAW_NOISE = 2.0 * math.pi / 180.0
YAW_RATE_NOISE = 0.05
YAW_ACCELERATION_NOISE = 0.05
BASE_ACCEL_NOISE = 0.02
ANGLE_BIAS = 0.005
VELOCITY_BIAS = 0.010
SLOPE_BIAS = 1.0 * math.pi / 180.0
DIRECTION_BIAS = 5.0 * math.pi / 180.0
YAW_BIAS = 5.0 * math.pi / 180.0
YAW_RATE_BIAS = 0.05
BASE_ACCEL_BIAS = 0.02

TRUTH_RAW = np.concatenate([
    params.MASSES,
    np.array(params.COM).reshape(-1),
    np.array(params.INERTIA).reshape(-1),
    np.array(params.JOINT_OFFSETS).reshape(-1),
    params.DAMPING,
])


def build_model(raw: np.ndarray) -> gd.TcrgModel:
    """按 56 维原始参数构造模型：质量、质心、惯量、关节安装偏置、阻尼。"""
    raw = np.asarray(raw, dtype=float)
    return gd.make_model(raw[0:4], raw[4:16].reshape(4, 3),
                         raw[16:40].reshape(4, 6), raw[40:52].reshape(4, 3),
                         raw[52:56])


def design_model() -> gd.TcrgModel:
    return gd.make_model()


def identified_model(path: Path = Path("data/identified_final.npz")) -> gd.TcrgModel:
    return build_model(np.load(path)["model"])


def mismatch_raw() -> np.ndarray:
    """被控对象参数失配：质量 +10%、惯量 +12%、质心与安装偏置加偏、阻尼 ×1.5。"""
    raw = TRUTH_RAW.copy()
    raw[0:4] *= 1.10
    raw[16:40] *= 1.12
    com = raw[4:16].reshape(4, 3)
    com += np.array([0.0, 0.005, -0.004])
    offsets = raw[40:52].reshape(4, 3)
    offsets += np.array([0.0, 0.003, 0.002])
    raw[52:56] *= 1.5
    return raw


def generate_reference(rng: np.random.Generator, steps: int,
                       difficulty: float = 1.0):
    """多正弦加随机阶跃与 yaw 漂移的参考轨迹，允许超出关节限位。"""
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
        if LIMITED[joint]:
            signal = np.clip(signal, LOWER[joint] - OUT_OF_RANGE_MARGIN,
                             UPPER[joint] + OUT_OF_RANGE_MARGIN)
        angle[:, joint] = signal
    velocity = np.clip(np.gradient(angle, DT, axis=0), -3.0, 3.0)
    acceleration = np.gradient(velocity, DT, axis=0)
    return angle, velocity, acceleration


class Scenario:
    """一条固定场景：参考、地形、底盘运动、量测偏差与噪声水平。"""

    def __init__(self, seed: int = 7, steps: int = 1000, difficulty: float = 1.0):
        self.seed = int(seed)
        self.steps = int(steps)
        rng = np.random.default_rng(self.seed)
        self.reference_angle, self.reference_velocity, self.reference_acceleration = (
            generate_reference(rng, self.steps, difficulty))
        self.slope = float(rng.uniform(0.05, 0.17))
        self.direction = float(rng.uniform(0.0, 2.0 * math.pi))
        self.terrain = gd.make_terrain(self.slope, self.direction)
        self.yaw_center = float(rng.uniform(-math.pi, math.pi))
        self.yaw_amplitude = float(rng.uniform(0.0, 1.0) * BASE_YAW_AMPLITUDE)
        self.yaw_frequency = float(rng.uniform(*BASE_YAW_FREQUENCY))
        self.yaw_phase = float(rng.uniform(-math.pi, math.pi))
        self.acceleration_amplitude = float(rng.uniform(0.0, 1.0) * BASE_ACCELERATION)
        self.acceleration_frequency = float(rng.uniform(0.3, 2.0))
        self.acceleration_phase = float(rng.uniform(-math.pi, math.pi))
        self.acceleration_direction = float(rng.uniform(0.0, 2.0 * math.pi))
        self.angle_bias = rng.uniform(-1.0, 1.0, 4) * ANGLE_BIAS
        self.velocity_bias = rng.uniform(-1.0, 1.0, 4) * VELOCITY_BIAS
        self.slope_bias = float(rng.uniform(-1.0, 1.0) * SLOPE_BIAS)
        self.direction_bias = float(rng.uniform(-1.0, 1.0) * DIRECTION_BIAS)
        self.yaw_bias = float(rng.uniform(-1.0, 1.0) * YAW_BIAS)
        self.yaw_rate_bias = float(rng.uniform(-1.0, 1.0) * YAW_RATE_BIAS)
        self.base_accel_bias = rng.uniform(-1.0, 1.0, 3) * BASE_ACCEL_BIAS
        initial = self.reference_angle[0] + rng.uniform(-0.02, 0.02, 4)
        for joint in range(4):
            if not LIMITED[joint]:
                continue
            initial[joint] = np.clip(initial[joint],
                                     LOWER[joint] + 2.0 * LIMIT_MARGIN,
                                     UPPER[joint] - 2.0 * LIMIT_MARGIN)
        self.initial_angle = initial
        self.initial_velocity = self.reference_velocity[0].copy()

    def base_state(self, index: int) -> gd.TcrgBaseState:
        """第 index 拍起始时刻的真实底盘状态。"""
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
        return gd.make_base(acceleration=acceleration, yaw=yaw, yaw_rate=yaw_rate,
                            yaw_acceleration=yaw_acceleration)

    def window(self, index: int, horizon: int):
        """从下一拍开始的 horizon 个参考点，末尾不足处取最后一个点。"""
        indices = np.clip(index + np.arange(1, horizon + 1), 0, self.steps - 1)
        return self.reference_angle[indices], self.reference_velocity[indices]

    def measure(self, angle: np.ndarray, velocity: np.ndarray, index: int,
                rng: np.random.Generator) -> dict:
        """带偏差与噪声的量测：关节角速度、地形、底盘状态。"""
        angle_measured = angle + self.angle_bias + rng.normal(0.0, ANGLE_NOISE, 4)
        velocity_measured = (velocity + self.velocity_bias
                             + rng.normal(0.0, VELOCITY_NOISE, 4))
        terrain = gd.make_terrain(
            self.slope + self.slope_bias + rng.normal(0.0, SLOPE_NOISE),
            self.direction + self.direction_bias + rng.normal(0.0, DIRECTION_NOISE))
        base = self.base_state(index)
        acceleration = np.array([base.acceleration[i] for i in range(3)])
        base_measured = gd.make_base(
            acceleration=acceleration + self.base_accel_bias
            + rng.normal(0.0, BASE_ACCEL_NOISE, 3),
            yaw=base.yaw + self.yaw_bias + rng.normal(0.0, YAW_NOISE),
            yaw_rate=base.yaw_rate + self.yaw_rate_bias
            + rng.normal(0.0, YAW_RATE_NOISE),
            yaw_acceleration=base.yaw_acceleration
            + rng.normal(0.0, YAW_ACCELERATION_NOISE))
        return {"angle": angle_measured, "velocity": velocity_measured,
                "terrain": terrain, "base": base_measured}


def run_closed_loop(scenario: Scenario, controller, plant_model: gd.TcrgModel,
                    context: gd.Context, noise_seed: int = 0,
                    torque_limit: float = TORQUE_LIMIT) -> dict:
    """闭环仿真：控制器只用带噪声量测与给定模型，被控对象按硬挡块处理限位。"""
    steps = scenario.steps
    horizon = int(getattr(controller, "horizon", 1))
    rng = np.random.default_rng(noise_seed)
    state = gd.make_state(scenario.initial_angle, scenario.initial_velocity)
    angle_history = np.zeros((steps + 1, 4))
    velocity_history = np.zeros((steps + 1, 4))
    angle_history[0] = scenario.initial_angle
    velocity_history[0] = scenario.initial_velocity
    command_history = np.zeros((steps, 4))
    hit_history = np.zeros(steps, dtype=bool)
    solve_time = np.zeros(steps)
    iterations = np.zeros(steps, dtype=int)
    calls = np.zeros(steps, dtype=int)
    for index in range(steps):
        angle = np.array([state.angle[i] for i in range(4)])
        velocity = np.array([state.velocity[i] for i in range(4)])
        measured = scenario.measure(angle, velocity, index, rng)
        target_angle, target_velocity = scenario.window(index, horizon)
        command = np.asarray(controller.command(
            measured["angle"], measured["velocity"], target_angle, target_velocity,
            measured["terrain"], measured["base"]), dtype=float)
        command = np.clip(command, -torque_limit, torque_limit)
        command_history[index] = command
        info = getattr(controller, "last_info", {})
        solve_time[index] = float(info.get("solve_time", 0.0))
        iterations[index] = int(info.get("iterations", 0))
        calls[index] = int(info.get("calls", 0))
        state = context.forward_step(plant_model, scenario.terrain,
                                     scenario.base_state(index), state, command,
                                     DT, REFINEMENT)
        next_angle = np.array([state.angle[i] for i in range(4)])
        next_velocity = np.array([state.velocity[i] for i in range(4)])
        hit = False
        for joint in range(4):
            if not LIMITED[joint]:
                continue
            if next_angle[joint] < LOWER[joint]:
                next_angle[joint] = LOWER[joint]
                next_velocity[joint] = 0.0
                hit = True
            elif next_angle[joint] > UPPER[joint]:
                next_angle[joint] = UPPER[joint]
                next_velocity[joint] = 0.0
                hit = True
        next_velocity = np.clip(next_velocity, -VELOCITY_LIMIT, VELOCITY_LIMIT)
        state = gd.make_state(next_angle, next_velocity)
        hit_history[index] = hit
        angle_history[index + 1] = next_angle
        velocity_history[index + 1] = next_velocity
    return {"angle": angle_history, "velocity": velocity_history,
            "reference": scenario.reference_angle,
            "reference_velocity": scenario.reference_velocity,
            "command": command_history, "hits": hit_history,
            "solve_time": solve_time, "iterations": iterations, "calls": calls}
