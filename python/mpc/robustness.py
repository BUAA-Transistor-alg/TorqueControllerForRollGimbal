"""MPC 参数失真鲁棒性实验。

同一闭环场景下同时施加四类失真，控制器始终使用标称设计参数模型：

  1 被控对象参数：质量、惯量、质心、关节安装偏置、阻尼在给定范围内随机变化；
  2 观测状态：关节角、角速度、地形与底盘状态叠加高斯噪声与常值偏移；
  3 执行增益：四个轴的实际力矩分别为指令力矩乘一个 1 附近的正增益，增益不等于 1；
  4 通信时延：控制指令与观测按每拍随机步数延迟，指令零阶保持，观测取历史缓冲。

失真在每个回合按种子抽取一次（时延逐拍抽取）。评估给出跟踪误差、抖动、撞限位与
发散判定，并按单项失真与包络倍数扫描给出容忍边界。

用法（在 python 目录下）：
    python3 -m mpc.robustness --steps 600 --seeds 8 --workers 4 --output ../data/mpc/robustness
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np

import gimbal_dynamics as gd
import gimbal_env
import params
from mpc import scenarios
from mpc.controller import TORQUE_LIMIT, MpcController, controller_parameters

DT = gimbal_env.DT
REFINEMENT = gimbal_env.REFINEMENT
# 参考轨迹的随机阶跃按实车目标模式的移动速度限幅为斜坡
SCENARIO_STEP_RATE = params.PID_TARGET_RATE
# 随机阶跃只出现在回合前 60% 的时间，后 25% 作为无阶跃激励的收敛观察段
SCENARIO_STEP_FRACTION = 0.6
TAIL_FRACTION = 0.25
TRUTH_RAW = scenarios.TRUTH_RAW
LIMITED = scenarios.LIMITED
LOWER = scenarios.LOWER
UPPER = scenarios.UPPER
VELOCITY_LIMIT = scenarios.VELOCITY_LIMIT

AXIS_NAMES = ("big yaw", "roll", "small yaw", "pitch")
# 执行增益与 1 的最小偏离：保证每一轴的增益确实不等于 1
MIN_TORQUE_GAIN_DEVIATION = 0.02


@dataclass(frozen=True)
class Distortion:
    """失真范围。

    相对类字段为范围 δ，抽样为 1 + U(-δ, δ)；绝对类字段为范围端点；噪声类字段为
    标准差；时延类字段为每拍随机延迟的步数上限。
    """

    mass_relative: float = 0.0
    inertia_relative: float = 0.0
    com_absolute: float = 0.0
    offset_absolute: float = 0.0
    damping_relative: float = 0.0
    torque_gain_relative: float = 0.0
    angle_noise: float = 0.0
    angle_bias: float = 0.0
    velocity_noise: float = 0.0
    velocity_bias: float = 0.0
    slope_noise: float = 0.0
    slope_bias: float = 0.0
    direction_noise: float = 0.0
    direction_bias: float = 0.0
    yaw_noise: float = 0.0
    yaw_bias: float = 0.0
    yaw_rate_noise: float = 0.0
    yaw_rate_bias: float = 0.0
    yaw_acceleration_noise: float = 0.0
    base_acceleration_noise: float = 0.0
    base_acceleration_bias: float = 0.0
    control_delay_steps: int = 0
    observation_delay_steps: int = 0


# 标称量测：与 mpc.scenarios.Scenario.measure 的噪声与偏移水平一致
NOMINAL_DISTORTION = Distortion(
    angle_noise=scenarios.ANGLE_NOISE,
    angle_bias=scenarios.ANGLE_BIAS,
    velocity_noise=scenarios.VELOCITY_NOISE,
    velocity_bias=scenarios.VELOCITY_BIAS,
    slope_noise=scenarios.SLOPE_NOISE,
    slope_bias=scenarios.SLOPE_BIAS,
    direction_noise=scenarios.DIRECTION_NOISE,
    direction_bias=scenarios.DIRECTION_BIAS,
    yaw_noise=scenarios.YAW_NOISE,
    yaw_bias=scenarios.YAW_BIAS,
    yaw_rate_noise=scenarios.YAW_RATE_NOISE,
    yaw_rate_bias=scenarios.YAW_RATE_BIAS,
    yaw_acceleration_noise=scenarios.YAW_ACCELERATION_NOISE,
    base_acceleration_noise=scenarios.BASE_ACCEL_NOISE,
    base_acceleration_bias=scenarios.BASE_ACCEL_BIAS,
)

PARAMETER_DISTORTION = {
    "mass_relative": 0.15,
    "inertia_relative": 0.15,
    "com_absolute": 0.010,
    "offset_absolute": 0.008,
    "damping_relative": 0.5,
}
MEASUREMENT_DISTORTION = {
    "angle_noise": 0.008,
    "angle_bias": 0.02,
    "velocity_noise": 0.10,
    "velocity_bias": 0.05,
    "slope_noise": math.radians(1.5),
    "slope_bias": math.radians(3.0),
    "direction_noise": math.radians(4.0),
    "direction_bias": math.radians(10.0),
    "yaw_noise": math.radians(4.0),
    "yaw_bias": math.radians(10.0),
    "yaw_rate_noise": 0.10,
    "yaw_rate_bias": 0.10,
    "yaw_acceleration_noise": 0.10,
    "base_acceleration_noise": 0.05,
    "base_acceleration_bias": 0.05,
}
TORQUE_GAIN_DISTORTION = {"torque_gain_relative": 0.15}
CONTROL_DELAY_DISTORTION = {"control_delay_steps": 2}
OBSERVATION_DELAY_DISTORTION = {"observation_delay_steps": 2}

COMBINED_DISTORTION = replace(
    NOMINAL_DISTORTION,
    **PARAMETER_DISTORTION,
    **MEASUREMENT_DISTORTION,
    **TORQUE_GAIN_DISTORTION,
    **CONTROL_DELAY_DISTORTION,
    **OBSERVATION_DELAY_DISTORTION,
)

DISTORTION_GROUPS = {
    "parameter": PARAMETER_DISTORTION,
    "measurement": MEASUREMENT_DISTORTION,
    "torque gain": TORQUE_GAIN_DISTORTION,
    "control delay": CONTROL_DELAY_DISTORTION,
    "observation delay": OBSERVATION_DELAY_DISTORTION,
}


def distortion_conditions() -> dict[str, Distortion]:
    """标称、单项失真与全失真条件。"""
    return {
        "nominal": NOMINAL_DISTORTION,
        "parameter": replace(NOMINAL_DISTORTION, **PARAMETER_DISTORTION),
        "measurement": replace(NOMINAL_DISTORTION, **MEASUREMENT_DISTORTION),
        "torque gain": replace(NOMINAL_DISTORTION, **TORQUE_GAIN_DISTORTION),
        "control delay": replace(NOMINAL_DISTORTION, **CONTROL_DELAY_DISTORTION),
        "observation delay": replace(NOMINAL_DISTORTION, **OBSERVATION_DELAY_DISTORTION),
        "combined": COMBINED_DISTORTION,
    }


def scaled_group(kind: str, factor: float) -> Distortion:
    """把某一组失真的范围按 factor 从标称水平放大。"""
    overrides = DISTORTION_GROUPS[kind]
    values = {item.name: getattr(NOMINAL_DISTORTION, item.name)
              for item in fields(Distortion)}
    for name, amount in overrides.items():
        value = getattr(NOMINAL_DISTORTION, name) + factor * amount
        values[name] = int(round(value)) if name.endswith("_steps") else float(value)
    return Distortion(**values)


def scaled_combined(factor: float) -> Distortion:
    """全失真包络按 factor 缩放：0 为标称，1 为设计包络。"""
    values = {}
    for item in fields(Distortion):
        nominal = getattr(NOMINAL_DISTORTION, item.name)
        combined = getattr(COMBINED_DISTORTION, item.name)
        value = nominal + factor * (combined - nominal)
        values[item.name] = int(round(value)) if item.name.endswith("_steps") else float(value)
    return Distortion(**values)


@dataclass(frozen=True)
class DistortionSample:
    """一个回合抽取到的具体失真：被控对象参数、执行增益与观测常值偏移。"""

    distortion: Distortion
    raw: np.ndarray
    torque_gain: np.ndarray
    angle_bias: np.ndarray
    velocity_bias: np.ndarray
    slope_bias: float
    direction_bias: float
    yaw_bias: float
    yaw_rate_bias: float
    base_acceleration_bias: np.ndarray


def sample_distortion(rng: np.random.Generator, distortion: Distortion) -> DistortionSample:
    """按失真范围抽样被控对象参数、执行增益与观测偏移。

    无论失真范围是否为零都抽取同样数量的随机数，因此去掉某组失真时其余随机量保持
    一致，便于逐组定位。
    """
    raw = TRUTH_RAW.copy()
    raw[0:4] *= 1.0 + distortion.mass_relative * rng.uniform(-1.0, 1.0, 4)
    raw[4:16] += distortion.com_absolute * rng.uniform(-1.0, 1.0, 12)
    raw[16:40] *= 1.0 + distortion.inertia_relative * rng.uniform(-1.0, 1.0, 24)
    raw[40:52] += distortion.offset_absolute * rng.uniform(-1.0, 1.0, 12)
    raw[52:56] *= 1.0 + distortion.damping_relative * rng.uniform(-1.0, 1.0, 4)

    signs = rng.choice(np.array([-1.0, 1.0]), size=4)
    magnitudes = rng.uniform(0.0, 1.0, 4)
    torque_gain = np.ones(4)
    if distortion.torque_gain_relative > MIN_TORQUE_GAIN_DEVIATION:
        deviation = MIN_TORQUE_GAIN_DEVIATION + magnitudes * (
            distortion.torque_gain_relative - MIN_TORQUE_GAIN_DEVIATION)
        torque_gain = 1.0 + signs * deviation

    return DistortionSample(
        distortion=distortion,
        raw=raw,
        torque_gain=torque_gain,
        angle_bias=distortion.angle_bias * rng.uniform(-1.0, 1.0, 4),
        velocity_bias=distortion.velocity_bias * rng.uniform(-1.0, 1.0, 4),
        slope_bias=float(distortion.slope_bias * rng.uniform(-1.0, 1.0)),
        direction_bias=float(distortion.direction_bias * rng.uniform(-1.0, 1.0)),
        yaw_bias=float(distortion.yaw_bias * rng.uniform(-1.0, 1.0)),
        yaw_rate_bias=float(distortion.yaw_rate_bias * rng.uniform(-1.0, 1.0)),
        base_acceleration_bias=distortion.base_acceleration_bias
        * rng.uniform(-1.0, 1.0, 3),
    )


def _measure(scenario: scenarios.Scenario, trial: DistortionSample,
             angle_true: np.ndarray, velocity_true: np.ndarray, index: int,
             rng: np.random.Generator) -> dict:
    """带噪声与偏移的量测，底盘状态取与观测同一时刻的真实值。"""
    distortion = trial.distortion
    base_true = scenario.base_state(index)
    acceleration = np.array([base_true.acceleration[i] for i in range(3)])
    # 噪声一律先抽标准正态再乘标准差，标准差不改变随机数的消耗数量
    terrain = gd.make_terrain(
        scenario.slope + trial.slope_bias
        + distortion.slope_noise * rng.normal(0.0, 1.0),
        scenario.direction + trial.direction_bias
        + distortion.direction_noise * rng.normal(0.0, 1.0))
    base = gd.make_base(
        acceleration=acceleration + trial.base_acceleration_bias
        + distortion.base_acceleration_noise * rng.normal(0.0, 1.0, 3),
        yaw=base_true.yaw + trial.yaw_bias
        + distortion.yaw_noise * rng.normal(0.0, 1.0),
        yaw_rate=base_true.yaw_rate + trial.yaw_rate_bias
        + distortion.yaw_rate_noise * rng.normal(0.0, 1.0),
        yaw_acceleration=base_true.yaw_acceleration
        + distortion.yaw_acceleration_noise * rng.normal(0.0, 1.0))
    return {
        "angle": angle_true + trial.angle_bias
        + distortion.angle_noise * rng.normal(0.0, 1.0, 4),
        "velocity": velocity_true + trial.velocity_bias
        + distortion.velocity_noise * rng.normal(0.0, 1.0, 4),
        "terrain": terrain,
        "base": base,
    }


def run_closed_loop(scenario: scenarios.Scenario, controller, trial: DistortionSample,
                    context: gd.Context, torque_limit=TORQUE_LIMIT,
                    noise_seed: int = 0, delay_mode: str = "random") -> dict:
    """在失真条件下闭环仿真，记录指令力矩、实际力矩、真实状态与时延。

    delay_mode 为 "random" 时控制与观测时延逐拍独立抽取，"episode" 时每个回合各抽一个
    固定时延并保持整个回合。
    """
    if delay_mode not in ("random", "episode"):
        raise ValueError(f"未知的时延模式: {delay_mode}")
    limit = np.asarray(torque_limit, dtype=float)
    steps = scenario.steps
    horizon = int(getattr(controller, "horizon", 1))
    distortion = trial.distortion
    rng = np.random.default_rng(noise_seed)
    plant_model = scenarios.build_model(trial.raw)

    history_angle = np.zeros((steps + 1, 4))
    history_velocity = np.zeros((steps + 1, 4))
    history_angle[0] = scenario.initial_angle
    history_velocity[0] = scenario.initial_velocity
    command_history = np.zeros((steps, 4))
    applied_history = np.zeros((steps, 4))
    hit_history = np.zeros(steps, dtype=bool)
    solve_time = np.zeros(steps)
    iterations = np.zeros(steps, dtype=int)
    evaluations = np.zeros(steps, dtype=int)
    control_delay = np.zeros(steps, dtype=int)
    observation_delay = np.zeros(steps, dtype=int)

    # 未到达的指令：(发出拍号, 到达拍号, 力矩)。同一拍到达多拍指令时取最新发出的那条
    pending: list[tuple[int, int, np.ndarray]] = []
    held_command = np.zeros(4)
    state = gd.make_state(scenario.initial_angle, scenario.initial_velocity)
    episode_control_delay = int(rng.random() * (distortion.control_delay_steps + 1))
    episode_observation_delay = int(rng.random()
                                    * (distortion.observation_delay_steps + 1))

    for index in range(steps):
        if delay_mode == "random":
            observation_lag = int(rng.random()
                                  * (distortion.observation_delay_steps + 1))
        else:
            observation_lag = episode_observation_delay
        observed_index = max(0, index - observation_lag)
        measured = _measure(scenario, trial, history_angle[observed_index],
                            history_velocity[observed_index], observed_index, rng)
        target_angle = scenario.window(index, horizon)
        command = np.asarray(controller.command(
            measured["angle"], measured["velocity"], target_angle,
            measured["terrain"], measured["base"]), dtype=float)
        command = np.clip(command, -limit, limit)
        command_history[index] = command

        if delay_mode == "random":
            delay = int(rng.random() * (distortion.control_delay_steps + 1))
        else:
            delay = episode_control_delay
        control_delay[index] = delay
        observation_delay[index] = observation_lag
        pending.append((index, index + delay, command))
        arrived = [item for item in pending if item[1] <= index]
        if arrived:
            newest = max(arrived, key=lambda item: item[0])
            held_command = newest[2]
            pending = [item for item in pending if item[0] > newest[0]]
        else:
            pending = [item for item in pending if item[1] > index]

        applied = trial.torque_gain * held_command
        applied_history[index] = applied
        info = getattr(controller, "last_info", {})
        solve_time[index] = float(info.get("solve_time", 0.0))
        iterations[index] = int(info.get("iterations", 0))
        evaluations[index] = int(info.get("evaluations", 0))

        state = context.forward_step(plant_model, scenario.terrain,
                                     scenario.base_state(index), state, applied,
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
        history_angle[index + 1] = next_angle
        history_velocity[index + 1] = next_velocity

    return {
        "angle": history_angle,
        "velocity": history_velocity,
        "reference": scenario.reference_angle,
        "command": command_history,
        "applied": applied_history,
        "hits": hit_history,
        "solve_time": solve_time,
        "iterations": iterations,
        "evaluations": evaluations,
        "control_delay": control_delay,
        "observation_delay": observation_delay,
    }


def _high_frequency_ratio(signal: np.ndarray, window: int = 5) -> float:
    """信号相对其滑动平均的高频分量均方根与信号均方根之比。

    参考轨迹的频率不高于 1.2 Hz，5 点滑动平均（50 ms）几乎不改变参考驱动的运动，
    比值因此主要反映闭环抖动。
    """
    kernel = np.ones(window) / float(window)
    smoothed = np.stack([
        np.convolve(signal[:, joint], kernel, mode="same")
        for joint in range(signal.shape[1])], axis=1)
    difference = signal[window:-window] - smoothed[window:-window]
    reference = signal[window:-window]
    numerator = float(np.mean(np.sqrt(np.mean(difference ** 2, axis=0))))
    denominator = float(np.mean(np.sqrt(np.mean(reference ** 2, axis=0)))) + 1.0e-3
    return numerator / denominator


def _spectral_ratio(signal: np.ndarray, cutoff: float = 3.0) -> float:
    """信号中频率高于 cutoff Hz 的功率占比，用于识别持续的极限环震荡。"""
    centered = signal - np.mean(signal, axis=0)
    frequencies = np.fft.rfftfreq(signal.shape[0], DT)
    power = np.abs(np.fft.rfft(centered, axis=0)) ** 2
    total = float(np.sum(power))
    if total <= 0.0:
        return 0.0
    return float(np.sum(power[frequencies > cutoff]) / total)


def dominant_frequency(signal: np.ndarray) -> float:
    """信号功率最大的频率，单位 Hz。"""
    centered = signal - np.mean(signal, axis=0)
    frequencies = np.fft.rfftfreq(signal.shape[0], DT)
    power = np.sum(np.abs(np.fft.rfft(centered, axis=0)) ** 2, axis=1)
    return float(frequencies[int(np.argmax(power[1:]) + 1)])


def compute_metrics(records: dict, scenario: scenarios.Scenario) -> dict:
    """跟踪、抖动、撞限位与发散指标。角度取被控对象到达后的状态，与同一时刻参考比较。"""
    steps = scenario.steps
    angle = records["angle"][:-1]
    error = scenario.reference_angle - angle
    rmse = np.zeros(4)
    in_limit = np.zeros_like(error, dtype=bool)
    for joint in range(4):
        if LIMITED[joint]:
            in_limit[:, joint] = (
                (scenario.reference_angle[:, joint] >= LOWER[joint])
                & (scenario.reference_angle[:, joint] <= UPPER[joint]))
        else:
            in_limit[:, joint] = True
        mask = in_limit[:, joint]
        rmse[joint] = (float(np.sqrt(np.mean(error[mask, joint] ** 2)))
                       if np.any(mask) else float("nan"))

    command = records["command"]
    velocity = records["velocity"][:-1]
    changes = 0
    for joint in range(4):
        value = velocity[:, joint]
        changes += int(np.sum(value[1:] * value[:-1] < 0.0))
    sign_rate = changes / 4.0 / (steps * DT)
    delta = np.abs(np.diff(command, axis=0))
    churn = float(np.mean(np.mean(delta, axis=0)
                          / (np.mean(np.abs(command), axis=0) + 0.05)))
    # 指令贴到力矩上限的拍数占比，持续贴限是极限环的直接特征
    saturation = float(np.mean(np.any(
        np.abs(command) >= 0.98 * np.asarray(TORQUE_LIMIT), axis=1)))

    # 尾段没有随机阶跃激励，残留的误差与抖动反映是否真的收敛
    tail_start = int(steps * (1.0 - TAIL_FRACTION))
    tail_error = error[tail_start:]
    tail_mask = in_limit[tail_start:]
    tail_rmse_axes = np.zeros(4)
    for joint in range(4):
        mask = tail_mask[:, joint]
        tail_rmse_axes[joint] = (
            float(np.sqrt(np.mean(tail_error[mask, joint] ** 2)))
            if np.any(mask) else float("nan"))
    valid_tail = tail_rmse_axes[~np.isnan(tail_rmse_axes)]
    tail_rmse = (float(np.mean(valid_tail)) if valid_tail.size
                 else float(np.sqrt(np.mean(tail_error ** 2))))
    tail_max_error = (float(np.max(np.abs(tail_error[tail_mask])))
                      if np.any(tail_mask)
                      else float(np.max(np.abs(tail_error))))
    tail_command = command[tail_start:]
    tail_delta = np.abs(np.diff(tail_command, axis=0))
    tail_churn = float(np.mean(np.mean(tail_delta, axis=0)
                               / (np.mean(np.abs(tail_command), axis=0) + 0.05)))
    tail_saturation = float(np.mean(np.any(
        np.abs(tail_command) >= 0.98 * np.asarray(TORQUE_LIMIT), axis=1)))

    early = float(np.mean(np.abs(error[:max(1, steps // 5)])))
    late = float(np.mean(np.abs(error[-max(1, steps // 5):])))
    hit_steps = int(records["hits"].sum())
    finite = bool(np.all(np.isfinite(records["angle"]))
                  and np.all(np.isfinite(command)))
    diverged = (not finite) or hit_steps > 0.5 * steps
    diverged = diverged or float(np.max(np.abs(records["angle"]))) > 4.0
    diverged = diverged or (late > 0.25 and late > 2.5 * max(early, 1.0e-3))

    return {
        "rmse": rmse.tolist(),
        "mean_rmse": float(np.mean(rmse)),
        "worst_axis_rmse": float(np.nanmax(rmse)),
        "max_error": float(np.max(np.abs(error[in_limit]))),
        "hit_steps": hit_steps,
        "hit_fraction": hit_steps / steps,
        "torque_square": float(np.mean(command ** 2)),
        "torque_max": float(np.max(np.abs(command))),
        "torque_axis_max": np.max(np.abs(command), axis=0).tolist(),
        "applied_axis_max": np.max(np.abs(records["applied"]), axis=0).tolist(),
        "velocity_sign_rate": sign_rate,
        "command_churn": churn,
        "saturation_fraction": saturation,
        "tail_rmse": tail_rmse,
        "tail_rmse_axis": tail_rmse_axes.tolist(),
        "tail_max_error": tail_max_error,
        "tail_saturation": tail_saturation,
        "tail_churn": tail_churn,
        "velocity_hf_ratio": _high_frequency_ratio(velocity),
        "command_hf_ratio": _high_frequency_ratio(command),
        "velocity_hf_power": _spectral_ratio(velocity),
        "error_hf_power": _spectral_ratio(error),
        "velocity_dominant_hz": dominant_frequency(velocity),
        "error_dominant_hz": dominant_frequency(error),
        "angle_abs_max": float(np.max(np.abs(records["angle"]))),
        "velocity_abs_max": float(np.max(np.abs(records["velocity"]))),
        "late_error": late,
        "error_growth": late / max(early, 1.0e-3),
        "solve_ms": float(1000.0 * np.mean(records["solve_time"])),
        "solve_ms_max": float(1000.0 * np.max(records["solve_time"])),
        "iterations": float(np.mean(records["iterations"])),
        "control_delay_mean": float(np.mean(records["control_delay"])),
        "observation_delay_mean": float(np.mean(records["observation_delay"])),
        "diverged": bool(diverged),
    }


@dataclass(frozen=True)
class Criteria:
    """鲁棒性判定阈值。

    跟踪误差取限位内 RMSE；震荡用指令贴限占比与指令变化率判定，持续贴到力矩上限是
    极限环最直接的证据。
    """

    mean_rmse: float = 0.10
    worst_rmse: float = 0.15
    max_error: float = 0.90
    tail_rmse: float = 0.08
    tail_max_error: float = 0.30
    tail_saturation: float = 0.02
    hit_fraction: float = 0.01
    command_churn: float = 0.40


def summarize(trials: list[dict]) -> dict:
    """汇总同一控制器、同一失真条件下的全部回合。"""
    return {
        "mean_rmse": float(np.mean([item["mean_rmse"] for item in trials])),
        "worst_rmse": float(np.max([item["mean_rmse"] for item in trials])),
        "worst_axis_rmse": float(np.max([item["worst_axis_rmse"] for item in trials])),
        "max_error": float(np.max([item["max_error"] for item in trials])),
        "hit_fraction": float(np.mean([item["hit_fraction"] for item in trials])),
        "worst_hit_fraction": float(np.max([item["hit_fraction"] for item in trials])),
        "velocity_sign_rate": float(np.max([item["velocity_sign_rate"] for item in trials])),
        "command_churn": float(np.max([item["command_churn"] for item in trials])),
        "saturation_fraction": float(np.max(
            [item["saturation_fraction"] for item in trials])),
        "tail_rmse": float(np.mean([item["tail_rmse"] for item in trials])),
        "tail_worst_rmse": float(np.max([item["tail_rmse"] for item in trials])),
        "tail_max_error": float(np.max([item["tail_max_error"] for item in trials])),
        "tail_saturation": float(np.max([item["tail_saturation"] for item in trials])),
        "tail_churn": float(np.max([item["tail_churn"] for item in trials])),
        "velocity_hf_ratio": float(np.max([item["velocity_hf_ratio"] for item in trials])),
        "command_hf_ratio": float(np.max([item["command_hf_ratio"] for item in trials])),
        "velocity_hf_power": float(np.max([item["velocity_hf_power"] for item in trials])),
        "error_hf_power": float(np.max([item["error_hf_power"] for item in trials])),
        "velocity_dominant_hz": float(np.mean(
            [item["velocity_dominant_hz"] for item in trials])),
        "error_dominant_hz": float(np.mean(
            [item["error_dominant_hz"] for item in trials])),
        "angle_abs_max": float(np.max([item["angle_abs_max"] for item in trials])),
        "velocity_abs_max": float(np.max([item["velocity_abs_max"] for item in trials])),
        "error_growth": float(np.max([item["error_growth"] for item in trials])),
        "solve_ms": float(np.mean([item["solve_ms"] for item in trials])),
        "solve_ms_max": float(np.max([item["solve_ms_max"] for item in trials])),
        "iterations": float(np.mean([item["iterations"] for item in trials])),
        "control_delay_mean": float(np.mean([item["control_delay_mean"] for item in trials])),
        "observation_delay_mean": float(
            np.mean([item["observation_delay_mean"] for item in trials])),
        "diverged_trials": int(np.sum([item["diverged"] for item in trials])),
        "trials": len(trials),
    }


def verdict(summary: dict, criteria: Criteria) -> list[str]:
    """返回未通过的原因列表，空列表表示通过。"""
    reasons = []
    if summary["diverged_trials"] > 0:
        reasons.append("发散 %d/%d 回合" % (summary["diverged_trials"], summary["trials"]))
    if summary["mean_rmse"] > criteria.mean_rmse:
        reasons.append("平均 RMSE 超限")
    if summary["worst_rmse"] > criteria.worst_rmse:
        reasons.append("最差 RMSE 超限")
    if summary["max_error"] > criteria.max_error:
        reasons.append("最大误差超限")
    if summary["tail_rmse"] > criteria.tail_rmse:
        reasons.append("尾段 RMSE 超限")
    if summary["tail_max_error"] > criteria.tail_max_error:
        reasons.append("尾段最大误差超限")
    if summary["hit_fraction"] > criteria.hit_fraction:
        reasons.append("撞限位比例超限")
    if summary["tail_saturation"] > criteria.tail_saturation:
        reasons.append("尾段指令贴限占比超限")
    if summary["command_churn"] > criteria.command_churn:
        reasons.append("指令变化率超限")
    return reasons


# ---------------- 批量评估与报告 ----------------

DEFAULT_SEEDS = (7, 11, 23, 37, 51, 73, 91, 113, 137, 151, 173, 199)
DEFAULT_FACTORS = (0.0, 0.5, 1.0, 1.5, 2.0)
SWEEP_GROUPS = tuple(DISTORTION_GROUPS) + ("combined",)
CONTROLLERS = ("current", "robust")

_WORKER_CONTEXT = None
_WORKER_CONTROLLERS: dict = {}
_WORKER_THREADS = 4


def _worker_init(thread_count: int) -> None:
    global _WORKER_CONTEXT, _WORKER_CONTROLLERS, _WORKER_THREADS
    _WORKER_CONTEXT = gd.Context(int(thread_count))
    _WORKER_CONTROLLERS = {}
    _WORKER_THREADS = int(thread_count)


def _worker_controller(name: str) -> MpcController:
    controller = _WORKER_CONTROLLERS.get(name)
    if controller is None:
        controller = MpcController(scenarios.design_model(),
                                   torque_limit=TORQUE_LIMIT,
                                   thread_count=_WORKER_THREADS,
                                   **controller_parameters(name))
        _WORKER_CONTROLLERS[name] = controller
    return controller


def _task(controller: str, section: str, label: str, factor: float,
          distortion: Distortion, seed: int, steps: int, offset: int) -> dict:
    trial_seed = (seed * 1000003 + offset * 7919
                  + int(round(factor * 1000)) * 31 + 17)
    return {
        "controller": controller,
        "section": section,
        "label": label,
        "factor": float(factor),
        "distortion": distortion,
        "seed": int(seed),
        "steps": int(steps),
        "trial_seed": trial_seed,
        "noise_seed": trial_seed + 500000,
        "delay_mode": "random",
    }


def make_scenario(seed: int, steps: int) -> scenarios.Scenario:
    """鲁棒性实验场景：参考阶跃限幅为斜坡并集中在前 60% 的时间。"""
    return scenarios.Scenario(seed=seed, steps=steps,
                              step_fraction=SCENARIO_STEP_FRACTION,
                              step_rate=SCENARIO_STEP_RATE)


def _worker_run(task: dict) -> dict:
    controller = _worker_controller(task["controller"])
    controller.reset()
    scenario = make_scenario(task["seed"], task["steps"])
    trial = sample_distortion(np.random.default_rng(task["trial_seed"]),
                              task["distortion"])
    records = run_closed_loop(scenario, controller, trial, _WORKER_CONTEXT,
                              noise_seed=task["noise_seed"],
                              delay_mode=task["delay_mode"])
    metrics = compute_metrics(records, scenario)
    metrics.update({
        "controller": task["controller"],
        "section": task["section"],
        "label": task["label"],
        "factor": task["factor"],
        "seed": task["seed"],
    })
    return metrics


def _run_tasks(tasks: list[dict], workers: int, threads: int) -> list[dict]:
    pool = multiprocessing.get_context("fork").Pool(
        int(workers), initializer=_worker_init, initargs=(int(threads),))
    results = []
    total = len(tasks)
    with pool:
        for count, metrics in enumerate(
                pool.imap_unordered(_worker_run, tasks, chunksize=1), 1):
            results.append(metrics)
            if count % 20 == 0 or count == total:
                print("完成 %d/%d" % (count, total), flush=True)
    return results


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _pass_edge(sweeps: dict, group: str, factors, controller: str) -> float:
    """该组失真从 0 倍开始连续通过的最后一个倍数。"""
    edge = 0.0
    for factor in factors:
        if sweeps["%s x%.2f" % (group, factor)][controller]["passed"]:
            edge = factor
        else:
            break
    return edge


def _template_report(results: dict, criteria: Criteria, seeds, sweep_seeds,
                     factors) -> str:
    lines = [
        "# MPC 参数失真鲁棒性实验",
        "",
        "## 1 失真定义",
        "",
        "被控对象参数、量测通道、执行增益与时延同时失真，控制器始终使用标称设计参数模型。",
        "参考轨迹的随机阶跃按实车目标移动速度 params.PID_TARGET_RATE（%.1f rad/s）限幅为斜坡，"
        "并且只出现在回合前 %.0f%% 的时间，后 %.0f%% 作为无阶跃激励的收敛观察段。"
        % (SCENARIO_STEP_RATE, 100.0 * SCENARIO_STEP_FRACTION,
           100.0 * TAIL_FRACTION),
        "",
        "| 失真 | 范围 |",
        "|---|---|",
        "| 质量 | 每刚体 xU(0.85, 1.15) |",
        "| 惯量 | 每个分量 xU(0.85, 1.15) |",
        "| 质心 | 每个分量 ±10 mm |",
        "| 关节安装偏置 | 每个分量 ±8 mm |",
        "| 粘性阻尼 | xU(0.5, 1.5) |",
        "| 执行增益 | 每轴 x(1±U(0.02, 0.15))，正增益且不等于 1 |",
        "| 关节角噪声与偏移 | σ=0.008 rad，偏移 ±0.02 rad |",
        "| 关节角速度噪声与偏移 | σ=0.10 rad/s，偏移 ±0.05 rad/s |",
        "| 坡度噪声与偏移 | σ=1.5°，偏移 ±3° |",
        "| 坡向噪声与偏移 | σ=4°，偏移 ±10° |",
        "| 底盘 yaw 噪声与偏移 | σ=4°，偏移 ±10° |",
        "| 底盘 yaw 角速度噪声与偏移 | σ=0.10 rad/s，偏移 ±0.10 rad/s |",
        "| 底盘加速度噪声与偏移 | σ=0.05 m/s²，偏移 ±0.05 m/s² |",
        "| 控制时延 | 每拍随机 0~2 拍（0~20 ms） |",
        "| 观测时延 | 每拍随机 0~2 拍（0~20 ms） |",
        "| 参考目标阶跃 | 按 %.1f rad/s 斜率限幅，仅出现在前 %.0f%% 时间 |"
        % (SCENARIO_STEP_RATE, 100.0 * SCENARIO_STEP_FRACTION),
        "",
        "## 2 判定标准",
        "",
        "| 指标 | 阈值 |",
        "|---|---|",
        "| 平均限位内 RMSE | ≤ %.2f rad |" % criteria.mean_rmse,
        "| 最差回合平均 RMSE | ≤ %.2f rad |" % criteria.worst_rmse,
        "| 最大限位内误差 | ≤ %.2f rad |" % criteria.max_error,
        "| 尾段（后 %.0f%%）RMSE | ≤ %.2f rad |" % (
            100.0 * TAIL_FRACTION, criteria.tail_rmse),
        "| 尾段最大误差 | ≤ %.2f rad |" % criteria.tail_max_error,
        "| 撞硬限位步数占比 | ≤ %.1f%% |" % (100.0 * criteria.hit_fraction),
        "| 尾段指令贴力矩上限步数占比 | ≤ %.1f%% |" % (
            100.0 * criteria.tail_saturation),
        "| 指令变化率 | ≤ %.2f |" % criteria.command_churn,
        "| 发散 | 不允许 |",
        "",
        "回合数：条件对比 %d 个种子，包络扫描 %d 个种子，每个回合 %d 步（%.1f s）。"
        % (len(seeds), len(sweep_seeds), results["steps"],
           results["steps"] * DT),
        "",
        "## 3 控制器参数",
        "",
        "| 参数 | current | robust |",
        "|---|---|---|",
    ]
    current = results["parameters"]["current"]
    robust = results["parameters"]["robust"]
    lines.append("| 预测步数 | %d | %d |" % (current["horizon"], robust["horizon"]))
    lines.append("| 迭代上限 | %d | %d |" % (current["max_iter"], robust["max_iter"]))
    lines.append("| 力矩权重 | %s | %s |" % (
        np.array2string(np.asarray(current["weights"]["torque_weights"]), precision=3),
        np.array2string(np.asarray(robust["weights"]["torque_weights"]), precision=3)))
    lines.append("| 力矩变化率权重 | %s | %s |" % (
        np.array2string(np.asarray(current["weights"]["torque_rate_weights"]),
                        precision=3),
        np.array2string(np.asarray(robust["weights"]["torque_rate_weights"]),
                        precision=3)))
    lines.append("| 速度权重 | %s | %s |" % (
        np.array2string(np.asarray(current["weights"]["velocity_weights"]), precision=4),
        np.array2string(np.asarray(robust["weights"]["velocity_weights"]), precision=4)))
    lines.extend([
        "",
        "## 4 单项与全失真条件",
        "",
        "| 条件 | 控制器 | 平均 RMSE | 最差 RMSE | 最大误差 | 尾段 RMSE | 尾段最大误差 | 撞限位 | 贴限占比 | 指令变化率 | 求解 ms | 判定 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ])
    for label, item in results["conditions"].items():
        for controller in CONTROLLERS:
            value = item[controller]
            state = ("通过" if value["passed"]
                     else "未通过：" + "、".join(value["reasons"]))
            lines.append(
                "| %s | %s | %.4f | %.4f | %.3f | %.4f | %.3f | %.2f%% | %.2f%% "
                "| %.3f | %.2f | %s |"
                % (label, controller, value["mean_rmse"], value["worst_rmse"],
                   value["max_error"], value["tail_rmse"], value["tail_max_error"],
                   100.0 * value["hit_fraction"],
                   100.0 * value["saturation_fraction"], value["command_churn"],
                   value["solve_ms"], state))
    lines.extend([
        "",
        "## 5 包络倍数扫描",
        "",
        "表中倍数为该组失真范围相对设计包络的倍数，0 表示只有标称量测噪声与偏移。",
        "",
    ])
    for group in SWEEP_GROUPS:
        lines.append("### %s" % group)
        lines.append("")
        lines.append("| 倍数 | current RMSE | current 判定 | robust RMSE | robust 判定 |")
        lines.append("|---|---|---|---|---|")
        for factor in factors:
            label = "%s x%.2f" % (group, factor)
            item = results["sweeps"][label]
            current_value = item["current"]
            robust_value = item["robust"]
            lines.append("| %.2f | %.4f | %s | %.4f | %s |" % (
                factor, current_value["mean_rmse"],
                "通过" if current_value["passed"] else "未通过",
                robust_value["mean_rmse"],
                "通过" if robust_value["passed"] else "未通过"))
        lines.append("")
    lines.extend(["## 6 结论", ""])
    condition_labels = [label for label in results["conditions"] if label != "nominal"]
    current_failed = [label for label in condition_labels
                      if not results["conditions"][label]["current"]["passed"]]
    robust_failed = [label for label in condition_labels
                     if not results["conditions"][label]["robust"]["passed"]]
    if current_failed:
        lines.append("current 参数在以下条件未通过：%s。" % "、".join(current_failed))
    else:
        lines.append("current 参数在所有单项与全失真条件下通过。")
    if robust_failed:
        lines.append("robust 参数在以下条件未通过：%s。" % "、".join(robust_failed))
    else:
        lines.append("robust 参数在所有单项与全失真条件下通过。")
    combined = results["conditions"]["combined"]
    lines.extend([
        "",
        "全失真条件下 current 平均 RMSE 为 %.4f rad、尾段 RMSE %.4f rad、"
        "最大误差 %.3f rad、指令贴限占比 %.1f%%；robust 平均 RMSE 为 %.4f rad、"
        "尾段 RMSE %.4f rad、最大误差 %.3f rad、指令贴限占比 %.1f%%。" % (
            combined["current"]["mean_rmse"],
            combined["current"]["tail_rmse"],
            combined["current"]["max_error"],
            100.0 * combined["current"]["saturation_fraction"],
            combined["robust"]["mean_rmse"],
            combined["robust"]["tail_rmse"],
            combined["robust"]["max_error"],
            100.0 * combined["robust"]["saturation_fraction"]),
        "",
        "包络扫描给出容忍边界：",
        "",
    ])
    for group in SWEEP_GROUPS:
        current_edge = _pass_edge(results["sweeps"], group, factors, "current")
        robust_edge = _pass_edge(results["sweeps"], group, factors, "robust")
        lines.append("- %s：current 通过到 %.2f 倍，robust 通过到 %.2f 倍。"
                     % (group, current_edge, robust_edge))
    lines.extend([
        "",
        "图：combined_tracking.png 为全失真代表性回合的跟踪与力矩，"
        "conditions_metrics.png 为条件指标对比，envelope.png 为包络扫描。",
        "",
    ])
    return "\n".join(lines)


def _plot_conditions(results: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(results["conditions"])
    specs = [("mean in-limit RMSE (rad)", "mean_rmse"),
             ("max in-limit error (rad)", "max_error"),
             ("torque saturation fraction", "saturation_fraction"),
             ("command churn", "command_churn")]
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 7.5))
    positions = np.arange(len(labels))
    width = 0.36
    for axis, (title, key) in zip(axes.reshape(-1), specs):
        for offset, controller in enumerate(CONTROLLERS):
            values = [results["conditions"][label][controller][key]
                      for label in labels]
            axis.bar(positions + (offset - 0.5) * width, values, width,
                     label=controller)
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=25, fontsize=8)
        axis.set_title(title, fontsize=9)
        axis.grid(alpha=0.3, axis="y")
        axis.legend(fontsize=8)
    figure.suptitle("distortion condition comparison")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _plot_sweeps(results: dict, factors, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tolerance = Criteria().mean_rmse
    figure, axes = plt.subplots(2, 3, figsize=(14.0, 7.0), sharex=True, sharey=True)
    for axis, group in zip(axes.reshape(-1), SWEEP_GROUPS):
        for controller in CONTROLLERS:
            values = [results["sweeps"]["%s x%.2f" % (group, factor)][controller]["mean_rmse"]
                      for factor in factors]
            axis.plot(factors, values, marker="o", linewidth=1.2, label=controller)
        axis.axhline(tolerance, color="tab:red", linestyle="--", linewidth=0.9)
        axis.set_title(group + " distortion")
        axis.grid(alpha=0.3)
        axis.set_xlabel("envelope factor")
    axes[0, 0].set_ylabel("mean in-limit RMSE (rad)")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("tolerance boundary against distortion magnitude")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _trace_records(controller_name: str, seed: int, steps: int, threads: int,
                   condition_index: int):
    context = gd.Context(int(threads))
    controller = MpcController(scenarios.design_model(),
                               torque_limit=TORQUE_LIMIT,
                               thread_count=int(threads),
                               **controller_parameters(controller_name))
    scenario = make_scenario(seed, steps)
    trial = sample_distortion(
        np.random.default_rng(seed * 1000003 + condition_index * 7919 + 17),
        COMBINED_DISTORTION)
    controller.reset()
    records = run_closed_loop(scenario, controller, trial, context,
                              noise_seed=seed * 1000003 + condition_index * 7919 + 500017)
    controller.close()
    context.close()
    return scenario, records


def _plot_trajectories(seed: int, steps: int, threads: int, output: Path,
                       condition_index: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traces = {name: _trace_records(name, seed, steps, threads, condition_index)
              for name in CONTROLLERS}
    scenario = traces["current"][0]
    time = np.arange(steps) * DT
    time_state = np.arange(steps + 1) * DT
    limit = np.asarray(TORQUE_LIMIT)
    figure, axes = plt.subplots(4, 2, figsize=(14.0, 12.0), sharex=True)
    for joint in range(4):
        axis = axes[joint, 0]
        axis.plot(time, scenario.reference_angle[:, joint], "k:",
                  linewidth=1.2, label="reference")
        for name in CONTROLLERS:
            axis.plot(time_state, traces[name][1]["angle"][:, joint],
                      linewidth=1.1, label=name)
        axis.set_ylabel("angle (rad)")
        axis.set_title(AXIS_NAMES[joint])
        axis.grid(alpha=0.3)
        axis = axes[joint, 1]
        for name in CONTROLLERS:
            axis.plot(time, traces[name][1]["command"][:, joint],
                      linewidth=1.0, label=name)
        axis.axhline(limit[joint], color="tab:red", linewidth=0.8)
        axis.axhline(-limit[joint], color="tab:red", linewidth=0.8)
        axis.set_ylabel("torque (N·m)")
        axis.set_title(AXIS_NAMES[joint] + " torque")
        axis.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("time (s)")
    figure.suptitle("combined distortion, seed %d" % seed)
    figure.tight_layout()
    figure.savefig(output / "combined_tracking.png", dpi=140)
    plt.close(figure)

    records = traces["current"][1]
    figure, axes = plt.subplots(2, 1, figsize=(12.0, 5.0), sharex=True)
    axes[0].plot(time, records["control_delay"], linewidth=0.9, label="control delay")
    axes[0].plot(time, records["observation_delay"], linewidth=0.9,
                 label="observation delay")
    axes[0].set_ylabel("delay (steps)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].plot(time_state,
                 np.max(np.abs(records["velocity"]), axis=1))
    axes[1].set_ylabel("max |velocity| (rad/s)")
    axes[1].set_xlabel("time (s)")
    axes[1].grid(alpha=0.3)
    figure.suptitle("delay realization and current controller velocity magnitude")
    figure.tight_layout()
    figure.savefig(output / "combined_delay.png", dpi=140)
    plt.close(figure)


def run_experiment(seeds, steps: int, sweep_seeds, factors, workers: int,
                   threads: int, output: Path) -> dict:
    conditions = distortion_conditions()
    sweep_items = []
    for group in SWEEP_GROUPS:
        for factor in factors:
            distortion = (scaled_combined(factor) if group == "combined"
                          else scaled_group(group, factor))
            sweep_items.append((group, float(factor), distortion))
    labels = list(conditions)
    labels.extend("%s x%.2f" % (group, factor)
                  for group, factor, _ in sweep_items)
    offsets = {label: index for index, label in enumerate(labels)}
    tasks = []
    for label, distortion in conditions.items():
        for seed in seeds:
            for controller in CONTROLLERS:
                tasks.append(_task(controller, "condition", label, 0.0, distortion,
                                   seed, steps, offsets[label]))
    for group, factor, distortion in sweep_items:
        label = "%s x%.2f" % (group, factor)
        for seed in sweep_seeds:
            for controller in CONTROLLERS:
                tasks.append(_task(controller, "sweep", label, factor, distortion,
                                   seed, steps, offsets[label]))
    print("任务总数 %d，工作进程 %d，每进程线程 %d"
          % (len(tasks), workers, threads), flush=True)
    metrics = _run_tasks(tasks, workers, threads)
    criteria = Criteria()
    grouped = {}
    for item in metrics:
        key = (item["section"], item["label"], item["controller"])
        grouped.setdefault(key, []).append(item)
    results = {
        "criteria": asdict(criteria),
        "seeds": list(seeds),
        "sweep_seeds": list(sweep_seeds),
        "factors": list(factors),
        "steps": int(steps),
        "sweep_groups": list(SWEEP_GROUPS),
        "parameters": {name: controller_parameters(name) for name in CONTROLLERS},
        "conditions": {},
        "sweeps": {},
    }
    for (section, label, controller), trials in grouped.items():
        summary = summarize(trials)
        summary["reasons"] = verdict(summary, criteria)
        summary["passed"] = not summary["reasons"]
        container = results["conditions"] if section == "condition" else results["sweeps"]
        container.setdefault(label, {})[controller] = summary
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(_jsonable(results), ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_conditions(results, output / "conditions_metrics.png")
    _plot_sweeps(results, factors, output / "envelope.png")
    condition_index = list(conditions).index("combined")
    _plot_trajectories(int(seeds[0]), steps, threads, output, condition_index)
    report = _template_report(results, criteria, seeds, sweep_seeds, factors)
    (output / "report.md").write_text(report, encoding="utf-8")
    print("%-16s %-8s %8s %8s %8s %8s %8s" % (
        "condition", "ctrl", "rmse", "maxerr", "sat", "churn", "verdict"))
    for label, item in results["conditions"].items():
        for controller in CONTROLLERS:
            value = item[controller]
            print("%-16s %-8s %8.4f %8.4f %8.4f %8.4f %s"
                  % (label, controller, value["mean_rmse"], value["max_error"],
                     value["saturation_fraction"], value["command_churn"],
                     "pass" if value["passed"] else "fail:" + ",".join(value["reasons"])))
    print("报告与图已写入", output)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="MPC 参数失真鲁棒性实验")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--sweep-seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--factors", default=",".join(
        "%g" % item for item in DEFAULT_FACTORS))
    parser.add_argument("--output", type=Path, default=Path("../data/mpc/robustness"))
    arguments = parser.parse_args()
    if arguments.seeds > len(DEFAULT_SEEDS) or arguments.sweep_seeds > len(DEFAULT_SEEDS):
        raise ValueError(f"种子数量不能超过 {len(DEFAULT_SEEDS)}")
    seeds = DEFAULT_SEEDS[:arguments.seeds]
    sweep_seeds = DEFAULT_SEEDS[:arguments.sweep_seeds]
    factors = tuple(float(item) for item in arguments.factors.split(","))
    run_experiment(seeds, arguments.steps, sweep_seeds, factors,
                   arguments.workers, arguments.threads, arguments.output)


if __name__ == "__main__":
    main()

