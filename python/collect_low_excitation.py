"""低激励采集协议：参数化的幅值、频率、坡度与增益，用于寻找满足精度要求的最小激励。

协议与 collect_closed_loop.py 一致：真实模型 + PID 前馈闭环，段内底盘静止，
每段 3 s、300 点、dt=0.01，随后按固定噪声标准加噪。所有关节目标为小幅多正弦，
另混入静态保持段（速度接近零，仍能提供重力方向信息）。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

import gimbal_dynamics as gd
import params
from gimbal_env import GimbalEnv

DT = 0.01
POINTS = 300
ROLL_LIMIT = (-45.0 * math.pi / 180.0, 5.0 * math.pi / 180.0)
SMALL_YAW_LIMIT = (-50.0 * math.pi / 180.0, 50.0 * math.pi / 180.0)
PITCH_LIMIT = (-45.0 * math.pi / 180.0, 45.0 * math.pi / 180.0)
LIMITS = (None, ROLL_LIMIT, SMALL_YAW_LIMIT, PITCH_LIMIT)
ROLL_CENTER = -0.18


def multisine(rng, center, amplitude, frequency_low, frequency_high, modes=2):
    time = np.arange(POINTS) * DT
    angle = np.full(POINTS, center)
    velocity = np.zeros(POINTS)
    normalizer = 0.0
    components = []
    for _ in range(modes):
        frequency = rng.uniform(max(frequency_low, 0.05), max(frequency_high, frequency_low * 1.2))
        phase = rng.uniform(-math.pi, math.pi)
        weight = rng.uniform(0.5, 1.0)
        components.append((frequency, phase, weight))
        normalizer += weight
    for frequency, phase, weight in components:
        angle += amplitude * weight / normalizer * np.sin(frequency * time + phase)
        velocity += amplitude * weight / normalizer * frequency * np.cos(frequency * time + phase)
    return angle, velocity


AMPLITUDES = [0.35, 0.12, 0.15, 0.15]
FREQUENCIES = [0.15, 0.20, 0.20, 0.20]
BANDWIDTH = 0.40
SLOW_AMPLITUDES = None
SLOW_FREQUENCIES = None
FAST_AMPLITUDES = None
FAST_FREQUENCIES = None


def reference(rng, kind: str, amplitude_scale: float, frequency_scale: float,
              static_amplitude: float):
    angle = np.zeros((POINTS, 4))
    velocity = np.zeros((POINTS, 4))
    if kind == "static":
        for joint, center in ((0, 0.0), (1, ROLL_CENTER), (2, 0.0), (3, 0.0)):
            offset = rng.uniform(-1.0, 1.0) * static_amplitude
            angle[:, joint] = center + offset
            velocity[:, joint] = 0.0
        return angle, velocity
    time = np.arange(POINTS) * DT
    for joint in range(4):
        center = ROLL_CENTER if joint == 1 else 0.0
        if SLOW_AMPLITUDES is None:
            angle[:, joint], velocity[:, joint] = multisine(
                rng, center, AMPLITUDES[joint] * amplitude_scale,
                FREQUENCIES[joint] * frequency_scale,
                (FREQUENCIES[joint] + BANDWIDTH) * frequency_scale, 2)
            continue
        # 双频带：慢分量提供姿态行程，快分量提供加速度，二者幅值独立控制
        signal = np.full(POINTS, center)
        derivative = np.zeros(POINTS)
        phase_slow = rng.uniform(-math.pi, math.pi)
        signal += SLOW_AMPLITUDES[joint] * amplitude_scale * np.sin(
            SLOW_FREQUENCIES[joint] * frequency_scale * time + phase_slow)
        derivative += SLOW_AMPLITUDES[joint] * amplitude_scale * SLOW_FREQUENCIES[joint] * frequency_scale * np.cos(
            SLOW_FREQUENCIES[joint] * frequency_scale * time + phase_slow)
        for _ in range(2):
            frequency = FAST_FREQUENCIES[joint] * frequency_scale * rng.uniform(0.7, 1.3)
            phase = rng.uniform(-math.pi, math.pi)
            signal += 0.5 * FAST_AMPLITUDES[joint] * amplitude_scale * np.sin(
                frequency * time + phase)
            derivative += 0.5 * FAST_AMPLITUDES[joint] * amplitude_scale * frequency * np.cos(
                frequency * time + phase)
        angle[:, joint] = signal
        velocity[:, joint] = derivative
    return angle, velocity


def simulate_segment(rng, kind, amplitude_scale, frequency_scale, static_amplitude,
                     tilt_limit, gain_scale):
    env = GimbalEnv()
    slope = rng.uniform(0.0, tilt_limit)
    direction = rng.uniform(0.0, 2.0 * math.pi)
    yaw = rng.uniform(-math.pi, math.pi)
    terrain = gd.make_terrain(slope, direction)
    base = gd.make_base(yaw=yaw)
    target_q, target_v = reference(rng, kind, amplitude_scale, frequency_scale,
                                   static_amplitude)
    gains_kp = np.array([8.0, 2.0, 2.0, 2.0]) * gain_scale * rng.uniform(0.9, 1.1, 4)
    gains_ki = np.array([0.2, 0.2, 0.2, 0.2]) * gain_scale * rng.uniform(0.9, 1.1, 4)
    gains_kd = np.array([0.5, 0.2, 0.15, 0.1]) * gain_scale * rng.uniform(0.9, 1.1, 4)
    observation = env.reset(target_q[0], target_v[0], terrain=terrain, base=base)
    q = np.zeros((POINTS, 4))
    qdot = np.zeros((POINTS, 4))
    tau = np.zeros((POINTS, 4))
    integral = np.zeros(4)
    integral_clip = 1.0
    for k in range(POINTS):
        current = observation.angle
        velocity = observation.velocity
        q[k] = current
        qdot[k] = velocity
        error = target_q[k] - current
        integral = np.clip(integral + error * DT, -integral_clip, integral_clip)
        feedforward = env.bias_torque()
        command = np.clip(feedforward + gains_kp * error + gains_ki * integral
                          + gains_kd * (target_v[k] - velocity), -10.0, 10.0)
        tau[k] = command
        if k + 1 < POINTS:
            observation = env.step(command)
    env.close()
    for joint in (1, 2, 3):
        low, high = LIMITS[joint]
        if np.any(q[:, joint] < low) or np.any(q[:, joint] > high):
            return None
    return dict(q=q, qdot=qdot, tau=tau, target=target_q, target_velocity=target_v,
                terrain=np.array([slope, direction]), base_yaw=yaw,
                kp=gains_kp, ki=gains_ki, kd=gains_kd)


def collect(segments: int, seed: int, amplitude_scale: float, frequency_scale: float,
            static_amplitude: float, tilt_limit: float, gain_scale: float,
            static_fraction: float, output_clean: Path, output_noisy: Path, noise_seed: int):
    rng = np.random.default_rng(seed)
    keys = ("q", "qdot", "tau", "target", "target_velocity")
    storage = {key: np.zeros((segments, POINTS, 4)) for key in keys}
    terrain = np.zeros((segments, 2))
    base_yaw = np.zeros(segments)
    gains = {key: np.zeros((segments, 4)) for key in ("kp", "ki", "kd")}
    for segment in range(segments):
        kind = "static" if rng.uniform() < static_fraction else "multi"
        result = None
        for _ in range(6):
            result = simulate_segment(rng, kind, amplitude_scale, frequency_scale,
                                      static_amplitude, tilt_limit, gain_scale)
            if result is not None:
                break
            amplitude_scale *= 0.7
        if result is None:
            raise RuntimeError("第 %d 段无法满足限位" % segment)
        for key in keys:
            storage[key][segment] = result[key]
        terrain[segment] = result["terrain"]
        base_yaw[segment] = result["base_yaw"]
        for key in gains:
            gains[key][segment] = result[key]
    fields = dict(**storage, terrain=terrain, base_yaw=base_yaw,
                  initial=storage["target"][:, 0].copy(),
                  initial_velocity=storage["target_velocity"][:, 0].copy(),
                  torque_limit=np.full(segments, 10.0), integral_limit=np.full(segments, 1.0),
                  dt=DT, points=POINTS, mass=np.array(params.MASSES),
                  com=np.array(params.COM), inertia=np.array(params.INERTIA),
                  joint_offsets=np.array(params.JOINT_OFFSETS),
                  damping=np.array(params.DAMPING), gravity=params.GRAVITY, **gains)
    output_clean.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_clean, **fields)
    noise_rng = np.random.default_rng(noise_seed)
    noisy = dict(fields)
    noisy["q"] = fields["q"] + noise_rng.normal(0.0, 0.001, fields["q"].shape)
    noisy["qdot"] = fields["qdot"] + noise_rng.normal(0.0, 0.1, fields["qdot"].shape)
    noisy["tau"] = fields["tau"] + noise_rng.normal(0.0, 0.01, fields["tau"].shape)
    noisy.update({"q_clean": fields["q"], "qdot_clean": fields["qdot"],
                  "tau_clean": fields["tau"],
                  "initial": noisy["q"][:, 0].copy(),
                  "initial_velocity": noisy["qdot"][:, 0].copy(),
                  "initial_clean": fields["q"][:, 0].copy(),
                  "initial_velocity_clean": fields["qdot"][:, 0].copy(),
                  "noise_position_std": 0.001, "noise_velocity_std": 0.1,
                  "noise_torque_std": 0.01})
    np.savez_compressed(output_noisy, **noisy)
    velocity = np.abs(fields["qdot"])
    torque = np.abs(fields["tau"])
    print("agenda 段数 %d" % segments)
    print("峰值速度 %s（最大 %.3f rad/s）" % (
        np.array2string(velocity.max(axis=(0, 1)), precision=3), velocity.max()))
    print("峰值力矩 %s（最大 %.3f N·m）" % (
        np.array2string(torque.max(axis=(0, 1)), precision=3), torque.max()))
    print("位置范围 %s" % np.array2string(
        np.concatenate([fields["q"].min(axis=(0, 1))[:, None],
                        fields["q"].max(axis=(0, 1))[:, None]], axis=1), precision=3))
    print("坡度最大 %.1f 度" % np.degrees(terrain[:, 0].max()))


def main() -> None:
    parser = argparse.ArgumentParser(description="低激励闭环采集")
    parser.add_argument("--segments", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261220)
    parser.add_argument("--noise-seed", type=int, default=20261221)
    parser.add_argument("--amplitude-scale", type=float, default=1.0)
    parser.add_argument("--frequency-scale", type=float, default=1.0)
    parser.add_argument("--amplitudes", type=str, default=None,
                        help="四个关节的幅值，逗号分隔")
    parser.add_argument("--frequencies", type=str, default=None,
                        help="四个关节的最低频率，逗号分隔")
    parser.add_argument("--bandwidth", type=float, default=None, help="频率带宽")
    parser.add_argument("--roll-center", type=float, default=None)
    parser.add_argument("--slow-amplitudes", type=str, default=None)
    parser.add_argument("--slow-frequencies", type=str, default=None)
    parser.add_argument("--fast-amplitudes", type=str, default=None)
    parser.add_argument("--fast-frequencies", type=str, default=None)
    parser.add_argument("--static-amplitude", type=float, default=0.15)
    parser.add_argument("--tilt", type=float, default=20.0, help="最大坡度，单位度")
    parser.add_argument("--gain-scale", type=float, default=1.0)
    parser.add_argument("--static-fraction", type=float, default=0.3)
    parser.add_argument("--clean", type=Path, default=Path("data/low_data.npz"))
    parser.add_argument("--noisy", type=Path, default=Path("data/low_data_noisy.npz"))
    arguments = parser.parse_args()
    if arguments.amplitudes is not None:
        AMPLITUDES[:] = [float(value) for value in arguments.amplitudes.split(",")]
    if arguments.frequencies is not None:
        FREQUENCIES[:] = [float(value) for value in arguments.frequencies.split(",")]
    if arguments.bandwidth is not None:
        global BANDWIDTH
        BANDWIDTH = arguments.bandwidth
    if arguments.roll_center is not None:
        global ROLL_CENTER
        ROLL_CENTER = arguments.roll_center
    if arguments.slow_amplitudes is not None:
        global SLOW_AMPLITUDES, SLOW_FREQUENCIES, FAST_AMPLITUDES, FAST_FREQUENCIES
        SLOW_AMPLITUDES = [float(value) for value in arguments.slow_amplitudes.split(",")]
        SLOW_FREQUENCIES = [float(value) for value in arguments.slow_frequencies.split(",")]
        FAST_AMPLITUDES = [float(value) for value in arguments.fast_amplitudes.split(",")]
        FAST_FREQUENCIES = [float(value) for value in arguments.fast_frequencies.split(",")]
    collect(arguments.segments, arguments.seed, arguments.amplitude_scale,
            arguments.frequency_scale, arguments.static_amplitude,
            math.radians(arguments.tilt), arguments.gain_scale,
            arguments.static_fraction, arguments.clean, arguments.noisy,
            arguments.noise_seed)


if __name__ == "__main__":
    main()