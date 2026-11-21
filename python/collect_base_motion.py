"""采集带底盘晃动（偏航摆动）与滑移（水平加速度）的闭环数据，用于检验静态底盘假设的容忍性。

底盘状态按模型支持的形式给出：偏航角随时间摆动（含角速度与角加速度），
底盘参考点带水平线加速度（滑移）。控制器前馈使用真实底盘状态，因此记录力矩里
含有底盘运动引起的真实力矩。
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
ROLL_CENTER = -0.12
SLOW_AMPLITUDES = [0.45, 0.10, 0.15, 0.15]
FAST_AMPLITUDES = [0.10, 0.06, 0.08, 0.08]
FAST_FREQUENCIES = [5.0, 5.0, 5.0, 5.0]
LIMITS = (None, (-45.0 * math.pi / 180.0, 5.0 * math.pi / 180.0),
          (-50.0 * math.pi / 180.0, 50.0 * math.pi / 180.0),
          (-45.0 * math.pi / 180.0, 45.0 * math.pi / 180.0))


def reference(rng):
    time = np.arange(POINTS) * DT
    angle = np.zeros((POINTS, 4))
    velocity = np.zeros((POINTS, 4))
    for joint in range(4):
        center = ROLL_CENTER if joint == 1 else 0.0
        signal = np.full(POINTS, center)
        derivative = np.zeros(POINTS)
        phase = rng.uniform(-math.pi, math.pi)
        signal += SLOW_AMPLITUDES[joint] * np.sin(0.5 * time + phase)
        derivative += SLOW_AMPLITUDES[joint] * 0.5 * np.cos(0.5 * time + phase)
        for _ in range(2):
            frequency = FAST_FREQUENCIES[joint] * rng.uniform(0.7, 1.3)
            local_phase = rng.uniform(-math.pi, math.pi)
            signal += 0.5 * FAST_AMPLITUDES[joint] * np.sin(frequency * time + local_phase)
            derivative += 0.5 * FAST_AMPLITUDES[joint] * frequency * np.cos(
                frequency * time + local_phase)
        angle[:, joint] = signal
        velocity[:, joint] = derivative
    return angle, velocity


def base_sequence(rng, yaw_amplitude, yaw_frequency, slip_amplitude, slip_frequency):
    """返回每一拍的 (加速度, 偏航角, 偏航角速度, 偏航角加速度)。"""
    time = np.arange(POINTS) * DT
    yaw0 = rng.uniform(-math.pi, math.pi)
    phase = rng.uniform(-math.pi, math.pi)
    direction = rng.uniform(0.0, 2.0 * math.pi)
    slip_phase = rng.uniform(-math.pi, math.pi)
    yaw = yaw0 + yaw_amplitude * np.sin(yaw_frequency * time + phase)
    yaw_rate = yaw_amplitude * yaw_frequency * np.cos(yaw_frequency * time + phase)
    yaw_acceleration = -yaw_amplitude * yaw_frequency ** 2 * np.sin(
        yaw_frequency * time + phase)
    acceleration = np.outer(slip_amplitude * np.sin(slip_frequency * time + slip_phase),
                            np.array([math.cos(direction), math.sin(direction), 0.0]))
    return acceleration, yaw, yaw_rate, yaw_acceleration, yaw0


def simulate_segment(rng, yaw_amplitude, yaw_frequency, slip_amplitude, slip_frequency,
                     tilt_limit, gain_scale):
    env = GimbalEnv()
    slope = rng.uniform(0.0, tilt_limit)
    direction = rng.uniform(0.0, 2.0 * math.pi)
    terrain = gd.make_terrain(slope, direction)
    target_q, target_v = reference(rng)
    acceleration, yaw, yaw_rate, yaw_acceleration, yaw0 = base_sequence(
        rng, yaw_amplitude, yaw_frequency, slip_amplitude, slip_frequency)
    gains_kp = np.array([8.0, 2.0, 2.0, 2.0]) * gain_scale * rng.uniform(0.9, 1.1, 4)
    gains_ki = np.array([0.2] * 4) * gain_scale * rng.uniform(0.9, 1.1, 4)
    gains_kd = np.array([0.5, 0.2, 0.15, 0.1]) * gain_scale * rng.uniform(0.9, 1.1, 4)
    base = gd.make_base(acceleration=acceleration[0], yaw=float(yaw[0]),
                        yaw_rate=float(yaw_rate[0]),
                        yaw_acceleration=float(yaw_acceleration[0]))
    observation = env.reset(target_q[0], target_v[0], terrain=terrain, base=base)
    q = np.zeros((POINTS, 4))
    qdot = np.zeros((POINTS, 4))
    tau = np.zeros((POINTS, 4))
    integral = np.zeros(4)
    for k in range(POINTS):
        base = gd.make_base(acceleration=acceleration[k], yaw=float(yaw[k]),
                            yaw_rate=float(yaw_rate[k]),
                            yaw_acceleration=float(yaw_acceleration[k]))
        env.set_scene(base=base)
        current = observation.angle
        velocity = observation.velocity
        q[k] = current
        qdot[k] = velocity
        error = target_q[k] - current
        integral = np.clip(integral + error * DT, -1.0, 1.0)
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
                terrain=np.array([slope, direction]), base_yaw=yaw0,
                kp=gains_kp, ki=gains_ki, kd=gains_kd)


def collect(segments, seed, yaw_amplitude_deg, yaw_frequency, slip_amplitude,
            slip_frequency, tilt_limit, output_clean, output_noisy, noise_seed):
    rng = np.random.default_rng(seed)
    keys = ("q", "qdot", "tau", "target", "target_velocity")
    storage = {key: np.zeros((segments, POINTS, 4)) for key in keys}
    terrain = np.zeros((segments, 2))
    base_yaw = np.zeros(segments)
    gains = {key: np.zeros((segments, 4)) for key in ("kp", "ki", "kd")}
    yaw_amplitude = math.radians(yaw_amplitude_deg)
    for segment in range(segments):
        result = None
        for _ in range(6):
            result = simulate_segment(rng, yaw_amplitude, yaw_frequency, slip_amplitude,
                                      slip_frequency, tilt_limit, 1.0)
            if result is not None:
                break
        if result is None:
            raise RuntimeError("第 %d 段超出限位" % segment)
        for key in keys:
            storage[key][segment] = result[key]
        terrain[segment] = result["terrain"]
        base_yaw[segment] = result["base_yaw"]
        for key in gains:
            gains[key][segment] = result[key]
    fields = dict(**storage, terrain=terrain, base_yaw=base_yaw,
                  initial=storage["target"][:, 0].copy(),
                  initial_velocity=storage["target_velocity"][:, 0].copy(),
                  torque_limit=np.full(segments, 10.0),
                  integral_limit=np.full(segments, 1.0), dt=DT, points=POINTS,
                  mass=np.array(params.MASSES), com=np.array(params.COM),
                  inertia=np.array(params.INERTIA),
                  joint_offsets=np.array(params.JOINT_OFFSETS),
                  damping=np.array(params.DAMPING), gravity=params.GRAVITY,
                  base_motion=np.array([math.degrees(yaw_amplitude), yaw_frequency,
                                        slip_amplitude, slip_frequency]), **gains)
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
    print("峰值速度 %.3f，峰值力矩 %.3f" % (
        np.abs(fields["qdot"]).max(), np.abs(fields["tau"]).max()))


def main() -> None:
    parser = argparse.ArgumentParser(description="带底盘晃动与滑移的数据采集")
    parser.add_argument("--segments", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261220)
    parser.add_argument("--noise-seed", type=int, default=20261221)
    parser.add_argument("--yaw-amplitude", type=float, default=0.0, help="偏航摆幅（度）")
    parser.add_argument("--yaw-frequency", type=float, default=2.0)
    parser.add_argument("--slip-amplitude", type=float, default=0.0, help="滑移加速度幅值 m/s^2")
    parser.add_argument("--slip-frequency", type=float, default=1.0)
    parser.add_argument("--tilt", type=float, default=10.0)
    parser.add_argument("--clean", type=Path, default=Path("data/base_motion_data.npz"))
    parser.add_argument("--noisy", type=Path, default=Path("data/base_motion_data_noisy.npz"))
    arguments = parser.parse_args()
    collect(arguments.segments, arguments.seed, arguments.yaw_amplitude,
            arguments.yaw_frequency, arguments.slip_amplitude, arguments.slip_frequency,
            math.radians(arguments.tilt), arguments.clean, arguments.noisy,
            arguments.noise_seed)


if __name__ == "__main__":
    main()
