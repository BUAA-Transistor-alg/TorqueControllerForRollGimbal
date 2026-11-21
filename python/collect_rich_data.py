"""富激励闭环 PID 数据采集：限位内多正弦 + 静态保持 + 快速阶跃，随后加噪。

采集协议与 collect_closed_loop.py 一致：真实模型 + PID 前馈闭环，段内底盘静止，
每段 3 s、300 点、dt=0.01；随后按同一噪声标准加高斯噪声（新种子）。
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
TILT_LIMIT = 20.0 * math.pi / 180.0
LIMITS = (None, ROLL_LIMIT, SMALL_YAW_LIMIT, PITCH_LIMIT)


def multisine(rng: np.random.Generator, low: float, high: float, margin: float,
              modes: int, frequency_low: float, frequency_high: float):
    """在 [low, high] 内构造多正弦参考，按预测极值缩放幅值。"""
    time = np.arange(POINTS) * DT
    center = rng.uniform(low + 0.35 * (high - low), high - 0.35 * (high - low))
    span = min(center - low, high - center) * margin
    angle = np.full(POINTS, center)
    velocity = np.zeros(POINTS)
    frequency_low = max(frequency_low, 0.3)
    frequency_high = max(frequency_high, frequency_low * 1.2)
    for _ in range(modes):
        frequency = rng.uniform(frequency_low, frequency_high)
        phase = rng.uniform(-math.pi, math.pi)
        weight = rng.uniform(0.4, 1.0)
        angle += weight * np.sin(frequency * time + phase)
        velocity += weight * frequency * np.cos(frequency * time + phase)
    deviation = np.max(np.abs(angle - center))
    if deviation > 1.0e-9:
        gain = span / deviation
        angle = center + gain * (angle - center)
        velocity = gain * velocity
    return angle, velocity


def step_reference(rng: np.random.Generator, low: float, high: float, margin: float):
    """随机分段常值目标，形成快速阶跃与斜坡。"""
    time = np.arange(POINTS) * DT
    span = (high - low) * 0.5 * margin
    center = rng.uniform(low + span, high - span)
    values = np.zeros(POINTS)
    velocity = np.zeros(POINTS)
    level = center
    k = 0
    safe_low = low + 0.12 * (high - low)
    safe_high = high - 0.14 * (high - low)
    while k < POINTS:
        hold = int(rng.integers(15, 70))
        target = center + rng.uniform(-1.0, 1.0) * span
        target = float(np.clip(target, safe_low, safe_high))
        ramp = int(rng.integers(15, 40))
        values[k:k + ramp] = np.linspace(level, target, ramp)[:max(0, min(ramp, POINTS - k))]
        velocity[k:k + ramp] = (target - level) / (ramp * DT)
        k += ramp
        values[k:k + hold] = target
        level = target
        k += hold
    return values[:POINTS], velocity[:POINTS]


def reference(rng: np.random.Generator, kind: str, strength: float = 1.0):
    angle = np.zeros((POINTS, 4))
    velocity = np.zeros((POINTS, 4))
    # 大 yaw 不限位，但转速过高会通过耦合把 roll 甩出限位，因此频率上限取 3 rad/s
    angle[:, 0], velocity[:, 0] = multisine(rng, -1.1, 1.1, 0.95,
                                            int(rng.integers(2, 4)), 0.5, 3.0)
    for joint in (1, 2, 3):
        low, high = LIMITS[joint]
        if kind == "static":
            span = (high - low) * 0.22
            center = rng.uniform(low + span * 1.2, high - span * 1.2)
            angle[:, joint] = center
            velocity[:, joint] = 0.0
        elif kind == "step":
            angle[:, joint], velocity[:, joint] = step_reference(rng, low, high,
                                                                 0.85 * strength)
        else:
            frequency_high = rng.uniform(2.5, 5.5)
            safe_low = low + 0.10 * (high - low)
            safe_high = high - 0.12 * (high - low)
            angle[:, joint], velocity[:, joint] = multisine(
                rng, safe_low, safe_high, rng.uniform(0.85, 0.97) * strength,
                int(rng.integers(2, 5)), 0.6, frequency_high * strength)
    return angle, velocity


def simulate_segment(rng, kind: str, strength: float, slope: float, direction: float,
                     yaw: float, diagnostics=None):
    """生成参考并做一次闭环仿真；返回数据或 None（超出限位）。"""
    env = GimbalEnv()
    terrain_seg = gd.make_terrain(slope, direction)
    base = gd.make_base(yaw=yaw)
    target_q, target_v = reference(rng, kind, strength)
    q = np.zeros((POINTS, 4))
    qdot = np.zeros((POINTS, 4))
    tau = np.zeros((POINTS, 4))
    observation = env.reset(target_q[0], target_v[0], terrain=terrain_seg, base=base)
    # roll 是倒立摆，增益过低会让闭环失稳，因此 roll 的增益缩放取下界更高
    kp_scale = rng.uniform(0.8, 1.5, 4)
    kp_scale[1] = rng.uniform(1.0, 1.6)
    gains_kp = np.array([8.0, 2.0, 2.0, 2.0]) * kp_scale
    gains_ki = np.array([0.2, 0.2, 0.2, 0.2]) * rng.uniform(0.5, 1.8, 4)
    gains_kd = np.array([0.5, 0.2, 0.15, 0.1]) * rng.uniform(0.8, 1.6, 4)
    limit = 10.0
    integral = np.zeros(4)
    integral_clip = 1.0
    saturation = 0
    for k in range(POINTS):
        current = observation.angle
        velocity = observation.velocity
        q[k] = current
        qdot[k] = velocity
        error = target_q[k] - current
        integral = np.clip(integral + error * DT, -integral_clip, integral_clip)
        feedforward = env.bias_torque()
        command = (feedforward + gains_kp * error + gains_ki * integral
                   + gains_kd * (target_v[k] - velocity))
        limited = np.clip(command, -limit, limit)
        saturation += int(np.sum(np.abs(command) > limit + 1.0e-9))
        tau[k] = limited
        if k + 1 < POINTS:
            observation = env.step(limited)
    env.close()
    for joint in (1, 2, 3):
        low, high = LIMITS[joint]
        if np.any(q[:, joint] < low) or np.any(q[:, joint] > high):
            if diagnostics is not None:
                diagnostics.append((joint, float(q[:, joint].min()), float(q[:, joint].max()),
                                    kind, strength, float(target_q[:, joint].min()),
                                    float(target_q[:, joint].max())))
            return None
    return dict(q=q, qdot=qdot, tau=tau, target=target_q, target_velocity=target_v,
                kp=gains_kp, ki=gains_ki, kd=gains_kd, saturation=saturation)


def collect(segments: int, seed: int, output_clean: Path, output_noisy: Path,
            noise_seed: int) -> None:
    rng = np.random.default_rng(seed)
    q = np.zeros((segments, POINTS, 4))
    qdot = np.zeros((segments, POINTS, 4))
    tau = np.zeros((segments, POINTS, 4))
    target = np.zeros((segments, POINTS, 4))
    target_velocity = np.zeros((segments, POINTS, 4))
    terrain = np.zeros((segments, 2))
    base_yaw = np.zeros(segments)
    initial = np.zeros((segments, 4))
    initial_velocity = np.zeros((segments, 4))
    kp = np.zeros((segments, 4))
    ki = np.zeros((segments, 4))
    kd = np.zeros((segments, 4))
    torque_limit_values = np.zeros(segments)
    integral_limit_values = np.zeros(segments)
    kinds = []
    saturation = 0
    total_steps = 0
    for segment in range(segments):
        selector = segment % 10
        kind = "static" if selector < 2 else ("step" if selector < 5 else "multi")
        slope = rng.uniform(0.0, TILT_LIMIT)
        direction = rng.uniform(0.0, 2.0 * math.pi)
        yaw = rng.uniform(-math.pi, math.pi)
        strength = 1.0
        result = None
        diagnostics = []
        for _ in range(6):
            candidate = simulate_segment(rng, kind, strength, slope, direction, yaw,
                                         diagnostics)
            if candidate is not None:
                result = candidate
                break
            strength *= 0.55
        if result is None:
            candidate = simulate_segment(rng, "static", 1.0, slope, direction, yaw,
                                         diagnostics)
            if candidate is not None:
                result = candidate
                kind = "static"
        if result is None:
            print("失败诊断：")
            for item in diagnostics:
                print("  关节 %d 实际范围 %.4f 到 %.4f，类型 %s，强度 %.3f，目标范围 %.4f 到 %.4f"
                      % item)
            raise RuntimeError("第 %d 段无法生成满足限位的轨迹" % segment)
        q[segment] = result["q"]
        qdot[segment] = result["qdot"]
        tau[segment] = result["tau"]
        target[segment] = result["target"]
        target_velocity[segment] = result["target_velocity"]
        kp[segment] = result["kp"]
        ki[segment] = result["ki"]
        kd[segment] = result["kd"]
        saturation += result["saturation"]
        total_steps += 4 * POINTS
        kinds.append(kind)
        terrain[segment] = [slope, direction]
        base_yaw[segment] = yaw
        initial[segment] = result["target"][0]
        initial_velocity[segment] = result["target_velocity"][0]
        torque_limit_values[segment] = 10.0
        integral_limit_values[segment] = 1.0
        if segment % 100 == 99:
            print("已采集 %d 段" % (segment + 1))

    fields = dict(q=q, qdot=qdot, tau=tau, target=target, target_velocity=target_velocity,
                  terrain=terrain, base_yaw=base_yaw, initial=initial,
                  initial_velocity=initial_velocity, kp=kp, ki=ki, kd=kd,
                  torque_limit=torque_limit_values, integral_limit=integral_limit_values,
                  dt=DT, points=POINTS, kind=np.array(kinds),
                  mass=np.array(params.MASSES), com=np.array(params.COM),
                  inertia=np.array(params.INERTIA),
                  joint_offsets=np.array(params.JOINT_OFFSETS),
                  damping=np.array(params.DAMPING), gravity=params.GRAVITY)
    output_clean.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_clean, **fields)

    noise_rng = np.random.default_rng(noise_seed)
    position_std, velocity_std, torque_std = 0.001, 0.1, 0.01
    noisy = dict(fields)
    noisy["q"] = q + noise_rng.normal(0.0, position_std, q.shape)
    noisy["qdot"] = qdot + noise_rng.normal(0.0, velocity_std, qdot.shape)
    noisy["tau"] = tau + noise_rng.normal(0.0, torque_std, tau.shape)
    noisy.update({"q_clean": q, "qdot_clean": qdot, "tau_clean": tau,
                  "initial": noisy["q"][:, 0].copy(),
                  "initial_velocity": noisy["qdot"][:, 0].copy(),
                  "initial_clean": q[:, 0].copy(),
                  "initial_velocity_clean": qdot[:, 0].copy(),
                  "noise_position_std": position_std, "noise_velocity_std": velocity_std,
                  "noise_torque_std": torque_std})
    np.savez_compressed(output_noisy, **noisy)
    print("已保存", output_clean, "与", output_noisy)
    print("力矩饱和比例 %.4f" % (saturation / total_steps))
    for joint in range(4):
        print("关节 %d 位置范围 %.3f 到 %.3f，速度绝对值最大 %.2f，加速度绝对值最大 %.2f" % (
            joint + 1, q[:, :, joint].min(), q[:, :, joint].max(),
            np.max(np.abs(qdot[:, :, joint])),
            np.max(np.abs(np.gradient(qdot[:, :, joint], DT, axis=1)))))
    print("坡度范围 %.3f 到 %.3f" % (terrain[:, 0].min(), terrain[:, 0].max()))


def main() -> None:
    parser = argparse.ArgumentParser(description="富激励闭环数据采集")
    parser.add_argument("--segments", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20261205)
    parser.add_argument("--noise-seed", type=int, default=20261206)
    parser.add_argument("--clean", type=Path, default=Path("data/rich2_data.npz"))
    parser.add_argument("--noisy", type=Path, default=Path("data/rich2_data_noisy.npz"))
    arguments = parser.parse_args()
    collect(arguments.segments, arguments.seed, arguments.clean, arguments.noisy,
            arguments.noise_seed)


if __name__ == "__main__":
    main()