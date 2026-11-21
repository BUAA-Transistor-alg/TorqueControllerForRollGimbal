"""闭环 PID 数据采集：保存 PID 参数、目标位置序列、状态与力矩。"""

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


def reference(rng: np.random.Generator, points: int):
    time = np.arange(points) * DT
    angle = np.zeros((points, 4))
    velocity = np.zeros((points, 4))
    # 大 yaw 允许连续旋转
    center = rng.uniform(-0.6, 0.6)
    amplitude = rng.uniform(0.5, 1.2)
    frequency = rng.uniform(0.6, 1.5)
    phase = rng.uniform(-math.pi, math.pi)
    ramp = rng.uniform(-0.5, 0.5)
    angle[:, 0] = center + amplitude * np.sin(frequency * time + phase) + ramp * time
    velocity[:, 0] = amplitude * frequency * np.cos(frequency * time + phase) + ramp
    for joint, (low, high, center_low, center_high) in enumerate([
        (ROLL_LIMIT[0], ROLL_LIMIT[1], -0.55, -0.05),
        (SMALL_YAW_LIMIT[0], SMALL_YAW_LIMIT[1], -0.25, 0.25),
        (PITCH_LIMIT[0], PITCH_LIMIT[1], -0.25, 0.25),
    ], start=1):
        c = rng.uniform(center_low, center_high)
        margin = min(c - low, high - c)
        a = rng.uniform(0.3, 0.7) * margin
        f = rng.uniform(0.5, 1.5)
        p = rng.uniform(-math.pi, math.pi)
        angle[:, joint] = c + a * np.sin(f * time + p)
        velocity[:, joint] = a * f * np.cos(f * time + p)
    return angle, velocity


def collect(segments: int, seed: int, output: Path) -> None:
    rng = np.random.default_rng(seed)
    env = GimbalEnv()
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
    torque_limit = np.zeros(segments)
    integral_limit = np.zeros(segments)

    for segment in range(segments):
        target_q, target_v = reference(rng, POINTS)
        slope = rng.uniform(0.0, TILT_LIMIT)
        direction = rng.uniform(0.0, 2.0 * math.pi)
        yaw = rng.uniform(-math.pi, math.pi)
        terrain_seg = gd.make_terrain(slope, direction)
        base = gd.make_base(yaw=yaw)
        initial_value = env.reset({"angle": target_q[0], "velocity": target_v[0],
                                   "terrain": {"slope_angle": slope,
                                               "slope_direction": direction},
                                   "base": {"yaw": yaw}})
        gains_kp = np.array([8.0, 2.0, 2.0, 2.0]) * rng.uniform(0.8, 1.2, 4)
        gains_ki = np.array([0.2, 0.2, 0.2, 0.2]) * rng.uniform(0.7, 1.3, 4)
        gains_kd = np.array([0.5, 0.2, 0.15, 0.1]) * rng.uniform(0.8, 1.2, 4)
        limit = 10.0
        integral = np.zeros(4)
        integral_clip = 1.0
        observation = initial_value
        for k in range(POINTS):
            q[segment, k] = observation["angle"]
            qdot[segment, k] = observation["velocity"]
            target[segment, k] = target_q[k]
            target_velocity[segment, k] = target_v[k]
            error = target_q[k] - q[segment, k]
            integral = np.clip(integral + error * DT, -integral_clip, integral_clip)
            feedforward = env.feedforward_torque()
            command = (feedforward + gains_kp * error + gains_ki * integral
                       + gains_kd * (target_v[k] - qdot[segment, k]))
            command = np.clip(command, -limit, limit)
            if k + 1 < POINTS:
                observation = env.step({"torque": command})
                tau[segment, k] = observation["torque"]
            else:
                tau[segment, k] = command
        terrain[segment] = [slope, direction]
        base_yaw[segment] = yaw
        initial[segment] = initial_value["angle"]
        initial_velocity[segment] = initial_value["velocity"]
        kp[segment] = gains_kp
        ki[segment] = gains_ki
        kd[segment] = gains_kd
        torque_limit[segment] = limit
        integral_limit[segment] = integral_clip
        if (np.any(q[segment, :, 1] < ROLL_LIMIT[0]) or np.any(q[segment, :, 1] > ROLL_LIMIT[1]) or
                np.any(q[segment, :, 2] < SMALL_YAW_LIMIT[0]) or np.any(q[segment, :, 2] > SMALL_YAW_LIMIT[1]) or
                np.any(q[segment, :, 3] < PITCH_LIMIT[0]) or np.any(q[segment, :, 3] > PITCH_LIMIT[1])):
            raise RuntimeError(f"第 {segment} 段超出关节限位")
    env.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, q=q, qdot=qdot, tau=tau, target=target,
        target_velocity=target_velocity, terrain=terrain, base_yaw=base_yaw,
        initial=initial, initial_velocity=initial_velocity,
        kp=kp, ki=ki, kd=kd, torque_limit=torque_limit, integral_limit=integral_limit,
        dt=DT, points=POINTS,
        mass=np.array(params.MASSES), com=np.array(params.COM),
        inertia=np.array(params.INERTIA), joint_offsets=np.array(params.JOINT_OFFSETS),
        damping=np.array(params.DAMPING), gravity=params.GRAVITY,
    )
    print("已保存", segments, "段闭环数据到", output)
    print("roll range", q[:, :, 1].min(), q[:, :, 1].max())
    print("small yaw range", q[:, :, 2].min(), q[:, :, 2].max())
    print("pitch range", q[:, :, 3].min(), q[:, :, 3].max())
    print("tilt max", terrain[:, 0].max())


def main() -> None:
    parser = argparse.ArgumentParser(description="闭环 PID 数据采集")
    parser.add_argument("--segments", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--output", type=Path, default=Path("data/closed_loop_data.npz"))
    arguments = parser.parse_args()
    collect(arguments.segments, arguments.seed, arguments.output)


if __name__ == "__main__":
    main()
