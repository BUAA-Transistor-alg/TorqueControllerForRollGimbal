"""在关节限位与底盘静止约束下采集控制-响应序列。

只对数据采集过程施加机械/电控限位，不修改动力学模型与仿真本身。

每段 3 s，300 个时间点，dt = 0.01 s。段内底盘不移动、不旋转，仅通过坡角与
坡向体现不超过 20 度的底盘倾角。采集内容保存为 npz：
  q, qdot, tau        (segments, 300, 4)
  terrain             (segments, 2)，slope_angle, slope_direction
  base_yaw            (segments,)
  initial_angle, initial_velocity (segments, 4)
  model_*             真实参数
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def reference_trajectory(rng: np.random.Generator, points: int):
    times = np.arange(points, dtype=float) * DT

    def sinusoid(center: float, amplitude: float, frequency: float, phase: float):
        angle = center + amplitude * np.sin(frequency * times + phase)
        velocity = amplitude * frequency * np.cos(frequency * times + phase)
        acceleration = -amplitude * frequency * frequency * np.sin(frequency * times + phase)
        return angle, velocity, acceleration

    # 大 yaw 允许多圈，用低频正弦叠加恒速旋转
    yaw_center = rng.uniform(-0.6, 0.6)
    yaw_amplitude = rng.uniform(0.6, 1.4)
    yaw_frequency = rng.uniform(0.8, 1.8)
    yaw_phase = rng.uniform(-math.pi, math.pi)
    yaw_velocity = rng.uniform(-0.35, 0.35)
    angle1 = yaw_center + yaw_amplitude * np.sin(yaw_frequency * times + yaw_phase) + yaw_velocity * times
    velocity1 = yaw_amplitude * yaw_frequency * np.cos(yaw_frequency * times + yaw_phase) + yaw_velocity
    acceleration1 = -yaw_amplitude * yaw_frequency**2 * np.sin(yaw_frequency * times + yaw_phase)

    def bounded(limit_low: float, limit_high: float, center_low: float, center_high: float):
        center = rng.uniform(center_low, center_high)
        margin = min(center - limit_low, limit_high - center)
        amplitude = rng.uniform(0.3, 0.75) * margin
        frequency = rng.uniform(0.8, 2.2)
        phase = rng.uniform(-math.pi, math.pi)
        return sinusoid(center, amplitude, frequency, phase)

    angle2, velocity2, acceleration2 = bounded(ROLL_LIMIT[0], ROLL_LIMIT[1], -0.55, -0.05)
    angle3, velocity3, acceleration3 = bounded(SMALL_YAW_LIMIT[0], SMALL_YAW_LIMIT[1], -0.25, 0.25)
    angle4, velocity4, acceleration4 = bounded(PITCH_LIMIT[0], PITCH_LIMIT[1], -0.25, 0.25)

    angles = np.stack([angle1, angle2, angle3, angle4], axis=1)
    velocities = np.stack([velocity1, velocity2, velocity3, velocity4], axis=1)
    accelerations = np.stack([acceleration1, acceleration2, acceleration3, acceleration4], axis=1)
    if np.any(angles[:, 1] < ROLL_LIMIT[0]) or np.any(angles[:, 1] > ROLL_LIMIT[1]):
        raise RuntimeError("roll 参考超出限位")
    if np.any(angles[:, 2] < SMALL_YAW_LIMIT[0]) or np.any(angles[:, 2] > SMALL_YAW_LIMIT[1]):
        raise RuntimeError("小 yaw 参考超出限位")
    if np.any(angles[:, 3] < PITCH_LIMIT[0]) or np.any(angles[:, 3] > PITCH_LIMIT[1]):
        raise RuntimeError("pitch 参考超出限位")
    return angles, velocities, accelerations


def collect(segments: int, seed: int, output: Path) -> None:
    rng = np.random.default_rng(seed)
    env = GimbalEnv()
    q = np.zeros((segments, POINTS, 4))
    qdot = np.zeros((segments, POINTS, 4))
    tau = np.zeros((segments, POINTS, 4))
    terrain = np.zeros((segments, 2))
    base_yaw = np.zeros(segments)
    initial_angle = np.zeros((segments, 4))
    initial_velocity = np.zeros((segments, 4))

    for index in range(segments):
        for _attempt in range(20):
            angles, velocities, accelerations = reference_trajectory(rng, POINTS)
            slope_angle = rng.uniform(0.0, TILT_LIMIT)
            slope_direction = rng.uniform(0.0, 2.0 * math.pi)
            yaw = rng.uniform(-math.pi, math.pi)
            terrain_segment = gd.make_terrain(slope_angle, slope_direction)
            base_segment = gd.make_base(yaw=yaw)
            initial_value = env.reset({"angle": angles[0], "velocity": velocities[0],
                                       "terrain": {"slope_angle": slope_angle,
                                                   "slope_direction": slope_direction},
                                       "base": {"yaw": yaw}})
            trajectory_tau = np.zeros((POINTS, 4))
            for k in range(POINTS):
                trajectory_tau[k, :] = env.inverse_dynamics(accelerations[k])
                if k + 1 < POINTS:
                    env.step({"torque": trajectory_tau[k]})
            # 实际状态重新用同一组力矩推进并记录
            observation = env.reset({"angle": angles[0], "velocity": velocities[0]})
            actual_q = np.zeros((POINTS, 4))
            actual_qdot = np.zeros((POINTS, 4))
            valid = True
            for k in range(POINTS):
                actual_q[k] = observation["angle"]
                actual_qdot[k] = observation["velocity"]
                if not (ROLL_LIMIT[0] <= actual_q[k, 1] <= ROLL_LIMIT[1]):
                    valid = False
                    break
                if not (SMALL_YAW_LIMIT[0] <= actual_q[k, 2] <= SMALL_YAW_LIMIT[1]):
                    valid = False
                    break
                if not (PITCH_LIMIT[0] <= actual_q[k, 3] <= PITCH_LIMIT[1]):
                    valid = False
                    break
                if k + 1 < POINTS:
                    observation = env.step({"torque": trajectory_tau[k]})
            if valid:
                q[index] = actual_q
                qdot[index] = actual_qdot
                tau[index] = trajectory_tau
                terrain[index] = [slope_angle, slope_direction]
                base_yaw[index] = yaw
                initial_angle[index] = initial_value["angle"]
                initial_velocity[index] = initial_value["velocity"]
                break
        else:
            raise RuntimeError(f"第 {index} 段始终超出关节限位，请降低参考幅值")
    env.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        q=q,
        qdot=qdot,
        tau=tau,
        terrain=terrain,
        base_yaw=base_yaw,
        initial_angle=initial_angle,
        initial_velocity=initial_velocity,
        dt=DT,
        points=POINTS,
        mass=np.array(params.MASSES),
        com=np.array(params.COM),
        inertia=np.array(params.INERTIA),
        joint_offsets=np.array(params.JOINT_OFFSETS),
        damping=np.array(params.DAMPING),
        gravity=params.GRAVITY,
    )
    print(f"已保存 {segments} 段到 {output}")
    print(f"roll 范围 [{q[:, :, 1].min():.4f}, {q[:, :, 1].max():.4f}] rad")
    print(f"小 yaw 范围 [{q[:, :, 2].min():.4f}, {q[:, :, 2].max():.4f}] rad")
    print(f"pitch 范围 [{q[:, :, 3].min():.4f}, {q[:, :, 3].max():.4f}] rad")
    print(f"底盘坡角范围 [0, {terrain[:, 0].max():.4f}] rad")


def main() -> None:
    parser = argparse.ArgumentParser(description="采集带限位的控制-响应序列")
    parser.add_argument("--segments", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--output", type=Path, default=Path("data/identification_data.npz"))
    arguments = parser.parse_args()
    collect(arguments.segments, arguments.seed, arguments.output)


if __name__ == "__main__":
    main()
