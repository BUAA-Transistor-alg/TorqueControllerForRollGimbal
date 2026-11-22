"""真参数教师控制器：逆动力学前馈 + 比例反馈 + 限位制动 + 有界力矩分配。

教师知道真实动力学参数与每个关节的电机力矩系数，用它们构造期望角加速度并求解力���；
控制器只通过带噪声的观测与循环状态推断这些量，因此教师的动作可以作为参数知识上限的参照，
也是监督预训练与行为克隆辅助目标的来源。

所有输入都来自单步观测（sin/cos 角、角速度、当前参考、20 步目标预瞄），不使用超过 0.2 s 的未来参考。
"""

from __future__ import annotations

import numpy as np

import tracking_env as te

# 标称物理特征加动作残差滑动平均的结构化自适应控制器权重。
# 每行一个关节，四列分别对应 [模型基前馈需求, 偏置力矩, 残差滑动平均, 常数]，
# 由 research/calibrate_structured.py 在真参数教师驱动的回合上用岭回归标定。
STRUCTURED_WEIGHTS = np.array([[0.9753, 1.3625, 0.2348, -0.0187],
                               [1.0112, 0.8489, 0.7720, 0.0383],
                               [1.0303, 0.7832, 0.7191, -0.0284],
                               [1.2029, 1.0908, 0.9858, -0.0182]])
LOAD_RATE = 0.1


class StructuredTeacher:
    """标称模型前馈 + 偏置 + 动作残差滑动平均的结构化自适应控制器。

    输入只来自观测：最后 12 维是标称特征（前馈需求、偏置力矩、模型反演动作），
    残差滑动平均由观测里的瞬时失配逐步累积，代表对执行器增益与参数偏差的在线估计。
    """

    def __init__(self, env=None, weights=None, load_rate: float = LOAD_RATE):
        self.weights = (STRUCTURED_WEIGHTS if weights is None
                        else np.asarray(weights, dtype=float))
        self.load_rate = float(load_rate)
        self.load = np.zeros(4)

    def reset(self):
        self.load = np.zeros(4)

    def action(self, step_index: int = 0, observation=None) -> np.ndarray:
        values = np.asarray(observation, dtype=float)
        demand = values[-16:-12] / te.NOMINAL_OBSERVATION_SCALE
        bias = values[-12:-8] / te.NOMINAL_OBSERVATION_SCALE
        mismatch = values[-8:-4]
        self.load = (1.0 - self.load_rate) * self.load + self.load_rate * mismatch
        features = np.stack([demand, bias, self.load, np.ones(4)], axis=1)
        return np.clip(np.sum(features * self.weights, axis=1), -1.0, 1.0).astype(np.float32)


class TeacherController:
    """mode 取 full、feedforward 或 feedback，用于消融前馈与反馈的贡献。"""

    def __init__(self, env, kp: float = 196.0, kd: float = 28.0, margin: float = 0.035,
                 brake: float = 14.0, accel_limit: float = 55.0, mode: str = "full"):
        self.env = env
        self.kp = float(kp)
        self.kd = float(kd)
        self.margin = float(margin)
        self.brake = float(brake)
        self.accel_limit = float(accel_limit)
        if mode not in ("full", "feedforward", "feedback"):
            raise ValueError("未知控制模式 %s" % mode)
        self.mode = mode
        self.lower = np.array([-np.inf] + [item[0] for item in te.LIMITS[1:]]) + self.margin
        self.upper = np.array([np.inf] + [item[1] for item in te.LIMITS[1:]]) - self.margin
        self.terrain = te.gd.make_terrain()
        self.base = te.gd.make_base()
        self.times = te.DT * np.arange(1, te.PREVIEW_STEPS + 1)
        self.accel_weights = 2.0 * self.times ** 2 / np.sum(self.times ** 4)

    def action(self, step_index: int, observation=None) -> np.ndarray:
        env = self.env
        if step_index != env.step_index:
            raise ValueError("step_index 必须对应当前可执行动作的时间步")
        if observation is None:
            preview = np.clip(env._preview(step_index), -5.0 * te.PREVIEW_SCALE,
                              5.0 * te.PREVIEW_SCALE).reshape(-1, 4)
            angle, velocity = env.measurement()[:2]
            angle = te.wrap_angle(angle)
            velocity = np.clip(velocity, -5.0 * te.VELOCITY_SCALE, 5.0 * te.VELOCITY_SCALE)
        else:
            values = np.asarray(observation, dtype=float)
            if values.shape != (te.OBSERVATION_DIM,):
                raise ValueError("观测维度与 TrackingEnv 不符")
            preview = values[28:108].reshape(-1, 4) * te.PREVIEW_SCALE
            angle = np.arctan2(values[0:4], values[4:8])
            velocity = values[8:12] * te.VELOCITY_SCALE

        reference = env.angle_ref[step_index]
        target = np.clip(reference, self.lower, self.upper)
        target_velocity = env.velocity_ref[step_index].copy()
        target_velocity[target != reference] = 0.0
        future = np.clip(reference + preview, self.lower, self.upper)
        count = min(te.PREVIEW_STEPS, env.episode_steps - 1 - step_index)
        if count > 0:
            times = self.times[:count]
            weights = 2.0 * times ** 2 / np.sum(times ** 4)
            feedforward = np.clip(
                weights @ (future[:count] - target - times[:, None] * target_velocity), -12.0, 12.0)
        else:
            feedforward = np.zeros(4)
        desired = np.zeros(4) if self.mode == "feedback" else feedforward
        if self.mode != "feedforward":
            desired = (desired + self.kp * te.wrap_angle(target - angle)
                       + self.kd * (target_velocity - velocity))
        desired = np.clip(desired, -self.accel_limit, self.accel_limit)

        # 由 v*DT + v^2/(2b) <= d 求每个方向的安全速度，并对期望加速度做连续限制
        lag = self.brake * te.DT
        distances = np.stack((angle - self.lower, self.upper - angle))
        speeds = np.sqrt(lag ** 2 + 2.0 * self.brake * np.maximum(distances, 0.0)) - lag
        lower_speed, upper_speed = speeds + (2.0 * self.kp / self.kd) * np.minimum(
            distances, 0.0)
        desired = np.clip(desired, self.kd * (-lower_speed - velocity),
                          self.kd * (upper_speed - velocity))
        desired = np.clip(desired, -self.accel_limit, self.accel_limit)

        state = te.gd.make_state(angle, velocity)
        bias = np.asarray(env.context.inverse_dynamics(env.model, self.terrain, self.base,
                                                       state, np.zeros(4)))
        demand = np.asarray(env.context.inverse_dynamics(env.model, self.terrain, self.base,
                                                         state, desired)) - bias
        motor = np.asarray(env.motor_scale)
        anchor = np.zeros(4) if self.mode == "feedback" else bias / max(
            1.0, float(np.max(np.abs(bias) / motor)))
        headroom = motor - np.sign(demand) * anchor
        ratio = np.divide(headroom, np.abs(demand), out=np.full(4, np.inf),
                          where=demand != 0.0)
        alpha = float(np.clip(np.min(ratio), 0.0, 1.0))
        return np.clip((anchor + alpha * demand) / motor, -1.0, 1.0).astype(np.float32)
