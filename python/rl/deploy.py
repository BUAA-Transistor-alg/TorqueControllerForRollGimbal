"""把训练好的循环残差 SAC 控制器部署到 gimbal_dynamics 的状态上。

控制器每 10 ms 更新一次：维护最近 0.2 s 的目标位置缓冲，用缓冲的斜率外推出未来 0.2 s 的预瞄
参考（相对当前目标），再按训练时的观测布局组装 142 维观测，输出归一化动作并乘以标称电机力矩
系数得到各关节力矩。目标位置的产生方式与界面里的 PID 模式相同：按住按键以固定速度移动目标。

用法（脚本内）：
    from deploy import PolicyController
    controller = PolicyController("data/rl_final.pt")
    torque = controller.torque(state, target, terrain, base)   # 每 10 ms 调用一次
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "identification"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gimbal_dynamics as gd
import params
import tracking_env as te

CONTROL_PERIOD = te.DT
PREVIEW_COUNT = te.PREVIEW_STEPS
HISTORY_LENGTH = 21
VELOCITY_LIMIT = 3.0


def target_limits():
    """控制器接受的目标位置范围：大 yaw 不限，其余三轴取硬限位内侧 0.07 rad。"""
    return [(float(low), float(high)) for low, high in zip(te.SOFT_LOW, te.SOFT_HIGH)]


class PolicyController:
    """用训练好的策略把四个关节驱动到目标位置。"""

    def __init__(self, policy_path, context=None):
        from evaluate import build_agent

        self.context = gd.Context(1) if context is None else context
        self.owns_context = context is None
        self.agent = build_agent(Path(policy_path))
        self.model, self.scale = te.nominal_model(self.context)
        self.agent.actor.eval()
        self.reset()

    def close(self) -> None:
        if self.owns_context:
            self.context.close()

    def reset(self, target=None) -> None:
        self.hidden, _, _ = self.agent.initial_hidden(1)
        self.previous_action = np.zeros(4)
        self.previous_action_ema = np.zeros(4)
        self.filtered_velocity = np.zeros(4)
        self.previous_filtered_velocity = np.zeros(4)
        self.load_estimate = np.zeros(4)
        initial = np.zeros(4) if target is None else np.asarray(target, dtype=float).copy()
        self.target_history = deque([initial.copy() for _ in range(HISTORY_LENGTH)],
                                    maxlen=HISTORY_LENGTH)
        self.reference = initial.copy()
        self.velocity_reference = np.zeros(4)
        self.preview = np.zeros(PREVIEW_COUNT * 4)

    def references(self, target):
        """把当前目标压入 0.2 s 缓冲，用缓冲斜率外推未来 0.2 s 的预瞄参考。"""
        self.target_history.append(np.asarray(target, dtype=float).copy())
        history = np.stack(self.target_history)
        times = np.arange(len(history)) * CONTROL_PERIOD
        centered = times - times.mean()
        denominator = max(float(np.sum(centered ** 2)), 1.0e-9)
        slope = ((centered[:, None] * (history - history.mean(axis=0))).sum(axis=0)
                 / denominator)
        current = history[-1]
        steps = np.arange(1, PREVIEW_COUNT + 1) * CONTROL_PERIOD
        future = np.clip(current[None, :] + slope[None, :] * steps[:, None],
                         te.SOFT_LOW, te.SOFT_HIGH)
        self.reference = current.copy()
        self.velocity_reference = np.clip(slope, -VELOCITY_LIMIT, VELOCITY_LIMIT)
        self.preview = (future - current[None, :]).reshape(-1)
        return self.reference, self.velocity_reference, self.preview

    def torque(self, state, target, terrain=None, base=None):
        """返回 4 维力矩（N·m）；调用周期应为 10 ms。"""
        if terrain is None:
            terrain = gd.make_terrain()
        if base is None:
            base = gd.make_base()
        reference, velocity_reference, preview = self.references(target)
        angle = np.array([state.angle[i] for i in range(4)], dtype=float)
        velocity = np.array([state.velocity[i] for i in range(4)], dtype=float)
        self.filtered_velocity = ((1.0 - te.ACCELERATION_FILTER) * self.filtered_velocity
                                  + te.ACCELERATION_FILTER * velocity)
        filtered_acceleration = ((self.filtered_velocity - self.previous_filtered_velocity)
                                 / CONTROL_PERIOD)
        self.previous_filtered_velocity = self.filtered_velocity.copy()
        nominal = te.nominal_features(self.context, angle, velocity, filtered_acceleration,
                                      reference, velocity_reference, preview)
        self.previous_action_ema = ((1.0 - te.ACTION_EMA_RATE) * self.previous_action_ema
                                    + te.ACTION_EMA_RATE * self.previous_action)
        mismatch = np.clip(self.previous_action_ema - nominal[8:12],
                           -te.NOMINAL_MISMATCH_LIMIT, te.NOMINAL_MISMATCH_LIMIT)
        observation = te.observation_from_parts(
            angle, velocity, terrain.slope_angle, terrain.slope_direction,
            base.yaw, base.yaw_rate,
            np.array([base.acceleration[i] for i in range(3)], dtype=float),
            reference, velocity_reference, preview, self.previous_action,
            filtered_acceleration, nominal, mismatch, np.zeros(4))
        tensor = torch.as_tensor(observation[None], dtype=torch.float32)
        with torch.no_grad():
            action, _, self.hidden = self.agent.act_partial(tensor, self.hidden, True)
            self.load_estimate = self.agent.actor.predict_load(self.hidden).numpy()[0]
        self.previous_action = action.numpy()[0].astype(float)
        return (self.previous_action * self.scale).tolist()
