"""循环网络：快慢两路上下文、结构化前馈加循环残差策略、动作价值头与辅助头。

控制器维护两路状态：

- 快路（门控线性递归，`h = a * h + (1 - a) * W x`，时间常数几个控制周期）吃基础观测与当前
  标称模型物理特征，负责跟踪当前状态，支撑反馈控制；
- 慢路（GRU）只吃动作失配与负载特征，按一阶滑动平均解析初始化，负责在线估计等效扰动；
  它的输出经线性负载头给出控制器使用的负载估计，是策略在线适应的载体。

动作 = clip(结构化动作 + 残差幅度 × tanh(残差均值 + 噪声), -1, 1)。结构化动作由标称物理特征与
负载估计按标定权重线性组合（等价于一个固定初值的自适应控制器），循环残差学习上下文相关的修正。
慢路另外接参数回归头、响应回归头与动力学预测头（见 `sac.py`），把动力学信息压进隐藏状态。
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0


class GatedLinearRNN(nn.Module):
    """线性记忆：h_t = a_t * h_{t-1} + (1 - a_t) * W x_t。"""

    def __init__(self, input_dim: int, hidden_dim: int, gate_bias: float = 2.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.gate = nn.Linear(input_dim, hidden_dim)
        nn.init.constant_(self.gate.bias, gate_bias)

    def initial_state(self, batch: int, device=None):
        return torch.zeros(batch, self.hidden_dim, device=device)

    def step(self, x, hidden):
        gate = torch.sigmoid(self.gate(x))
        return gate * hidden + (1.0 - gate) * self.projection(x)


class DualEncoder(nn.Module):
    """快路门控线性递归 + 慢路 GRU；输入向量按 `fast_input_dim` 切成两段分别送入。"""

    def __init__(self, observation_dim: int, fast_input_dim: int, fast_dim: int = 64,
                 slow_dim: int = 64, fast_bias: float = 2.0, slow_gate_bias: float = 2.0,
                 load_rate: float = 0.1, load_dim: int = 4):
        super().__init__()
        self.fast_input_dim = fast_input_dim
        self.slow_input_dim = observation_dim - fast_input_dim
        self.fast_dim = fast_dim
        self.slow_dim = slow_dim
        self.hidden_dim = fast_dim + slow_dim
        self.fast = GatedLinearRNN(fast_input_dim, fast_dim, fast_bias)
        self.slow = nn.GRUCell(self.slow_input_dim, slow_dim)
        # 慢路一开始就实现动作失配的一阶滑动平均 h_t = (1-rate) h_{t-1} + rate * 失配：
        # 更新门偏置取 logit(1-rate)，候选状态的前 load_dim 维等于输入的前 load_dim 维，
        # 其余输入权重与循环权重置零。这样在线负载估计直接落在隐藏状态里，策略从一开始
        # 就与标定好的结构化控制器等价。
        keep = 1.0 - float(load_rate)
        gate_bias = math.log(keep / (1.0 - keep)) if keep < 1.0 else slow_gate_bias
        with torch.no_grad():
            self.slow.bias_ih.zero_()
            self.slow.bias_hh.zero_()
            self.slow.bias_hh[slow_dim:2 * slow_dim].fill_(gate_bias)
            self.slow.weight_ih.zero_()
            self.slow.weight_hh.zero_()
            for index in range(min(load_dim, self.slow_input_dim, slow_dim)):
                self.slow.weight_ih[2 * slow_dim + index, index] = 1.0

    def initial_state(self, batch: int, device=None):
        return torch.zeros(batch, self.hidden_dim, device=device)

    def step(self, observation, hidden):
        fast_input = observation[..., :self.fast_input_dim]
        slow_input = observation[..., self.fast_input_dim:]
        fast = self.fast.step(fast_input, hidden[:, :self.fast_dim])
        slow = self.slow(slow_input, hidden[:, self.fast_dim:])
        return torch.cat([fast, slow], dim=-1)

    def forward(self, inputs, initial=None):
        batch, steps = inputs.shape[0], inputs.shape[1]
        hidden = self.initial_state(batch, inputs.device) if initial is None else initial
        states = []
        for index in range(steps):
            hidden = self.step(inputs[:, index], hidden)
            states.append(hidden)
        return torch.stack(states, dim=1)

    def split(self, hidden):
        return hidden[..., :self.fast_dim], hidden[..., self.fast_dim:]


class LoadHead(nn.Module):
    """由慢路原始状态线性读出动作残差滑动平均（控制器在线估计的等效扰动）。"""

    def __init__(self, slow_dim: int, output_dim: int = 4):
        super().__init__()
        self.projection = nn.Linear(slow_dim, output_dim)

    def forward(self, slow):
        return self.projection(slow)


class RecurrentActor(nn.Module):
    """结构化自适应控制器加循环残差策略。

    输出动作 = clip(结构化动作 + tanh(残差均值 + 高斯噪声), -1, 1)。结构化动作由观测里的
    标称物理特征（模型基前馈需求、偏置力矩）与动作失配滑动平均按标定权重线性组合，相当于一个
    固定初值的自适应控制器；循环网络学习对它做上下文相关修正。策略一开始就具备教师级别的
    跟踪与限位行为，强化学习只需学残差，避开了从零模仿高增益控制器的精度瓶颈。
    """

    def __init__(self, observation_dim: int, action_dim: int, fast_input_dim: int,
                 fast_dim: int = 64, slow_dim: int = 64, layer_dim: int = 256,
                 fast_bias: float = 2.0, slow_gate_bias: float = 2.0,
                 structured_weights=None, nominal_scale: float = 5.0,
                 demand_start: int = 126, bias_start: int = 130, load_start: int = 138,
                 residual_scale: float = 0.3, load_from_state: bool = False,
                 load_mean=None, load_std=None, log_std_init: float = -2.0,
                 fixed_log_std: bool = True):
        super().__init__()
        self.action_dim = action_dim
        self.encoder = DualEncoder(observation_dim, fast_input_dim, fast_dim, slow_dim,
                                   fast_bias, slow_gate_bias)
        self.hidden_dim = self.encoder.hidden_dim
        self.fast_normalization = nn.LayerNorm(fast_dim)
        self.slow_normalization = nn.LayerNorm(slow_dim)
        self.body = nn.Sequential(nn.Linear(self.hidden_dim, layer_dim), nn.ReLU(),
                                  nn.Linear(layer_dim, layer_dim), nn.ReLU())
        self.mean = nn.Linear(layer_dim, action_dim)
        self.log_std = nn.Linear(layer_dim, action_dim)
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        nn.init.constant_(self.log_std.bias, log_std_init)
        # 残差是在一个已经工作的控制器上的小修正，探索噪声必须小且稳定：
        # 让标准差固定为常数，避免价值梯度把噪声不断推大。
        self.fixed_log_std = bool(fixed_log_std)
        self.log_std_value = float(log_std_init)
        weights = np.zeros((4, 4)) if structured_weights is None else np.asarray(
            structured_weights, dtype=float)
        self.register_buffer("structured_weights",
                             torch.as_tensor(weights, dtype=torch.float32))
        self.nominal_scale = float(nominal_scale)
        self.residual_scale = float(residual_scale)
        self.demand_start = int(demand_start)
        self.bias_start = int(bias_start)
        self.load_start = int(load_start)
        self.load_from_state = bool(load_from_state)
        self.load_head = LoadHead(slow_dim)
        self.register_buffer("load_mean", torch.zeros(4) if load_mean is None else (
            torch.as_tensor(np.asarray(load_mean), dtype=torch.float32)))
        self.register_buffer("load_std", torch.ones(4) if load_std is None else (
            torch.as_tensor(np.asarray(load_std), dtype=torch.float32)))
        with torch.no_grad():
            self.load_head.projection.weight.zero_()
            self.load_head.projection.bias.zero_()
            for index in range(min(4, slow_dim)):
                self.load_head.projection.weight[index, index] = (
                    1.0 / float(self.load_std[index]))
            self.load_head.projection.bias.copy_(-self.load_mean / self.load_std)

    def predict_load(self, hidden):
        """由慢路原始状态预测动作残差滑动平均（与观测里的负载特征同量纲）。"""
        slow = self.encoder.split(hidden)[1]
        return self.load_head(slow) * self.load_std + self.load_mean

    def structured_action(self, observation, hidden=None):
        """由标称物理特征与负载估计得到结构化动作（与标定教师完全相同的固定线性律）。

        `load_from_state` 为真时，负载由慢路上下文预测，此时在线适应完全依靠循环状态；
        为假时直接使用观测里的负载特征。
        """
        demand = observation[..., self.demand_start:self.demand_start + 4] / self.nominal_scale
        bias = observation[..., self.bias_start:self.bias_start + 4] / self.nominal_scale
        if self.load_from_state:
            if hidden is None:
                raise ValueError("load_from_state 为真时必须提供隐藏状态")
            load = self.predict_load(hidden)
        else:
            load = observation[..., self.load_start:self.load_start + 4]
        features = torch.stack([demand, bias, load, torch.ones_like(load)], dim=-1)
        return torch.clamp((features * self.structured_weights).sum(dim=-1), -1.0, 1.0)

    def initial_state(self, batch: int, device=None):
        return self.encoder.initial_state(batch, device)

    def slow_features(self, hidden):
        return self.slow_normalization(self.encoder.split(hidden)[1])

    def features_from_hidden(self, hidden):
        fast, slow = self.encoder.split(hidden)
        return torch.cat([self.fast_normalization(fast),
                          self.slow_normalization(slow)], dim=-1)

    def head_from_hidden(self, hidden):
        features = self.body(self.features_from_hidden(hidden))
        mean = self.mean(features)
        if self.fixed_log_std:
            log_std = torch.full_like(mean, self.log_std_value)
        else:
            log_std = torch.clamp(self.log_std(features), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def step(self, observation, hidden):
        hidden = self.encoder.step(observation, hidden)
        mean, log_std = self.head_from_hidden(hidden)
        return mean, log_std, hidden

    def sample(self, mean, log_std, structured=None):
        std = log_std.exp()
        noise = torch.randn_like(mean)
        value = mean + std * noise
        delta = torch.tanh(value)
        if structured is None:
            structured = torch.zeros_like(delta)
        action = torch.clamp(structured + self.residual_scale * delta, -1.0, 1.0)
        log_probability = (-0.5 * (((value - mean) / std) ** 2 + 2.0 * log_std
                                   + math.log(2.0 * math.pi))).sum(dim=-1)
        log_probability -= torch.log(1.0 - delta ** 2 + 1.0e-6).sum(dim=-1)
        return action, log_probability

    def deterministic(self, mean, structured):
        return torch.clamp(structured + self.residual_scale * torch.tanh(mean), -1.0, 1.0)


class RecurrentCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, fast_input_dim: int,
                 fast_dim: int = 64, slow_dim: int = 64, layer_dim: int = 256,
                 fast_bias: float = 2.0, slow_gate_bias: float = 2.0):
        super().__init__()
        self.encoder = DualEncoder(observation_dim, fast_input_dim, fast_dim, slow_dim,
                                   fast_bias, slow_gate_bias)
        self.hidden_dim = self.encoder.hidden_dim
        self.fast_normalization = nn.LayerNorm(fast_dim)
        self.slow_normalization = nn.LayerNorm(slow_dim)
        self.body = nn.Sequential(
            nn.Linear(self.hidden_dim + action_dim, layer_dim), nn.ReLU(),
            nn.Linear(layer_dim, layer_dim), nn.ReLU(), nn.Linear(layer_dim, 1))

    def initial_state(self, batch: int, device=None):
        return self.encoder.initial_state(batch, device)

    def features_from_hidden(self, hidden):
        fast, slow = self.encoder.split(hidden)
        return torch.cat([self.fast_normalization(fast),
                          self.slow_normalization(slow)], dim=-1)

    def step(self, observation, action, hidden):
        hidden = self.encoder.step(observation, hidden)
        features = torch.cat([self.features_from_hidden(hidden), action], dim=-1)
        return self.body(features).squeeze(-1), hidden

    def evaluate(self, hidden, action):
        features = torch.cat([self.features_from_hidden(hidden), action], dim=-1)
        return self.body(features).squeeze(-1)


class DynamicsHead(nn.Module):
    """由慢路上下文、当前动作与真实状态预测下一拍关节角加速度。"""

    def __init__(self, slow_dim: int, action_dim: int, state_dim: int,
                 layer_dim: int = 128, output_dim: int = 4):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(slow_dim + action_dim + state_dim, layer_dim),
                                  nn.ReLU(), nn.Linear(layer_dim, layer_dim), nn.ReLU(),
                                  nn.Linear(layer_dim, output_dim))

    def forward(self, slow, action, state):
        return self.body(torch.cat([slow, action, state], dim=-1))


class ParameterHead(nn.Module):
    """由慢路上下文回归规范固定后的参数标签（28 维可辨识坐标 + 4 维相对电机增益）。"""

    def __init__(self, slow_dim: int, output_dim: int = 32, layer_dim: int = 128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(slow_dim, layer_dim), nn.ReLU(),
                                  nn.Linear(layer_dim, layer_dim), nn.ReLU(),
                                  nn.Linear(layer_dim, output_dim))

    def forward(self, slow):
        return self.body(slow)


class ResponseHead(nn.Module):
    """由慢路上下文与当前真实状态回归当前等效扰动与有效惯量（动作单位）。"""

    def __init__(self, slow_dim: int, state_dim: int, output_dim: int = 8,
                 layer_dim: int = 128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(slow_dim + state_dim, layer_dim), nn.ReLU(),
                                  nn.Linear(layer_dim, layer_dim), nn.ReLU(),
                                  nn.Linear(layer_dim, output_dim))

    def forward(self, slow, state):
        return self.body(torch.cat([slow, state], dim=-1))
