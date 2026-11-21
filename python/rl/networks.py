"""循环网络：门控线性递归上下文、策略头与动作价值头。

上下文递推为线性形式 h_t = a_t * h_{t-1} + (1 - a_t) * W x_t，其中 a_t 由输入决定。
这样 h_t 是历史输入的加权累加，单个固定大小的向量承载全部历史信息；梯度沿时间展开
由 PyTorch 自动完成。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0


class GatedLinearRNN(nn.Module):
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

    def forward(self, inputs, initial=None):
        batch, steps = inputs.shape[0], inputs.shape[1]
        hidden = self.initial_state(batch, inputs.device) if initial is None else initial
        states = []
        for index in range(steps):
            hidden = self.step(inputs[:, index], hidden)
            states.append(hidden)
        return torch.stack(states, dim=1)


class RecurrentActor(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64,
                 layer_dim: int = 128, gate_bias: float = 2.0, log_std_init: float = -2.0):
        super().__init__()
        self.action_dim = action_dim
        self.encoder = GatedLinearRNN(observation_dim, hidden_dim, gate_bias)
        self.normalization = nn.LayerNorm(hidden_dim)
        self.body = nn.Sequential(nn.Linear(hidden_dim, layer_dim), nn.ReLU(),
                                  nn.Linear(layer_dim, layer_dim), nn.ReLU())
        self.mean = nn.Linear(layer_dim, action_dim)
        self.log_std = nn.Linear(layer_dim, action_dim)
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        nn.init.constant_(self.log_std.bias, log_std_init)

    def initial_state(self, batch: int, device=None):
        return self.encoder.initial_state(batch, device)

    def step(self, observation, hidden):
        hidden = self.encoder.step(observation, hidden)
        features = self.body(self.normalization(hidden))
        mean = self.mean(features)
        log_std = torch.clamp(self.log_std(features), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std, hidden

    def sample(self, mean, log_std):
        std = log_std.exp()
        noise = torch.randn_like(mean)
        value = mean + std * noise
        action = torch.tanh(value)
        log_probability = (-0.5 * (((value - mean) / std) ** 2 + 2.0 * log_std
                                   + math.log(2.0 * math.pi))).sum(dim=-1)
        log_probability -= torch.log(1.0 - action ** 2 + 1.0e-6).sum(dim=-1)
        return action, log_probability


class RecurrentCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64,
                 layer_dim: int = 128, gate_bias: float = 2.0):
        super().__init__()
        self.encoder = GatedLinearRNN(observation_dim, hidden_dim, gate_bias)
        self.normalization = nn.LayerNorm(hidden_dim)
        self.body = nn.Sequential(nn.Linear(hidden_dim + action_dim, layer_dim), nn.ReLU(),
                                  nn.Linear(layer_dim, layer_dim), nn.ReLU(),
                                  nn.Linear(layer_dim, 1))

    def initial_state(self, batch: int, device=None):
        return self.encoder.initial_state(batch, device)

    def step(self, observation, action, hidden):
        hidden = self.encoder.step(observation, hidden)
        features = torch.cat([self.normalization(hidden), action], dim=-1)
        return self.body(features).squeeze(-1), hidden

    def evaluate(self, hidden, action):
        features = torch.cat([self.normalization(hidden), action], dim=-1)
        return self.body(features).squeeze(-1)


class SoftUpdate:
    def __init__(self, source: nn.Module, target: nn.Module, tau: float):
        self.source = source
        self.target = target
        self.tau = tau

    def __call__(self):
        with torch.no_grad():
            for source_parameter, target_parameter in zip(self.source.parameters(),
                                                          self.target.parameters()):
                target_parameter.mul_(1.0 - self.tau).add_(self.tau * source_parameter)
