"""带序列回放的 SAC：结构化残差循环策略、循环动作价值网络与辅助头。

回放缓冲按回合保存观测、动作、奖励、终止标志、各网络每一步的上下文，以及辅助目标需要的
监督量（真实关节状态、真实角加速度、规范固定参数标签、当前等效扰动与有效惯量、负载、教师动作）。
更新时抽取固定长度窗口，窗口起点用回放时保存的上下文作为初始状态，梯度只覆盖窗口长度。

辅助损失把梯度回传到策略编码器：

- 动力学预测：慢路上下文 + 当前动作 + 真实状态 → 下一拍真实角加速度；
- 参数回归：慢路上下文 → 规范固定参数标签（28 维可辨识坐标 + 4 维相对电机增益）；
- 响应回归：慢路上下文 + 当前真实状态 → 当前等效扰动与有效惯量（动作单位）；
- 负载回归：慢路原始状态 → 动作失配滑动平均，监督在线扰动估计的递推；
- 残差行为克隆：策略残差与"教师动作减结构化动作"比较，权重很小，只作为锚定。
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from networks import (DynamicsHead, LoadHead, ParameterHead, RecurrentActor,
                      RecurrentCritic, ResponseHead)

ACCELERATION_TARGET_SCALE = 50.0
PARAMETER_DIMENSION = 32
RESPONSE_DIMENSION = 12


class SequenceBuffer:
    def __init__(self, capacity_episodes: int, device=None):
        self.capacity = capacity_episodes
        self.episodes = []
        self.device = device or torch.device("cpu")

    def add(self, episode: dict):
        self.episodes.append(episode)
        while len(self.episodes) > self.capacity:
            self.episodes.pop(0)

    def __len__(self):
        return len(self.episodes)

    def sample_windows(self, batch: int, length: int):
        indices = np.random.randint(0, len(self.episodes), batch)
        starts = [np.random.randint(0, len(self.episodes[i]["action"]) - length)
                  for i in indices]

        def stack(key, offset=0, extra=0):
            return np.stack([self.episodes[i][key][s + offset:s + offset + length + extra]
                             for i, s in zip(indices, starts)])

        arrays = {"observation": stack("observation", 0, 1), "action": stack("action"),
                  "reward": stack("reward"), "done": stack("done"),
                  "hidden_actor": np.stack([self.episodes[i]["hidden_actor"][s]
                                            for i, s in zip(indices, starts)]),
                  "hidden_critic_one": np.stack([self.episodes[i]["hidden_critic_one"][s]
                                                 for i, s in zip(indices, starts)]),
                  "hidden_critic_two": np.stack([self.episodes[i]["hidden_critic_two"][s]
                                                 for i, s in zip(indices, starts)]),
                  "state": stack("state"), "acceleration": stack("acceleration"),
                  "target": stack("target"), "response": stack("response"),
                  "teacher": stack("teacher")}
        return {key: torch.as_tensor(value, dtype=torch.float32, device=self.device)
                for key, value in arrays.items()}


class RecurrentSAC:
    def __init__(self, observation_dim: int, action_dim: int, fast_input_dim: int,
                 fast_dim: int = 64, slow_dim: int = 64, state_dim: int = 18,
                 layer_dim: int = 256, gamma: float = 0.99, tau: float = 0.005,
                 learning_rate: float = 3.0e-4, alpha_learning_rate: float = 3.0e-4,
                 target_entropy: float = None, fast_bias: float = 2.0,
                 slow_gate_bias: float = 2.0, dynamics_weight: float = 0.2,
                 parameter_weight: float = 0.2, response_weight: float = 0.2,
                 load_weight: float = 1.0, behavior_weight: float = 0.0,
                 auxiliary_learning_rate: float = 3.0e-4,
                 auxiliary_weight_decay: float = 1.0e-2, parameter_mean=None,
                 parameter_std=None, response_mean=None, response_std=None,
                 structured_weights=None, residual_scale: float = 0.3,
                 head_learning_rate: float = 3.0e-4, alpha_init: float = -2.0,
                 load_from_state: bool = False, fixed_log_std: bool = True,
                 log_std_init: float = -2.0):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.tau = tau
        self.fast_dim = fast_dim
        self.slow_dim = slow_dim
        self.hidden_dim = fast_dim + slow_dim
        self.dynamics_weight = float(dynamics_weight)
        self.parameter_weight = float(parameter_weight)
        self.response_weight = float(response_weight)
        self.load_weight = float(load_weight)
        self.behavior_weight = float(behavior_weight)
        self.actor = RecurrentActor(observation_dim, action_dim, fast_input_dim, fast_dim,
                                    slow_dim, layer_dim, fast_bias, slow_gate_bias,
                                    structured_weights=structured_weights,
                                    residual_scale=residual_scale,
                                    load_from_state=load_from_state,
                                    fixed_log_std=fixed_log_std,
                                    log_std_init=log_std_init,
                                    load_mean=None if response_mean is None else np.asarray(
                                        response_mean)[8:],
                                    load_std=None if response_std is None else np.asarray(
                                        response_std)[8:])
        self.critic_one = RecurrentCritic(observation_dim, action_dim, fast_input_dim,
                                          fast_dim, slow_dim, layer_dim, fast_bias,
                                          slow_gate_bias)
        self.critic_two = RecurrentCritic(observation_dim, action_dim, fast_input_dim,
                                          fast_dim, slow_dim, layer_dim, fast_bias,
                                          slow_gate_bias)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic_one = copy.deepcopy(self.critic_one)
        self.target_critic_two = copy.deepcopy(self.critic_two)
        for network in (self.target_actor, self.target_critic_one, self.target_critic_two):
            for parameter in network.parameters():
                parameter.requires_grad_(False)
        # 慢路与负载头构成在线扰动估计器：按一阶滑动平均解析初始化后冻结，
        # 只让快路、残差头与辅助探针头训练，避免训练过程破坏前馈补偿所依赖的估计。
        for parameter in self.actor.encoder.slow.parameters():
            parameter.requires_grad_(False)
        for parameter in self.actor.load_head.parameters():
            parameter.requires_grad_(False)
        self.dynamics_head = DynamicsHead(slow_dim, action_dim, state_dim)
        self.parameter_head = ParameterHead(slow_dim, PARAMETER_DIMENSION)
        self.response_head = ResponseHead(slow_dim, state_dim, RESPONSE_DIMENSION - 4)
        self.actor_optimizer = torch.optim.Adam(
            list(self.actor.body.parameters()) + list(self.actor.mean.parameters()),
            lr=learning_rate)
        head_parameters = (list(self.dynamics_head.parameters())
                           + list(self.parameter_head.parameters())
                           + list(self.response_head.parameters()))
        self.auxiliary_optimizer = torch.optim.Adam([
            {"params": list(self.actor.parameters()), "lr": auxiliary_learning_rate},
            {"params": head_parameters, "lr": head_learning_rate,
             "weight_decay": auxiliary_weight_decay}])
        self.critic_one_optimizer = torch.optim.Adam(self.critic_one.parameters(),
                                                     lr=learning_rate)
        self.critic_two_optimizer = torch.optim.Adam(self.critic_two.parameters(),
                                                     lr=learning_rate)
        self.log_alpha = torch.tensor(float(alpha_init), requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_learning_rate)
        self.target_entropy = float(-action_dim if target_entropy is None else target_entropy)
        self.parameter_mean = self._tensor(parameter_mean, PARAMETER_DIMENSION)
        self.parameter_std = self._tensor(parameter_std, PARAMETER_DIMENSION, ones=True)
        self.response_mean = self._tensor(response_mean, RESPONSE_DIMENSION)
        self.response_std = self._tensor(response_std, RESPONSE_DIMENSION, ones=True)
        with torch.no_grad():
            self.actor.load_from_state = bool(load_from_state)
        self.load_in_observation = True

    @staticmethod
    def _tensor(values, dimension, ones=False):
        if values is None:
            return torch.ones(dimension) if ones else torch.zeros(dimension)
        return torch.as_tensor(np.asarray(values), dtype=torch.float32)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def initial_hidden(self, batch: int):
        return (self.actor.initial_state(batch), self.critic_one.initial_state(batch),
                self.critic_two.initial_state(batch))

    @torch.no_grad()
    def act(self, observation, hidden, deterministic: bool = False):
        mean, log_std, hidden_actor = self.actor.step(observation, hidden[0])
        structured = self.actor.structured_action(observation, hidden_actor)
        if deterministic:
            action = self.actor.deterministic(mean, structured)
            log_probability = torch.zeros(mean.shape[0])
        else:
            action, log_probability = self.actor.sample(mean, log_std, structured)
        return action, log_probability, (hidden_actor, hidden[1], hidden[2])

    @torch.no_grad()
    def act_partial(self, observation, hidden_actor, deterministic: bool = False):
        mean, log_std, hidden_actor = self.actor.step(observation, hidden_actor)
        structured = self.actor.structured_action(observation, hidden_actor)
        if deterministic:
            action = self.actor.deterministic(mean, structured)
        else:
            action, _ = self.actor.sample(mean, log_std, structured)
        return action, None, hidden_actor

    @torch.no_grad()
    def residual_action(self, observation, hidden_actor):
        """给定观测与上下文返回残差动作（不含结构化前馈），用于诊断。"""
        structured = self.actor.structured_action(observation)
        mean, _ = self.actor.head_from_hidden(hidden_actor)
        return self.actor.residual_scale * torch.tanh(mean), structured

    @torch.no_grad()
    def parameter_prediction(self, observation_window, initial_hidden=None):
        hidden = self.actor.encoder(observation_window, initial_hidden)
        slow = self.actor.slow_features(hidden)
        prediction = self.parameter_head(slow)
        return prediction * self.parameter_std + self.parameter_mean

    @torch.no_grad()
    def response_prediction(self, observation_window, state, initial_hidden=None):
        hidden = self.actor.encoder(observation_window, initial_hidden)
        slow = self.actor.slow_features(hidden)
        prediction = self.response_head(slow, state)
        return prediction * self.response_std[:8] + self.response_mean[:8]

    @torch.no_grad()
    def load_prediction(self, observation_window, initial_hidden=None):
        hidden = self.actor.encoder(observation_window, initial_hidden)
        slow = self.actor.slow_features(hidden)
        return self.actor.predict_load(hidden)

    def update_critics(self, batch):
        observation = batch["observation"]
        action = batch["action"]
        reward = batch["reward"]
        done = batch["done"]
        length = action.shape[1]
        with torch.no_grad():
            hidden_actor = self.target_actor.encoder(observation, batch["hidden_actor"])
            hidden_one = self.target_critic_one.encoder(observation,
                                                        batch["hidden_critic_one"])
            hidden_two = self.target_critic_two.encoder(observation,
                                                        batch["hidden_critic_two"])
            mean, log_std = self.target_actor.head_from_hidden(
                hidden_actor[:, 1:].reshape(-1, self.hidden_dim))
            next_structured = self.target_actor.structured_action(
                observation[:, 1:], hidden_actor[:, 1:]).reshape(-1, self.action_dim)
            next_action, _ = self.target_actor.sample(mean, log_std, next_structured)
            next_action = next_action.reshape(observation.shape[0], length, self.action_dim)
            target_one = self.target_critic_one.evaluate(
                hidden_one[:, 1:].reshape(-1, self.hidden_dim),
                next_action.reshape(-1, self.action_dim))
            target_two = self.target_critic_two.evaluate(
                hidden_two[:, 1:].reshape(-1, self.hidden_dim),
                next_action.reshape(-1, self.action_dim))
            target_value = torch.minimum(target_one, target_two).reshape(reward.shape)
            target = reward + self.gamma * (1.0 - done) * target_value
        hidden_one = self.critic_one.encoder(observation[:, :length],
                                             batch["hidden_critic_one"])
        hidden_two = self.critic_two.encoder(observation[:, :length],
                                             batch["hidden_critic_two"])
        prediction_one = self.critic_one.evaluate(
            hidden_one.reshape(-1, self.hidden_dim),
            action.reshape(-1, self.action_dim)).reshape(reward.shape)
        prediction_two = self.critic_two.evaluate(
            hidden_two.reshape(-1, self.hidden_dim),
            action.reshape(-1, self.action_dim)).reshape(reward.shape)
        loss_one = F.mse_loss(prediction_one, target)
        loss_two = F.mse_loss(prediction_two, target)
        self.critic_one_optimizer.zero_grad()
        loss_one.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_one.parameters(), 10.0)
        self.critic_one_optimizer.step()
        self.critic_two_optimizer.zero_grad()
        loss_two.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_two.parameters(), 10.0)
        self.critic_two_optimizer.step()
        return float(loss_one.detach()), float(target.mean())

    def auxiliary_losses(self, auxiliary_batch):
        """辅助监督损失：参数回归、响应回归、动力学预测与行为克隆。"""
        observation = auxiliary_batch["observation"][:, :-1]
        hidden = self.actor.encoder(observation, auxiliary_batch["hidden_actor"])
        return self.auxiliary_from_hidden(auxiliary_batch, hidden)

    def auxiliary_from_hidden(self, auxiliary_batch, hidden):
        observation = auxiliary_batch["observation"][:, :-1]
        slow = self.actor.slow_features(hidden).detach()
        parameter_target = ((auxiliary_batch["target"] - self.parameter_mean)
                            / self.parameter_std)
        parameter_loss = F.mse_loss(self.parameter_head(slow), parameter_target)
        response_target = ((auxiliary_batch["response"] - self.response_mean)
                           / self.response_std)
        response_loss = F.mse_loss(
            self.response_head(slow, auxiliary_batch["state"]), response_target[..., :8])
        raw_slow = self.actor.encoder.split(hidden)[1]
        load_loss = F.mse_loss(self.actor.load_head(raw_slow.detach()),
                                response_target[..., 8:])
        dynamics_loss = F.mse_loss(
            self.dynamics_head(slow, auxiliary_batch["action"], auxiliary_batch["state"]),
            auxiliary_batch["acceleration"] / ACCELERATION_TARGET_SCALE)
        mean, _ = self.actor.head_from_hidden(hidden.reshape(-1, self.hidden_dim))
        teacher = auxiliary_batch["teacher"].reshape(-1, self.action_dim)
        structured = self.actor.structured_action(
            observation, hidden).detach().reshape(-1, self.action_dim)
        behavior_loss = F.mse_loss(torch.tanh(mean), teacher - structured)
        total = (self.parameter_weight * parameter_loss
                 + self.response_weight * response_loss
                 + self.load_weight * load_loss
                 + self.dynamics_weight * dynamics_loss
                 + self.behavior_weight * behavior_loss)
        return total, {"parameter_loss": float(parameter_loss.detach()),
                       "response_loss": float(response_loss.detach()),
                       "load_loss": float(load_loss.detach()),
                       "dynamics_loss": float(dynamics_loss.detach()),
                       "behavior_loss": float(behavior_loss.detach())}

    def update_actor(self, batch, auxiliary_batch=None):
        observation = batch["observation"][:, :-1]
        # 策略损失训练快路与残差头；慢路状态（负载估计）视为固定特征，
        # 避免价值梯度改写结构化前馈所依赖的在线估计。
        hidden_actor = self.actor.encoder(observation, batch["hidden_actor"])
        hidden_actor = torch.cat([hidden_actor[..., :self.fast_dim],
                                  hidden_actor[..., self.fast_dim:].detach()], dim=-1)
        with torch.no_grad():
            hidden_critic_one = self.critic_one.encoder(observation,
                                                        batch["hidden_critic_one"])
            hidden_critic_two = self.critic_two.encoder(observation,
                                                        batch["hidden_critic_two"])
        structured = self.actor.structured_action(
            observation, hidden_actor).reshape(-1, self.action_dim)
        mean, log_std = self.actor.head_from_hidden(
            hidden_actor.reshape(-1, self.hidden_dim))
        action, log_probability = self.actor.sample(mean, log_std, structured)
        value_one = self.critic_one.evaluate(
            hidden_critic_one.reshape(-1, self.hidden_dim), action)
        value_two = self.critic_two.evaluate(
            hidden_critic_two.reshape(-1, self.hidden_dim), action)
        value = torch.minimum(value_one, value_two)
        loss_actor = (self.alpha.detach() * log_probability - value).mean()
        statistics = {"actor_loss": 0.0, "alpha": float(self.alpha.detach()),
                      "log_probability": float(log_probability.mean().detach())}
        if auxiliary_batch is not None:
            auxiliary_loss, auxiliary_statistics = self.auxiliary_losses(auxiliary_batch)
            loss_actor = loss_actor + auxiliary_loss
            statistics.update(auxiliary_statistics)
            self.auxiliary_optimizer.zero_grad()
            loss_actor.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            for head in (self.parameter_head, self.response_head, self.actor.load_head,
                         self.dynamics_head):
                torch.nn.utils.clip_grad_norm_(head.parameters(), 10.0)
            self.auxiliary_optimizer.step()
        else:
            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.body.parameters(), 10.0)
            torch.nn.utils.clip_grad_norm_(self.actor.mean.parameters(), 10.0)
            self.actor_optimizer.step()
        statistics["actor_loss"] = float(loss_actor.detach())
        alpha_loss = -(self.log_alpha * (log_probability.detach()
                                         + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        with torch.no_grad():
            self.log_alpha.clamp_(-10.0, 2.0)
        self.soft_update()
        return statistics

    def update_auxiliary_supervised(self, auxiliary_batch):
        """只用监督目标更新编码器与辅助头，返回损失统计与窗口末端的上下文（用于分块展开）。"""
        observation = auxiliary_batch["observation"][:, :-1]
        hidden = self.actor.encoder(observation, auxiliary_batch["hidden_actor"])
        auxiliary_loss, statistics = self.auxiliary_from_hidden(auxiliary_batch, hidden)
        self.auxiliary_optimizer.zero_grad()
        auxiliary_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        for head in (self.parameter_head, self.response_head, self.actor.load_head,
                     self.dynamics_head):
            torch.nn.utils.clip_grad_norm_(head.parameters(), 10.0)
        self.auxiliary_optimizer.step()
        return statistics, hidden[:, -1].detach()

    def soft_update(self):
        with torch.no_grad():
            for source, target in ((self.actor, self.target_actor),
                                   (self.critic_one, self.target_critic_one),
                                   (self.critic_two, self.target_critic_two)):
                for source_parameter, target_parameter in zip(source.parameters(),
                                                              target.parameters()):
                    target_parameter.mul_(1.0 - self.tau).add_(self.tau * source_parameter)

    def save(self, path: Path):
        torch.save({"actor": self.actor.state_dict(),
                    "critic_one": self.critic_one.state_dict(),
                    "critic_two": self.critic_two.state_dict(),
                    "target_actor": self.target_actor.state_dict(),
                    "target_critic_one": self.target_critic_one.state_dict(),
                    "target_critic_two": self.target_critic_two.state_dict(),
                    "dynamics_head": self.dynamics_head.state_dict(),
                    "parameter_head": self.parameter_head.state_dict(),
                    "response_head": self.response_head.state_dict(),
                    "log_alpha": self.log_alpha.detach(),
                    "load_from_state": bool(self.actor.load_from_state),
                    "load_in_observation": bool(self.load_in_observation),
                    "residual_scale": float(self.actor.residual_scale),
                    "parameter_mean": self.parameter_mean,
                    "parameter_std": self.parameter_std,
                    "response_mean": self.response_mean,
                    "response_std": self.response_std}, path)

    def load(self, path: Path, load_actor_only: bool = False):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if load_actor_only:
            self.actor.load_state_dict(payload["actor"])
            return
        self.actor.load_state_dict(payload["actor"])
        self.critic_one.load_state_dict(payload["critic_one"])
        self.critic_two.load_state_dict(payload["critic_two"])
        self.target_actor.load_state_dict(payload["target_actor"])
        self.target_critic_one.load_state_dict(payload["target_critic_one"])
        self.target_critic_two.load_state_dict(payload["target_critic_two"])
        self.dynamics_head.load_state_dict(payload["dynamics_head"])
        self.parameter_head.load_state_dict(payload["parameter_head"])
        self.response_head.load_state_dict(payload["response_head"])
        if "load_from_state" in payload:
            self.actor.load_from_state = bool(payload["load_from_state"])
        if "load_in_observation" in payload:
            self.load_in_observation = bool(payload["load_in_observation"])
        if "residual_scale" in payload:
            self.actor.residual_scale = float(payload["residual_scale"])
        self.parameter_mean = payload["parameter_mean"]
        self.parameter_std = payload["parameter_std"]
        self.response_mean = payload["response_mean"]
        self.response_std = payload["response_std"]
        with torch.no_grad():
            self.log_alpha.copy_(payload["log_alpha"])
