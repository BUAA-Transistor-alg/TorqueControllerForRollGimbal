"""带序列回放的 SAC：循环策略与循环动作价值网络。

回放缓冲按回合保存观测、动作、奖励、终止标志与各网络在每一步的上下文，
更新时抽取固定长度窗口并沿时间展开求梯度；窗口起点处用回放时保存的上下文作为
初始状态，从而让梯度只需要覆盖窗口长度。

第二阶段额外加入两项针对上下文的软约束：
- 稳定项：控制一段时间之后，规范化上下文的逐步变化尽量小；
- 熵项：收敛上下文的分布熵尽量大，防止上下文退化为固定值；该系数按 SAC 的
  最大熵思想用对偶上升自动调节，目标是预先给定的上下文熵。
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

from networks import RecurrentActor, RecurrentCritic


class ParameterExtractor(nn.Module):
    """把收敛上下文映射到可辨识参数坐标，梯度回传到策略编码器。"""

    def __init__(self, input_dim: int, output_dim: int, layer_dim: int = 128):
        super().__init__()
        self.body = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, layer_dim),
                                  nn.ReLU(), nn.Linear(layer_dim, output_dim))

    def forward(self, inputs):
        return self.body(inputs)


class SequenceBuffer:
    def __init__(self, capacity_episodes: int, observation_dim: int, action_dim: int,
                 hidden_dim: int, device=None):
        self.capacity = capacity_episodes
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.episodes = []
        self.device = device or torch.device("cpu")

    def add(self, episode: dict):
        self.episodes.append(episode)
        while len(self.episodes) > self.capacity:
            self.episodes.pop(0)

    def __len__(self):
        return len(self.episodes)

    def sample_windows(self, batch: int, length: int, generator=None):
        indices = np.random.randint(0, len(self.episodes), batch)
        starts = []
        for index in indices:
            episode = self.episodes[index]
            starts.append(np.random.randint(0, len(episode["action"]) - length))
        observation = np.stack([self.episodes[i]["observation"][s:s + length + 1]
                                for i, s in zip(indices, starts)])
        action = np.stack([self.episodes[i]["action"][s:s + length]
                           for i, s in zip(indices, starts)])
        reward = np.stack([self.episodes[i]["reward"][s:s + length]
                           for i, s in zip(indices, starts)])
        done = np.stack([self.episodes[i]["done"][s:s + length]
                         for i, s in zip(indices, starts)])
        hidden_actor = np.stack([self.episodes[i]["hidden_actor"][s]
                                 for i, s in zip(indices, starts)])
        hidden_critic_one = np.stack([self.episodes[i]["hidden_critic_one"][s]
                                      for i, s in zip(indices, starts)])
        hidden_critic_two = np.stack([self.episodes[i]["hidden_critic_two"][s]
                                      for i, s in zip(indices, starts)])
        steps = np.stack([np.arange(s, s + length) for s in starts])
        return {
            "observation": torch.as_tensor(observation, dtype=torch.float32, device=self.device),
            "action": torch.as_tensor(action, dtype=torch.float32, device=self.device),
            "reward": torch.as_tensor(reward, dtype=torch.float32, device=self.device),
            "done": torch.as_tensor(done, dtype=torch.float32, device=self.device),
            "hidden_actor": torch.as_tensor(hidden_actor, dtype=torch.float32, device=self.device),
            "hidden_critic_one": torch.as_tensor(hidden_critic_one, dtype=torch.float32, device=self.device),
            "hidden_critic_two": torch.as_tensor(hidden_critic_two, dtype=torch.float32, device=self.device),
            "steps": torch.as_tensor(steps, dtype=torch.long),
        }

    def sample_converged(self, batch: int, length: int):
        """从每个回合末尾取固定长度的窗口，返回观测、初始上下文与参数可辨识坐标。"""
        indices = np.random.randint(0, len(self.episodes), batch)
        windows = []
        hidden = []
        targets = []
        for index in indices:
            episode = self.episodes[index]
            total = len(episode["action"])
            start = max(0, total - length)
            windows.append(episode["observation"][start:start + length])
            hidden.append(episode["hidden_actor"][start])
            targets.append(episode["coordinates"])
        return (torch.as_tensor(np.stack(windows), dtype=torch.float32, device=self.device),
                torch.as_tensor(np.stack(hidden), dtype=torch.float32, device=self.device),
                torch.as_tensor(np.stack(targets), dtype=torch.float32, device=self.device))


class RecurrentSAC:
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64,
                 layer_dim: int = 128, gamma: float = 0.99, tau: float = 0.005,
                 learning_rate: float = 3e-4, alpha_learning_rate: float = 3e-4,
                 target_entropy: float = None, gate_bias: float = 2.0,
                 stability_weight: float = 0.0, stability_warmup: int = 0,
                 context_entropy_target: float = 0.0,
                 context_entropy_learning_rate: float = 1e-3,
                 context_entropy_weight: float = 0.0,
                 extractor_weight: float = 0.0,
                 extractor_layer_dim: int = 128,
                 extractor_learning_rate: float = 3e-4,
                 coordinate_weights=None):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.stability_weight = stability_weight
        self.stability_warmup = stability_warmup
        self.context_entropy_weight = context_entropy_weight
        self.context_entropy_target = context_entropy_target
        self.actor = RecurrentActor(observation_dim, action_dim, hidden_dim, layer_dim, gate_bias)
        self.critic_one = RecurrentCritic(observation_dim, action_dim, hidden_dim, layer_dim, gate_bias)
        self.critic_two = RecurrentCritic(observation_dim, action_dim, hidden_dim, layer_dim, gate_bias)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic_one = copy.deepcopy(self.critic_one)
        self.target_critic_two = copy.deepcopy(self.critic_two)
        for network in (self.target_actor, self.target_critic_one, self.target_critic_two):
            for parameter in network.parameters():
                parameter.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_one_optimizer = torch.optim.Adam(self.critic_one.parameters(),
                                                     lr=learning_rate)
        self.critic_two_optimizer = torch.optim.Adam(self.critic_two.parameters(),
                                                     lr=learning_rate)
        self.log_alpha = torch.tensor(0.0, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_learning_rate)
        self.target_entropy = float(-action_dim if target_entropy is None else target_entropy)
        self.log_context_lambda = torch.tensor(0.0, requires_grad=True)
        self.context_optimizer = torch.optim.Adam([self.log_context_lambda],
                                                  lr=context_entropy_learning_rate)
        self.extractor_weight = float(extractor_weight)
        self.extractor = ParameterExtractor(hidden_dim, 28, extractor_layer_dim)
        self.extractor_optimizer = torch.optim.Adam(self.extractor.parameters(),
                                                    lr=extractor_learning_rate,
                                                    weight_decay=1.0e-2)
        self.coordinate_weights = (torch.ones(28) if coordinate_weights is None
                                   else torch.as_tensor(coordinate_weights, dtype=torch.float32))

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def context_lambda(self):
        return self.log_context_lambda.exp()

    def initial_hidden(self, batch: int):
        return (self.actor.initial_state(batch), self.critic_one.initial_state(batch),
                self.critic_two.initial_state(batch))

    @torch.no_grad()
    def act(self, observation, hidden, deterministic: bool = False):
        mean, log_std, hidden_actor = self.actor.step(observation, hidden[0])
        if deterministic:
            action = torch.tanh(mean)
            log_probability = torch.zeros(mean.shape[0])
        else:
            action, log_probability = self.actor.sample(mean, log_std)
        return action, log_probability, (hidden_actor, hidden[1], hidden[2])

    @torch.no_grad()
    def act_partial(self, observation, hidden_actor, deterministic: bool = False):
        """只推进策略编码器，返回动作与新的上下文。"""
        mean, log_std, hidden_actor = self.actor.step(observation, hidden_actor)
        if deterministic:
            action = torch.tanh(mean)
        else:
            action, _ = self.actor.sample(mean, log_std)
        return action, None, hidden_actor

    def update_critics(self, batch):
        observation = batch["observation"]
        action = batch["action"]
        reward = batch["reward"]
        done = batch["done"]
        length = action.shape[1]
        with torch.no_grad():
            hidden_actor = self.target_actor.encoder(observation,
                                                     batch["hidden_actor"])
            hidden_one = self.target_critic_one.encoder(observation,
                                                        batch["hidden_critic_one"])
            hidden_two = self.target_critic_two.encoder(observation,
                                                        batch["hidden_critic_two"])
            mean, log_std, _ = self.target_actor.step(observation[:, 1:].reshape(
                -1, self.observation_dim), hidden_actor[:, 1:].reshape(-1, hidden_actor.shape[-1]))
            next_action, _ = self.target_actor.sample(mean, log_std)
            next_action = next_action.reshape(observation.shape[0], length, self.action_dim)
            target_one = self.target_critic_one.evaluate(
                hidden_one[:, 1:].reshape(-1, hidden_one.shape[-1]), next_action.reshape(-1, self.action_dim))
            target_two = self.target_critic_two.evaluate(
                hidden_two[:, 1:].reshape(-1, hidden_two.shape[-1]), next_action.reshape(-1, self.action_dim))
            target_value = torch.minimum(target_one, target_two).reshape(reward.shape)
            target = reward + self.gamma * (1.0 - done) * target_value
        hidden_one = self.critic_one.encoder(observation[:, :length],
                                             batch["hidden_critic_one"])
        hidden_two = self.critic_two.encoder(observation[:, :length],
                                             batch["hidden_critic_two"])
        prediction_one = self.critic_one.evaluate(
            hidden_one.reshape(-1, hidden_one.shape[-1]), action.reshape(-1, self.action_dim)).reshape(reward.shape)
        prediction_two = self.critic_two.evaluate(
            hidden_two.reshape(-1, hidden_two.shape[-1]), action.reshape(-1, self.action_dim)).reshape(reward.shape)
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

    def update_actor(self, batch, buffer, converged_length: int = 32,
                     converged_batch: int = 32):
        observation = batch["observation"][:, :-1]
        length = observation.shape[1]
        hidden_actor = self.actor.encoder(observation, batch["hidden_actor"])
        hidden_critic_one = self.critic_one.encoder(observation, batch["hidden_critic_one"])
        hidden_critic_two = self.critic_two.encoder(observation, batch["hidden_critic_two"])
        mean, log_std, _ = self.actor.step(observation.reshape(-1, self.observation_dim),
                                           hidden_actor.reshape(-1, hidden_actor.shape[-1]))
        action, log_probability = self.actor.sample(mean, log_std)
        value_one = self.critic_one.evaluate(hidden_critic_one.reshape(
            -1, hidden_critic_one.shape[-1]), action)
        value_two = self.critic_two.evaluate(hidden_critic_two.reshape(
            -1, hidden_critic_two.shape[-1]), action)
        value = torch.minimum(value_one, value_two)
        loss_actor = (self.alpha.detach() * log_probability - value).mean()

        # 上下文正则都作用在方向归一化后的上下文上，消除靠放大尺度绕过惩罚的可能
        stability = torch.tensor(0.0)
        if self.stability_weight > 0.0:
            directions = F.normalize(hidden_actor, dim=-1)
            differences = ((directions[:, 1:] - directions[:, :-1]) ** 2).sum(dim=-1)
            mask = (batch["steps"][:, 1:] > self.stability_warmup).float()
            stability = (differences * mask).sum() / mask.sum().clamp(min=1.0)
            loss_actor = loss_actor + self.stability_weight * stability

        context_entropy = torch.tensor(0.0)
        extractor_loss = torch.tensor(0.0)
        if self.context_entropy_weight > 0.0 or self.extractor_weight > 0.0:
            converged_observation, converged_hidden, targets = buffer.sample_converged(
                converged_batch, converged_length)
            converged = self.actor.encoder(converged_observation, converged_hidden)
            context_mean = self.actor.normalization(converged).mean(dim=1)
            if self.context_entropy_weight > 0.0:
                # 归一化熵：各维方差相对均匀分布的比值，均匀分布时为 0，常值为负无穷
                variance = context_mean.var(dim=0) + 1.0e-4
                context_entropy = 0.5 * torch.log(
                    variance * self.actor.encoder.hidden_dim).mean()
                loss_actor = loss_actor - self.context_lambda * context_entropy
            if self.extractor_weight > 0.0:
                prediction = self.extractor(context_mean)
                extractor_loss = (((prediction - targets) ** 2)
                                  * self.coordinate_weights).mean()
                loss_actor = loss_actor + self.extractor_weight * extractor_loss

        self.actor_optimizer.zero_grad()
        self.extractor_optimizer.zero_grad()
        loss_actor.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optimizer.step()
        if self.extractor_weight > 0.0:
            torch.nn.utils.clip_grad_norm_(self.extractor.parameters(), 10.0)
            self.extractor_optimizer.step()

        # 温度系数：熵高于目标则降低 alpha
        alpha_loss = -(self.log_alpha * (log_probability.detach()
                                         + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        with torch.no_grad():
            self.log_alpha.clamp_(-10.0, 2.0)

        if self.context_entropy_weight > 0.0:
            context_loss = self.log_context_lambda * (context_entropy.detach()
                                                      - self.context_entropy_target)
            self.context_optimizer.zero_grad()
            context_loss.backward()
            self.context_optimizer.step()
            with torch.no_grad():
                self.log_context_lambda.clamp_(-8.0, 4.0)

        self.soft_update()
        return {"actor_loss": float(loss_actor.detach()),
                "alpha": float(self.alpha.detach()),
                "stability": float(stability.detach()),
                "context_entropy": float(context_entropy.detach()),
                "extractor_loss": float(extractor_loss.detach()),
                "context_lambda": float(self.context_lambda.detach()),
                "log_probability": float(log_probability.mean().detach())}

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
                    "log_alpha": self.log_alpha.detach(),
                    "log_context_lambda": self.log_context_lambda.detach(),
                    "extractor": self.extractor.state_dict()}, path)

    def load(self, path: Path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.actor.load_state_dict(payload["actor"])
        self.critic_one.load_state_dict(payload["critic_one"])
        self.critic_two.load_state_dict(payload["critic_two"])
        self.target_actor.load_state_dict(payload["target_actor"])
        self.target_critic_one.load_state_dict(payload["target_critic_one"])
        self.target_critic_two.load_state_dict(payload["target_critic_two"])
        if "extractor" in payload:
            try:
                self.extractor.load_state_dict(payload["extractor"])
            except RuntimeError:
                print("提取器结构变化，重新初始化")
        with torch.no_grad():
            self.log_alpha.copy_(payload["log_alpha"])
            self.log_context_lambda.copy_(payload["log_context_lambda"])