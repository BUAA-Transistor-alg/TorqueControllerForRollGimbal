"""训练循环策略的 SAC：第一阶段固定参数，第二阶段随机参数并使用上下文正则。

用法（在仓库根目录）：
    python3 python/rl/train.py --stage 1 --iterations 200 --save data/rl_stage1.pt
    python3 python/rl/train.py --stage 2 --parameter-scale 0.35 --iterations 200 \
        --resume data/rl_stage1.pt --stability-weight 2.0 --context-entropy-target -40 \
        --save data/rl_stage2.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "identification"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sac import RecurrentSAC, SequenceBuffer
from tracking_env import OBSERVATION_DIM, TrackingEnv, identifiable_coordinates, sample_parameters


class RewardNormalizer:
    """奖励标准化：用运行均值与标准差把奖励压到 O(1)，避免动作价值量级过大。"""

    def __init__(self):
        self.count = 1.0e-4
        self.mean = -1.0
        self.variance = 25.0

    def update(self, rewards: np.ndarray):
        batch_mean = float(rewards.mean())
        batch_variance = float(rewards.var())
        batch_count = float(rewards.size)
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / total
        self.variance = (self.variance * self.count + batch_variance * batch_count
                         + delta ** 2 * self.count * batch_count / total) / total
        self.count = total

    def normalize(self, rewards: np.ndarray) -> np.ndarray:
        return np.clip((rewards - self.mean) / math.sqrt(self.variance + 1.0e-6), -10.0, 10.0)


def rollout(agent: RecurrentSAC, envs, buffer: SequenceBuffer, observation_batch: np.ndarray,
            episode_steps: int, deterministic: bool = False, collect: bool = True,
            normalizer: RewardNormalizer = None):
    count = len(envs)
    hidden_actor, hidden_one, hidden_two = agent.initial_hidden(count)
    records = {"observation": [observation_batch.copy()], "action": [], "reward": [],
               "done": [], "hidden_actor": [], "hidden_critic_one": [],
               "hidden_critic_two": [], "reference": []}
    errors = []
    raw_rewards = []
    for step in range(episode_steps):
        observation = torch.as_tensor(observation_batch, dtype=torch.float32)
        with torch.no_grad():
            action, _, hidden_actor = agent.act_partial(observation, hidden_actor,
                                                        deterministic)
            hidden_one = agent.critic_one.encoder.step(observation, hidden_one)
            hidden_two = agent.critic_two.encoder.step(observation, hidden_two)
        records["hidden_actor"].append(hidden_actor.numpy())
        records["hidden_critic_one"].append(hidden_one.numpy())
        records["hidden_critic_two"].append(hidden_two.numpy())
        actions = action.numpy()
        next_observations = np.zeros_like(observation_batch)
        rewards = np.zeros(count)
        dones = np.zeros(count)
        for index, env in enumerate(envs):
            next_observation, reward, done, info = env.step(actions[index])
            next_observations[index] = next_observation
            rewards[index] = reward
            dones[index] = float(done)
            errors.append(np.abs(info["error"]))
            records["reference"].append(env.angle_ref[min(env.step_index - 1,
                                                          env.episode_steps - 1)].copy())
        records["action"].append(actions)
        raw_rewards.append(rewards.copy())
        records["reward"].append(normalizer.normalize(rewards) if normalizer is not None
                                 else rewards)
        records["done"].append(dones)
        records["observation"].append(next_observations.copy())
        observation_batch = next_observations
    with torch.no_grad():
        observation = torch.as_tensor(observation_batch, dtype=torch.float32)
        final_actor = agent.actor.encoder.step(observation, hidden_actor)
        final_one = agent.critic_one.encoder.step(observation, hidden_one)
        final_two = agent.critic_two.encoder.step(observation, hidden_two)
    records["hidden_actor"].append(final_actor.numpy())
    records["hidden_critic_one"].append(final_one.numpy())
    records["hidden_critic_two"].append(final_two.numpy())
    if collect:
        coordinates = np.stack([env.info()["coordinates"] for env in envs])
        for index in range(count):
            episode = {
                "coordinates": coordinates[index],
                "observation": np.stack([item[index] for item in records["observation"]]),
                "action": np.stack([item[index] for item in records["action"]]),
                "reward": np.array([item[index] for item in records["reward"]]),
                "done": np.array([item[index] for item in records["done"]]),
                "hidden_actor": np.stack([item[index] for item in records["hidden_actor"]]),
                "hidden_critic_one": np.stack([item[index] for item in records["hidden_critic_one"]]),
                "hidden_critic_two": np.stack([item[index] for item in records["hidden_critic_two"]]),
            }
            buffer.add(episode)
    returns = np.sum(np.stack(raw_rewards), axis=0)
    return observation_batch, np.array(errors), float(returns.mean()), np.concatenate(raw_rewards)


def collect_extractor_data(agent, envs, steps: int, window: int = 128):
    """用当前策略在若干新回合上采集提取器训练样本（观测窗口、起始上下文、参数坐标）。"""
    windows, hiddens, coordinates = [], [], []
    for env in envs:
        observation = env.reset()
        hidden, _, _ = agent.initial_hidden(1)
        observations = []
        hidden_states = []
        info = None
        for _ in range(steps):
            observations.append(observation)
            hidden_states.append(hidden[0].numpy())
            tensor = torch.as_tensor(observation[None], dtype=torch.float32)
            with torch.no_grad():
                action, _, hidden = agent.act_partial(tensor, hidden, True)
            observation, reward, done, info = env.step(action.numpy()[0])
        start = max(0, len(observations) - window)
        windows.append(np.stack(observations[start:]))
        hiddens.append(hidden_states[start])
        coordinates.append(info["coordinates"])
    window_tensor = torch.as_tensor(np.stack(windows), dtype=torch.float32)
    hidden_tensor = torch.as_tensor(np.stack(hiddens), dtype=torch.float32)
    with torch.no_grad():
        context = agent.converged_context(window_tensor, hidden_tensor).numpy()
    return {"window": window_tensor, "hidden": hidden_tensor, "context": context,
            "coordinates": np.stack(coordinates).astype(np.float32)}


def evaluate(agent: RecurrentSAC, seeds, episode_steps: int, parameter_scale: float,
             deterministic: bool = True):
    """多回合评估：返回整体跟踪误差、前段与后段误差、以及收敛上下文。"""
    results = []
    for seed in seeds:
        env = TrackingEnv(seed=seed, episode_steps=episode_steps,
                          parameter_scale=parameter_scale)
        observation = env.reset()
        hidden_actor, hidden_one, hidden_two = agent.initial_hidden(1)
        errors = []
        for step in range(episode_steps):
            tensor = torch.as_tensor(observation[None], dtype=torch.float32)
            with torch.no_grad():
                action, _, hidden_actor = agent.act_partial(tensor, hidden_actor,
                                                            deterministic)
            observation, reward, done, info = env.step(action.numpy()[0])
            errors.append(np.abs(info["error"]))
        with torch.no_grad():
            tensor = torch.as_tensor(observation[None], dtype=torch.float32)
            final_hidden = agent.actor.encoder.step(tensor, hidden_actor)
            context = agent.actor.normalization(final_hidden)[0].numpy()
        errors = np.array(errors)
        results.append({"seed": seed, "errors": errors, "context": context,
                        "parameters": env.info()["raw"]})
        env.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="循环 SAC 训练")
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--updates", type=int, default=64)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--parameter-scale", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--layer-dim", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-episodes", type=int, default=400)
    parser.add_argument("--stability-weight", type=float, default=0.0)
    parser.add_argument("--stability-warmup", type=int, default=300)
    parser.add_argument("--context-entropy-target", type=float, default=-0.05)
    parser.add_argument("--extractor-weight", type=float, default=0.0)
    parser.add_argument("--extractor-converged-length", type=int, default=128)
    parser.add_argument("--extractor-layer-dim", type=int, default=64)
    parser.add_argument("--extractor-episodes", type=int, default=4)
    parser.add_argument("--extractor-steps", type=int, default=4)
    parser.add_argument("--extractor-batch", type=int, default=128)
    parser.add_argument("--extractor-capacity", type=int, default=2048)
    parser.add_argument("--extractor-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--context-entropy-weight", type=float, default=0.0)
    parser.add_argument("--reset-context-lambda", action="store_true")
    parser.add_argument("--difficulty-start", type=float, default=1.0)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--save", type=Path, default=Path("data/rl_policy.pt"))
    parser.add_argument("--log", type=Path, default=Path("data/rl_training.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evaluate-every", type=int, default=20)
    parser.add_argument("--evaluate-episodes", type=int, default=4)
    arguments = parser.parse_args()

    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.set_num_threads(8)

    envs = [TrackingEnv(seed=arguments.seed * 1000 + index,
                        episode_steps=arguments.episode_steps,
                        parameter_scale=arguments.parameter_scale)
            for index in range(arguments.envs)]
    observation_batch = np.stack([env.reset() for env in envs])
    buffer = SequenceBuffer(arguments.buffer_episodes, OBSERVATION_DIM, 4,
                            arguments.hidden_dim)
    # 坐标权重：按采样分布的标准差归一化，使各维贡献可比
    sampled = []
    for _ in range(12):
        for env in envs:
            env.reset()
            sampled.append(env.info()["coordinates"])
    coordinate_mean = np.stack(sampled).mean(axis=0)
    coordinate_std = np.stack(sampled).std(axis=0) + 1.0e-8
    print("坐标标准差范围 %.4f 到 %.4f" % (coordinate_std.min(), coordinate_std.max()))
    agent = RecurrentSAC(OBSERVATION_DIM, 4, arguments.hidden_dim, arguments.layer_dim,
                         gamma=arguments.gamma, learning_rate=arguments.learning_rate,
                         stability_weight=arguments.stability_weight,
                         stability_warmup=arguments.stability_warmup,
                         context_entropy_target=arguments.context_entropy_target,
                         context_entropy_weight=arguments.context_entropy_weight,
                         extractor_weight=0.0,
                         extractor_layer_dim=arguments.extractor_layer_dim,
                         probe_learning_rate=arguments.extractor_learning_rate,
                         coordinate_mean=coordinate_mean, coordinate_std=coordinate_std)
    if arguments.resume is not None:
        agent.load(arguments.resume)
        if arguments.reset_context_lambda:
            with torch.no_grad():
                agent.log_context_lambda.fill_(0.0)
        print("从 %s 继续训练" % arguments.resume)

    extractor_envs = [TrackingEnv(seed=900000 + arguments.seed * 100 + index,
                                  episode_steps=arguments.episode_steps,
                                  parameter_scale=max(arguments.parameter_scale, 0.2))
                      for index in range(arguments.extractor_episodes)]
    extractor_contexts = []
    extractor_coordinates = []
    history = []
    normalizer = RewardNormalizer()
    start = time.time()
    raw_rewards = []
    for iteration in range(arguments.iterations):
        difficulty = min(1.0, arguments.difficulty_start
                         + (1.0 - arguments.difficulty_start)
                         * iteration / max(1.0, 0.6 * arguments.iterations))
        for env in envs:
            env.difficulty = difficulty
        observation_batch = np.stack([env.reset() for env in envs]) if iteration > 0 else observation_batch
        observation_batch, errors, episode_return, raw_rewards = rollout(
            agent, envs, buffer, observation_batch, arguments.episode_steps,
            normalizer=normalizer)
        normalizer.update(raw_rewards)
        statistics = []
        for _ in range(arguments.updates):
            if len(buffer) < 4:
                break
            batch = buffer.sample_windows(arguments.batch, arguments.window)
            critic_one_loss, target_value = agent.update_critics(batch)
            actor_statistics = agent.update_actor(batch, buffer)
            actor_statistics["critic_loss"] = critic_one_loss
            actor_statistics["target_value"] = target_value
            statistics.append(actor_statistics)
        probe_loss, auxiliary_loss = 0.0, 0.0
        if arguments.extractor_weight > 0.0:
            for env in extractor_envs:
                env.difficulty = difficulty
            fresh = collect_extractor_data(agent, extractor_envs,
                                           arguments.episode_steps, 128)
            extractor_contexts.append(fresh["context"])
            extractor_coordinates.append(fresh["coordinates"])
            while len(extractor_contexts) > arguments.extractor_capacity:
                extractor_contexts.pop(0)
                extractor_coordinates.pop(0)
            all_context = torch.as_tensor(np.concatenate(extractor_contexts), dtype=torch.float32)
            all_coordinates = torch.as_tensor(np.concatenate(extractor_coordinates),
                                              dtype=torch.float32)
            probe_loss = agent.update_extractor_probe(all_context, all_coordinates,
                                                      steps=arguments.extractor_steps,
                                                      batch=arguments.extractor_batch)
            auxiliary_loss = agent.actor_auxiliary(
                fresh["window"], fresh["hidden"],
                torch.as_tensor(fresh["coordinates"], dtype=torch.float32),
                arguments.extractor_weight)
        record = {"iteration": iteration, "episode_return": float(episode_return),
                  "mean_error": float(errors.mean()),
                  "mean_error_late": float(errors[len(errors) // 2:].mean()),
                  "buffer": len(buffer), "probe_samples": sum(len(item) for item in extractor_contexts),
                  "probe_loss": float(probe_loss), "auxiliary_loss": float(auxiliary_loss)}
        if statistics:
            for key in statistics[0]:
                record[key] = float(np.mean([item[key] for item in statistics]))
        if iteration % 5 == 0 or iteration == arguments.iterations - 1:
            log_line = "迭代 %3d 回报 %7.2f 误差 %.4f 后段 %.4f alpha %.3f" % (
                iteration, record["episode_return"], record["mean_error"],
                record["mean_error_late"], record.get("alpha", 0.0))
            if "stability" in record:
                log_line += " 稳定项 %.4f 上下文熵 %.1f lambda %.3f" % (
                    record["stability"], record["context_entropy"],
                    record["context_lambda"])
            print(log_line, flush=True)
        if arguments.evaluate_every > 0 and (iteration % arguments.evaluate_every == 0
                                             or iteration == arguments.iterations - 1):
            results = evaluate(agent, list(range(5000, 5000 + arguments.evaluate_episodes)),
                               arguments.episode_steps, arguments.parameter_scale)
            early = np.mean([item["errors"][:arguments.episode_steps // 4].mean()
                             for item in results])
            late = np.mean([item["errors"][-arguments.episode_steps // 4:].mean()
                            for item in results])
            record["evaluation_early_error"] = float(early)
            record["evaluation_late_error"] = float(late)
            print("  评估：前四分之一误差 %.4f，后四分之一误差 %.4f" % (early, late),
                  flush=True)
        history.append(record)
        arguments.log.write_text(json.dumps(history, indent=2, ensure_ascii=False))
        if (iteration + 1) % 20 == 0 or iteration == arguments.iterations - 1:
            agent.save(arguments.save)
    agent.save(arguments.save)
    elapsed = time.time() - start
    print("训练完成，用时 %.1f s，已保存 %s" % (elapsed, arguments.save))
    for env in envs:
        env.close()


if __name__ == "__main__":
    main()