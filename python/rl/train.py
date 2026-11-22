"""训练结构化残差循环 SAC 自适应控制器：参数范围课程 + 回合内参数漂移。

用法（在仓库根目录）：
    OPENBLAS_NUM_THREADS=6 python3 -u python/rl/train.py --iterations 300 \
        --load-from-state --hide-load --save data/rl_adaptive.pt --log data/rl_adaptive.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "identification"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sac import PARAMETER_DIMENSION, RESPONSE_DIMENSION, RecurrentSAC, SequenceBuffer
from teacher import STRUCTURED_WEIGHTS, StructuredTeacher, TeacherController
from tracking_env import FAST_INPUT_DIM, LIMITS, OBSERVATION_DIM, TrackingEnv


def parameter_target(transition) -> np.ndarray:
    return np.concatenate([transition["coordinates"],
                           np.log(np.maximum(transition["relative_motor"], 1.0e-6))])


def collect_episode_records(agent, envs, observation_batch: np.ndarray, episode_steps: int,
                          true_teacher: bool = False):
    count = len(envs)
    if true_teacher:
        teachers = [TeacherController(env) for env in envs]
    else:
        teachers = [StructuredTeacher(env) for env in envs]
    hidden_actor, hidden_one, hidden_two = agent.initial_hidden(count)
    observations = [observation_batch.copy()]
    actions, rewards, dones, teacher_actions = [], [], [], []
    hidden_actor_records, hidden_one_records, hidden_two_records = [], [], []
    states, accelerations, targets, responses = [], [], [], []
    errors, hits = [], np.zeros(count)
    for _ in range(episode_steps):
        observation = torch.as_tensor(observation_batch, dtype=torch.float32)
        with torch.no_grad():
            action, _, hidden_actor = agent.act_partial(observation, hidden_actor)
            hidden_one = agent.critic_one.encoder.step(observation, hidden_one)
            hidden_two = agent.critic_two.encoder.step(observation, hidden_two)
        hidden_actor_records.append(hidden_actor.numpy())
        hidden_one_records.append(hidden_one.numpy())
        hidden_two_records.append(hidden_two.numpy())
        actions.append(action.numpy())
        teacher_actions.append(np.stack([
            teachers[index].action(env.step_index, observation_batch[index])
            for index, env in enumerate(envs)]))
        next_observations = np.zeros_like(observation_batch)
        step_rewards = np.zeros(count)
        step_dones = np.zeros(count)
        step_states, step_accelerations, step_targets, step_responses = [], [], [], []
        for index, env in enumerate(envs):
            next_observation, reward, done, info = env.step(actions[-1][index])
            next_observations[index] = next_observation
            step_rewards[index] = reward
            step_dones[index] = float(done)
            errors.append(np.abs(info["error"]))
            hits[index] = info["hits"]
            transition = info["transition"]
            step_states.append(transition["state"])
            step_accelerations.append(transition["acceleration"])
            step_targets.append(parameter_target(transition))
            step_responses.append(transition["response"])
        observations.append(next_observations.copy())
        rewards.append(step_rewards)
        dones.append(step_dones)
        states.append(step_states)
        accelerations.append(step_accelerations)
        targets.append(step_targets)
        responses.append(step_responses)
        observation_batch = next_observations
    with torch.no_grad():
        observation = torch.as_tensor(observation_batch, dtype=torch.float32)
        hidden_actor_records.append(agent.actor.encoder.step(observation, hidden_actor).numpy())
        hidden_one_records.append(agent.critic_one.encoder.step(observation, hidden_one).numpy())
        hidden_two_records.append(agent.critic_two.encoder.step(observation, hidden_two).numpy())
    episodes = []
    for index in range(count):
        episodes.append({
            "observation": np.stack([item[index] for item in observations]),
            "action": np.stack([item[index] for item in actions]),
            "reward": np.array([item[index] for item in rewards]),
            "done": np.array([item[index] for item in dones]),
            "hidden_actor": np.stack([item[index] for item in hidden_actor_records]),
            "hidden_critic_one": np.stack([item[index] for item in hidden_one_records]),
            "hidden_critic_two": np.stack([item[index] for item in hidden_two_records]),
            "state": np.stack([item[index] for item in states]),
            "acceleration": np.stack([item[index] for item in accelerations]),
            "target": np.stack([item[index] for item in targets]),
            "response": np.stack([item[index] for item in responses]),
            "teacher": np.stack([item[index] for item in teacher_actions]),
        })
    return episodes, np.array(errors), hits


def evaluate(agent, seeds, episode_steps: int, parameter_scale: float, difficulty: float = 1.0,
             drift_scale: float = 0.0, drift_motor: float = 0.0):
    totals, per_joint, inside, hits = [], [], [], []
    for seed in seeds:
        env = TrackingEnv(seed=seed, episode_steps=episode_steps,
                          parameter_scale=parameter_scale, drift_scale=drift_scale,
                          drift_motor=drift_motor, difficulty=difficulty,
                          load_in_observation=getattr(agent, "load_in_observation", True))
        observation = env.reset()
        hidden_actor, _, _ = agent.initial_hidden(1)
        errors = []
        for step in range(episode_steps):
            tensor = torch.as_tensor(observation[None], dtype=torch.float32)
            with torch.no_grad():
                action, _, hidden_actor = agent.act_partial(tensor, hidden_actor, True)
            observation, reward, done, info = env.step(action.numpy()[0])
            errors.append(info["error"].copy())
        errors = np.abs(np.array(errors))
        reference = env.angle_ref[:episode_steps]
        outside = np.zeros(errors.shape, dtype=bool)
        for joint, limits in enumerate(LIMITS):
            if limits is not None:
                outside[:, joint] = ((reference[:, joint] < limits[0])
                                     | (reference[:, joint] > limits[1]))
        totals.append(errors.mean())
        per_joint.append(errors.mean(axis=0))
        inside.append(errors[~outside].mean())
        hits.append(env.hits)
        env.close()
    return {"mean_error": float(np.mean(totals)),
            "per_joint": np.mean(per_joint, axis=0).tolist(),
            "in_range_error": float(np.mean(inside)),
            "hits_per_episode": float(np.mean(hits)), "max_hits": float(np.max(hits)),
            "per_episode": [float(item) for item in totals]}


def sample_auxiliary_statistics(envs, rng, scale_max: float, count: int = 4):
    parameters, responses = [], []
    for _ in range(count):
        for env in envs:
            env.parameter_scale = rng.uniform(0.0, scale_max)
            env.drift_scale = 0.0
            env.drift_motor = 0.0
            env.reset()
            observation = env.observation()
            for _ in range(80):
                observation, reward, done, info = env.step(rng.uniform(-1.0, 1.0, 4))
                transition = info["transition"]
                parameters.append(parameter_target(transition))
                responses.append(transition["response"])
    parameters = np.stack(parameters)
    responses = np.stack(responses)
    return (parameters.mean(axis=0), parameters.std(axis=0) + 1.0e-6,
            responses.mean(axis=0), responses.std(axis=0) + 1.0e-6)


def main() -> None:
    parser = argparse.ArgumentParser(description="循环 SAC 自适应训练")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--episode-steps", type=int, default=1500)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--updates", type=int, default=40)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--auxiliary-batch", type=int, default=8)
    parser.add_argument("--auxiliary-window", type=int, default=128)
    parser.add_argument("--actor-warmup", type=int, default=30)
    parser.add_argument("--residual-scale", type=float, default=0.3)
    parser.add_argument("--head-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--alpha-init", type=float, default=-2.0)
    parser.add_argument("--load-from-state", action="store_true")
    parser.add_argument("--hide-load", action="store_true")
    parser.add_argument("--no-fixed-log-std", action="store_true")
    parser.add_argument("--log-std-init", type=float, default=-2.0)
    parser.add_argument("--save-best", type=Path, default=None)
    parser.add_argument("--true-teacher", action="store_true")
    parser.add_argument("--supervised", action="store_true")
    parser.add_argument("--fast-dim", type=int, default=64)
    parser.add_argument("--slow-dim", type=int, default=64)
    parser.add_argument("--layer-dim", type=int, default=256)
    parser.add_argument("--fast-bias", type=float, default=2.0)
    parser.add_argument("--slow-gate-bias", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--alpha-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--auxiliary-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--target-entropy", type=float, default=-4.0)
    parser.add_argument("--buffer-episodes", type=int, default=160)
    parser.add_argument("--dynamics-weight", type=float, default=0.2)
    parser.add_argument("--parameter-weight", type=float, default=0.2)
    parser.add_argument("--response-weight", type=float, default=0.2)
    parser.add_argument("--load-weight", type=float, default=1.0)
    parser.add_argument("--behavior-weight", type=float, default=0.1)
    parser.add_argument("--reward-scale", type=float, default=0.05)
    parser.add_argument("--limit-weight", type=float, default=30.0)
    parser.add_argument("--hit-penalty", type=float, default=100.0)
    parser.add_argument("--overshoot-weight", type=float, default=1000.0)
    parser.add_argument("--limit-margin", type=float, default=0.0349)
    parser.add_argument("--limit-lookahead", type=float, default=0.04)
    parser.add_argument("--parameter-scale-start", type=float, default=0.6)
    parser.add_argument("--parameter-scale-final", type=float, default=1.25)
    parser.add_argument("--curriculum-fraction", type=float, default=0.35)
    parser.add_argument("--drift-final", type=float, default=0.35)
    parser.add_argument("--drift-motor-final", type=float, default=0.3)
    parser.add_argument("--drift-start-iteration", type=float, default=0.2)
    parser.add_argument("--difficulty-start", type=float, default=0.85)
    parser.add_argument("--pretrain", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--save", type=Path, default=Path("data/rl_adaptive.pt"))
    parser.add_argument("--log", type=Path, default=Path("data/rl_adaptive.json"))
    parser.add_argument("--evaluate-every", type=int, default=20)
    parser.add_argument("--evaluate-episodes", type=int, default=4)
    parser.add_argument("--evaluate-scales", type=str, default="0.0,1.25")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.set_num_threads(arguments.threads)
    rng = np.random.default_rng(arguments.seed)

    envs = [TrackingEnv(seed=arguments.seed * 1000 + index,
                        episode_steps=arguments.episode_steps,
                        limit_weight=arguments.limit_weight,
                        hit_penalty=arguments.hit_penalty,
                        overshoot_weight=arguments.overshoot_weight,
                        reward_scale=arguments.reward_scale,
                        load_in_observation=not arguments.hide_load,
                        limit_margin=arguments.limit_margin,
                        limit_lookahead=arguments.limit_lookahead)
            for index in range(arguments.envs)]
    source = arguments.pretrain if arguments.pretrain is not None else arguments.resume
    if source is not None:
        payload = torch.load(source, map_location="cpu", weights_only=False)
        parameter_mean = payload["parameter_mean"].numpy()
        parameter_std = payload["parameter_std"].numpy()
        response_mean = payload["response_mean"].numpy()
        response_std = payload["response_std"].numpy()
    else:
        parameter_mean, parameter_std, response_mean, response_std = (
            sample_auxiliary_statistics(envs, rng, arguments.parameter_scale_final))
    agent = RecurrentSAC(OBSERVATION_DIM, 4, FAST_INPUT_DIM, arguments.fast_dim,
                         arguments.slow_dim, 18, arguments.layer_dim,
                         gamma=arguments.gamma, learning_rate=arguments.learning_rate,
                         alpha_learning_rate=arguments.alpha_learning_rate,
                         target_entropy=arguments.target_entropy,
                         fast_bias=arguments.fast_bias,
                         slow_gate_bias=arguments.slow_gate_bias,
                         dynamics_weight=arguments.dynamics_weight,
                         parameter_weight=arguments.parameter_weight,
                         response_weight=arguments.response_weight,
                         load_weight=arguments.load_weight,
                         behavior_weight=arguments.behavior_weight,
                         auxiliary_learning_rate=arguments.auxiliary_learning_rate,
                         structured_weights=STRUCTURED_WEIGHTS,
                         residual_scale=arguments.residual_scale,
                         head_learning_rate=arguments.head_learning_rate,
                         alpha_init=arguments.alpha_init,
                         load_from_state=arguments.load_from_state,
                         fixed_log_std=not arguments.no_fixed_log_std,
                         log_std_init=arguments.log_std_init,
                         parameter_mean=parameter_mean, parameter_std=parameter_std,
                         response_mean=response_mean, response_std=response_std)
    if arguments.pretrain is not None:
        agent.load(arguments.pretrain, load_actor_only=True)
        print("已加载预训练编码器 %s" % arguments.pretrain)
    if arguments.resume is not None:
        agent.load(arguments.resume)
        print("从 %s 继续训练" % arguments.resume)

    agent.load_in_observation = not arguments.hide_load
    if arguments.load_from_state:
        agent.actor.load_from_state = True
    buffer = SequenceBuffer(arguments.buffer_episodes)
    history = []
    start_time = time.time()
    error_sum = 0.0
    best_score = float("inf")
    for iteration in range(arguments.iterations):
        progress = iteration / max(1, arguments.iterations - 1)
        scale_max = arguments.parameter_scale_start + (
            arguments.parameter_scale_final - arguments.parameter_scale_start) * min(
            1.0, progress / arguments.curriculum_fraction)
        drift_progress = min(1.0, max(
            0.0, (progress - arguments.drift_start_iteration)
            / max(1.0e-6, 1.0 - arguments.drift_start_iteration)))
        drift = arguments.drift_final * drift_progress
        drift_motor = arguments.drift_motor_final * drift_progress
        difficulty = min(1.0, arguments.difficulty_start
                         + (1.0 - arguments.difficulty_start) * min(1.0, progress / 0.6))
        for env in envs:
            env.difficulty = difficulty
            env.drift_scale = drift * rng.uniform()
            env.drift_motor = drift_motor * rng.uniform()
            env.parameter_scale = rng.uniform(0.0, scale_max)
        observation_batch = np.stack([env.reset() for env in envs])
        episodes, errors, hits = collect_episode_records(
            agent, envs, observation_batch, arguments.episode_steps,
            true_teacher=arguments.true_teacher)
        for episode in episodes:
            buffer.add(episode)
        statistics = []
        for _ in range(arguments.updates):
            if arguments.supervised:
                auxiliary_batch = buffer.sample_windows(arguments.auxiliary_batch,
                                                        arguments.auxiliary_window)
                supervised_statistics = agent.update_auxiliary_supervised(auxiliary_batch)[0]
                statistics.append(supervised_statistics)
                continue
            batch = buffer.sample_windows(arguments.batch, arguments.window)
            critic_loss, target_value = agent.update_critics(batch)
            if iteration < arguments.actor_warmup:
                # 预热阶段只训练价值网络与辅助头：价值网络先拟合回报，辅助头先塑造编码器，
                # 避免随机初始化的价值与编码器改变已经工作的结构化策略。
                auxiliary_batch = buffer.sample_windows(arguments.auxiliary_batch,
                                                        arguments.auxiliary_window)
                warmup_statistics = agent.update_auxiliary_supervised(auxiliary_batch)[0]
                warmup_statistics["critic_loss"] = critic_loss
                warmup_statistics["target_value"] = target_value
                statistics.append(warmup_statistics)
                agent.soft_update()
                continue
            auxiliary_batch = buffer.sample_windows(arguments.auxiliary_batch,
                                                    arguments.auxiliary_window)
            actor_statistics = agent.update_actor(batch, auxiliary_batch)
            actor_statistics["critic_loss"] = critic_loss
            actor_statistics["target_value"] = target_value
            statistics.append(actor_statistics)
        record = {"iteration": iteration, "scale_max": float(scale_max),
                  "drift": float(drift), "drift_motor": float(drift_motor),
                  "difficulty": float(difficulty),
                  "mean_error": float(errors.mean()),
                  "mean_error_late": float(errors[len(errors) // 2:].mean()),
                  "mean_hits": float(hits.mean()), "max_hits": float(hits.max()),
                  "buffer": len(buffer)}
        for key in (statistics[0] if statistics else {}):
            record[key] = float(np.mean([item[key] for item in statistics]))
        record["alpha"] = float(agent.alpha.detach())
        error_sum = 0.9 * error_sum + 0.1 * record["mean_error"]
        record["mean_error_smoothed"] = error_sum
        if arguments.evaluate_every > 0 and (iteration % arguments.evaluate_every == 0
                                             or iteration == arguments.iterations - 1):
            seeds = list(range(7000, 7000 + arguments.evaluate_episodes))
            for scale in [float(item) for item in arguments.evaluate_scales.split(",")]:
                result = evaluate(agent, seeds, arguments.episode_steps, scale)
                record["evaluation_%s" % scale] = result
                if scale == 0.0:
                    record["mean_error_smoothed"] = result["in_range_error"]
            if arguments.save_best is not None:
                score = 0.0
                for key, value in record.items():
                    if key.startswith("evaluation_"):
                        score += value["in_range_error"] + 0.02 * value["max_hits"]
                record["selection_score"] = float(score)
                if score < best_score:
                    best_score = float(score)
                    agent.save(arguments.save_best)
                    print("  新的最佳检查点，选择分 %.4f" % score, flush=True)
        if iteration % 2 == 0 or iteration == arguments.iterations - 1:
            line = ("迭代 %3d 尺度上限 %.2f 漂移 %.2f 误差 %.4f 平滑 %.4f 撞限位 %.2f/%.0f "
                    "alpha %.3f 参数 %.3f 响应 %.3f 动力学 %.3f 克隆 %.3f" % (
                        iteration, scale_max, drift, record["mean_error"], error_sum,
                        record["mean_hits"], record["max_hits"], record.get("alpha", 0.0),
                        record.get("parameter_loss", 0.0), record.get("response_loss", 0.0),
                        record.get("dynamics_loss", 0.0), record.get("behavior_loss", 0.0)))
            if "evaluation_0.0" in record:
                line += " | 固定参数 %.4f 撞限位 %.2f/%.0f" % (
                    record["evaluation_0.0"]["in_range_error"],
                    record["evaluation_0.0"]["hits_per_episode"],
                    record["evaluation_0.0"]["max_hits"])
            print(line, flush=True)
        history.append(record)
        if iteration % 5 == 0 or iteration == arguments.iterations - 1:
            arguments.log.write_text(json.dumps(history, indent=2, ensure_ascii=False))
        if (iteration + 1) % 20 == 0 or iteration == arguments.iterations - 1:
            agent.save(arguments.save)
    agent.save(arguments.save)
    print("训练完成，用时 %.1f s，已保存 %s" % (time.time() - start_time, arguments.save))
    for env in envs:
        env.close()


if __name__ == "__main__":
    main()
