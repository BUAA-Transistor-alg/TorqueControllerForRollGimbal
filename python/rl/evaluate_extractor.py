"""评估隐藏状态是否真的编码动力学参数与当前扰动，以及策略是否真的在用这些信息。

指标：
- 参数回归：每步从慢路上下文预测规范固定参数标签（28 维可辨识坐标 + 4 维相对电机增益），
  在留出回合上报告标准化 R^2（整体、坐标、电机增益）与线性探针 R^2；
- 响应回归：慢路上下文加当前真实状态预测当前等效扰动与有效惯量（动作单位），报告 R^2；
- 漂移跟踪：带回合内参数漂移时分时段报告 R^2 与预测变化方向与真实变化方向的余弦相似度；
- 上下文消融：把慢路状态每步清零，比较跟踪误差；
- 上下文交换：两个不同参数回合在指定步交换慢路状态，检验性能是否随之改变。

用法（在仓库根目录）：
    python3 python/rl/evaluate_extractor.py --policy data/rl_adaptive.pt --episodes 24
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "identification"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import build_agent
from tracking_env import TrackingEnv


def parameter_target(transition):
    return np.concatenate([transition["coordinates"],
                           np.log(np.maximum(transition["relative_motor"], 1.0e-6))])


def rollout(agent, seed: int, episode_steps: int, parameter_scale: float,
            drift_scale: float = 0.0, drift_motor: float = 0.0, ablate: str = "none",
            swap_step: int = None, swap_hidden=None, swap_length: int = 1,
            freeze_step: int = None, record_hidden_at: int = None):
    env = TrackingEnv(seed=seed, episode_steps=episode_steps,
                      parameter_scale=parameter_scale, drift_scale=drift_scale,
                      drift_motor=drift_motor,
                      load_in_observation=getattr(agent, "load_in_observation", True))
    observation = env.reset()
    hidden, _, _ = agent.initial_hidden(1)
    observations, targets, responses, states, errors = [], [], [], [], []
    hidden_at = None
    frozen = None
    for step in range(episode_steps):
        if freeze_step is not None and step == freeze_step:
            frozen = hidden[:, agent.fast_dim:].clone()
        if frozen is not None and step >= freeze_step:
            hidden = hidden.clone()
            hidden[:, agent.fast_dim:] = frozen
        if (swap_step is not None and swap_hidden is not None
                and swap_step <= step < swap_step + swap_length):
            hidden = hidden.clone()
            hidden[:, agent.fast_dim:] = swap_hidden
        if ablate == "slow":
            hidden = hidden.clone()
            hidden[..., agent.fast_dim:] = 0.0
        elif ablate == "all":
            hidden = agent.actor.initial_state(1)
        if record_hidden_at is not None and step == record_hidden_at:
            hidden_at = hidden[:, agent.fast_dim:].clone()
        tensor = torch.as_tensor(observation[None], dtype=torch.float32)
        with torch.no_grad():
            action, _, hidden = agent.act_partial(tensor, hidden, True)
        observations.append(observation)
        observation, reward, done, info = env.step(action.numpy()[0])
        transition = info["transition"]
        targets.append(parameter_target(transition))
        responses.append(transition["response"])
        states.append(transition["state"])
        errors.append(info["error"].copy())
    result = {"observation": np.stack(observations), "target": np.stack(targets),
              "response": np.stack(responses), "state": np.stack(states),
              "error": np.abs(np.array(errors)), "hits": env.hits,
              "hidden_at": None if hidden_at is None else hidden_at.numpy()}
    env.close()
    return result


def predict(agent, result):
    with torch.no_grad():
        observation = torch.as_tensor(result["observation"][None], dtype=torch.float32)
        state = torch.as_tensor(result["state"][None], dtype=torch.float32)
        hidden = agent.actor.encoder(observation)
        slow = agent.actor.slow_features(hidden)
        parameter = agent.parameter_head(slow) * agent.parameter_std + agent.parameter_mean
        response = (agent.response_head(slow, state) * agent.response_std[:8]
                    + agent.response_mean[:8])
        load = agent.actor.predict_load(hidden)[0]
    return parameter[0].numpy(), response[0].numpy(), load.numpy(), slow[0].numpy()


def standardized_r_squared(prediction, target, scale):
    residual = (prediction - target) / scale
    return 1.0 - float(np.mean(residual ** 2))


def linear_probe(features, target, split=0.7):
    features = features - features.mean(axis=0)
    target = target - target.mean(axis=0)
    count = int(split * len(features))
    left = features[:count].T @ features[:count] + 1.0e-1 * np.eye(features.shape[1])
    weights = np.linalg.solve(left, features[:count].T @ target[:count])
    prediction = features[count:] @ weights
    residual = prediction - target[count:]
    return 1.0 - float((residual ** 2).sum()) / float((target[count:] ** 2).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description="隐藏状态信息评估")
    parser.add_argument("--policy", type=Path, default=Path("data/rl_adaptive.pt"))
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--episode-steps", type=int, default=1500)
    parser.add_argument("--parameter-scale", type=float, default=1.25)
    parser.add_argument("--drift", type=float, default=0.0)
    parser.add_argument("--drift-motor", type=float, default=0.3)
    parser.add_argument("--swap-step", type=int, default=400)
    parser.add_argument("--seed-base", type=int, default=21000)
    parser.add_argument("--report", type=Path, default=Path("data/rl_extractor.json"))
    parser.add_argument("--threads", type=int, default=4)
    arguments = parser.parse_args()
    torch.set_num_threads(arguments.threads)
    agent = build_agent(arguments.policy)
    seeds = [arguments.seed_base + index for index in range(arguments.episodes)]
    drift = arguments.drift
    drift_motor = arguments.drift_motor if drift > 0.0 else 0.0

    parameters, targets, responses, response_targets, features, errors = (
        [], [], [], [], [], [])
    loads, load_targets = [], []
    for seed in seeds:
        result = rollout(agent, seed, arguments.episode_steps, arguments.parameter_scale,
                         drift_scale=drift, drift_motor=drift_motor)
        parameter, response, load, slow = predict(agent, result)
        parameters.append(parameter)
        targets.append(result["target"])
        responses.append(response)
        response_targets.append(result["response"][:, :8])
        loads.append(load)
        load_targets.append(result["response"][:, 8:])
        features.append(slow)
        errors.append(result["error"].mean())
    parameter = np.stack(parameters)
    target = np.stack(targets)
    response = np.stack(responses)
    response_target = np.stack(response_targets)
    features = np.stack(features)
    load = np.stack(loads)
    load_target = np.stack(load_targets)
    parameter_scale = agent.parameter_std.numpy()[None, None, :]
    response_scale = agent.response_std.numpy()[None, None, :8]
    load_scale = agent.response_std.numpy()[None, None, 8:]
    report = {"episodes": arguments.episodes, "drift": drift,
              "parameter_scale": arguments.parameter_scale,
              "parameter_r_squared": standardized_r_squared(parameter, target, parameter_scale),
              "coordinate_r_squared": standardized_r_squared(
                  parameter[:, :, :28], target[:, :, :28], parameter_scale[:, :, :28]),
              "motor_r_squared": standardized_r_squared(
                  parameter[:, :, 28:], target[:, :, 28:], parameter_scale[:, :, 28:]),
              "response_r_squared": standardized_r_squared(
                  response, response_target, response_scale),
              "load_r_squared": standardized_r_squared(load, load_target, load_scale),
              "mean_error": float(np.mean(errors))}
    standardized_target = ((target - agent.parameter_mean.numpy())
                           / agent.parameter_std.numpy())
    report["linear_probe_r_squared"] = linear_probe(
        features.reshape(-1, features.shape[-1]),
        standardized_target.reshape(-1, target.shape[-1]))
    print("参数回归：整体 R^2 %.4f（坐标 %.4f，相对电机增益 %.4f），线性探针 R^2 %.4f；"
          "响应回归 R^2 %.4f；负载回归 R^2 %.4f" % (
              report["parameter_r_squared"], report["coordinate_r_squared"],
              report["motor_r_squared"], report["linear_probe_r_squared"],
              report["response_r_squared"], report["load_r_squared"]), flush=True)

    if drift > 0.0:
        windows = np.array_split(np.arange(arguments.episode_steps), 5)
        parameter_windows = [standardized_r_squared(parameter[:, window], target[:, window],
                                                    parameter_scale) for window in windows]
        response_windows = [standardized_r_squared(response[:, window],
                                                   response_target[:, window],
                                                   response_scale) for window in windows]
        load_windows = [standardized_r_squared(load[:, window], load_target[:, window],
                                               load_scale) for window in windows]

        def delta_cosine(prediction_values, target_values):
            delta_prediction = np.diff(prediction_values, axis=1)
            delta_target = np.diff(target_values, axis=1)
            numerator = np.sum(delta_prediction * delta_target, axis=2)
            denominator = (np.linalg.norm(delta_prediction, axis=2)
                           * np.linalg.norm(delta_target, axis=2) + 1.0e-12)
            return float(np.mean(numerator / denominator))

        report["drift_parameter_windows"] = [float(item) for item in parameter_windows]
        report["drift_response_windows"] = [float(item) for item in response_windows]
        report["drift_load_windows"] = [float(item) for item in load_windows]
        report["drift_parameter_delta_cosine"] = delta_cosine(parameter, target)
        report["drift_load_delta_cosine"] = delta_cosine(load, load_target)
        report["drift_load_mean_error"] = float(np.mean(np.abs(load - load_target)))
        print("漂移跟踪：负载五段 R^2 %s（变化方向余弦 %.4f），参数五段 R^2 %s，"
              "响应五段 R^2 %s" % (
                  np.round(load_windows, 4), report["drift_load_delta_cosine"],
                  np.round(parameter_windows, 4), np.round(response_windows, 4)), flush=True)

    full_errors, slow_errors, all_errors = [], [], []
    for seed in seeds[:max(4, arguments.episodes // 4)]:
        full_errors.append(rollout(agent, seed, arguments.episode_steps,
                                   arguments.parameter_scale)["error"].mean())
        slow_errors.append(rollout(agent, seed, arguments.episode_steps,
                                   arguments.parameter_scale, ablate="slow")["error"].mean())
        all_errors.append(rollout(agent, seed, arguments.episode_steps,
                                  arguments.parameter_scale, ablate="all")["error"].mean())
    report["ablation_full_error"] = float(np.mean(full_errors))
    report["ablation_slow_zero_error"] = float(np.mean(slow_errors))
    report["ablation_all_zero_error"] = float(np.mean(all_errors))
    print("消融：完整 %.4f，慢路清零 %.4f，全部清零 %.4f rad" % (
        report["ablation_full_error"], report["ablation_slow_zero_error"],
        report["ablation_all_zero_error"]), flush=True)

    frozen_errors, window_errors, swapped, original = [], [], [], []
    for index in range(max(2, arguments.episodes // 8)):
        donor = rollout(agent, 31000 + index, arguments.episode_steps,
                        arguments.parameter_scale, record_hidden_at=arguments.swap_step)
        base = rollout(agent, seeds[index], arguments.episode_steps, arguments.parameter_scale)
        frozen_result = rollout(agent, seeds[index], arguments.episode_steps,
                                arguments.parameter_scale, freeze_step=arguments.swap_step)
        window_result = rollout(agent, seeds[index], arguments.episode_steps,
                                arguments.parameter_scale, swap_step=arguments.swap_step,
                                swap_length=100,
                                swap_hidden=torch.as_tensor(donor["hidden_at"]))
        result = rollout(agent, seeds[index], arguments.episode_steps,
                         arguments.parameter_scale, swap_step=arguments.swap_step,
                         swap_hidden=torch.as_tensor(donor["hidden_at"]))
        original.append(base["error"].mean())
        swapped.append(result["error"].mean())
        frozen_errors.append(frozen_result["error"].mean())
        window_errors.append(window_result["error"].mean())
    report["swap_original_error"] = float(np.mean(original))
    report["swap_donor_error"] = float(np.mean(swapped))
    report["freeze_error"] = float(np.mean(frozen_errors))
    report["swap_window_error"] = float(np.mean(window_errors))
    print("上下文干预：原始 %.4f，冻结慢路 %.4f（%.2f 倍），单次交换 %.4f（%.2f 倍），"
          "持续交换 100 步 %.4f（%.2f 倍）" % (
              report["swap_original_error"], report["freeze_error"],
              report["freeze_error"] / report["swap_original_error"],
              report["swap_donor_error"],
              report["swap_donor_error"] / report["swap_original_error"],
              report["swap_window_error"],
              report["swap_window_error"] / report["swap_original_error"]), flush=True)
    arguments.report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("已保存", arguments.report)


if __name__ == "__main__":
    main()
