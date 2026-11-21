"""评估训练好的循环策略：跟踪精度、限位行为、阶跃响应、随时间与参数的变化。"""

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

from sac import RecurrentSAC
from tracking_env import (BIG_YAW_LIMIT, OBSERVATION_DIM, PITCH_LIMIT, ROLL_LIMIT,
                          SMALL_YAW_LIMIT, TrackingEnv)


def run_episode(agent: RecurrentSAC, seed: int, episode_steps: int, parameter_scale: float,
                deterministic: bool = True, noise_scale: float = 1.0,
                bias_scale: float = 1.0):
    env = TrackingEnv(seed=seed, episode_steps=episode_steps,
                      parameter_scale=parameter_scale, noise_scale=noise_scale,
                      bias_scale=bias_scale)
    observation = env.reset()
    hidden_actor, _, _ = agent.initial_hidden(1)
    errors, actions, angles, references, contexts = [], [], [], [], []
    for step in range(episode_steps):
        tensor = torch.as_tensor(observation[None], dtype=torch.float32)
        with torch.no_grad():
            action, _, hidden_actor = agent.act_partial(tensor, hidden_actor, deterministic)
        with torch.no_grad():
            direction = torch.nn.functional.normalize(hidden_actor, dim=-1)[0].numpy()
        contexts.append(direction.copy())
        observation, reward, done, info = env.step(action.numpy()[0])
        errors.append(info["error"].copy())
        actions.append(action.numpy()[0].copy())
        index = min(env.step_index, env.episode_steps - 1)
        angles.append(np.array([env.state.angle[i] for i in range(4)]))
        references.append(env.angle_ref[index].copy())
    with torch.no_grad():
        tensor = torch.as_tensor(observation[None], dtype=torch.float32)
        final_hidden = agent.actor.encoder.step(tensor, hidden_actor)
        context = agent.actor.normalization(final_hidden)[0].numpy()
    result = {"seed": seed, "error": np.array(errors), "action": np.array(actions),
              "angle": np.array(angles), "reference": np.array(references),
              "context": context, "contexts": np.array(contexts),
              "hidden": final_hidden[0].numpy(), "hits": env.hits,
              "parameters": env.info()["raw"]}
    env.close()
    return result


def step_response(references, errors, threshold: float = 0.01, window: int = 200):
    """检测参考轨迹的阶跃并统计峰值误差与恢复时间；输入形状为 (回合, 时间, 关节)。"""
    references = np.asarray(references)
    errors = np.asarray(errors)
    peaks, recoveries = [], []
    for episode in range(references.shape[0]):
        difference = np.abs(np.diff(references[episode], axis=0)).max(axis=1)
        for index in np.where(difference > 0.08)[0]:
            if index + window >= references.shape[1]:
                continue
            segment = np.abs(errors[episode, index:index + window]).max(axis=1)
            peaks.append(float(segment.max()))
            settled = np.where(segment < threshold)[0]
            after = settled[settled > 5]
            recoveries.append(float(after[0]) if after.size else float(window))
    return peaks, recoveries


def main() -> None:
    parser = argparse.ArgumentParser(description="评估循环策略")
    parser.add_argument("--policy", type=Path, default=Path("data/rl_stage2.pt"))
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--episode-steps", type=int, default=1500)
    parser.add_argument("--parameter-scale", type=float, default=0.0)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--bias-scale", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layer-dim", type=int, default=128)
    parser.add_argument("--report", type=Path, default=Path("data/rl_evaluation.json"))
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    agent = RecurrentSAC(OBSERVATION_DIM, 4, arguments.hidden_dim, arguments.layer_dim)
    agent.load(arguments.policy)
    results = [run_episode(agent, 7000 + index, arguments.episode_steps,
                           arguments.parameter_scale, noise_scale=arguments.noise_scale,
                           bias_scale=arguments.bias_scale)
               for index in range(arguments.episodes)]
    errors = np.stack([item["error"] for item in results])
    references = np.stack([item["reference"] for item in results])
    length = arguments.episode_steps
    windows = np.array_split(np.arange(length), 10)
    limits = (BIG_YAW_LIMIT, ROLL_LIMIT, SMALL_YAW_LIMIT, PITCH_LIMIT)
    outside = np.zeros(errors.shape[:2], dtype=bool)
    for joint, (low, high) in enumerate(limits):
        outside |= ((references[:, :, joint] < low) | (references[:, :, joint] > high))
    summary = {
        "episodes": arguments.episodes, "parameter_scale": arguments.parameter_scale,
        "noise_scale": arguments.noise_scale, "bias_scale": arguments.bias_scale,
        "mean_absolute_error": float(np.abs(errors).mean()),
        "per_joint": np.abs(errors).mean(axis=(0, 1)).tolist(),
        "window_errors": [float(np.abs(errors[:, window]).mean()) for window in windows],
        "error_in_range": float(np.abs(errors[~outside]).mean()),
        "error_out_of_range": float(np.abs(errors[outside]).mean()) if outside.any() else None,
        "out_of_range_fraction": float(outside.mean()),
        "hits_per_episode": float(np.mean([item["hits"] for item in results])),
        "peak_limit_usage": float(np.abs(np.stack([item["angle"] for item in results])
                                          / np.array([np.pi, 0.785, 0.873, 0.785])).max()),
    }
    contexts = np.stack([item["contexts"] for item in results])
    changes = np.linalg.norm(contexts[:, 1:] - contexts[:, :-1], axis=-1)
    change_windows = np.array_split(np.arange(changes.shape[1]), 10)
    summary["context_change_windows"] = [float(changes[:, window].mean())
                                         for window in change_windows]
    peaks, recoveries = step_response(references, errors)
    summary["step_peak_error"] = float(np.mean(peaks)) if peaks else None
    summary["step_recovery_steps"] = float(np.mean(recoveries)) if recoveries else None
    summary["step_count"] = len(peaks)
    print("平均绝对误差 %.4f rad，各关节 %s" % (
        summary["mean_absolute_error"], np.round(summary["per_joint"], 4)))
    print("限位内误差 %.4f，限位外目标误差 %s，限位外比例 %.2f" % (
        summary["error_in_range"],
        None if summary["error_out_of_range"] is None
        else round(summary["error_out_of_range"], 4), summary["out_of_range_fraction"]))
    print("每回合撞限位 %.2f 次，最大限位占用 %.3f" % (
        summary["hits_per_episode"], summary["peak_limit_usage"]))
    if summary["step_count"]:
        print("阶跃响应：%d 个阶跃，峰值误差 %.4f，恢复步数 %.0f" % (
            summary["step_count"], summary["step_peak_error"],
            summary["step_recovery_steps"]))
    else:
        print("阶跃响应：未检出阶跃（峰值跳变阈值 0.08 rad）")
    print("十个时段误差 %s" % np.round(summary["window_errors"], 4))
    summary["per_episode"] = [float(np.abs(item["error"]).mean()) for item in results]
    arguments.report.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    np.savez_compressed(arguments.report.with_suffix(".npz"),
                        error=errors, action=np.stack([i["action"] for i in results]),
                        angle=np.stack([i["angle"] for i in results]),
                        reference=references,
                        context=np.stack([i["context"] for i in results]),
                        contexts=contexts,
                        hidden=np.stack([i["hidden"] for i in results]),
                        parameters=np.stack([i["parameters"] for i in results]))
    print("已保存", arguments.report)


if __name__ == "__main__":
    main()
