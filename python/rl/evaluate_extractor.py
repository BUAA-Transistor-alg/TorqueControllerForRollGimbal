"""评估联合训练的上下文提取器：在新回合上把收敛上下文映射成可辨识参数坐标。"""

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
from tracking_env import OBSERVATION_DIM, TrackingEnv, identifiable_coordinates


def collect(agent, episodes: int, steps: int, scale: float, seed: int = 21000,
            converged_length: int = 128):
    contexts, targets, raws, predictions = [], [], [], []
    for index in range(episodes):
        env = TrackingEnv(seed=seed + index, episode_steps=steps, parameter_scale=scale)
        observation = env.reset()
        hidden, _, _ = agent.initial_hidden(1)
        normalized_history = []
        for _ in range(steps):
            tensor = torch.as_tensor(observation[None], dtype=torch.float32)
            with torch.no_grad():
                action, _, hidden = agent.act_partial(tensor, hidden, True)
                normalized_history.append(agent.actor.normalization(hidden)[0].numpy())
            observation, reward, done, info = env.step(action.numpy()[0])
        context = np.mean(normalized_history[-converged_length:], axis=0)
        with torch.no_grad():
            prediction = agent.extractor(
                torch.as_tensor(context, dtype=torch.float32)[None])[0].numpy()
        contexts.append(context)
        targets.append(info["coordinates"])
        predictions.append(prediction)
        raws.append(env.raw.copy())
        env.close()
    return (np.stack(contexts), np.stack(targets), np.stack(raws), np.stack(predictions))


def main() -> None:
    parser = argparse.ArgumentParser(description="评估上下文提取器")
    parser.add_argument("--policy", type=Path, default=Path("data/rl_v5_stage2.pt"))
    parser.add_argument("--episodes", type=int, default=48)
    parser.add_argument("--episode-steps", type=int, default=1500)
    parser.add_argument("--parameter-scale", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--layer-dim", type=int, default=256)
    parser.add_argument("--report", type=Path, default=Path("data/rl_v5_extractor.json"))
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    agent = RecurrentSAC(OBSERVATION_DIM, 4, arguments.hidden_dim, arguments.layer_dim)
    agent.load(arguments.policy)
    contexts, targets, raws, predictions = collect(
        agent, arguments.episodes, arguments.episode_steps, arguments.parameter_scale)
    residual = predictions - targets
    total = ((targets - targets.mean(axis=0)) ** 2).sum()
    r_squared = 1.0 - float((residual ** 2).sum()) / float(total)
    per_dimension = 1.0 - (residual ** 2).sum(axis=0) / (
        (targets - targets.mean(axis=0)) ** 2).sum(axis=0)
    normalized_error = float(np.sqrt(((residual ** 2) / (targets.std(axis=0) ** 2 + 1e-8)).mean()))
    print("提取器：R^2 %.4f，逐维中位数 %.4f，归一化均方根误差 %.4f" % (
        r_squared, float(np.median(per_dimension)), normalized_error))
    # 线性探针基线
    features = contexts - contexts.mean(axis=0)
    target = targets - targets.mean(axis=0)
    split = int(0.7 * len(features))
    weights = np.linalg.solve(features[:split].T @ features[:split] + np.eye(features.shape[1]),
                              features[:split].T @ target[:split])
    prediction = features[split:] @ weights
    linear_r2 = 1.0 - ((prediction - target[split:]) ** 2).sum() / (target[split:] ** 2).sum()
    print("线性探针 R^2 %.4f" % linear_r2)
    report = {"episodes": arguments.episodes, "r_squared": float(r_squared),
              "per_dimension_median": float(np.median(per_dimension)),
              "normalized_rmse": normalized_error, "linear_r_squared": float(linear_r2),
              "per_dimension": per_dimension.tolist()}
    arguments.report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("已保存", arguments.report)


if __name__ == "__main__":
    main()
