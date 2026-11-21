"""评估训练好的循环策略：跟踪精度、随时间的变化、以及随机参数下的适应过程。"""

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
from tracking_env import OBSERVATION_DIM, TrackingEnv


def run_episode(agent: RecurrentSAC, seed: int, episode_steps: int, parameter_scale: float,
                deterministic: bool = True):
    env = TrackingEnv(seed=seed, episode_steps=episode_steps,
                      parameter_scale=parameter_scale)
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
              "hidden": final_hidden[0].numpy(),
              "parameters": env.info()["raw"]}
    env.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="评估循环策略")
    parser.add_argument("--policy", type=Path, default=Path("data/rl_stage1.pt"))
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--episode-steps", type=int, default=1500)
    parser.add_argument("--parameter-scale", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layer-dim", type=int, default=128)
    parser.add_argument("--report", type=Path, default=Path("data/rl_evaluation.json"))
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    agent = RecurrentSAC(OBSERVATION_DIM, 4, arguments.hidden_dim, arguments.layer_dim)
    agent.load(arguments.policy)
    results = [run_episode(agent, 7000 + index, arguments.episode_steps,
                           arguments.parameter_scale)
               for index in range(arguments.episodes)]
    errors = np.stack([item["error"] for item in results])
    length = arguments.episode_steps
    windows = np.array_split(np.arange(length), 10)
    summary = {"episodes": arguments.episodes, "parameter_scale": arguments.parameter_scale,
               "mean_absolute_error": float(np.abs(errors).mean()),
               "per_joint": np.abs(errors).mean(axis=(0, 1)).tolist(),
               "window_errors": [float(np.abs(errors[:, window]).mean()) for window in windows]}
    contexts = np.stack([item["contexts"] for item in results])
    changes = np.linalg.norm(contexts[:, 1:] - contexts[:, :-1], axis=-1)
    change_windows = np.array_split(np.arange(changes.shape[1]), 10)
    summary["context_change_windows"] = [float(changes[:, window].mean())
                                         for window in change_windows]
    print("平均绝对误差 %.4f rad，各关节 %s" % (
        summary["mean_absolute_error"], np.round(summary["per_joint"], 4)))
    print("十个时段误差 %s" % np.round(summary["window_errors"], 4))
    print("十个时段上下文变化 %s" % np.round(summary["context_change_windows"], 4))
    summary["per_episode"] = [float(np.abs(item["error"]).mean()) for item in results]
    arguments.report.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    np.savez_compressed(arguments.report.with_suffix(".npz"),
                        error=errors, action=np.stack([i["action"] for i in results]),
                        angle=np.stack([i["angle"] for i in results]),
                        reference=np.stack([i["reference"] for i in results]),
                        context=np.stack([i["context"] for i in results]),
                        contexts=contexts,
                        hidden=np.stack([i["hidden"] for i in results]),
                        parameters=np.stack([i["parameters"] for i in results]))
    print("已保存", arguments.report)


if __name__ == "__main__":
    main()
