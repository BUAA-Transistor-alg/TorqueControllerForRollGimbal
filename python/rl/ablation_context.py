"""消融：把上下文清零（无记忆策略）与保留历史对比，检验策略是否真的用到上下文。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "identification"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sac import RecurrentSAC
from tracking_env import OBSERVATION_DIM, TrackingEnv


def run(agent, seed: int, steps: int, scale: float, memory: bool, parameter_scale_seed=None):
    env = TrackingEnv(seed=seed, episode_steps=steps, parameter_scale=scale)
    observation = env.reset()
    hidden_actor, _, _ = agent.initial_hidden(1)
    errors = []
    for step in range(steps):
        if not memory:
            hidden_actor = agent.actor.initial_state(1)
        tensor = torch.as_tensor(observation[None], dtype=torch.float32)
        with torch.no_grad():
            action, _, hidden_actor = agent.act_partial(tensor, hidden_actor, True)
        observation, reward, done, info = env.step(action.numpy()[0])
        errors.append(info["error"])
    env.close()
    return float(np.abs(np.array(errors)).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="上下文消融")
    parser.add_argument("--policy", type=Path, default=Path("data/rl_stage2b.pt"))
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--episode-steps", type=int, default=1500)
    parser.add_argument("--parameter-scale", type=float, default=0.8)
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    agent = RecurrentSAC(OBSERVATION_DIM, 4, 96, 256)
    agent.load(arguments.policy)
    full, stateless = [], []
    for index in range(arguments.episodes):
        seed = 9000 + index
        full.append(run(agent, seed, arguments.episode_steps, arguments.parameter_scale, True))
        stateless.append(run(agent, seed, arguments.episode_steps, arguments.parameter_scale, False))
    print("保留上下文 %.4f rad，上下文清零 %.4f rad" % (np.mean(full), np.mean(stateless)))


if __name__ == "__main__":
    main()
