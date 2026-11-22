"""可适应范围扫描：参数范围、观测噪声、观测偏差。

对每个配置跑若干回合，统计限位内误差、限位外目标误差、撞限位次数与阶跃响应，
用于确定训练后的控制器能适应的范围。
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

from evaluate import run_episode, step_response
from sac import RecurrentSAC
from tracking_env import LIMITS, OBSERVATION_DIM


def measure(agent, episodes: int, steps: int, parameter_scale: float, noise_scale: float,
            bias_scale: float, seed: int = 11000):
    limits = LIMITS
    errors, outside, hits, peaks, recoveries = [], [], [], [], []
    contexts = []
    for index in range(episodes):
        result = run_episode(agent, seed + index, steps, parameter_scale,
                             noise_scale=noise_scale, bias_scale=bias_scale)
        reference = result["reference"]
        mask = np.zeros(reference.shape[0], dtype=bool)
        for joint, limits in enumerate(LIMITS):
            if limits is None:
                continue
            mask |= (reference[:, joint] < limits[0]) | (reference[:, joint] > limits[1])
        errors.append(result["error"])
        outside.append(mask)
        hits.append(result["hits"])
        peak, recovery = step_response(reference[None], result["error"][None])
        peaks.extend(peak)
        recoveries.extend(recovery)
        contexts.append(result["contexts"])
    errors = np.stack(errors)
    outside = np.stack(outside)
    changes = np.stack([np.linalg.norm(item[1:] - item[:-1], axis=-1) for item in contexts])
    return {
        "error": float(np.abs(errors).mean()),
        "error_in_range": float(np.abs(errors[~outside]).mean()) if (~outside).any() else None,
        "error_out_of_range": float(np.abs(errors[outside]).mean()) if outside.any() else None,
        "out_of_range_fraction": float(outside.mean()),
        "hits": float(np.mean(hits)),
        "step_peak": float(np.mean(peaks)) if peaks else None,
        "step_recovery": float(np.mean(recoveries)) if recoveries else None,
        "context_change": [float(changes[:, 0:150].mean()),
                           float(changes[:, 300:600].mean()),
                           float(changes[:, 900:].mean())],
        "error_windows": [float(np.abs(errors[:, window]).mean())
                          for window in np.array_split(np.arange(errors.shape[1]), 5)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="可适应范围扫描")
    parser.add_argument("--policy", type=Path, default=Path("data/rl_noisy_stage2.pt"))
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--episode-steps", type=int, default=1200)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--layer-dim", type=int, default=256)
    parser.add_argument("--extractor-layer-dim", type=int, default=64)
    parser.add_argument("--report", type=Path, default=Path("data/rl_range.json"))
    parser.add_argument("--sweep", default="all")
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    agent = RecurrentSAC(OBSERVATION_DIM, 4, arguments.hidden_dim, arguments.layer_dim)
    agent.load(arguments.policy)
    report = {}
    if arguments.sweep in ("all", "parameter"):
        report["parameter"] = {}
        for scale in (0.5, 1.0, 1.5, 2.0, 2.5):
            result = measure(agent, arguments.episodes, arguments.episode_steps, scale, 1.0, 1.0)
            report["parameter"][str(scale)] = result
            print("参数尺度 %.1f：误差 %.4f（限位内 %.4f，限位外 %s），撞限位 %.1f" % (
                scale, result["error"], result["error_in_range"],
                None if result["error_out_of_range"] is None
                else round(result["error_out_of_range"], 4), result["hits"]), flush=True)
    if arguments.sweep in ("all", "noise"):
        report["noise"] = {}
        for scale in (1.0, 2.0, 4.0, 8.0):
            result = measure(agent, arguments.episodes, arguments.episode_steps, 1.5, scale, 1.0)
            report["noise"][str(scale)] = result
            print("噪声倍数 %.1f：误差 %.4f，撞限位 %.1f" % (
                scale, result["error"], result["hits"]), flush=True)
    if arguments.sweep in ("all", "bias"):
        report["bias"] = {}
        for scale in (0.0, 1.0, 2.0, 3.0):
            result = measure(agent, arguments.episodes, arguments.episode_steps, 1.5, 1.0, scale)
            report["bias"][str(scale)] = result
            print("偏差倍数 %.1f：误差 %.4f，撞限位 %.1f" % (
                scale, result["error"], result["hits"]), flush=True)
    arguments.report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("已保存", arguments.report)


if __name__ == "__main__":
    main()