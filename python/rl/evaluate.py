"""评估自适应控制器的跟踪性能、限位行为与漂移适应。

用法（在仓库根目录）：
    python3 python/rl/evaluate.py --policy data/rl_adaptive.pt --episodes 8 \
        --scales 0.0,0.5,1.0,1.5,2.0,2.5 --report data/rl_eval_adaptive.json
    python3 python/rl/evaluate.py --policy data/rl_adaptive.pt --compare data/rl_fixed.pt
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

from sac import RecurrentSAC
from tracking_env import FAST_INPUT_DIM, LIMITS, OBSERVATION_DIM, TrackingEnv


def build_agent(policy: Path) -> RecurrentSAC:
    payload = torch.load(policy, map_location="cpu", weights_only=False)
    actor = payload["actor"]
    fast_dim = actor["encoder.fast.projection.weight"].shape[0]
    slow_dim = actor["encoder.slow.weight_ih"].shape[0] // 3
    layer_dim = actor["body.0.weight"].shape[0]
    agent = RecurrentSAC(OBSERVATION_DIM, 4, FAST_INPUT_DIM, fast_dim, slow_dim, 18,
                         layer_dim)
    agent.load(policy)
    agent.actor.eval()
    return agent


def run_episode(agent: RecurrentSAC, seed: int, episode_steps: int, parameter_scale: float,
                drift_scale: float = 0.0, drift_motor: float = 0.0, difficulty: float = 1.0,
                noise_scale: float = 1.0, bias_scale: float = 1.0,
                deterministic: bool = True, ablate: str = "none",
                freeze_drift: bool = False):
    env = TrackingEnv(seed=seed, episode_steps=episode_steps,
                      parameter_scale=parameter_scale, drift_scale=drift_scale,
                      drift_motor=drift_motor, difficulty=difficulty,
                      noise_scale=noise_scale, bias_scale=bias_scale,
                      load_in_observation=getattr(agent, "load_in_observation", True),
                      freeze_drift=freeze_drift)
    observation = env.reset()
    hidden_actor, _, _ = agent.initial_hidden(1)
    errors, angles, references, actions = [], [], [], []
    for _ in range(episode_steps):
        if ablate == "slow":
            hidden_actor = hidden_actor.clone()
            hidden_actor[..., agent.fast_dim:] = 0.0
        elif ablate == "all":
            hidden_actor = agent.actor.initial_state(1)
        tensor = torch.as_tensor(observation[None], dtype=torch.float32)
        with torch.no_grad():
            action, _, hidden_actor = agent.act_partial(tensor, hidden_actor, deterministic)
        observation, reward, done, info = env.step(action.numpy()[0])
        errors.append(info["error"].copy())
        actions.append(action.numpy()[0].copy())
        angles.append(np.array([env.state.angle[i] for i in range(4)]))
        references.append(env.angle_ref[min(env.step_index, episode_steps - 1)].copy())
    result = {"seed": seed, "error": np.array(errors), "action": np.array(actions),
              "angle": np.array(angles), "reference": np.array(references),
              "hits": env.hits, "raw": env.raw.copy()}
    env.close()
    return result


def closest_approach(angle: np.ndarray) -> float:
    """受限关节在整段回合里离最近硬限位的最小距离（度）。"""
    closest = np.inf
    for joint, limits in enumerate(LIMITS):
        if limits is None:
            continue
        values = angle[:, :, joint]
        distance = np.minimum(values - limits[0], limits[1] - values)
        closest = min(closest, float(distance.min()))
    return float(np.degrees(closest))


def step_response(references, errors, threshold: float = 0.01, window: int = 200):
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


def summarize_episodes(results, episode_steps: int) -> dict:
    errors = np.stack([item["error"] for item in results])
    references = np.stack([item["reference"] for item in results])
    angles = np.stack([item["angle"] for item in results])
    outside = np.zeros(errors.shape, dtype=bool)
    for joint, limits in enumerate(LIMITS):
        if limits is None:
            continue
        outside[:, :, joint] = ((references[:, :, joint] < limits[0])
                                | (references[:, :, joint] > limits[1]))
    hits = np.array([item["hits"] for item in results])
    peaks, recoveries = step_response(references, errors)
    per_joint_in_range = []
    for joint in range(4):
        mask = ~outside[:, :, joint]
        per_joint_in_range.append(float(np.abs(errors[:, :, joint][mask]).mean()))
    return {"mean_error": float(np.abs(errors).mean()),
            "per_joint": np.abs(errors).mean(axis=(0, 1)).tolist(),
            "per_joint_in_range": per_joint_in_range,
            "in_range_error": float(np.abs(errors[~outside]).mean()),
            "out_of_range_error": (float(np.abs(errors[outside]).mean())
                                   if outside.any() else None),
            "out_of_range_fraction": float(outside.mean()),
            "hits_per_episode": float(hits.mean()), "max_hits": int(hits.max()),
            "episodes_with_hits": int((hits > 0).sum()),
            "closest_approach_deg": closest_approach(angles),
            "step_peak_error": float(np.mean(peaks)) if peaks else None,
            "step_recovery_steps": float(np.mean(recoveries)) if recoveries else None,
            "step_count": len(peaks),
            "error_windows": [float(np.abs(errors[:, window]).mean())
                              for window in np.array_split(np.arange(episode_steps), 5)],
            "per_episode": [float(np.abs(item["error"]).mean()) for item in results],
            "per_episode_hits": hits.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description="自适应控制器评估")
    parser.add_argument("--policy", type=Path, default=Path("data/rl_adaptive.pt"))
    parser.add_argument("--compare", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--episode-steps", type=int, default=1500)
    parser.add_argument("--scales", type=str, default="0.0,0.5,1.0,1.5,2.0,2.5")
    parser.add_argument("--drift-scales", type=str, default="0.0,0.2,0.35")
    parser.add_argument("--drift-motor", type=float, default=0.3)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--bias-scale", type=float, default=1.0)
    parser.add_argument("--seed-base", type=int, default=7000)
    parser.add_argument("--report", type=Path, default=Path("data/rl_eval_adaptive.json"))
    parser.add_argument("--threads", type=int, default=4)
    arguments = parser.parse_args()
    torch.set_num_threads(arguments.threads)
    seeds = list(range(arguments.seed_base, arguments.seed_base + arguments.episodes))
    agent = build_agent(arguments.policy)
    baseline = build_agent(arguments.compare) if arguments.compare is not None else None

    report = {"policy": str(arguments.policy), "compare": None if baseline is None
              else str(arguments.compare), "episodes": arguments.episodes,
              "episode_steps": arguments.episode_steps, "seed_base": arguments.seed_base,
              "parameter_scale": {}, "drift": {}, "drift_frozen": {},
              "drift_ratio": {}, "noise": {}, "bias": {}}

    def run(agent_, **kwargs):
        return [run_episode(agent_, seed, arguments.episode_steps,
                            noise_scale=arguments.noise_scale,
                            bias_scale=arguments.bias_scale, **kwargs)
                for seed in seeds]

    for scale in [float(item) for item in arguments.scales.split(",")]:
        result = summarize_episodes(run(agent, parameter_scale=scale), arguments.episode_steps)
        report["parameter_scale"][str(scale)] = result
        line = ("参数尺度 %.1f：限位内误差 %.4f 限位内分关节 %s 撞限位 平均 %.2f/最大 %d "
                "最近限位距离 %.1f 度 阶跃峰值 %s") % (
            scale, result["in_range_error"],
            np.round(result["per_joint_in_range"], 4),
            result["hits_per_episode"], result["max_hits"],
            result["closest_approach_deg"],
            None if result["step_peak_error"] is None else round(result["step_peak_error"], 3))
        print(line, flush=True)
        if baseline is not None:
            other = summarize_episodes(run(baseline, parameter_scale=scale),
                                       arguments.episode_steps)
            report.setdefault("compare_parameter_scale", {})[str(scale)] = other
            print("  对照：限位内误差 %.4f 分关节 %s 撞限位 平均 %.2f/最大 %d" % (
                other["in_range_error"], np.round(other["per_joint"], 4),
                other["hits_per_episode"], other["max_hits"]), flush=True)

    for drift in [float(item) for item in arguments.drift_scales.split(",")]:
        result = summarize_episodes(
            run(agent, parameter_scale=1.0, drift_scale=drift,
                drift_motor=arguments.drift_motor * (1.0 if drift > 0.0 else 0.0)),
            arguments.episode_steps)
        report["drift"][str(drift)] = result
        line = ("漂移 %.2f：限位内误差 %.4f 限位内分关节 %s 撞限位 平均 %.2f/最大 %d "
                "五个时段 %s" % (
                    drift, result["in_range_error"],
                    np.round(result["per_joint_in_range"], 4),
                    result["hits_per_episode"], result["max_hits"],
                    np.round(result["error_windows"], 4)))
        if drift > 0.0:
            frozen = summarize_episodes(
                run(agent, parameter_scale=1.0, drift_scale=drift,
                    drift_motor=arguments.drift_motor, freeze_drift=True),
                arguments.episode_steps)
            report["drift_frozen"][str(drift)] = frozen
            report["drift_ratio"][str(drift)] = (
                result["in_range_error"] / frozen["in_range_error"])
            line += " | 冻结对照 %.4f（比值 %.2f）撞限位 平均 %.2f/最大 %d" % (
                frozen["in_range_error"], report["drift_ratio"][str(drift)],
                frozen["hits_per_episode"], frozen["max_hits"])
        print(line, flush=True)

    for name, kwargs in (("noise", {"noise_scale": 4.0}), ("bias", {"bias_scale": 3.0})):
        results = [run_episode(agent, seed, arguments.episode_steps, 1.5,
                               noise_scale=kwargs.get("noise_scale", 1.0),
                               bias_scale=kwargs.get("bias_scale", 1.0))
                   for seed in seeds]
        result = summarize_episodes(results, arguments.episode_steps)
        report[name]["scale"] = result
        print("%s 放大：限位内误差 %.4f 撞限位 平均 %.2f/最大 %d" % (
            "噪声" if name == "noise" else "偏差", result["in_range_error"],
            result["hits_per_episode"], result["max_hits"]), flush=True)

    arguments.report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    archive_results = run(agent, parameter_scale=0.0)
    np.savez_compressed(arguments.report.with_suffix(".npz"),
                        error=np.stack([item["error"] for item in archive_results]),
                        angle=np.stack([item["angle"] for item in archive_results]),
                        reference=np.stack([item["reference"] for item in archive_results]))
    print("已保存", arguments.report)


if __name__ == "__main__":
    main()
