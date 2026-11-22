"""参考控制器评估：真参数教师（参数知识上限）与结构化自适应控制器。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "identification"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from teacher import StructuredTeacher, TeacherController
from tracking_env import LIMITS, TrackingEnv


def measure(kind: str, seeds, episode_steps: int, parameter_scale: float,
            drift_scale: float = 0.0, drift_motor: float = 0.0):
    totals, inside, hits = [], [], []
    for seed in seeds:
        env = TrackingEnv(seed=seed, episode_steps=episode_steps,
                          parameter_scale=parameter_scale, drift_scale=drift_scale,
                          drift_motor=drift_motor)
        observation = env.reset()
        if kind == "true":
            controller = TeacherController(env)
        else:
            controller = StructuredTeacher(env)
            controller.reset()
        errors = []
        for step in range(episode_steps):
            action = controller.action(env.step_index, observation)
            observation, reward, done, info = env.step(action)
            errors.append(info["error"].copy())
        errors = np.abs(np.array(errors))
        reference = env.angle_ref[:episode_steps]
        outside = np.zeros(errors.shape, dtype=bool)
        for joint, limits in enumerate(LIMITS):
            if limits is not None:
                outside[:, joint] = ((reference[:, joint] < limits[0])
                                     | (reference[:, joint] > limits[1]))
        totals.append(errors.mean())
        inside.append(errors[~outside].mean())
        hits.append(env.hits)
        env.close()
    return {"mean_error": float(np.mean(totals)),
            "in_range_error": float(np.mean(inside)),
            "hits_per_episode": float(np.mean(hits)), "max_hits": int(np.max(hits)),
            "per_episode": [float(item) for item in totals]}


def main() -> None:
    parser = argparse.ArgumentParser(description="参考控制器评估")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--episode-steps", type=int, default=1500)
    parser.add_argument("--scales", type=str, default="0.0,1.0,1.75,2.5")
    parser.add_argument("--drift", type=str, default="0.0,0.35")
    parser.add_argument("--seed-base", type=int, default=7000)
    parser.add_argument("--report", type=Path, default=Path("data/rl_reference.json"))
    arguments = parser.parse_args()
    seeds = list(range(arguments.seed_base, arguments.seed_base + arguments.episodes))
    report = {"true_teacher": {}, "structured": {}, "structured_drift": {},
              "true_teacher_drift": {}}
    for scale in [float(item) for item in arguments.scales.split(",")]:
        for kind, key in (("true", "true_teacher"), ("structured", "structured")):
            result = measure(kind, seeds, arguments.episode_steps, scale)
            report[key][str(scale)] = result
            print("%s 尺度 %.2f：限位内 %.4f 全域 %.4f 撞限位 %.2f/最大 %d" % (
                "真参数教师" if kind == "true" else "结构化控制器", scale,
                result["in_range_error"], result["mean_error"],
                result["hits_per_episode"], result["max_hits"]), flush=True)
    for drift in [float(item) for item in arguments.drift.split(",")]:
        if drift <= 0.0:
            continue
        for kind, key in (("true", "true_teacher_drift"), ("structured", "structured_drift")):
            result = measure(kind, seeds, arguments.episode_steps, 1.0, drift, drift * 0.85)
            report[key][str(drift)] = result
            print("%s 漂移 %.2f：限位内 %.4f 全域 %.4f 撞限位 %.2f/最大 %d" % (
                "真参数教师" if kind == "true" else "结构化控制器", drift,
                result["in_range_error"], result["mean_error"],
                result["hits_per_episode"], result["max_hits"]), flush=True)
    arguments.report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("已保存", arguments.report)


if __name__ == "__main__":
    main()
