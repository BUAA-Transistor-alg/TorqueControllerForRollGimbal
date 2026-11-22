"""标定结构化自适应控制器：用真参数教师的动作做岭回归，输出可直接写入 teacher.py 的权重。

用法（在仓库根目录）：
    OPENBLAS_NUM_THREADS=4 python3 -u python/rl/calibrate_structured.py \
        --report data/structured_weights.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "identification"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tracking_env as te
from teacher import LOAD_RATE, StructuredTeacher, TeacherController
from tracking_env import LIMITS, TrackingEnv


def configure(margin: float, mismatch_clip: float, inertia_margin: float):
    te.SOFT_LOW = np.array([-np.inf] + [LIMITS[i][0] + margin for i in (1, 2, 3)])
    te.SOFT_HIGH = np.array([np.inf] + [LIMITS[i][1] - margin for i in (1, 2, 3)])
    te.NOMINAL_MISMATCH_LIMIT = mismatch_clip
    te.INERTIA_MARGIN = inertia_margin
    te._NOMINAL.clear()


def features_and_labels(env, teacher, controller, steps: int, noise: float, rng):
    observation = env.reset()
    controller.reset()
    features, labels, errors = [], [], []
    for _ in range(steps):
        label = teacher.action(env.step_index, observation)
        values = np.asarray(observation, dtype=float)
        controller.load = (1.0 - controller.load_rate) * controller.load + (
            controller.load_rate * values[-4:])
        block = np.stack([values[-12:-8], values[-8:-4], controller.load, np.ones(4)], axis=1)
        features.append(block)
        labels.append(label)
        action = np.clip(np.sum(block * controller.weights, axis=1), -1.0, 1.0)
        applied = np.clip(action + rng.normal(0.0, noise, 4), -1.0, 1.0)
        observation, reward, done, info = env.step(applied)
        errors.append(info["error"].copy())
    reference = env.angle_ref[:steps]
    outside = np.zeros((steps, 4), dtype=bool)
    for joint, limits in enumerate(LIMITS):
        if limits is not None:
            outside[:, joint] = ((reference[:, joint] < limits[0])
                                 | (reference[:, joint] > limits[1]))
    inside = np.abs(np.array(errors))[~outside].mean()
    return np.stack(features), np.stack(labels), float(inside)


def fit(features, labels, prior_load: float = 0.85):
    x = np.concatenate(features)
    y = np.concatenate(labels)
    prior = np.array([1.0, 1.0, prior_load, 0.0])
    regularizer = np.diag([2.0, 2.0, 12.0, 2.0])
    return np.array([np.linalg.solve(x[:, joint].T @ x[:, joint] + regularizer,
                                     x[:, joint].T @ y[:, joint] + regularizer @ prior)
                     for joint in range(4)])


def evaluate(controller, seeds, steps: int, scale: float, drift: float = 0.0):
    totals, inside, hits = [], [], []
    for seed in seeds:
        env = TrackingEnv(seed=seed, episode_steps=steps, parameter_scale=scale,
                          drift_scale=drift, drift_motor=drift * 0.85)
        observation = env.reset()
        controller.reset()
        errors = []
        for _ in range(steps):
            action = controller.action(env.step_index, observation)
            observation, reward, done, info = env.step(action)
            errors.append(info["error"].copy())
        errors = np.abs(np.array(errors))
        reference = env.angle_ref[:steps]
        outside = np.zeros((steps, 4), dtype=bool)
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
            "hits_per_episode": float(np.mean(hits)), "max_hits": int(np.max(hits))}


def main() -> None:
    parser = argparse.ArgumentParser(description="结构化控制器权重标定")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--episode-steps", type=int, default=800)
    parser.add_argument("--scales", type=str, default="0.0,0.75,1.5")
    parser.add_argument("--noise", type=str, default="0.0,0.08,0.05,0.03")
    parser.add_argument("--evaluate-seeds", type=int, default=6)
    parser.add_argument("--load-rate", type=float, default=LOAD_RATE)
    parser.add_argument("--margin", type=float, default=0.035)
    parser.add_argument("--mismatch-clip", type=float, default=1.5)
    parser.add_argument("--inertia-margin", type=float, default=20.0)
    parser.add_argument("--report", type=Path, default=Path("data/structured_weights.json"))
    arguments = parser.parse_args()
    configure(arguments.margin, arguments.mismatch_clip, arguments.inertia_margin)
    rng = np.random.default_rng(0)
    scales = [float(item) for item in arguments.scales.split(",")]
    noises = [float(item) for item in arguments.noise.split(",")]
    controller = StructuredTeacher(weights=np.tile([1.0, 1.0, 0.85, 0.0], (4, 1)),
                                   load_rate=arguments.load_rate)
    features, labels = [], []
    evaluation_seeds = list(range(52000, 52000 + arguments.evaluate_seeds))
    history = []
    for round_index in range(arguments.rounds):
        noise = noises[min(round_index, len(noises) - 1)]
        for index, scale in enumerate(scales):
            for episode in range(arguments.episodes):
                seed = 50500 + 100 * round_index + 13 * index + episode
                env = TrackingEnv(seed=seed, episode_steps=arguments.episode_steps,
                                  parameter_scale=scale)
                teacher = TeacherController(env)
                block, target, inside = features_and_labels(
                    env, teacher, controller, arguments.episode_steps, noise, rng)
                env.close()
                features.append(block)
                labels.append(target)
        controller.weights = fit(features, labels)
        record = {"round": round_index, "noise": noise,
                  "weights": controller.weights.tolist(), "evaluation": {}}
        for scale in (0.0, 0.5, 1.0, 1.75, 2.5):
            record["evaluation"][str(scale)] = evaluate(
                controller, evaluation_seeds, arguments.episode_steps, scale)
        record["evaluation"]["drift"] = evaluate(
            controller, evaluation_seeds, arguments.episode_steps, 1.0, 0.35)
        history.append(record)
        print("第 %d 轮 噪声 %.2f：%s" % (round_index, noise, json.dumps(
            {key: {"in": round(value["in_range_error"], 4),
                   "hits": round(value["hits_per_episode"], 2)}
             for key, value in record["evaluation"].items()})), flush=True)
    arguments.report.write_text(json.dumps({"load_rate": arguments.load_rate,
                                            "weights": controller.weights.tolist(),
                                            "history": history}, indent=2,
                                           ensure_ascii=False))
    print("最终权重：", np.round(controller.weights, 4).tolist())
    print("已保存", arguments.report)


if __name__ == "__main__":
    main()
