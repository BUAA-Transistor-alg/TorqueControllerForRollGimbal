"""开环仿真精度评估：比较辨识参数与真实参数在同一初始状态、同一力矩序列下的开环轨迹。

真实参数只用于评估，不参与辨识。本文件同时提供可复用的评估接口：

    import eval_openloop as open_loop
    result = open_loop.evaluate(identified_x)          # identified_x 为 56 维原始参数
    print(open_loop.format_report(result))

指标：
- horizon 处的逐关节位置 RMSE 与最大误差；
- 初始加速度误差与初始运动方向一致比例；
- 静止状态下零力矩与小幅力矩的初始加速度方向一致比例；
- 物理一致性（质量、惯量正定）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import gimbal_dynamics as gd
import identify_physical as ip

HORIZONS = (5, 10, 25, 50, 100, 300)
DIRECTION_HORIZONS = (5, 10, 25, 50)
JOINT_NAMES = ("大 yaw", "roll", "小 yaw", "pitch")
DEFAULT_DATA = Path("data/closed_loop_data_noisy.npz")
DIRECTION_THRESHOLD = 2.0e-3


def simulate(context: gd.Context, model, terrain, base, initial_q, initial_v, torques):
    steps = len(torques)
    state = gd.make_state(initial_q, initial_v)
    angles = np.zeros((steps, 4))
    velocities = np.zeros((steps, 4))
    for k in range(steps):
        angles[k] = [state.angle[i] for i in range(4)]
        velocities[k] = [state.velocity[i] for i in range(4)]
        state = context.forward_step(model, terrain, base, state, torques[k], 0.01, 2)
    return angles, velocities


def physical_report(x: np.ndarray) -> str:
    lines = []
    for body in range(4):
        inertia = np.array(
            [[x[16 + 6 * body], x[16 + 6 * body + 3], x[16 + 6 * body + 4]],
             [x[16 + 6 * body + 3], x[17 + 6 * body], x[16 + 6 * body + 5]],
             [x[16 + 6 * body + 4], x[16 + 6 * body + 5], x[18 + 6 * body]]])
        lines.append("  B%d 质量 %+.5f  惯量特征值最小 %+.3e"
                     % (body + 1, x[body], np.linalg.eigvalsh(inertia)[0]))
    lines.append("  阻尼 %s" % np.array2string(x[52:56], precision=5))
    return "\n".join(lines)


def load_data(path: Path = DEFAULT_DATA):
    return np.load(path)


def evaluate(identified_x: np.ndarray, data=DEFAULT_DATA, segments: int = 0,
             initial: str = "clean", torque: str = "clean",
             segment_range: tuple[int, int] | None = None) -> dict:
    """评估辨识参数相对真实参数的开环轨迹偏差。segments=0 表示全部段。

    segment_range 为 (起, 止) 半开区间，用于只看留出段。
    """
    if not isinstance(data, np.ndarray) and not hasattr(data, "files"):
        data = np.load(data)
    total = data["q_clean"].shape[0]
    count = total if segments in (0, None) else min(segments, total)
    first = 0 if segment_range is None else int(segment_range[0])
    last = count if segment_range is None else min(int(segment_range[1]), count)
    count = last - first
    identified = ip.build_model(np.asarray(identified_x, dtype=float))
    truth = gd.make_model()
    if initial == "clean":
        q0, v0 = data["q_clean"][first:last, 0], data["qdot_clean"][first:last, 0]
    else:
        q0, v0 = data["q"][first:last, 0], data["qdot"][first:last, 0]
    torque_sequence = (data["tau_clean"] if torque == "clean" else data["tau"])[first:last]

    steps = torque_sequence.shape[1]
    truth_angles = np.zeros((count, steps, 4))
    ident_angles = np.zeros((count, steps, 4))
    qdd_truth = np.zeros((count, 4))
    qdd_ident = np.zeros((count, 4))
    context = gd.Context(1)
    try:
        for segment in range(count):
            scene = first + segment
            terrain = gd.make_terrain(float(data["terrain"][scene, 0]),
                                      float(data["terrain"][scene, 1]))
            base = gd.make_base(yaw=float(data["base_yaw"][scene]))
            truth_angles[segment], _ = simulate(context, truth, terrain, base, q0[segment],
                                                v0[segment], torque_sequence[segment])
            ident_angles[segment], _ = simulate(context, identified, terrain, base, q0[segment],
                                                v0[segment], torque_sequence[segment])
            state = gd.make_state(q0[segment], v0[segment])
            qdd_truth[segment] = context.inverse_dynamics(truth, terrain, base, state, np.zeros(4))
            qdd_ident[segment] = context.inverse_dynamics(identified, terrain, base, state,
                                                          np.zeros(4))
        direction = static_direction_report(context, truth, identified, data, scene_range=(first, last))
    finally:
        context.close()

    difference = ident_angles - truth_angles
    metrics = {"segments": count, "steps": steps, "difference": difference,
               "truth_angles": truth_angles, "ident_angles": ident_angles,
               "qdd_truth": qdd_truth, "qdd_ident": qdd_ident,
               "reference": data["q_clean"][first:last], "static_direction": direction}
    metrics["rmse"] = {horizon: np.sqrt(np.mean(difference[:, :horizon, :] ** 2, axis=(0, 1)))
                       for horizon in HORIZONS}
    metrics["maximum"] = {horizon: np.max(np.abs(difference[:, :horizon, :]), axis=(0, 1))
                          for horizon in HORIZONS}
    reference_error = ident_angles - data["q_clean"][first:last]
    metrics["reference_rmse"] = {horizon: np.sqrt(
        np.mean(reference_error[:, :horizon, :] ** 2, axis=(0, 1))) for horizon in HORIZONS}
    agree = {}
    for horizon in DIRECTION_HORIZONS:
        displacement_truth = truth_angles[:, horizon, :] - truth_angles[:, 0, :]
        displacement_ident = ident_angles[:, horizon, :] - ident_angles[:, 0, :]
        mask = np.abs(displacement_truth) > DIRECTION_THRESHOLD
        fraction = np.zeros(4)
        samples = np.zeros(4, dtype=int)
        for joint in range(4):
            selected = mask[:, joint]
            samples[joint] = int(np.sum(selected))
            if samples[joint] > 0:
                fraction[joint] = np.mean(np.sign(displacement_truth[selected, joint]) ==
                                          np.sign(displacement_ident[selected, joint]))
        agree[horizon] = (fraction, samples)
    metrics["direction"] = agree
    wrong = 0
    for horizon in DIRECTION_HORIZONS:
        fraction, samples = agree[horizon]
        wrong += int(round(np.sum((1.0 - fraction) * samples)))
    metrics["direction_wrong"] = wrong
    return metrics


def static_direction_report(context: gd.Context, truth, identified, data, torques=(0.0, 0.5),
                            scene_range=None):
    """静止状态下施加零力矩与小幅正力矩，比较初始加速度方向。"""
    report = {}
    first = 0 if scene_range is None else int(scene_range[0])
    last = data["terrain"].shape[0] if scene_range is None else int(scene_range[1])
    for torque_value in torques:
        matches = 0
        total = 0
        relative = []
        for segment in range(first, last):
            terrain = gd.make_terrain(float(data["terrain"][segment, 0]),
                                      float(data["terrain"][segment, 1]))
            base = gd.make_base(yaw=float(data["base_yaw"][segment]))
            state = gd.make_state(data["q_clean"][segment, 0], np.zeros(4))
            command = np.full(4, torque_value)
            truth_qdd = np.array(context.inverse_dynamics(truth, terrain, base, state, command))
            ident_qdd = np.array(context.inverse_dynamics(identified, terrain, base, state, command))
            magnitude = np.maximum(np.abs(truth_qdd), 1.0e-3)
            for joint in range(4):
                total += 1
                if np.sign(truth_qdd[joint]) == np.sign(ident_qdd[joint]):
                    matches += 1
                relative.append(abs(ident_qdd[joint] - truth_qdd[joint]) / magnitude[joint])
        report[torque_value] = (matches / max(total, 1), float(np.mean(relative)))
    return report


def format_report(metrics: dict) -> str:
    lines = ["评估段数 %d，步长 0.01 s" % metrics["segments"],
             "\n开环位置误差（真值参数 vs 辨识参数）："]
    for horizon in HORIZONS:
        lines.append("  %3d 步 (%5.2f s) RMSE %s" % (
            horizon, horizon * 0.01,
            " ".join("%.3e" % value for value in metrics["rmse"][horizon])))
        lines.append("                MAX  %s" % " ".join(
            "%.3e" % value for value in metrics["maximum"][horizon]))
    lines.append("\n初始运动方向一致比例（阈值 %.0e rad）：" % DIRECTION_THRESHOLD)
    for horizon in DIRECTION_HORIZONS:
        fraction, samples = metrics["direction"][horizon]
        lines.append("  %3d 步  %s  （样本数 %s）" % (
            horizon,
            " ".join("%s %.3f" % (JOINT_NAMES[joint], fraction[joint]) for joint in range(4)),
            " ".join("%d" % samples[joint] for joint in range(4))))
    lines.append("  方向不一致总数 %d" % metrics["direction_wrong"])
    lines.append("\n静止状态初始加速度方向一致比例 / 相对误差：")
    for torque_value, (fraction, relative) in metrics["static_direction"].items():
        lines.append("  力矩 %.2f：方向 %.4f，相对误差 %.4f" % (torque_value, fraction, relative))
    lines.append("\n初始加速度误差 RMS（rad/s^2）：%s" % np.array2string(
        np.sqrt(np.mean((metrics["qdd_ident"] - metrics["qdd_truth"]) ** 2, axis=0)), precision=4))
    return "\n".join(lines)


def score(metrics: dict) -> float:
    """单一评分：0.5 s 开环 RMSE 的逐关节最大值，方向不一致按 0.1 权重叠加。"""
    return float(np.max(metrics["rmse"][50]) + 0.1 * metrics["direction_wrong"])


def load_model(path: Path) -> np.ndarray:
    return np.load(path)["model"]


def main() -> None:
    parser = argparse.ArgumentParser(description="开环仿真精度评估")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--identified", type=Path, default=Path("data/identified_final.npz"))
    parser.add_argument("--initial", choices=("clean", "noisy"), default="clean")
    parser.add_argument("--torque", choices=("clean", "noisy"), default="clean")
    parser.add_argument("--segments", type=int, default=0)
    arguments = parser.parse_args()
    metrics = evaluate(load_model(arguments.identified), arguments.data, arguments.segments,
                       arguments.initial, arguments.torque)
    print(format_report(metrics))
    print("\n辨识参数物理一致性：")
    print(physical_report(load_model(arguments.identified)))
    print("\n评分（越小越好）：%.6e" % score(metrics))


if __name__ == "__main__":
    main()
