"""闭环评估：同一场景、同一力矩上限下对比 PID 与两种 MPC。

PID 使用 params.py 的 KP/KI/KD 与 bias_torque 前馈；MPC 分别使用设计参数模型与
data/identified_final.npz 的辨识参数模型。被控对象默认为失配参数，关节限位按硬挡块
处理。指标包含限位内跟踪 RMSE、最大误差、撞限位步数、力矩均方与单步求解时间，
图表保存到 data/mpc/。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gimbal_dynamics as gd
import params
from mpc import scenarios
from mpc.controller import HORIZON, TORQUE_LIMIT, MpcController

LABELS = ["big yaw", "roll", "small yaw", "pitch"]
COLORS = {"PID": "C0", "MPC (design)": "C1", "MPC (identified)": "C2"}
LINE_STYLES = {"PID": "-", "MPC (design)": "--", "MPC (identified)": "-."}


class PidController:
    """params.py 增益加 bias_torque 前馈的 PID，微分作用在测量角速度上，输出限幅与 MPC 一致。"""

    horizon = 1

    def __init__(self, context: gd.Context, model: gd.TcrgModel,
                 torque_limit: float = TORQUE_LIMIT, dt: float = scenarios.DT):
        self.context = context
        self.model = model
        self.torque_limit = float(torque_limit)
        self.dt = float(dt)
        self.kp = np.asarray(params.PID_KP, dtype=float)
        self.ki = np.asarray(params.PID_KI, dtype=float)
        self.kd = np.asarray(params.PID_KD, dtype=float)
        self.integral_limit = float(params.PID_INTEGRAL_LIMIT)
        self.integral = np.zeros(4)
        self.last_info = {"solve_time": 0.0, "iterations": 0}

    def command(self, angle, velocity, target_angle,
                terrain: gd.TcrgTerrain, base: gd.TcrgBaseState) -> np.ndarray:
        started = time.perf_counter()
        angle = np.asarray(angle, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        error = np.asarray(target_angle[0], dtype=float) - angle
        self.integral = np.clip(self.integral + error * self.dt,
                                -self.integral_limit, self.integral_limit)
        state = gd.make_state(angle, velocity)
        feedforward = np.asarray(self.context.bias_torque(
            self.model, terrain, base, state))
        output = (feedforward + self.kp * error + self.ki * self.integral
                  - self.kd * velocity)
        output = np.clip(output, -self.torque_limit, self.torque_limit)
        self.last_info = {"solve_time": time.perf_counter() - started,
                          "iterations": 0}
        return output


def compute_metrics(records: dict, scenario: scenarios.Scenario) -> dict:
    """限位内跟踪指标与执行代价。角度取被控对象到达后的状态，与同一时刻参考比较。"""
    steps = scenario.steps
    angle = records["angle"][:-1]
    error = scenario.reference_angle - angle
    rmse = np.zeros(4)
    in_limit = np.zeros_like(error, dtype=bool)
    for joint in range(4):
        if scenarios.LIMITED[joint]:
            in_limit[:, joint] = ((scenario.reference_angle[:, joint] >= scenarios.LOWER[joint])
                                  & (scenario.reference_angle[:, joint] <= scenarios.UPPER[joint]))
        else:
            in_limit[:, joint] = True
        mask = in_limit[:, joint]
        rmse[joint] = float(np.sqrt(np.mean(error[mask, joint] ** 2))) if np.any(mask) else float("nan")
    command = records["command"]
    return {
        "rmse": rmse,
        "mean_rmse": float(np.mean(rmse)),
        "max_error": float(np.max(np.abs(error[in_limit]))) if np.any(in_limit) else float("nan"),
        "hit_steps": int(np.sum(records["hits"])),
        "torque_square": float(np.mean(command ** 2)),
        "torque_max": float(np.max(np.abs(command))),
        "solve_ms": float(1000.0 * np.mean(records["solve_time"])),
        "solve_ms_max": float(1000.0 * np.max(records["solve_time"])),
        "iterations": float(np.mean(records["iterations"])),
        "evaluations": float(np.mean(records["evaluations"])),
    }


def print_table(metrics: dict) -> None:
    header = ("%-18s %9s %9s %9s %9s %9s %11s %8s %11s %9s %11s %7s"
              % ("controller", "RMSE j1", "RMSE j2", "RMSE j3", "RMSE j4",
                 "RMSE mean", "max err", "hits", "tau mean sq", "|tau|max",
                 "solve ms", "evals"))
    print(header)
    print("-" * len(header))
    for name, values in metrics.items():
        print("%-18s %9.4f %9.4f %9.4f %9.4f %9.4f %11.4f %8d %11.4f %9.3f %11.2f %7.1f"
              % (name, values["rmse"][0], values["rmse"][1], values["rmse"][2],
                 values["rmse"][3], values["mean_rmse"], values["max_error"],
                 values["hit_steps"], values["torque_square"], values["torque_max"],
                 values["solve_ms"], values["evaluations"]))
    print("限位内 RMSE 只统计参考落在该关节硬限位内的时刻；hits 为至少一个关节撞挡块的步数")
    for name, values in metrics.items():
        if name.startswith("MPC"):
            print("%s 求解时间: 平均 %.2f ms 最大 %.2f ms，平均迭代 %.1f 次，每拍代价求值 %.1f 次"
                  % (name, values["solve_ms"], values["solve_ms_max"],
                     values["iterations"], values["evaluations"]))


def plot_tracking(scenario, runs, output: Path) -> None:
    time = np.arange(scenario.steps) * scenarios.DT
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 7.5), sharex=True)
    for joint in range(4):
        axis = axes.reshape(-1)[joint]
        axis.plot(time, scenario.reference_angle[:, joint], color="0.25",
                  linewidth=1.2, linestyle=":", label="reference")
        for name, records in runs.items():
            axis.plot(time, records["angle"][:-1, joint], color=COLORS[name],
                      linestyle=LINE_STYLES[name], linewidth=1.3, label=name)
        if scenarios.LIMITED[joint]:
            axis.axhline(scenarios.UPPER[joint], color="tab:red", linewidth=0.9,
                         linestyle="--")
            axis.axhline(scenarios.LOWER[joint], color="tab:red", linewidth=0.9,
                         linestyle="--")
        axis.set_ylabel("angle (rad)")
        axis.set_title(LABELS[joint] + (" (hard limits: red)" if scenarios.LIMITED[joint] else ""))
        axis.grid(alpha=0.3)
    axes.reshape(-1)[0].legend(fontsize=8, ncol=2)
    for axis in axes[-1]:
        axis.set_xlabel("time (s)")
    figure.suptitle("closed-loop tracking on mismatched plant")
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)


def plot_error(scenario, runs, output: Path) -> None:
    time = np.arange(scenario.steps) * scenarios.DT
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 7.5), sharex=True)
    for joint in range(4):
        axis = axes.reshape(-1)[joint]
        for name, records in runs.items():
            error = scenario.reference_angle[:, joint] - records["angle"][:-1, joint]
            axis.plot(time, error, color=COLORS[name], linestyle=LINE_STYLES[name],
                      linewidth=1.2, label=name)
        axis.axhline(0.0, color="0.4", linewidth=0.8)
        axis.set_ylabel("error (rad)")
        axis.set_title(LABELS[joint])
        axis.grid(alpha=0.3)
    axes.reshape(-1)[0].legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("time (s)")
    figure.suptitle("tracking error (reference minus angle)")
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)


def plot_torque(scenario, runs, output: Path, torque_limit: float) -> None:
    time = np.arange(scenario.steps) * scenarios.DT
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 7.5), sharex=True)
    for joint in range(4):
        axis = axes.reshape(-1)[joint]
        for name, records in runs.items():
            axis.plot(time, records["command"][:, joint], color=COLORS[name],
                      linestyle=LINE_STYLES[name], linewidth=1.1, label=name)
        axis.axhline(torque_limit, color="tab:red", linewidth=0.9, linestyle="--")
        axis.axhline(-torque_limit, color="tab:red", linewidth=0.9, linestyle="--")
        axis.set_ylabel("torque (N·m)")
        axis.set_title(LABELS[joint])
        axis.grid(alpha=0.3)
    axes.reshape(-1)[0].legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("time (s)")
    figure.suptitle("commanded torque (limit ±%.1f N·m)" % torque_limit)
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)


def plot_metrics(metrics: dict, output: Path) -> None:
    names = list(metrics.keys())
    categories = [("mean in-limit RMSE (rad)", "mean_rmse", 1.0),
                  ("limit hit steps", "hit_steps", 1.0),
                  ("mean torque square (N·m)^2", "torque_square", 1.0),
                  ("mean solve time (ms)", "solve_ms", 1.0)]
    figure, axes = plt.subplots(1, 4, figsize=(14.0, 4.0))
    for axis, (title, key, scale) in zip(axes, categories):
        values = [metrics[name][key] * scale for name in names]
        axis.bar(range(len(names)), values,
                 color=[COLORS[name] for name in names])
        axis.set_ylim(0.0, max(max(values) * 1.15, 1.0 if key == "hit_steps" else 1e-3))
        axis.set_xticks(range(len(names)))
        axis.set_xticklabels(names, rotation=20, fontsize=8)
        axis.set_title(title, fontsize=9)
        axis.grid(alpha=0.3, axis="y")
    figure.suptitle("metric comparison")
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="PID 与 MPC 闭环对比")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--plant", choices=("mismatch", "truth"), default="mismatch")
    parser.add_argument("--torque-limit", type=float, default=TORQUE_LIMIT)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("data/mpc"))
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    scenario = scenarios.Scenario(seed=arguments.seed, steps=arguments.steps)
    if arguments.plant == "mismatch":
        plant_model = scenarios.build_model(scenarios.mismatch_raw())
        plant_name = "mismatch (mass +10%, inertia +12%, damping ×1.5)"
    else:
        plant_model = scenarios.design_model()
        plant_name = "truth parameters"
    print("场景: seed=%d steps=%d dt=%.3f 坡度=%.2f° 坡向=%.2f° 参考 yaw 漂移极值=%.2f rad"
          % (scenario.seed, scenario.steps, scenarios.DT,
             np.degrees(scenario.slope), np.degrees(scenario.direction),
             scenario.reference_angle[:, 0].max() - scenario.reference_angle[:, 0].min()))
    print("被控对象: " + plant_name)
    runs = {}
    metrics = {}
    # 前向仿真与 MPC 求解共享同一线程数；MPC 求解器在 C++ 内部并行阶段 Jacobian，
    # 控制结果与线程数无关。
    with gd.Context(arguments.threads) as context:
        controllers = {
            "PID": PidController(context, scenarios.design_model(),
                                 torque_limit=arguments.torque_limit),
            "MPC (design)": MpcController(scenarios.design_model(),
                                          horizon=arguments.horizon,
                                          torque_limit=arguments.torque_limit,
                                          max_iter=arguments.max_iter,
                                          thread_count=arguments.threads),
            "MPC (identified)": MpcController(scenarios.identified_model(),
                                              horizon=arguments.horizon,
                                              torque_limit=arguments.torque_limit,
                                              max_iter=arguments.max_iter,
                                              thread_count=arguments.threads),
        }
        for name, controller in controllers.items():
            started = time.perf_counter()
            records = scenarios.run_closed_loop(
                scenario, controller, plant_model, context,
                noise_seed=arguments.seed + 1000,
                torque_limit=arguments.torque_limit)
            elapsed = time.perf_counter() - started
            runs[name] = records
            metrics[name] = compute_metrics(records, scenario)
            print("完成 %-18s 耗时 %6.2f s  平均迭代 %5.1f 次"
                  % (name, elapsed, metrics[name]["iterations"]))
    print()
    print_table(metrics)
    plot_tracking(scenario, runs, arguments.output_dir / "tracking.png")
    plot_error(scenario, runs, arguments.output_dir / "error.png")
    plot_torque(scenario, runs, arguments.output_dir / "torque.png", arguments.torque_limit)
    plot_metrics(metrics, arguments.output_dir / "metrics.png")
    print("图已保存到", arguments.output_dir)

    passed = True
    for name in ("MPC (design)", "MPC (identified)"):
        if metrics[name]["hit_steps"] != 0:
            passed = False
            print("[未通过] %s 撞限位步数 %d" % (name, metrics[name]["hit_steps"]))
        if metrics[name]["mean_rmse"] > metrics["PID"]["mean_rmse"]:
            passed = False
            print("[未通过] %s 限位内平均 RMSE %.4f 高于 PID %.4f"
                  % (name, metrics[name]["mean_rmse"], metrics["PID"]["mean_rmse"]))
    for name, values in metrics.items():
        if values["torque_max"] > arguments.torque_limit + 1e-9:
            passed = False
            print("[未通过] %s 力矩越界 %.4f" % (name, values["torque_max"]))
    print("结果:", "通过" if passed else "未通过")
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
