"""第三阶段：把控制收敛后的上下文翻译成动力学参数（可辨识坐标）。

解码器输入有两种表示：
  方向平均：回合末尾 100 步单位化上下文的平均；
  原始上下文：回合最后一拍的 64 维隐状态。
对两种表示分别做岭回归与带权重衰减的小网络，用留出回合计算 R^2；
再把解码出的可辨识坐标投影成物理合法参数，比较开环轨迹误差与常量先验的差距。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "identification"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gimbal_dynamics as gd
import identify_physical as ip
import param_basis as pb
from evaluate import run_episode
from sac import RecurrentSAC
from tracking_env import OBSERVATION_DIM


def collect_dataset(agent, episodes: int, episode_steps: int, parameter_scale: float,
                    seed: int = 12345):
    basis = pb.compute_basis()[0]
    directions, hiddens, targets, raws = [], [], [], []
    for index in range(episodes):
        result = run_episode(agent, seed + index, episode_steps, parameter_scale)
        directions.append(result["contexts"][-100:].mean(axis=0))
        hiddens.append(result["hidden"])
        raw = result["parameters"]
        raws.append(raw)
        targets.append(basis.T @ pb.inertial_from_raw(raw))
    return (np.stack(directions).astype(np.float32), np.stack(hiddens).astype(np.float32),
            np.stack(targets).astype(np.float32), np.stack(raws), basis)


class Decoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, layer_dim: int = 64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(input_dim, layer_dim), nn.ReLU(),
                                  nn.Linear(layer_dim, output_dim))

    def forward(self, inputs):
        return self.body(inputs)


def ridge_probe(features, targets, train_index, test_index, penalty: float = 1.0):
    design = features[train_index] - features[train_index].mean(axis=0)
    target = targets[train_index] - targets[train_index].mean(axis=0)
    weights = np.linalg.solve(design.T @ design + penalty * np.eye(design.shape[1]),
                              design.T @ target)
    prediction = (features[test_index] - features[train_index].mean(axis=0)) @ weights
    truth = targets[test_index] - targets[train_index].mean(axis=0)
    return 1.0 - ((prediction - truth) ** 2).sum() / (truth ** 2).sum()


def train_decoder(features, targets, train_index, validation_index, test_index, epochs: int,
                  learning_rate: float, weight_decay: float):
    inputs = torch.as_tensor(features)
    values = torch.as_tensor(targets)
    decoder = Decoder(features.shape[1], targets.shape[1])
    optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    loss_function = nn.MSELoss()
    best_state, best_validation, best_epoch = None, float("inf"), 0
    for epoch in range(epochs):
        decoder.train()
        optimizer.zero_grad()
        loss = loss_function(decoder(inputs[train_index]), values[train_index])
        loss.backward()
        optimizer.step()
        decoder.eval()
        with torch.no_grad():
            validation = float(loss_function(decoder(inputs[validation_index]),
                                             values[validation_index]))
        if validation < best_validation:
            best_validation, best_epoch = validation, epoch
            best_state = {key: value.clone() for key, value in decoder.state_dict().items()}
    decoder.load_state_dict(best_state)
    decoder.eval()
    with torch.no_grad():
        prediction = decoder(inputs[test_index]).numpy()
    truth = values[test_index].numpy()
    r_squared = 1.0 - ((prediction - truth) ** 2).sum() / ((truth - truth.mean(axis=0)) ** 2).sum()
    per_dimension = 1.0 - ((prediction - truth) ** 2).sum(axis=0) / (
        (truth - truth.mean(axis=0)) ** 2).sum(axis=0)
    return decoder, float(r_squared), per_dimension, best_epoch


def open_loop_error(true_raw, candidate_raw, steps: int = 150):
    """同一力矩序列下两个模型的开环轨迹差；参数退化导致无法仿真时返回 NaN。"""
    context = gd.Context(1)
    truth_model = ip.build_model(true_raw)
    candidate_model = ip.build_model(candidate_raw)
    terrain = gd.make_terrain(0.12, 0.5)
    base = gd.make_base(yaw=0.4)
    state_true = gd.make_state(np.zeros(4), np.zeros(4))
    state_candidate = gd.make_state(np.zeros(4), np.zeros(4))
    torques = np.random.default_rng(3).normal(0.0, 0.4, (steps, 4))
    difference = []
    try:
        for torque in torques:
            state_true = context.forward_step(truth_model, terrain, base, state_true,
                                              torque, 0.01, 2)
            state_candidate = context.forward_step(candidate_model, terrain, base,
                                                   state_candidate, torque, 0.01, 2)
            difference.append(np.abs(np.array([state_true.angle[i] for i in range(4)])
                                     - np.array([state_candidate.angle[i] for i in range(4)])))
    except RuntimeError:
        context.close()
        return float("nan")
    context.close()
    return float(np.array(difference).mean())


def valid(raw):
    try:
        ip.build_model(raw)
        return True
    except RuntimeError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="上下文到参数的解码器")
    parser.add_argument("--policy", type=Path, default=Path("data/rl_stage2b.pt"))
    parser.add_argument("--episodes", type=int, default=520)
    parser.add_argument("--episode-steps", type=int, default=1200)
    parser.add_argument("--parameter-scale", type=float, default=0.8)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--layer-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-3)
    parser.add_argument("--report", type=Path, default=Path("data/rl_decoder.json"))
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    agent = RecurrentSAC(OBSERVATION_DIM, 4, arguments.hidden_dim, arguments.layer_dim)
    agent.load(arguments.policy)
    cache = arguments.report.with_name("rl_decoder_dataset.npz")
    if cache.exists():
        payload = np.load(cache)
        if "directions" in payload.files and "hiddens" in payload.files:
            directions, hiddens = payload["directions"], payload["hiddens"]
            targets, raws, basis = payload["targets"], payload["raws"], payload["basis"]
        else:
            cache.unlink()
    if not cache.exists():
        directions, hiddens, targets, raws, basis = collect_dataset(
            agent, arguments.episodes, arguments.episode_steps, arguments.parameter_scale)
        np.savez_compressed(cache, directions=directions, hiddens=hiddens, targets=targets,
                            raws=raws, basis=basis)
    print("数据：%d 个回合，方向表示 %s，原始上下文 %s，目标 %s" % (
        len(directions), directions.shape, hiddens.shape, targets.shape))
    target_mean = targets.mean(axis=0)
    target_scale = targets.std(axis=0) + 1.0e-8
    normalized = (targets - target_mean) / target_scale
    permutation = np.random.permutation(len(directions))
    split = int(0.7 * len(directions))
    validation_split = int(0.85 * len(directions))
    train_index = permutation[:split]
    validation_index = permutation[split:validation_split]
    test_index = permutation[validation_split:]

    report = {"episodes": int(len(directions)), "target_dim": int(targets.shape[1]),
              "context_dim": int(directions.shape[1]), "representations": {}}
    for name, features in (("方向平均", directions), ("原始上下文", hiddens)):
        features = (features - features[train_index].mean(axis=0)) / (
            features[train_index].std(axis=0) + 1.0e-8)
        linear = ridge_probe(features, normalized, train_index, test_index)
        decoder, r_squared, per_dimension, best_epoch = train_decoder(
            features, normalized, train_index, validation_index, test_index,
            arguments.epochs, arguments.learning_rate, arguments.weight_decay)
        print("%s：岭回归 R^2 %.4f，小网络 R^2 %.4f（第 %d 轮），逐维中位数 %.4f" % (
            name, linear, r_squared, best_epoch, float(np.median(per_dimension))))
        report["representations"][name] = {
            "ridge_r_squared": float(linear), "network_r_squared": float(r_squared),
            "per_dimension": per_dimension.tolist(), "best_epoch": int(best_epoch)}
        if name == "方向平均":
            decoded_coordinates = decoder(torch.as_tensor(features[test_index])).detach().numpy()
            decoded_coordinates = decoded_coordinates * target_scale + target_mean

    # 可辨识坐标误差：坐标本身决定动力学，比较解码结果与常量先验的差距
    truth_coordinates = targets[test_index]
    prior_coordinates = np.broadcast_to(target_mean, decoded_coordinates.shape)
    decoded_error = np.abs(decoded_coordinates - truth_coordinates) / target_scale
    prior_error = np.abs(prior_coordinates - truth_coordinates) / target_scale
    report["decoded_coordinate_error"] = float(decoded_error.mean())
    report["prior_coordinate_error"] = float(prior_error.mean())
    report["coordinate_improvement"] = float(1.0 - decoded_error.mean() / prior_error.mean())
    print("可辨识坐标误差：解码 %.4f，常量先验 %.4f，提升 %.1f%%" % (
        report["decoded_coordinate_error"], report["prior_coordinate_error"],
        100.0 * report["coordinate_improvement"]))
    # 物理可行性：解码结果投影后最小伪惯量特征值，以及投影移动量
    prior_model = pb.prior_model()
    deviations = []
    for position in range(min(24, len(decoded_coordinates))):
        _, deviation, eigenvalue = pb.physical_projection(decoded_coordinates[position],
                                                          basis, prior_model)
        deviations.append((float(deviation), float(eigenvalue)))
    report["projection_deviation"] = float(np.mean([item[0] for item in deviations]))
    report["projected_min_eigenvalue"] = float(np.mean([item[1] for item in deviations]))
    print("投影：坐标移动量 %.4f，最小伪惯量特征值 %.3e" % (
        report["projection_deviation"], report["projected_min_eigenvalue"]))
    arguments.report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    torch.save({"decoder": decoder.state_dict(), "target_mean": target_mean,
                "target_scale": target_scale, "basis": basis},
               arguments.report.with_suffix(".pt"))
    print("已保存", arguments.report)


if __name__ == "__main__":
    main()
