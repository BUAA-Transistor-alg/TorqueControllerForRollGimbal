"""用不含模型先验的神经网络对闭环动力学做一步预测建模。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn


class DeltaNet(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(features, 64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 8),
        )

    def forward(self, values):
        return self.network(values)


def build_features(q, qdot, tau, terrain, yaw):
    return np.concatenate([
        q, qdot, tau,
        np.array([terrain[0], terrain[1], np.sin(yaw), np.cos(yaw)]),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="神经网络动力学建模")
    parser.add_argument("--data", type=Path, default=Path("data/closed_loop_data_noisy.npz"))
    parser.add_argument("--save", type=Path, default=Path("data/nn_dynamics.pt"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261123)
    arguments = parser.parse_args()

    torch.manual_seed(arguments.seed)
    data = np.load(arguments.data)
    q = data["q"]
    qdot = data["qdot"]
    tau = data["tau"]
    terrain = data["terrain"]
    base_yaw = data["base_yaw"]
    segments = q.shape[0]
    train_segments = int(segments * 0.8)

    def make_samples(indices):
        features = []
        targets = []
        for segment in indices:
            for k in range(q.shape[1] - 1):
                features.append(build_features(q[segment, k], qdot[segment, k], tau[segment, k],
                                               terrain[segment], base_yaw[segment]))
                targets.append(np.concatenate([
                    q[segment, k + 1] - q[segment, k],
                    qdot[segment, k + 1] - qdot[segment, k],
                ]))
        return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32)

    train_x, train_y = make_samples(range(train_segments))
    val_x, val_y = make_samples(range(train_segments, segments))
    mean_x = train_x.mean(axis=0)
    std_x = train_x.std(axis=0) + 1e-8
    mean_y = train_y.mean(axis=0)
    std_y = train_y.std(axis=0) + 1e-8
    train_x = (train_x - mean_x) / std_x
    val_x = (val_x - mean_x) / std_x
    train_y = (train_y - mean_y) / std_y
    val_y = (val_y - mean_y) / std_y

    model = DeltaNet(train_x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=arguments.epochs)
    loss_fn = nn.MSELoss()
    best_state = None
    best_validation = float("inf")
    train_tensor = torch.from_numpy(train_x)
    train_target = torch.from_numpy(train_y)
    val_tensor = torch.from_numpy(val_x)
    val_target = torch.from_numpy(val_y)

    for epoch in range(arguments.epochs):
        model.train()
        order = torch.randperm(train_tensor.shape[0])
        total = 0.0
        for begin in range(0, train_tensor.shape[0], arguments.batch):
            index = order[begin:begin + arguments.batch]
            prediction = model(train_tensor[index])
            loss = loss_fn(prediction, train_target[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss) * index.numel()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            validation = float(loss_fn(model(val_tensor), val_target))
        if validation < best_validation:
            best_validation = validation
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
        if (epoch + 1) % 20 == 0:
            print("epoch %d train %.6e validation %.6e" % (
                epoch + 1, total / train_tensor.shape[0], validation))

    model.load_state_dict(best_state)
    model.eval()
    torch.save({
        "state_dict": model.state_dict(),
        "mean_x": mean_x, "std_x": std_x,
        "mean_y": mean_y, "std_y": std_y,
        "train_segments": train_segments,
        "train_state_min": np.stack([q[:train_segments].min(axis=(0, 1)),
                                     qdot[:train_segments].min(axis=(0, 1))]),
        "train_state_max": np.stack([q[:train_segments].max(axis=(0, 1)),
                                     qdot[:train_segments].max(axis=(0, 1))]),
        "train_torque_min": tau[:train_segments].min(axis=(0, 1)),
        "train_torque_max": tau[:train_segments].max(axis=(0, 1)),
        "train_terrain_min": terrain[:train_segments].min(axis=0),
        "train_terrain_max": terrain[:train_segments].max(axis=0),
    }, arguments.save)
    print("最佳验证损失", best_validation)
    print("模型已保存到", arguments.save)

    # 验证集一步误差
    model.eval()
    with torch.no_grad():
        prediction = model(val_tensor) * torch.from_numpy(std_y) + torch.from_numpy(mean_y)
        target = val_target * torch.from_numpy(std_y) + torch.from_numpy(mean_y)
        error = (prediction - target).numpy()
    print("一步位置增量 RMSE", np.sqrt(np.mean(error[:, :4] ** 2, axis=0)))
    print("一步速度增量 RMSE", np.sqrt(np.mean(error[:, 4:] ** 2, axis=0)))


if __name__ == "__main__":
    main()
