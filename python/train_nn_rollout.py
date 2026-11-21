"""用多步开环滚动训练神经网络动力学模型。

支持 MLP、GRU、LSTM 三种结构；损失在 10 到 300 步的自由滚动轨迹上计算。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn


class MlpDynamics(nn.Module):
    def __init__(self, features: int, width: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(features, width), nn.GELU(), nn.LayerNorm(width), nn.Dropout(0.1),
            nn.Linear(width, width), nn.GELU(), nn.LayerNorm(width), nn.Dropout(0.1),
            nn.Linear(width, width), nn.GELU(), nn.LayerNorm(width), nn.Dropout(0.1),
            nn.Linear(width, 4),
        )

    def forward(self, features, hidden=None):
        return self.network(features), hidden


class RecurrentDynamics(nn.Module):
    def __init__(self, features: int, hidden_size: int = 128, kind: str = "gru"):
        super().__init__()
        self.kind = kind
        if kind == "gru":
            self.rnn = nn.GRU(features, hidden_size, batch_first=True)
        else:
            self.rnn = nn.LSTM(features, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 4))

    def forward(self, features, hidden=None):
        output, hidden = self.rnn(features, hidden)
        return self.head(output), hidden


def feature_vector(q, qdot, tau, terrain, yaw, mean_x, std_x):
    values = np.concatenate([q, qdot, tau,
                             np.array([terrain[0], terrain[1], np.sin(yaw), np.cos(yaw)])])
    return (values - mean_x) / std_x


def main() -> None:
    parser = argparse.ArgumentParser(description="多步开环神经网络训练")
    parser.add_argument("--data", type=Path, default=Path("data/closed_loop_data_noisy.npz"))
    parser.add_argument("--save", type=Path, default=Path("data/nn_rollout_mlp.pt"))
    parser.add_argument("--model", choices=("mlp", "gru", "lstm"), default="mlp")
    parser.add_argument("--stages", type=str, default="10,25,50,100,200,300")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20261125)
    arguments = parser.parse_args()

    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)
    data = np.load(arguments.data)
    q = data["q"].astype(np.float64)
    qdot = data["qdot"].astype(np.float64)
    tau = data["tau"].astype(np.float64)
    terrain = data["terrain"]
    base_yaw = data["base_yaw"]
    dt = float(data["dt"])
    segments = q.shape[0]
    training_segments = list(range(int(segments * 0.8)))
    validation_segments = list(range(int(segments * 0.8), segments))

    features = []
    for segment in training_segments:
        for k in range(q.shape[1]):
            features.append(np.concatenate([q[segment, k], qdot[segment, k], tau[segment, k],
                                            [terrain[segment, 0], terrain[segment, 1],
                                             np.sin(base_yaw[segment]), np.cos(base_yaw[segment])]]))
    features = np.asarray(features)
    mean_x = features.mean(axis=0)
    std_x = features.std(axis=0) + 1e-8
    q_mean, q_std = q[training_segments].mean(axis=(0, 1)), q[training_segments].std(axis=(0, 1)) + 1e-8
    qdot_mean = qdot[training_segments].mean(axis=(0, 1))
    qdot_std = qdot[training_segments].std(axis=(0, 1)) + 1e-8
    qdd_scale = 20.0

    input_size = 16
    if arguments.model == "mlp":
        model = MlpDynamics(input_size)
    else:
        model = RecurrentDynamics(input_size, kind=arguments.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    mean_tensor = torch.tensor(mean_x, dtype=torch.float32)
    std_tensor = torch.tensor(std_x, dtype=torch.float32)
    qstd_tensor = torch.tensor(q_std, dtype=torch.float32)
    vstd_tensor = torch.tensor(qdot_std, dtype=torch.float32)
    terrain_tensor = torch.tensor(
        np.concatenate([terrain, np.sin(base_yaw)[:, None], np.cos(base_yaw)[:, None]], axis=1),
        dtype=torch.float32)

    def rollout_loss(segment, start, length, train=True):
        qn = torch.tensor(q[segment, start], dtype=torch.float32)
        vn = torch.tensor(qdot[segment, start], dtype=torch.float32)
        hidden = None
        losses = []
        for k in range(length):
            tau_tensor = torch.tensor(tau[segment, start + k], dtype=torch.float32)
            terrain_features = terrain_tensor[segment]
            raw = torch.cat([qn, vn, tau_tensor, terrain_features])
            feature = ((raw - mean_tensor) / std_tensor).unsqueeze(0)
            prediction, hidden = model(feature.unsqueeze(0), hidden)
            acceleration = prediction.reshape(4) * qdd_scale
            vn_next = vn + dt * acceleration
            qn_next = qn + dt * vn_next
            target_q = torch.tensor(q[segment, start + k + 1], dtype=torch.float32)
            target_v = torch.tensor(qdot[segment, start + k + 1], dtype=torch.float32)
            losses.append(torch.mean(((qn_next - target_q) / qstd_tensor) ** 2))
            losses.append(0.1 * torch.mean(((vn_next - target_v) / vstd_tensor) ** 2))
            qn, vn = qn_next, vn_next
        return sum(losses) / len(losses)

    def rollout_loss_batch(segments_list, start, length):
        value = 0.0
        for segment in segments_list:
            value += float(rollout_loss(segment, start, length, train=False))
        return value / len(segments_list)

    stages = [int(value) for value in arguments.stages.split(",")]
    for stage_index, length in enumerate(stages):
        model.train()
        for epoch in range(arguments.epochs):
            order = np.random.permutation(training_segments)
            total = 0.0
            count = 0
            for begin in range(0, len(order), arguments.batch):
                batch = order[begin:begin + arguments.batch]
                optimizer.zero_grad()
                loss = 0.0
                for segment in batch:
                    start = int(np.random.randint(0, q.shape[1] - length))
                    loss = loss + rollout_loss(segment, start, length, train=True)
                loss = loss / len(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach())
                count += 1
            if True:
                model.eval()
                with torch.no_grad():
                    validation = rollout_loss_batch(validation_segments[:4], 0, length)
                model.train()
                print("阶段 %d 长度 %d epoch %d train %.6e val %.6e" % (
                    stage_index + 1, length, epoch + 1, total / count, validation))
        model.eval()
        with torch.no_grad():
            validation = rollout_loss_batch(validation_segments, 0, length)
        print("阶段 %d 长度 %d 验证损失 %.6e" % (stage_index + 1, length, validation))

    torch.save({
        "state_dict": model.state_dict(),
        "kind": arguments.model,
        "mean_x": mean_x, "std_x": std_x,
        "q_mean": q_mean, "q_std": q_std,
        "qdot_mean": qdot_mean, "qdot_std": qdot_std,
        "qdd_scale": qdd_scale,
    }, arguments.save)
    print("模型已保存到", arguments.save)


if __name__ == "__main__":
    main()
