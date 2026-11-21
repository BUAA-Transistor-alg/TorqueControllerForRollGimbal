"""大网络 Neural ODE 动力学建模，多步开环训练。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn


class OdeFunction(nn.Module):
    def __init__(self, features: int, width: int, depth: int):
        super().__init__()
        layers = [nn.Linear(features, width), nn.GELU(), nn.LayerNorm(width)]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.GELU(), nn.LayerNorm(width)]
        layers += [nn.Linear(width, 4)]
        self.network = nn.Sequential(*layers)

    def forward(self, state, torque, terrain):
        return self.network(torch.cat([state, torque, terrain], dim=-1))


def integrate_step(function, state, torque, terrain, dt, scale, method="heun"):
    def derivative(value):
        q = value[:, :4]
        v = value[:, 4:]
        acceleration = function(value, torque, terrain) * scale
        return torch.cat([v, acceleration], dim=1)

    if method == "euler":
        return state + dt * derivative(state)
    k1 = derivative(state)
    if method == "heun":
        k2 = derivative(state + dt * k1)
        return state + 0.5 * dt * (k1 + k2)
    k2 = derivative(state + 0.5 * dt * k1)
    k3 = derivative(state + 0.5 * dt * k2)
    k4 = derivative(state + dt * k3)
    return state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def main() -> None:
    parser = argparse.ArgumentParser(description="大网络 Neural ODE 训练")
    parser.add_argument("--data", type=Path, default=Path("data/closed_loop_data_noisy.npz"))
    parser.add_argument("--save", type=Path, default=Path("data/nn_ode.pt"))
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--stages", type=str, default="10,25,50,100")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20261126)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--integrator", choices=("euler", "heun", "rk4"), default="heun")
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
    q_std = q[training_segments].std(axis=(0, 1)) + 1e-8
    qdot_std = qdot[training_segments].std(axis=(0, 1)) + 1e-8
    qdd_scale = 20.0
    terrain_tensor = torch.tensor(
        np.concatenate([terrain, np.sin(base_yaw)[:, None], np.cos(base_yaw)[:, None]], axis=1),
        dtype=torch.float32)
    mean_tensor = torch.tensor(mean_x, dtype=torch.float32)
    std_tensor = torch.tensor(std_x, dtype=torch.float32)
    q_std_tensor = torch.tensor(q_std, dtype=torch.float32)
    v_std_tensor = torch.tensor(qdot_std, dtype=torch.float32)

    function = OdeFunction(16, arguments.width, arguments.depth)
    model = function
    if arguments.resume and arguments.save.exists():
        checkpoint = torch.load(arguments.save, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate, weight_decay=1e-4)

    def rollout_loss(segment, start, length):
        state = torch.tensor(np.concatenate([q[segment, start], qdot[segment, start]]),
                             dtype=torch.float32).unsqueeze(0)
        losses = []
        for k in range(length):
            torque = torch.tensor(tau[segment, start + k], dtype=torch.float32).unsqueeze(0)
            terrain_features = terrain_tensor[segment].unsqueeze(0)
            state = integrate_step(function, state, torque, terrain_features, dt,
                                   qdd_scale, arguments.integrator)
            target_q = torch.tensor(q[segment, start + k + 1], dtype=torch.float32)
            target_v = torch.tensor(qdot[segment, start + k + 1], dtype=torch.float32)
            losses.append(torch.mean(((state[0, :4] - target_q) / q_std_tensor) ** 2))
            losses.append(0.1 * torch.mean(((state[0, 4:] - target_v) / v_std_tensor) ** 2))
        return sum(losses) / len(losses)

    def validation(length):
        model.eval()
        total = 0.0
        with torch.no_grad():
            for segment in validation_segments:
                total += float(rollout_loss(segment, 0, length))
        model.train()
        return total / len(validation_segments)

    stages = [int(value) for value in arguments.stages.split(",")]
    for stage_index, length in enumerate(stages):
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
                    loss = loss + rollout_loss(segment, start, length)
                loss = loss / len(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach())
                count += 1
            if (epoch + 1) % 5 == 0:
                print("阶段 %d 长度 %d epoch %d train %.6e val %.6e" % (
                    stage_index + 1, length, epoch + 1, total / count, validation(length)),
                      flush=True)
        print("阶段 %d 长度 %d 验证损失 %.6e" % (stage_index + 1, length, validation(length)),
              flush=True)
        torch.save({
            "state_dict": model.state_dict(),
            "width": arguments.width,
            "depth": arguments.depth,
            "mean_x": mean_x, "std_x": std_x,
            "q_std": q_std, "qdot_std": qdot_std,
            "qdd_scale": qdd_scale,
            "integrator": arguments.integrator,
        }, arguments.save)
    torch.save({
        "state_dict": model.state_dict(),
        "width": arguments.width,
        "depth": arguments.depth,
        "mean_x": mean_x, "std_x": std_x,
        "q_std": q_std, "qdot_std": qdot_std,
        "qdd_scale": qdd_scale,
        "integrator": arguments.integrator,
    }, arguments.save)
    print("模型已保存到", arguments.save)


if __name__ == "__main__":
    main()
