"""统计各数据集的实际激励范围：角度、速度、加速度、力矩、增益与坡度。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

FILES = [("标准 100 段", "data/closed_loop_data_noisy.npz"),
         ("同协议另采 1200 段", "data/more_data_noisy.npz"),
         ("富激励 1000 段", "data/rich2_data_noisy.npz"),
         ("富激励 2000 段", "data/rich3_data_noisy.npz")]
LABELS = ["大 yaw", "roll", "小 yaw", "pitch"]

for title, path in FILES:
    data = np.load(path)
    q = data["q_clean"]
    v = data["qdot_clean"]
    a = np.gradient(v, 0.01, axis=1)
    torque = data["tau_clean"]
    print("=== %s（%d 段）===" % (title, q.shape[0]))
    print("  关节    位置范围 (rad)        位置范围 (度)        速度最大  速度 99.9%  加速度最大  加速度 99.9%")
    for joint in range(4):
        values = q[:, :, joint]
        velocity = np.abs(v[:, :, joint])
        acceleration = np.abs(a[:, :, joint])
        print("  %-7s %+.3f 到 %+.3f   %+6.1f 到 %+6.1f   %6.2f   %6.2f     %8.1f   %8.1f" % (
            LABELS[joint], values.min(), values.max(),
            np.degrees(values.min()), np.degrees(values.max()),
            velocity.max(), np.quantile(velocity, 0.999),
            acceleration.max(), np.quantile(acceleration, 0.999)))
    print("  力矩：绝对值最大 %.2f N·m，99.9%% 分位 %.2f N·m，限幅 10 N·m" % (
        np.abs(torque).max(), np.quantile(np.abs(torque), 0.999)))
    print("  力矩达到 9.99 N·m 的样本比例 %.4f%%" % (
        100.0 * np.mean(np.abs(torque) > 9.99)))
    print("  坡度 %.1f 到 %.1f 度；底盘偏航范围 ±180 度" % (
        np.degrees(data["terrain"][:, 0].min()), np.degrees(data["terrain"][:, 0].max())))
    print("  kp 范围 %s" % np.array2string(np.array([data["kp"][:, j].min() for j in range(4)]),
                                          precision=2),
          "到", np.array2string(np.array([data["kp"][:, j].max() for j in range(4)]), precision=2))
    print("  kd 范围 %s" % np.array2string(np.array([data["kd"][:, j].min() for j in range(4)]),
                                          precision=2),
          "到", np.array2string(np.array([data["kd"][:, j].max() for j in range(4)]), precision=2))
    print()
