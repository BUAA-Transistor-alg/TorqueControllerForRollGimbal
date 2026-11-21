"""不含真值的留出数据验证：用含噪实测数据评估模型的开环预测能力。

对每个验证窗口：
  - 初始状态取重平滑状态（仅用含噪 q 计算）；
  - 用记录力矩推进 horizon 步；
  - 先用 8 维初始状态残差做最小二乘消去平滑误差（避免初始速度噪声主导评分）；
  - 最后统计位置残差均方值。
真实的测量噪声水平为位置 0.001 rad，因此均方残差的地板约为 1e-6。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.signal import savgol_filter

import gimbal_dynamics as gd
import identify_physical as ip
import trajectory_fit as tf


def validation_score(raw: np.ndarray, data, segment_range, horizon: int = 100,
                     per_segment: int = 4, seed: int = 20261207, inner: int = 3):
    model = ip.build_model(raw)
    first, last = segment_range
    rng = np.random.default_rng(seed)
    context = gd.Context(1)
    errors = []
    try:
        for segment in range(first, last):
            terrain = gd.make_terrain(float(data["terrain"][segment, 0]),
                                      float(data["terrain"][segment, 1]))
            base = gd.make_base(yaw=float(data["base_yaw"][segment]))
            starts = rng.integers(0, data["q"].shape[1] - horizon, per_segment)
            for start in starts:
                window = tf.Window(int(segment), int(start), horizon, terrain, base,
                                   data["q"][segment, start:start + horizon],
                                   data["tau"][segment, start:start + horizon])
                smoothed_q = savgol_filter(data["q"][segment], 51, 5, axis=0)
                smoothed_v = savgol_filter(data["q"][segment], 51, 5, deriv=1, delta=0.01,
                                           axis=0)
                initial = np.concatenate([smoothed_q[start], smoothed_v[start]])
                initial, cost = tf.solve_initial_state(context, model, window, initial, inner)
                errors.append(cost / (horizon * 4))
    finally:
        context.close()
    return float(np.mean(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description="留出数据开环验证")
    parser.add_argument("--data", type=Path, default=Path("data/rich2_data_noisy.npz"))
    parser.add_argument("--models", type=Path, nargs="+",
                        default=[Path("data/identified_regressor.npz")])
    parser.add_argument("--start", type=int, default=800)
    parser.add_argument("--end", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--per-segment", type=int, default=4)
    arguments = parser.parse_args()
    data = np.load(arguments.data)
    print("数据 %s，验证段 %d 到 %d，窗口长度 %d" % (
        arguments.data.name, arguments.start, arguments.end, arguments.horizon))
    for path in arguments.models:
        raw = np.load(path)["model"]
        score = validation_score(raw, data, (arguments.start, arguments.end),
                                 arguments.horizon, arguments.per_segment)
        print("  %-40s 均方残差 %.4e（RMS %.4e rad）" % (path.name, score, np.sqrt(score)))


if __name__ == "__main__":
    main()
