"""导出最终辨识结果：模型文件、可读参数清单与评估指标。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import eval_openloop as open_loop
import identify_physical as ip
import param_basis as pb

JOINT_NAMES = ["大 yaw", "roll", "小 yaw", "pitch"]


def format_parameters(raw: np.ndarray) -> str:
    lines = ["四自由度云台辨识参数（56 个原始参数，SI 单位）", ""]
    lines.append("刚体惯性参数：")
    for body in range(4):
        inertia = np.array([[raw[16 + 6 * body], raw[16 + 6 * body + 3], raw[16 + 6 * body + 4]],
                            [raw[16 + 6 * body + 3], raw[17 + 6 * body], raw[16 + 6 * body + 5]],
                            [raw[16 + 6 * body + 4], raw[16 + 6 * body + 5], raw[18 + 6 * body]]])
        lines.append("  B%d 质量 %.6f kg" % (body + 1, raw[body]))
        lines.append("     质心 %s m" % np.array2string(raw[4 + 3 * body:7 + 3 * body], precision=6))
        lines.append("     绕质心惯量 xx=%.6f yy=%.6f zz=%.6f xy=%.6f xz=%.6f yz=%.6f" % tuple(
            [raw[16 + 6 * body], raw[17 + 6 * body], raw[18 + 6 * body],
             raw[16 + 6 * body + 3], raw[16 + 6 * body + 4], raw[16 + 6 * body + 5]]))
        lines.append("     惯量特征值 %s" % np.array2string(np.linalg.eigvalsh(inertia), precision=6))
    lines.append("")
    lines.append("关节安装偏置（父坐标系）：")
    for body in range(4):
        lines.append("  o%d = %s m" % (body + 1, np.array2string(raw[40 + 3 * body:43 + 3 * body],
                                                               precision=6)))
    lines.append("")
    lines.append("粘性阻尼：%s N·m·s/rad" % np.array2string(raw[52:56], precision=6))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="导出最终辨识结果")
    parser.add_argument("--identified", type=Path, default=Path("data/regressor_joint_relaxed.npz"))
    parser.add_argument("--data", type=Path, default=Path("data/closed_loop_data_noisy.npz"))
    parser.add_argument("--save", type=Path, default=Path("data/identified_final.npz"))
    parser.add_argument("--text", type=Path, default=Path("data/identified_final.txt"))
    parser.add_argument("--metrics", type=Path, default=Path("data/identified_final_metrics.json"))
    arguments = parser.parse_args()

    raw = np.asarray(np.load(arguments.identified)["model"], dtype=float)
    ip.build_model(raw)  # 立即检查物理合法性
    basis, singular, rank = pb.compute_basis()
    identifiable = basis.T @ pb.inertial_from_raw(raw)
    metrics = open_loop.evaluate(raw, arguments.data)
    np.savez_compressed(arguments.save, model=raw, identifiable=identifiable, basis=basis,
                        residual_rms=float(np.load(arguments.identified).get("loss", 0.0)))
    arguments.text.write_text(format_parameters(raw))
    payload = {
        "rmse": {str(horizon): metrics["rmse"][horizon].tolist()
                 for horizon in open_loop.HORIZONS},
        "maximum": {str(horizon): metrics["maximum"][horizon].tolist()
                    for horizon in open_loop.HORIZONS},
        "direction_wrong": metrics["direction_wrong"],
        "direction": {str(horizon): [metrics["direction"][horizon][0].tolist(),
                                     metrics["direction"][horizon][1].tolist()]
                      for horizon in open_loop.DIRECTION_HORIZONS},
        "static_direction": {str(key): list(value)
                             for key, value in metrics["static_direction"].items()},
        "initial_acceleration_error_rms": np.sqrt(
            np.mean((metrics["qdd_ident"] - metrics["qdd_truth"]) ** 2, axis=0)).tolist(),
    }
    arguments.metrics.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(format_parameters(raw))
    print()
    print(open_loop.format_report(metrics))
    print()
    print("已保存", arguments.save, arguments.text, arguments.metrics)


if __name__ == "__main__":
    main()
