"""批量评估候选模型：按 50 步开环 RMSE 与方向不一致次数排序。

用法：python3 leaderboard.py data/xxx.npz data/yyy.npz ...
"""

from __future__ import annotations

import sys
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import eval_openloop as open_loop


def main() -> None:
    paths = [Path(value) for value in sys.argv[1:]]
    if not paths:
        paths = sorted(Path("data").glob("*.npz"))
    rows = []
    for path in paths:
        try:
            payload = np.load(path)
            if "model" not in payload.files:
                continue
            model = np.asarray(payload["model"], dtype=float)
            if model.shape != (56,):
                continue
            metrics = open_loop.evaluate(model)
        except Exception as error:  # 非法模型直接跳过并打印原因
            print("%-40s 跳过：%s" % (path.name, error))
            continue
        rows.append((float(np.max(metrics["rmse"][50])), metrics["direction_wrong"],
                     float(np.max(metrics["maximum"][50])), path, metrics))
    rows.sort(key=lambda row: (row[0], row[1]))
    print("%-40s %12s %12s %6s   %s" % ("模型", "maxRMSE50", "maxMAX50", "方向错", "RMSE50 四轴"))
    for maximum, wrong, worst, path, metrics in rows:
        print("%-40s %12.4e %12.4e %6d   %s" % (
            path.name, maximum, worst, wrong,
            " ".join("%.2e" % value for value in metrics["rmse"][50])))
    if rows:
        print("\n最优模型：%s" % rows[0][3])
        print(open_loop.format_report(rows[0][4]))


if __name__ == "__main__":
    main()
