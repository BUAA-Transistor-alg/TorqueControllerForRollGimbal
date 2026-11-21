"""检查某个模型的方向一致率细节：按 horizon 与关节。"""
import sys
import numpy as np
import eval_openloop as ol

for path in sys.argv[1:]:
    metrics = ol.evaluate(np.load(path)["model"])
    print("=== %s" % path)
    for horizon in ol.DIRECTION_HORIZONS:
        fraction, samples = metrics["direction"][horizon]
        print("  %2d 步  %s  （样本数 %s）" % (
            horizon,
            " ".join("%s %.3f" % (ol.JOINT_NAMES[j], fraction[j]) for j in range(4)),
            " ".join("%d" % samples[j] for j in range(4))))
    print("  50 步 RMSE %s  方向不一致 %d" % (
        " ".join("%.2e" % v for v in metrics["rmse"][50]), metrics["direction_wrong"]))
