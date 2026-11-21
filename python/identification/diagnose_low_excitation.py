"""诊断低激励模型与标准模型的参数误差构成。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import param_basis as pb
import params

truth = np.concatenate([params.MASSES, np.array(params.COM).reshape(-1),
                        np.array(params.INERTIA).reshape(-1),
                        np.array(params.JOINT_OFFSETS).reshape(-1), params.DAMPING])
truth_values = pb.inertial_from_raw(truth)
basis, singular, rank = pb.compute_basis()
labels = []
for body in range(4):
    labels += ["B%d.m" % (body + 1), "B%d.hx" % (body + 1), "B%d.hy" % (body + 1),
               "B%d.hz" % (body + 1), "B%d.Ixx" % (body + 1), "B%d.Iyy" % (body + 1),
               "B%d.Izz" % (body + 1), "B%d.Ixy" % (body + 1), "B%d.Ixz" % (body + 1),
               "B%d.Iyz" % (body + 1)]
labels += ["d1", "d2", "d3", "d4"]

for name in ("data/identified_regressor.npz", "data/low_a_w71.npz"):
    raw = np.load(name)["model"]
    values = pb.inertial_from_raw(raw)
    difference = basis @ (basis.T @ (values - truth_values))
    print("=== %s：可辨识空间内的参数误差 ===" % name)
    print("  误差范数 %.4e（真值向量范数 %.4e）" % (
        np.linalg.norm(difference), np.linalg.norm(truth_values)))
    order = np.argsort(np.abs(difference))[::-1][:10]
    for index in order:
        print("    %-8s 真值 %+.5e  误差 %+.3e" % (labels[index], truth_values[index],
                                                  difference[index]))
    print()
