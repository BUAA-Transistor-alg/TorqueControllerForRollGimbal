"""放宽物理可行域后的重拟合：约束改为 C 库仿真所需的最小条件（质量为正、质心惯量正定）。

原实现约束的是绕坐标系原点的空间惯量与伪惯量半正定，比仿真所需更严；
真实连杆本身的惯量很小，严格约束会把解压到可行域边界，引入系统偏差。
"""
from __future__ import annotations

import sys

import numpy as np

import regressor_full as rf


def relaxed_feasibility(p: np.ndarray) -> np.ndarray:
    values = [np.array(p[40:44])]
    for body in range(4):
        mass = p[body * 10]
        values.append(np.array([mass - 0.02]))
        h = p[body * 10 + 1:body * 10 + 4]
        spatial = np.array([[p[body * 10 + 4], p[body * 10 + 7], p[body * 10 + 8]],
                            [p[body * 10 + 7], p[body * 10 + 5], p[body * 10 + 9]],
                            [p[body * 10 + 8], p[body * 10 + 9], p[body * 10 + 6]]])
        com = h / max(mass, 1.0e-6)
        centroidal = spatial - mass * ((com @ com) * np.eye(3) - np.outer(com, com))
        values.append(np.linalg.eigvalsh(centroidal) - 5.0e-5)
    return np.concatenate(values)


if __name__ == "__main__":
    rf.feasibility = relaxed_feasibility
    sys.argv = ["refit_relaxed.py"] + sys.argv[1:]
    rf.main()