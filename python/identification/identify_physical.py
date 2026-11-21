"""原始参数与动力学模型之间的转换，以及数据读取。"""

from __future__ import annotations

from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import gimbal_dynamics as gd
import params


def load_data(path: Path):
    """读取数据集，返回关节角、角速度、由角速度差分得到的角加速度、力矩、地形与底盘偏航。"""
    data = np.load(path)
    q = data["q"]
    qdot = data["qdot"]
    tau = data["tau"]
    dt = float(data["dt"])
    qdd = np.gradient(qdot, dt, axis=1)
    return q, qdot, qdd, tau, data["terrain"], data["base_yaw"]


def inertia_matrix(index: int) -> np.ndarray:
    """第 index 个刚体的质心惯量矩阵（由 params 的 6 分量形式还原）。"""
    q = params.INERTIA[index]
    return np.array([[q[0], q[3], q[4]], [q[3], q[1], q[5]], [q[4], q[5], q[2]]])


def spatial_inertia(index: int) -> np.ndarray:
    """第 index 个刚体绕本体系原点的空间惯量。"""
    m = params.MASSES[index]
    c = np.array(params.COM[index])
    return inertia_matrix(index) + m * ((c @ c) * np.eye(3) - np.outer(c, c))


def build_model(x: np.ndarray) -> gd.TcrgModel:
    """把 56 维原始参数转换成可仿真的模型。"""
    return gd.make_model(x[0:4], x[4:16].reshape(4, 3), x[16:40].reshape(4, 6),
                         x[40:52].reshape(4, 3), x[52:56])
