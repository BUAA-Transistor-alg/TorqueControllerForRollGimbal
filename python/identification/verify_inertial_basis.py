"""验证 tcrg_inertial_basis：44 列基（40 惯性 + 4 阻尼）应与逆动力学对空间惯量的导数一致。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import gimbal_dynamics as gd
import identify_physical as ip

# 一个物理上合法的基准模型（质量、质心、惯量都取安全值）
base_raw = np.zeros(56)
base_raw[0:4] = [2.0, 1.0, 1.0, 1.0]
for body in range(4):
    base_raw[4 + 3 * body:7 + 3 * body] = [0.01, 0.05, 0.02]
    base_raw[16 + 6 * body:22 + 6 * body] = [0.02, 0.02, 0.02, 0.0, 0.0, 0.0]
base_raw[40:52] = np.array([[0.0, 0.0, 0.0], [0.0, 0.09, 0.039],
                            [0.0, 0.0, 0.0521], [0.0, 0.0, 0.1579]]).reshape(-1)
base_raw[52:56] = 0.1


def inertial_from_raw(x):
    values = np.zeros(44)
    for body in range(4):
        m = x[body]
        c = x[4 + 3 * body:7 + 3 * body]
        inertia = np.array([[x[16 + 6 * body], x[16 + 6 * body + 3], x[16 + 6 * body + 4]],
                            [x[16 + 6 * body + 3], x[17 + 6 * body], x[16 + 6 * body + 5]],
                            [x[16 + 6 * body + 4], x[16 + 6 * body + 5], x[18 + 6 * body]]])
        spatial = inertia + m * ((c @ c) * np.eye(3) - np.outer(c, c))
        values[10 * body] = m
        values[10 * body + 1:10 * body + 4] = m * c
        values[10 * body + 4:10 * body + 7] = [spatial[0, 0], spatial[1, 1], spatial[2, 2]]
        values[10 * body + 7:10 * body + 10] = [spatial[0, 1], spatial[0, 2], spatial[1, 2]]
    values[40:44] = x[52:56]
    return values


def raw_from_inertial(values, offsets):
    x = np.zeros(56)
    for body in range(4):
        m = values[10 * body]
        h = values[10 * body + 1:10 * body + 4]
        c = h / m
        spatial = np.array([[values[10 * body + 4], values[10 * body + 7], values[10 * body + 8]],
                            [values[10 * body + 7], values[10 * body + 5], values[10 * body + 9]],
                            [values[10 * body + 8], values[10 * body + 9], values[10 * body + 6]]])
        inertia = spatial - m * ((c @ c) * np.eye(3) - np.outer(c, c))
        x[body] = m
        x[4 + 3 * body:7 + 3 * body] = c
        x[16 + 6 * body:22 + 6 * body] = [inertia[0, 0], inertia[1, 1], inertia[2, 2],
                                          inertia[0, 1], inertia[0, 2], inertia[1, 2]]
    x[40:52] = offsets
    x[52:56] = values[40:44]
    return x


context = gd.Context(1)
rng = np.random.default_rng(11)
reference = inertial_from_raw(base_raw)
offsets = base_raw[40:52]
step = 1e-5
for trial in range(3):
    terrain = gd.make_terrain(float(rng.uniform(0, 0.3)), float(rng.uniform(0, 2 * np.pi)))
    base = gd.make_base(yaw=float(rng.uniform(-np.pi, np.pi)))
    q = np.array([rng.uniform(-2, 2), rng.uniform(-0.7, 0.05), rng.uniform(-0.8, 0.8),
                  rng.uniform(-0.7, 0.7)])
    v = rng.uniform(-1, 1, 4)
    qdd = rng.uniform(-4, 4, 4)
    state = gd.make_state(q, v)
    basis = np.array(context.inertial_basis(ip.build_model(base_raw), terrain, base, state, qdd))
    errors = np.zeros(44)
    for column in range(44):
        plus = reference.copy()
        minus = reference.copy()
        plus[column] += step
        minus[column] -= step
        plus_torque = np.array(context.inverse_dynamics(
            ip.build_model(raw_from_inertial(plus, offsets)), terrain, base, state, qdd))
        minus_torque = np.array(context.inverse_dynamics(
            ip.build_model(raw_from_inertial(minus, offsets)), terrain, base, state, qdd))
        derivative = (plus_torque - minus_torque) / (2 * step)
        errors[column] = np.max(np.abs(derivative - basis[column]))
    scale = np.max(np.abs(basis)) + 1e-12
    print("第 %d 组：基与差分最大绝对误差 %.3e，相对 %.3e" % (
        trial + 1, np.max(errors), np.max(errors) / scale))
    worst = np.argsort(errors)[::-1][:4]
    print("   最差列", worst, "误差", np.array2string(errors[worst], precision=3))
context.close()
