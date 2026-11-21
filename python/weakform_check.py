"""弱形式辨识的代数性质验证脚本（weakform 前缀，仅供自检）。

验证三件事：
1. tcrg_inertial_basis 的 44 列在固定偏置下与逆动力学完全线性：
   tau = sum_s basis_s * p_s，p_s 为各刚体关于关节原点的空间惯量参数与阻尼。
2. 质量矩阵列提取：M[:, k] = basis(q, 0, e_k) - basis(q, 0, 0)。
3. 21 个惯性基参数 beta 是 40 个空间惯量参数的常数线性组合，
   对应的力矩基列可以由同样的组合系数从 basis 列线性重组得到。
"""

from __future__ import annotations

import numpy as np

import gimbal_dynamics as gd
import identify_physical as ip
import params


def spatial_params(x: np.ndarray) -> np.ndarray:
    """56 维原始参数 -> 44 维线性空间参数（每刚体 m,hx,hy,hz,I_o 六分量 + 4 阻尼）。"""
    p = np.zeros(44)
    for body in range(4):
        m = x[body]
        c = x[4 + 3 * body:7 + 3 * body]
        inertia = np.array([[x[16 + 6 * body], x[16 + 6 * body + 3], x[16 + 6 * body + 4]],
                            [x[16 + 6 * body + 3], x[17 + 6 * body], x[16 + 6 * body + 5]],
                            [x[16 + 6 * body + 4], x[16 + 6 * body + 5], x[18 + 6 * body]]])
        spatial = inertia + m * ((c @ c) * np.eye(3) - np.outer(c, c))
        base = body * 10
        p[base] = m
        p[base + 1:base + 4] = m * c
        p[base + 4:base + 10] = [spatial[0, 0], spatial[1, 1], spatial[2, 2],
                                 spatial[0, 1], spatial[0, 2], spatial[1, 2]]
    p[40:44] = x[52:56]
    return p


def beta_combination() -> np.ndarray:
    """25 维 (21 惯性 beta + 4 阻尼) 到 44 维空间参数的常数线性组合矩阵 C。"""
    c = np.zeros((25, 44))
    # B1：beta1 = I_o1,zz；beta2 = h1x；beta3 = h1y
    c[0, 0 * 10 + 4 + 2] = 1.0
    c[1, 0 * 10 + 1] = 1.0
    c[2, 0 * 10 + 2] = 1.0
    position = 3
    for body in (1, 2, 3):
        a = ip.AXES[body]
        p_axis = np.array(ip.PLANES[body][0])
        q_axis = np.array(ip.PLANES[body][1])
        base = body * 10
        c[position, base] = 1.0
        for axis in range(3):
            c[position + 1 + axis, base + 1 + axis] = 1.0
        # a^T I_o a：惯量槽位顺序 xx,yy,zz,xy,xz,yz
        slots = {(0, 0): 4, (1, 1): 5, (2, 2): 6, (0, 1): 7, (0, 2): 8, (1, 2): 9}
        for r in range(3):
            for s in range(3):
                c[position + 4, base + slots[tuple(sorted((r, s)))]] += a[r] * a[s]
        for r in range(3):
            for s in range(3):
                c[position + 5, base + slots[tuple(sorted((r, s)))]] += (
                    p_axis[r] * p_axis[s] - q_axis[r] * q_axis[s])
        position += 6
    for joint in range(4):
        c[21 + joint, 40 + joint] = 1.0
    return c


def main() -> None:
    context = gd.Context(1)
    try:
        rng = np.random.default_rng(0)
        x = np.concatenate([np.array(params.MASSES), np.array(params.COM).reshape(-1),
                            np.array(params.INERTIA).reshape(-1),
                            np.array(params.JOINT_OFFSETS).reshape(-1),
                            np.array(params.DAMPING)])
        model = gd.make_model()
        terrain = gd.make_terrain(0.25, 1.2)
        base = gd.make_base(yaw=0.6)
        q = rng.uniform(-0.5, 0.5, 4)
        qdot = rng.uniform(-1.0, 1.0, 4)
        qdd = rng.uniform(-1.0, 1.0, 4)
        state = gd.make_state(q, qdot)

        basis = np.array(context.inertial_basis(model, terrain, base, state, qdd))
        predicted = basis.T @ spatial_params(x)
        direct = np.array(context.inverse_dynamics(model, terrain, base, state, qdd))
        print("验证 1 线性基重组与逆动力学误差：", np.max(np.abs(predicted - direct)))

        columns = []
        for k in range(4):
            acceleration = np.zeros(4)
            acceleration[k] = 1.0
            with_k = np.array(context.inertial_basis(model, terrain, base, state, acceleration))
            columns.append(with_k - np.array(
                context.inertial_basis(model, terrain, base, state, np.zeros(4))))
        mass_columns = np.stack(columns, axis=2)  # (44, 4, 4)
        # 与有限差分质量矩阵对比（用真实模型的逆动力学）
        zero = np.zeros(4)
        matrix_fd = np.zeros((4, 4))
        for k in range(4):
            e_k = np.zeros(4)
            e_k[k] = 1.0
            matrix_fd[:, k] = (np.array(context.inverse_dynamics(
                model, terrain, base, gd.make_state(q, np.zeros(4)), e_k))
                - np.array(context.inverse_dynamics(
                    model, terrain, base, gd.make_state(q, np.zeros(4)), zero)))
        matrix_basis = mass_columns.T @ spatial_params(x)
        print("验证 2 质量矩阵列提取误差：", np.max(np.abs(matrix_basis - matrix_fd)))
        print("    对称性误差：", np.max(np.abs(matrix_basis - matrix_basis.T)))

        # 验证 3：每个 beta 基元素的力矩 = C 行重组
        c = beta_combination()
        bias_full = np.array(context.inertial_basis(model, terrain, base, state, qdd))
        for row in range(25):
            beta = np.zeros(35)
            beta[row if row < 21 else 21 + row] = 1.0
            if row >= 21:
                beta[31 + row - 21] = 1.0
            if row < 21:
                beta[row] = 1.0
            x_beta = ip.map_beta(beta, True)
            m_beta = ip.build_model(x_beta)
            direct_tau = np.array(context.inverse_dynamics(m_beta, terrain, base, state, qdd))
            recombined = bias_full.T @ c[row]
            error = np.max(np.abs(direct_tau - recombined))
            print("验证 3 基参数 %2d 重组误差：%.3e" % (row, error))
    finally:
        context.close()


if __name__ == "__main__":
    main()
