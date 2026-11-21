"""经验可辨识惯性基。

tau 对 40 个惯性原始参数（每个刚体：质量、一阶矩、绕本体系原点的空间惯量）与 4 个粘性阻尼
严格线性，且对应的灵敏度矩阵只依赖几何、底盘姿态、地形与状态采样，不依赖惯性参数本身。

本文件用 tcrg_inertial_basis 在大量随机状态上求灵敏度矩阵，做 SVD 得到数值行空间，
作为可辨识子空间的正交基。工程实现上这是唯一不依赖任何结构秩推导假设的做法。

用法：
    import param_basis as pb
    basis = pb.compute_basis()                 # 44 维参数空间中的 28 维可辨识基
    z = pb.to_identifiable(raw_model, basis)
    raw = pb.from_identifiable(z, basis, prior=pb.prior_model())
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import gimbal_dynamics as gd
import identify_physical as ip

# 标称几何：o1 取在关节 1 轴线上，o2.z 为关节 1、2 轴交点约定，其余为结构设计值
NOMINAL_OFFSETS = np.array([[0.0, 0.0, 0.0],
                            [0.0, 0.0900, 0.0390],
                            [0.0, 0.0, 0.0521],
                            [0.0, 0.0, 0.1579]])
NOMINAL_TOTAL_MASS = 6.0
INERTIAL_DIMENSION = 44


def nominal_model() -> np.ndarray:
    """不含真实参数的标称模型：总质量 6 kg，质心在偏置方向附近，惯量取 0.01。"""
    x = np.zeros(56)
    x[0:4] = NOMINAL_TOTAL_MASS / 4.0
    for body in range(4):
        x[4 + 3 * body:7 + 3 * body] = [0.0, 0.02, 0.0]
        x[16 + 6 * body:22 + 6 * body] = [0.01, 0.01, 0.01, 0.0, 0.0, 0.0]
    x[40:52] = NOMINAL_OFFSETS.reshape(-1)
    x[52:56] = 0.1
    return x


def prior_model(path: Path = Path("data/identified_analytic.npz")) -> np.ndarray:
    """先验模型：优先用已有辨识结果（物理合法、不含真实参数），否则用标称模型。"""
    if path.exists():
        x = np.load(path)["model"].astype(float)
        if np.all(x[0:4] > 0.0):
            return x
    return nominal_model()


def sample_states(count: int = 120, seed: int = 20261201):
    rng = np.random.default_rng(seed)
    states = []
    for _ in range(count):
        q = np.array([rng.uniform(-3.0, 3.0), rng.uniform(-0.75, 0.05),
                      rng.uniform(-0.85, 0.85), rng.uniform(-0.75, 0.75)])
        velocity = rng.uniform(-1.5, 1.5, 4)
        acceleration = rng.uniform(-5.0, 5.0, 4)
        slope = rng.uniform(0.0, 0.349)
        direction = rng.uniform(0.0, 2.0 * np.pi)
        yaw = rng.uniform(-np.pi, np.pi)
        states.append((q, velocity, acceleration, slope, direction, yaw))
    return states


def sensitivity(offsets: np.ndarray | None = None, count: int = 120,
                seed: int = 20261201, model: np.ndarray | None = None):
    """返回 44 维参数（40 惯性 + 4 阻尼）对关节力矩的灵敏度矩阵，形状 (4*count, 44)。"""
    offsets = NOMINAL_OFFSETS if offsets is None else np.asarray(offsets, dtype=float)
    reference = nominal_model() if model is None else np.asarray(model, dtype=float)
    reference = reference.copy()
    reference[40:52] = offsets.reshape(-1)
    reference_model = ip.build_model(reference)
    context = gd.Context(1)
    rows = np.zeros((4 * count, INERTIAL_DIMENSION))
    try:
        for index, (q, velocity, acceleration, slope, direction, yaw) in enumerate(
                sample_states(count, seed)):
            terrain = gd.make_terrain(float(slope), float(direction))
            base = gd.make_base(yaw=float(yaw))
            state = gd.make_state(q, velocity)
            basis = np.array(context.inertial_basis(reference_model, terrain, base, state,
                                                    acceleration))
            rows[4 * index:4 * index + 4, :] = basis.T
    finally:
        context.close()
    return rows


def compute_basis(offsets: np.ndarray | None = None, count: int = 120, seed: int = 20261201,
                  relative_threshold: float = 1.0e-8):
    """返回 (basis, singular_values, rank)。basis 形状 (44, rank)，列正交，张成可辨识子空间。

    惯性部分取 40 列灵敏度矩阵的行空间，阻尼 4 个方向独立拼接。
    """
    matrix = sensitivity(offsets, count, seed)
    inertial_matrix = matrix[:, :40]
    _, singular, right = np.linalg.svd(inertial_matrix, full_matrices=False)
    rank = int(np.sum(singular > singular[0] * relative_threshold))
    basis = np.zeros((INERTIAL_DIMENSION, rank + 4))
    basis[:40, :rank] = right[:rank, :].T
    basis[40:, rank:] = np.eye(4)
    return basis, singular, rank + 4


def inertial_from_raw(x: np.ndarray) -> np.ndarray:
    values = np.zeros(INERTIAL_DIMENSION)
    for body in range(4):
        mass = x[body]
        com = x[4 + 3 * body:7 + 3 * body]
        inertia = np.array([[x[16 + 6 * body], x[16 + 6 * body + 3], x[16 + 6 * body + 4]],
                            [x[16 + 6 * body + 3], x[17 + 6 * body], x[16 + 6 * body + 5]],
                            [x[16 + 6 * body + 4], x[16 + 6 * body + 5], x[18 + 6 * body]]])
        spatial = inertia + mass * ((com @ com) * np.eye(3) - np.outer(com, com))
        values[10 * body] = mass
        values[10 * body + 1:10 * body + 4] = mass * com
        values[10 * body + 4:10 * body + 7] = [spatial[0, 0], spatial[1, 1], spatial[2, 2]]
        values[10 * body + 7:10 * body + 10] = [spatial[0, 1], spatial[0, 2], spatial[1, 2]]
    values[40:44] = x[52:56]
    return values


def raw_from_inertial(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """由 44 维惯性参数与偏置构造 56 维原始参数。质量必须为正。"""
    x = np.zeros(56)
    for body in range(4):
        mass = values[10 * body]
        if not mass > 0.0:
            raise ValueError("惯性参数中的质量必须为正")
        h = values[10 * body + 1:10 * body + 4]
        com = h / mass
        spatial = np.array([[values[10 * body + 4], values[10 * body + 7],
                             values[10 * body + 8]],
                            [values[10 * body + 7], values[10 * body + 5],
                             values[10 * body + 9]],
                            [values[10 * body + 8], values[10 * body + 9],
                             values[10 * body + 6]]])
        inertia = spatial - mass * ((com @ com) * np.eye(3) - np.outer(com, com))
        x[body] = mass
        x[4 + 3 * body:7 + 3 * body] = com
        x[16 + 6 * body:22 + 6 * body] = [inertia[0, 0], inertia[1, 1], inertia[2, 2],
                                          inertia[0, 1], inertia[0, 2], inertia[1, 2]]
    x[40:52] = np.asarray(offsets, dtype=float).reshape(-1)
    x[52:56] = values[40:44]
    return x


def to_identifiable(raw: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return basis.T @ inertial_from_raw(raw)


def from_identifiable(z: np.ndarray, basis: np.ndarray, prior: np.ndarray,
                      offsets: np.ndarray | None = None) -> np.ndarray:
    """在保持先验不可辨识分量的前提下构造 56 维模型。"""
    prior_values = inertial_from_raw(prior)
    values = prior_values + basis @ (z - basis.T @ prior_values)
    effective_offsets = prior[40:52] if offsets is None else offsets
    return raw_from_inertial(values, effective_offsets)


def _pseudo_inertia(values: np.ndarray, body: int) -> np.ndarray:
    """由 (m, h, I_o) 构造 4x4 伪惯量矩阵，线性关系。"""
    mass = values[10 * body]
    h = values[10 * body + 1:10 * body + 4]
    spatial = np.array([[values[10 * body + 4], values[10 * body + 7], values[10 * body + 8]],
                        [values[10 * body + 7], values[10 * body + 5], values[10 * body + 9]],
                        [values[10 * body + 8], values[10 * body + 9], values[10 * body + 6]]])
    sigma = 0.5 * np.trace(spatial) * np.eye(3) - spatial
    matrix = np.zeros((4, 4))
    matrix[:3, :3] = sigma
    matrix[:3, 3] = h
    matrix[3, :3] = h
    matrix[3, 3] = mass
    return matrix


def _values_from_pseudo(matrix: np.ndarray, body: int, values: np.ndarray) -> None:
    mass = max(matrix[3, 3], 1.0e-3)
    h = matrix[:3, 3].copy()
    sigma = matrix[:3, :3]
    spatial = np.trace(sigma) * np.eye(3) - sigma
    values[10 * body] = mass
    values[10 * body + 1:10 * body + 4] = h
    values[10 * body + 4:10 * body + 7] = [spatial[0, 0], spatial[1, 1], spatial[2, 2]]
    values[10 * body + 7:10 * body + 10] = [spatial[0, 1], spatial[0, 2], spatial[1, 2]]


def project_physical_cone(values: np.ndarray) -> np.ndarray:
    """把 44 维参数投影到物理可行锥：每个刚体的伪惯量矩阵投影到半正定。"""
    result = values.copy()
    for body in range(4):
        matrix = _pseudo_inertia(values, body)
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        projected = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        _values_from_pseudo(projected, body, result)
    return result


def physical_projection(z_target: np.ndarray, basis: np.ndarray, prior: np.ndarray,
                        offsets: np.ndarray | None = None, iterations: int = 2000,
                        tolerance: float = 1.0e-12):
    """交替投影：保持可辨识坐标等于 z_target，并把参数投影到物理可行锥。

    可辨识约束是线性子空间，物理可行性（伪惯量半正定）在 (m, h, I_o) 参数化下是凸锥，
    两个集合都凸，交替投影收敛到交集内一点。返回 (raw, 可辨识坐标偏差, 最小伪惯量特征值)。
    """
    offsets = prior[40:52] if offsets is None else np.asarray(offsets, dtype=float)
    values = inertial_from_raw(prior)
    for _ in range(iterations):
        values = values + basis @ (z_target - basis.T @ values)
        projected = project_physical_cone(values)
        move = np.linalg.norm(projected - values)
        values = projected
        if move < tolerance:
            break
    values = values + basis @ (z_target - basis.T @ values)
    deviation = np.linalg.norm(basis.T @ values - z_target)
    eigenvalues = min(np.linalg.eigvalsh(_pseudo_inertia(values, body))[0] for body in range(4))
    return raw_from_inertial(values, offsets), deviation, float(eigenvalues)


def old_documented_directions() -> np.ndarray:
    """文档旧基的 21 个惯性方向在 44 维参数空间中的坐标（每行一个方向）。"""
    directions = np.zeros((40, 21))
    # B1：I_o1_zz、h1x、h1y
    directions[10 * 0 + 4 + 2, 0] = 1.0
    directions[1, 1] = 1.0
    directions[2, 2] = 1.0
    slots = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
    for body in range(1, 4):
        column = 3 + 6 * (body - 1)
        directions[10 * body, column] = 1.0
        directions[10 * body + 1, column + 1] = 1.0
        directions[10 * body + 2, column + 2] = 1.0
        directions[10 * body + 3, column + 3] = 1.0
        axis = ip.AXES[body]
        p_axis = np.array(ip.PLANES[body][0])
        q_axis = np.array(ip.PLANES[body][1])
        projection = np.outer(axis, axis)
        difference = np.outer(p_axis, p_axis) - np.outer(q_axis, q_axis)
        for slot, (row, col) in enumerate(slots):
            factor = 1.0 if row == col else 2.0
            directions[10 * body + 4 + slot, column + 4] = factor * projection[row, col]
            directions[10 * body + 4 + slot, column + 5] = factor * difference[row, col]
    return directions


def check_inertial_rank() -> None:
    """检查可辨识维度，并量化文档旧基的不可约模型误差。"""
    for name, offsets in (("标称几何", NOMINAL_OFFSETS), ("零偏置", np.zeros((4, 3)))):
        basis, singular, rank = compute_basis(offsets, count=80)
        print("%s：可辨识维数 %d，惯性部分最小奇异值 %.3e，最大 %.3e，0 空间最大 %.3e" % (
            name, rank, singular[24 - 1], singular[0], singular[24]))
    basis, singular, rank = compute_basis(count=200)
    print("200 状态采样：秩 %d" % rank)

    truth = np.concatenate([__import__("params").MASSES,
                            np.array(__import__("params").COM).reshape(-1),
                            np.array(__import__("params").INERTIA).reshape(-1),
                            np.array(__import__("params").JOINT_OFFSETS).reshape(-1),
                            __import__("params").DAMPING])
    matrix = sensitivity(count=80)
    truth_values = inertial_from_raw(truth)
    documented = np.zeros((INERTIAL_DIMENSION, 21))
    documented[:40, :] = old_documented_directions()
    columns = matrix @ documented
    coefficients, _, _, _ = np.linalg.lstsq(columns, matrix @ truth_values, rcond=None)
    residual = matrix @ truth_values - columns @ coefficients
    print("\n文档旧 21 维惯性基对真值的不可约力矩残差：")
    print("  逐行 RMS %.4e N·m，最大 %.4e N·m（力矩噪声标准差 0.01）" % (
        np.sqrt(np.mean(residual ** 2)), np.max(np.abs(residual))))
    print("  每关节 RMS %s" % np.array2string(
        np.sqrt(np.mean(residual.reshape(-1, 4) ** 2, axis=0)), precision=4))
    full = matrix @ basis
    coefficients, _, _, _ = np.linalg.lstsq(full, matrix @ truth_values, rcond=None)
    residual_full = matrix @ truth_values - full @ coefficients
    print("24 维经验基残差：逐行 RMS %.4e N·m" % np.sqrt(np.mean(residual_full ** 2)))

    reconstructed = from_identifiable(basis.T @ truth_values, basis, prior_model(),
                                      offsets=truth[40:52])
    print("\n由真值惯性参数重建 56 维模型后与真值的原始参数差：")
    print("  质量差 %s" % np.array2string(reconstructed[0:4] - truth[0:4], precision=5))
    print("  质心差最大 %.3e" % np.max(np.abs(reconstructed[4:16] - truth[4:16])))
    print("  惯量差最大 %.3e" % np.max(np.abs(reconstructed[16:40] - truth[16:40])))


if __name__ == "__main__":
    check_inertial_rank()