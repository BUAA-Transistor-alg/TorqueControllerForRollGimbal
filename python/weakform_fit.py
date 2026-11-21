"""积分弱形式参数辨识（weakform）。

在 44 维线性惯性参数空间（每刚体 m、h、I_o 共 10 个 + 4 个阻尼）上构造积分弱形式回归。
动力学方程在中点网格上按离散动量恒等式组装（RK4 数据中点二阶一致，无导数近似）：

    Σ_m w_m tau_m h = Σ_m w_m M_mid_m (v_{m+1} - v_m) + Σ_m w_m b_mid_m h

方程对 44 个参数严格线性。回归列由 tcrgg_inertial_basis 在中点状态上逐点构造，
8 个安装偏置在外层做非线性优化，内层在 param_basis 的经验可辨识子空间
（28 维正交基）内做加权 ridge 最小二乘，最后在零空间上求解物理可行性
（每个刚体的伪惯量矩阵半正定）得到 56 维物理合法模型。
"""

from __future__ import annotations

import argparse
import concurrent.futures
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_coeffs, savgol_filter

import gimbal_dynamics as gd
import param_basis as pb

DT = 0.01
SIGMA_TAU = 0.01
SIGMA_Q = 0.001
GRAVITY_SENSITIVITY = 3.0

# 8 个可辨识偏置在 (4,3) 偏置矩阵中的位置（o1 与 o2.z 为规范自由度，保持标称值）
OFFSET_SLOTS = ((1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2))


def offsets_to_full(offsets8: np.ndarray) -> np.ndarray:
    full = pb.NOMINAL_OFFSETS.copy()
    for (body, axis), value in zip(OFFSET_SLOTS, offsets8):
        full[body, axis] = value
    return full


def offsets_to_8(full: np.ndarray) -> np.ndarray:
    return np.array([full[body, axis] for body, axis in OFFSET_SLOTS])


def load_data(path: Path, sg_window: int, sg_order: int):
    """加载含噪数据，用 Savitzky-Golay 对位置求导得到平滑速度。"""
    data = np.load(path)
    q = data["q"]
    tau = data["tau"]
    terrain = data["terrain"]
    base_yaw = data["base_yaw"]
    qdot = savgol_filter(q, sg_window, sg_order, deriv=1, delta=DT, axis=1)
    coefficients = savgol_coeffs(sg_window, sg_order, deriv=1, delta=DT, use="conv")
    sigma_v = SIGMA_Q * float(np.sqrt(np.sum(coefficients ** 2)))
    return data, q, qdot, tau, terrain, base_yaw, sigma_v


def extract_segment(template, terrain, base, q_seg, qdot_seg):
    """在相邻采样点的中点求 44 列基。

    RK4（refinement=2）与零阶保持力矩下，步内平均加速度等于中点加速度加 O(dt^2)，
    因此动力学方程在中点状态上二阶一致。f[t,s,k,j] = M_s(q_mid_t)[j,k]，
    bias[t,s,j] = b_s(q_mid_t, v_mid_t)_j，t 取 0 到 T-2（T-1 个中点）。
    中点状态取 (q_t+q_{t+1})/2 与 (v_t+v_{t+1})/2，对测量噪声有平均作用。
    """
    context = gd.Context(1)
    try:
        points = q_seg.shape[0]
        q_mid = 0.5 * (q_seg[:-1] + q_seg[1:])
        v_mid = 0.5 * (qdot_seg[:-1] + qdot_seg[1:])
        f = np.zeros((points - 1, 44, 4, 4))
        bias = np.zeros((points - 1, 44, 4))
        zero4 = np.zeros(4)
        for t in range(points - 1):
            state = gd.make_state(q_mid[t], v_mid[t])
            state_still = gd.make_state(q_mid[t], zero4)
            basis_bias = np.array(context.inertial_basis(template, terrain, base, state, zero4))
            bias[t] = basis_bias
            zero_basis = np.array(context.inertial_basis(template, terrain, base, state_still, zero4))
            for k in range(4):
                e_k = np.zeros(4)
                e_k[k] = 1.0
                column_k = np.array(
                    context.inertial_basis(template, terrain, base, state_still, e_k))
                f[t, :, k, :] = column_k - zero_basis
        return f, bias
    finally:
        context.close()


def segment_rows(q_seg, qdot_seg, tau_seg, f, bias, length, hop, kmax, gap):
    """在一个数据段内按滑动窗口与测试函数 w=sin(kπ n/N) 组装回归行与噪声统计量。

    离散动量恒等式（中点形式，无任何导数近似）：
        Σ_m w_m M_mid_m (v_{m+gap} - v_m) = Σ_m (w_{m-1} M_mid_{m-1} - w_m M_mid_m) v_m
    gap 为动量差分跨步（奇数），加大跨步按 sqrt(gap) 降低速度噪声进入动量项的方差；
    动量在 gap 步的粗网格中点求值，偏置项与力矩项仍用细网格 Simpson 1/3 积分。
    """
    if length % 2 != 0 or length % gap != 0 or gap % 2 != 1:
        raise ValueError("窗口长度必须为偶数且被 gap 整除，gap 必须为奇数")
    points = q_seg.shape[0]
    simpson = np.ones(length + 1)
    simpson[1:-1:2] = 4.0
    simpson[2:-1:2] = 2.0
    simpson *= DT / 3.0
    coarse_steps = length // gap
    coarse_simpson = np.ones(coarse_steps + 1)
    coarse_simpson[1:-1:2] = 4.0
    coarse_simpson[2:-1:2] = 2.0
    coarse_simpson /= 3.0
    rows = []
    y_rows = []
    stats = []
    n = np.arange(length + 1)
    n_coarse = np.arange(coarse_steps + 1)
    offset_mid = (gap - 1) // 2
    for t0 in range(0, points - 1 - length + 1, hop):
        sl = slice(t0, t0 + length + 1)
        mid = slice(t0, t0 + length)
        bias_window = bias[mid]
        coarse_mid = slice(t0 + offset_mid, t0 + offset_mid + coarse_steps * gap, gap)
        f_coarse = f[coarse_mid]
        v_coarse = qdot_seg[t0:t0 + length + 1:gap]
        v_mid = 0.5 * (qdot_seg[sl][:-1] + qdot_seg[sl][1:])
        for k in range(1, kmax + 1):
            w = np.sin(k * np.pi * n / length) * simpson
            w_coarse = np.sin(k * np.pi * n_coarse / coarse_steps) * coarse_simpson
            for j in range(4):
                a = np.zeros(44)
                momentum = (np.einsum("m,mslj,ml->sj", w_coarse[:-2], f_coarse[:-1], v_coarse[1:-1])
                            - np.einsum("m,mslj,ml->sj", w_coarse[1:-1], f_coarse[1:], v_coarse[1:-1])
                            + w_coarse[-2] * np.einsum("slj,l->sj", f_coarse[-1], v_coarse[-1]))
                a[:40] = momentum[:40, j]
                a += np.einsum("m,msj->sj", w[:-1], bias_window)[:, j]
                y = np.sum(w * tau_seg[sl, j])
                sw2 = float(np.sum(w * w))
                f_sq = np.einsum("msk->ms", f_coarse[:, :, :, j] ** 2)
                nv2 = (np.einsum("m,ms->s", w_coarse[:-2] ** 2, f_sq[:-1])
                       + np.einsum("m,ms->s", w_coarse[1:-1] ** 2, f_sq[1:])
                       + w_coarse[-2] ** 2 * f_sq[-1])
                nbv = np.einsum("m,ms->s", w[:-1] ** 2,
                                np.einsum("msk,mk->ms",
                                          f[mid][:, :, :, j] ** 2, v_mid ** 2))
                rows.append(a)
                y_rows.append(y)
                stats.append((sw2, nv2, nbv, j))
    return np.asarray(rows), np.asarray(y_rows), stats


def assemble(offsets_full, data_pack, segments, length, hop, kmax, workers, gap=1):
    """并行组装全部数据段的回归矩阵。"""
    data, q, qdot, tau, terrain, base_yaw, sigma_v = data_pack
    template = gd.make_model([1.0] * 4, [[0.0] * 3] * 4,
                             [[0.01, 0.01, 0.01, 0.0, 0.0, 0.0]] * 4,
                             offsets_full, [0.0] * 4, 9.81)
    used = q.shape[0] if segments <= 0 else min(segments, q.shape[0])

    def work(index):
        terrain_seg = gd.make_terrain(float(terrain[index, 0]), float(terrain[index, 1]))
        base = gd.make_base(yaw=float(base_yaw[index]))
        f, bias = extract_segment(template, terrain_seg, base, q[index], qdot[index])
        return segment_rows(q[index], qdot[index], tau[index], f, bias, length, hop, kmax, gap)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(work, range(used)))
    matrix = np.concatenate([part[0] for part in results], axis=0)
    vector = np.concatenate([part[1] for part in results], axis=0)
    stats = [stat for part in results for stat in part[2]]
    return matrix, vector, stats, used


def row_weights(stats, p_hat, sigma_v):
    """按解析噪声传播计算每行权重：力矩噪声、速度噪声经动量项与科氏项、位置噪声经重力。

    统计量 sw2/nv2/nbv 已包含积分权重，直接等于对应积分噪声的方差系数。
    """
    weights = np.empty(len(stats))
    for index, (sw2, nv2, nbv, j) in enumerate(stats):
        variance = SIGMA_TAU ** 2 * sw2
        if p_hat is None:
            variance += (0.05 * sigma_v / DT) ** 2 * sw2
        else:
            variance += SIGMA_Q ** 2 * GRAVITY_SENSITIVITY ** 2 * float(np.sum(p_hat ** 2)) * sw2
            variance += float(np.sum(p_hat[:40] ** 2 * (2.0 * nv2[:40] + 4.0 * nbv[:40]))) * sigma_v ** 2
            variance += p_hat[40 + j] ** 2 * sigma_v ** 2 * sw2
        weights[index] = 1.0 / max(variance, 1.0e-20)
    return weights


def inner_solve(matrix, vector, stats, basis, z_prior, sigma_v, ridge_rel):
    """在 28 维可辨识子空间内做两遍加权 ridge 最小二乘，返回坐标与代价。"""
    design = matrix @ basis
    p_hat = None
    for _ in range(2):
        weights = row_weights(stats, p_hat, sigma_v)
        w_sqrt = np.sqrt(weights)
        weighted_design = design * w_sqrt[:, None]
        weighted_target = vector * w_sqrt
        gram = weighted_design.T @ weighted_design
        alpha = ridge_rel * float(np.mean(np.diag(gram)))
        z = np.linalg.solve(gram + alpha * np.eye(basis.shape[1]),
                            weighted_design.T @ weighted_target + alpha * z_prior)
        p_hat = basis @ z
    residual = w_sqrt * (design @ z - vector)
    cost = float(np.mean(residual ** 2))
    return z, cost, p_hat


def solve_offsets(offsets8, data_pack, prior56, config, basis_config):
    """给定偏置，内层求解，返回（代价开方、z、44 维参数）。供外层优化调用。"""
    full = offsets_to_full(offsets8)
    matrix, vector, stats, _ = assemble(full, data_pack, config["segments"], config["length"],
                                        config["hop"], config["kmax"], config["workers"],
                                        config["gap"])
    basis, _, _ = pb.compute_basis(full, count=basis_config["count"], seed=basis_config["seed"])
    p_prior = pb.inertial_from_raw(prior56)
    z_prior = basis.T @ p_prior
    z, cost, p_hat = inner_solve(matrix, vector, stats, basis, z_prior,
                                 data_pack[6], config["ridge_rel"])
    return cost ** 0.5, z, p_hat, basis


def null_basis(basis: np.ndarray) -> np.ndarray:
    """44 维参数空间中可辨识子空间的正交补基，形状 (44, 16)。"""
    projector = np.eye(44) - basis @ basis.T
    _, singular, vh = np.linalg.svd(projector)
    rank = 44 - basis.shape[1]
    if not np.all(singular[:rank] > 0.5):
        raise ValueError("零空间维度异常")
    return vh[:rank].T


def softplus(value):
    return np.log1p(np.exp(-np.abs(value))) + np.maximum(value, 0.0)


def inverse_softplus(value):
    return np.log(np.expm1(np.maximum(value, 1.0e-12)))


def theta_to_raw(theta: np.ndarray, offsets_full: np.ndarray) -> np.ndarray:
    """无约束物理参数 theta（44 维）-> 56 维原始参数。

    theta 布局：质量 softplus(4)、质心(12)、绕质心惯量 Cholesky 下三角(24)、
    阻尼 softplus(4)。参数化本身保证质量为正、绕质心惯量正定。
    """
    raw = np.zeros(56)
    raw[40:52] = np.asarray(offsets_full, dtype=float).reshape(-1)
    raw[0:4] = softplus(theta[0:4]) + 1.0e-3
    raw[4:16] = theta[4:16]
    raw[52:56] = softplus(theta[40:44]) + 1.0e-4
    position = 16
    for body in range(4):
        lower = np.zeros((3, 3))
        for row in range(3):
            for column in range(row + 1):
                lower[row, column] = theta[position]
                position += 1
        inertia = lower @ lower.T + 1.0e-6 * np.eye(3)
        raw[16 + 6 * body:22 + 6 * body] = [inertia[0, 0], inertia[1, 1], inertia[2, 2],
                                            inertia[0, 1], inertia[0, 2], inertia[1, 2]]
    return raw


def raw_to_theta(raw: np.ndarray) -> np.ndarray:
    theta = np.zeros(44)
    theta[0:4] = inverse_softplus(raw[0:4] - 1.0e-3)
    theta[4:16] = raw[4:16]
    theta[40:44] = inverse_softplus(raw[52:56] - 1.0e-4)
    position = 16
    for body in range(4):
        inertia = np.array([[raw[16 + 6 * body], raw[16 + 6 * body + 3], raw[16 + 6 * body + 4]],
                            [raw[16 + 6 * body + 3], raw[17 + 6 * body], raw[16 + 6 * body + 5]],
                            [raw[16 + 6 * body + 4], raw[16 + 6 * body + 5], raw[18 + 6 * body]]])
        lower = np.linalg.cholesky(inertia - 1.0e-6 * np.eye(3))
        for row in range(3):
            for column in range(row + 1):
                theta[position] = lower[row, column]
                position += 1
    return theta


SLOT = {(0, 0): 0, (1, 1): 1, (2, 2): 2, (0, 1): 3, (0, 2): 4, (1, 2): 5}


def theta_jacobian(theta: np.ndarray) -> np.ndarray:
    """theta -> 44 维空间惯性参数映射的解析雅可比，形状 (44, 44)。"""
    jac = np.zeros((44, 44))
    for body in range(4):
        base = 10 * body
        sigmoid_mass = 1.0 / (1.0 + np.exp(-theta[body]))
        mass = softplus(theta[body]) + 1.0e-3
        com = theta[4 + 3 * body:7 + 3 * body]
        jac[base, body] = sigmoid_mass
        jac[base + 1:base + 4, body] = sigmoid_mass * com
        jac[base + 1:base + 4, 4 + 3 * body:7 + 3 * body] = mass * np.eye(3)
        parallel = (com @ com) * np.eye(3) - np.outer(com, com)
        for row in range(3):
            for column in range(3):
                slot = SLOT[(min(row, column), max(row, column))]
                jac[base + 4 + slot, body] = sigmoid_mass * parallel[row, column]
        for axis in range(3):
            e_a = np.zeros(3)
            e_a[axis] = 1.0
            d_parallel = 2.0 * com[axis] * np.eye(3) - np.outer(e_a, com) - np.outer(com, e_a)
            for row in range(3):
                for column in range(3):
                    slot = SLOT[(min(row, column), max(row, column))]
                    jac[base + 4 + slot, 4 + 3 * body + axis] = mass * d_parallel[row, column]
        lower = np.zeros((3, 3))
        entries = ((0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2))
        for entry, (row, column) in enumerate(entries):
            lower[row, column] = theta[16 + 6 * body + entry]
        for entry, (row, column) in enumerate(entries):
            d_lower = np.zeros((3, 3))
            d_lower[row, column] = 1.0
            d_inertia = d_lower @ lower.T + lower @ d_lower.T
            for r in range(3):
                for c in range(3):
                    slot = SLOT[(min(r, c), max(r, c))]
                    jac[base + 4 + slot, 16 + 6 * body + entry] = d_inertia[r, c]
    for joint in range(4):
        jac[40 + joint, 40 + joint] = 1.0 / (1.0 + np.exp(-theta[40 + joint]))
    return jac


def final_model(z: np.ndarray, basis: np.ndarray, prior56: np.ndarray,
                offsets_full: np.ndarray):
    """构造物理合法的 56 维模型。

    在无约束物理参数空间（质量 softplus、绕质心惯量 Cholesky）内最小化：
    可辨识坐标严格逼近 z（28 项），零空间分量以 1e-3 权重拉向先验（16 项）。
    使用解析雅可比。参数化本身保证质量为正、惯量正定。
    """
    p_prior = pb.inertial_from_raw(prior56)
    p_target = p_prior + basis @ (z - basis.T @ p_prior)

    def residual(theta):
        raw = theta_to_raw(theta, offsets_full)
        p = pb.inertial_from_raw(raw)
        difference = p - p_target
        return np.concatenate([basis.T @ difference, 1.0e-3 * difference])

    def jacobian(theta):
        jac = theta_jacobian(theta)
        return np.vstack([basis.T @ jac, 1.0e-3 * jac])

    start = raw_to_theta(prior56)
    result = least_squares(residual, start, jac=jacobian, method="lm",
                           max_nfev=200, xtol=1e-13, ftol=1e-13)
    raw = theta_to_raw(result.x, offsets_full)
    mismatch = np.linalg.norm(basis.T @ (pb.inertial_from_raw(raw) - p_target))
    if mismatch > 1.0e-4:
        print("警告：物理参数化后可辨识坐标失配 %.3e" % mismatch)
    min_eigenvalue = float(np.min(np.linalg.eigvalsh(
        np.array([[raw[16], raw[19], raw[20]], [raw[19], raw[17], raw[21]],
                  [raw[20], raw[21], raw[18]]]))))
    for body in range(1, 4):
        inertia = np.array([[raw[16 + 6 * body], raw[16 + 6 * body + 3], raw[16 + 6 * body + 4]],
                            [raw[16 + 6 * body + 3], raw[17 + 6 * body], raw[16 + 6 * body + 5]],
                            [raw[16 + 6 * body + 4], raw[16 + 6 * body + 5], raw[18 + 6 * body]]])
        min_eigenvalue = min(min_eigenvalue, float(np.linalg.eigvalsh(inertia)[0]))
    return raw, min_eigenvalue


def fit_model(data_pack, prior56, config, basis_config, outer_config):
    """外层优化 8 个偏置，内层线性求解，返回 (56 维模型, z, offsets8, 外层代价)。"""
    initial = offsets_to_8(pb.NOMINAL_OFFSETS)
    scale = np.full(8, 0.05)

    def objective(offsets8):
        cost_root, _, _, _ = solve_offsets(offsets8, data_pack, prior56, outer_config,
                                           basis_config)
        return cost_root

    result = least_squares(objective, initial, bounds=(-0.3 + initial, 0.3 + initial),
                           x_scale=scale, jac="2-point", diff_step=1e-3,
                           max_nfev=outer_config["max_nfev"], verbose=1)
    best_offsets = np.asarray(result.x)
    _, z, p_hat, basis = solve_offsets(best_offsets, data_pack, prior56, config, basis_config)
    full = offsets_to_full(best_offsets)
    model, min_eigenvalue = final_model(z, basis, prior56, full)
    if min_eigenvalue < 0.0:
        print("警告：最终模型最小伪惯量特征值 %.3e" % min_eigenvalue)
    return model, z, best_offsets, float(result.cost)


def run_eval(model, data_path, segments):
    import eval_openloop
    metrics = eval_openloop.evaluate(model, data_path, segments)
    report = eval_openloop.format_report(metrics)
    print(report)
    rmse50 = metrics["rmse"][50]
    maximum50 = metrics["maximum"][50]
    wrong = metrics["direction_wrong"]
    print("50 步 RMSE %s" % " ".join("%.3e" % v for v in rmse50))
    print("50 步 MAX  %s" % " ".join("%.3e" % v for v in maximum50))
    print("direction_wrong %d  评分 %.6e" % (wrong, eval_openloop.score(metrics)))
    return metrics


def sweep(data_pack, prior56, config, basis_config, entries, data_path, sg_order):
    """在标称偏置下扫描窗口/平滑/正则配置，输出开环指标。"""
    print("扫描 %d 个配置" % len(entries))
    for entry in entries:
        sg_window, length, hop, kmax = (int(value) for value in entry.split("-"))
        if sg_window % 2 == 0 or sg_window <= sg_order:
            raise ValueError("sg_window 必须为大于 sg_order 的奇数")
        current_data_pack = load_data(data_path, sg_window, sg_order)
        current = dict(config)
        current.update({"length": length, "hop": hop, "kmax": kmax})
        _, z, _, basis = solve_offsets(offsets_to_8(pb.NOMINAL_OFFSETS), current_data_pack,
                                       prior56, current, basis_config)
        full = pb.NOMINAL_OFFSETS
        model, min_eigenvalue = final_model(z, basis, prior56, full)
        import eval_openloop
        metrics = eval_openloop.evaluate(model, data_path, 0)
        wrong = metrics["direction_wrong"]
        score = eval_openloop.score(metrics)
        print("配置 sg%d len%d hop%d kmax%d：50 步 RMSE %s，direction_wrong %d，评分 %.4e，最小特征值 %.1e"
              % (sg_window, length, hop, kmax,
                 " ".join("%.2e" % v for v in metrics["rmse"][50]), wrong, score, min_eigenvalue),
              flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="积分弱形式参数辨识")
    parser.add_argument("--data", type=Path,
                        default=Path("data/closed_loop_data_noisy.npz"))
    parser.add_argument("--save", type=Path, default=Path("data/identified_weakform.npz"))
    parser.add_argument("--prior", type=Path, default=Path("data/identified_analytic.npz"))
    parser.add_argument("--mode", choices=("sweep", "fit", "eval"), default="fit")
    parser.add_argument("--sg-window", type=int, default=7)
    parser.add_argument("--sg-order", type=int, default=3)
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--hop", type=int, default=50)
    parser.add_argument("--kmax", type=int, default=4)
    parser.add_argument("--gap", type=int, default=1)
    parser.add_argument("--segments", type=int, default=0)
    parser.add_argument("--ridge-rel", type=float, default=1.0e-9)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--outer-max-nfev", type=int, default=40)
    parser.add_argument("--outer-segments", type=int, default=40)
    parser.add_argument("--outer-length", type=int, default=80)
    parser.add_argument("--outer-hop", type=int, default=40)
    parser.add_argument("--outer-kmax", type=int, default=3)
    parser.add_argument("--no-outer", action="store_true")
    parser.add_argument("--sweep-configs", type=str, default="")
    parser.add_argument("--basis-count", type=int, default=120)
    arguments = parser.parse_args()

    if arguments.sg_window % 2 == 0 or arguments.sg_window <= arguments.sg_order:
        raise ValueError("sg_window 必须为大于 sg_order 的奇数")
    if arguments.length > 299:
        raise ValueError("窗口长度不能超过数据段点数")

    data_pack = load_data(arguments.data, arguments.sg_window, arguments.sg_order)
    prior56 = pb.prior_model(arguments.prior)
    if arguments.gap % 2 != 1 or arguments.length % arguments.gap != 0:
        raise ValueError("gap 必须为奇数且整除窗口长度")
    config = {"segments": arguments.segments, "length": arguments.length,
              "hop": arguments.hop, "kmax": arguments.kmax, "gap": arguments.gap,
              "ridge_rel": arguments.ridge_rel, "workers": arguments.workers}
    basis_config = {"count": arguments.basis_count, "seed": 20261201}
    outer_config = {"segments": arguments.outer_segments, "length": arguments.outer_length,
                    "hop": arguments.outer_hop, "kmax": arguments.outer_kmax,
                    "gap": arguments.gap,
                    "ridge_rel": arguments.ridge_rel, "workers": arguments.workers,
                    "max_nfev": arguments.outer_max_nfev}

    if arguments.mode == "sweep":
        if not arguments.sweep_configs:
            raise ValueError("sweep 模式需要 --sweep-configs")
        entries = arguments.sweep_configs.split(",")
        sweep(data_pack, prior56, config, basis_config, entries, arguments.data,
              arguments.sg_order)
        return
    if arguments.mode == "eval":
        model = np.load(arguments.save)["model"]
        run_eval(model, arguments.data, 0)
        return

    if arguments.no_outer:
        _, z, _, basis = solve_offsets(offsets_to_8(pb.NOMINAL_OFFSETS), data_pack, prior56,
                                       config, basis_config)
        best_offsets = offsets_to_8(pb.NOMINAL_OFFSETS)
        model, min_eigenvalue = final_model(z, basis, prior56, pb.NOMINAL_OFFSETS)
        if min_eigenvalue < 0.0:
            print("警告：最终模型最小伪惯量特征值 %.3e" % min_eigenvalue)
    else:
        model, z, best_offsets, outer_cost = fit_model(data_pack, prior56, config,
                                                       basis_config, outer_config)
        print("外层优化代价 %.6e，偏置 8 维 %s" % (outer_cost,
              " ".join("%.4f" % v for v in best_offsets)))

    reduced = np.concatenate([z, best_offsets])
    arguments.save.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.save, model=model, reduced=reduced, z=z,
                        offsets=best_offsets, prior=prior56)
    print("已保存到", arguments.save)
    print("辨识参数物理一致性：")
    import eval_openloop
    print(eval_openloop.physical_report(model))
    run_eval(model, arguments.data, 0)


if __name__ == "__main__":
    main()
