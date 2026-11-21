"""基于新参数类的 MAP 参数辨识。

使用 param_basis.py 提供的 28 维可辨识参数类（24 惯性 + 4 阻尼）。

策略：
1. 偏置外层固定（初值用标称几何）
2. 内层对 44 维惯性参数做线性最小二乘（每段优化初始状态）
3. 在零空间上把解拉向先验（加权最小范数）
4. 分段平滑：每窗独立优化初始状态，使仿真与观测位置通道匹配
"""

import numpy as np
import sys
from pathlib import Path
from typing import Tuple, List
import argparse
from dataclasses import dataclass
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
import gimbal_dynamics as gd
import identify_physical as ip
import param_basis as pb


# 噪声模型权重
WEIGHT_Q = 1.0 / (0.001 ** 2)  # 位置通道
WEIGHT_V = 0.01 / (0.1 ** 2)   # 速度通道（低权重）


@dataclass
class Segment:
    """数据段。"""
    q_obs: np.ndarray  # (steps, 4)
    v_obs: np.ndarray  # (steps, 4)
    tau_obs: np.ndarray  # (steps, 4)
    terrain: tuple
    base_yaw: float


def simulate_segment(context: gd.Context, model, segment: Segment, 
                    q0: np.ndarray, v0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """仿真单个数据段。"""
    steps = segment.q_obs.shape[0]
    q_sim = np.zeros((steps, 4))
    v_sim = np.zeros((steps, 4))
    
    terrain = gd.make_terrain(*segment.terrain)
    base = gd.make_base(yaw=segment.base_yaw)
    
    state = gd.make_state(q0, v0)
    
    for t in range(steps):
        for i in range(4):
            q_sim[t, i] = state.angle[i]
            v_sim[t, i] = state.velocity[i]
        
        if t < steps - 1:
            state = context.forward_step(
                model, terrain, base, state,
                segment.tau_obs[t], step=0.01, refinement=2
            )
    
    return q_sim, v_sim


def segment_loss(context: gd.Context, model, segment: Segment,
                q0: np.ndarray, v0: np.ndarray) -> float:
    """计算单段损失。"""
    q_sim, v_sim = simulate_segment(context, model, segment, q0, v0)
    
    dq = q_sim - segment.q_obs
    dv = v_sim - segment.v_obs
    
    loss = WEIGHT_Q * np.sum(dq ** 2) + WEIGHT_V * np.sum(dv ** 2)
    return loss


def optimize_segment_initial_state(context: gd.Context, model, segment: Segment,
                                   q0_init: np.ndarray, v0_init: np.ndarray,
                                   max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """优化单段的初始状态。"""
    x0 = np.concatenate([q0_init, v0_init])
    
    def objective(x):
        q0 = x[:4]
        v0 = x[4:]
        try:
            return segment_loss(context, model, segment, q0, v0)
        except Exception as e:
            return 1e10
    
    result = minimize(objective, x0, method='Nelder-Mead',
                     options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-9})
    
    return result.x[:4], result.x[4:]


def load_and_split_data(data_path: Path, window_steps: int, max_segments: int) -> List[Segment]:
    """加载并切分数据。"""
    q, qdot, qddot, tau, terrain, base_yaw = ip.load_data(data_path)
    
    print(f"加载数据：{q.shape[0]} 轨迹 × {q.shape[1]} 步")
    
    segments = []
    for traj_idx in range(q.shape[0]):
        num_windows = q.shape[1] // window_steps
        
        for win_idx in range(num_windows):
            start = win_idx * window_steps
            end = start + window_steps
            
            if end > q.shape[1]:
                break
            
            segment = Segment(
                q_obs=q[traj_idx, start:end, :].copy(),
                v_obs=qdot[traj_idx, start:end, :].copy(),
                tau_obs=tau[traj_idx, start:end, :].copy(),
                terrain=(float(terrain[traj_idx, 0]), float(terrain[traj_idx, 1])),
                base_yaw=float(base_yaw[traj_idx])
            )
            segments.append(segment)
            
            if len(segments) >= max_segments:
                break
        
        if len(segments) >= max_segments:
            break
    
    print(f"切分为 {len(segments)} 个窗口，每窗 {window_steps} 步")
    return segments


def compute_regressor_matrix(context: gd.Context, reference_model, segment: Segment,
                            q0: np.ndarray, v0: np.ndarray) -> np.ndarray:
    """计算单段的回归矩阵（对 44 维惯性参数）。
    
    Returns:
        Y: (steps*4, 44) 回归矩阵
    """
    steps = segment.q_obs.shape[0]
    Y = np.zeros((steps * 4, 44))
    
    terrain = gd.make_terrain(*segment.terrain)
    base = gd.make_base(yaw=segment.base_yaw)
    state = gd.make_state(q0, v0)
    
    # 逐步计算加速度和回归矩阵
    for t in range(steps):
        q_t = np.array([state.angle[i] for i in range(4)])
        v_t = np.array([state.velocity[i] for i in range(4)])
        
        # 数值微分加速度（向前差分）
        if t < steps - 1:
            state_next = context.forward_step(
                reference_model, terrain, base, state,
                segment.tau_obs[t], step=0.01, refinement=2
            )
            v_next = np.array([state_next.velocity[i] for i in range(4)])
            a_t = (v_next - v_t) / 0.01
        else:
            a_t = np.zeros(4)
        
        # 调用 inertial_basis 获取当前状态的惯性灵敏度
        basis_t = np.array(context.inertial_basis(
            reference_model, terrain, base, state, a_t
        ))  # (44, 4)
        
        Y[t*4:(t+1)*4, :] = basis_t.T  # (4, 44)
        
        state = state_next if t < steps - 1 else state
    
    return Y


def linear_ls_step(segments: List[Segment], initial_states: List[Tuple],
                  basis: np.ndarray, prior_raw: np.ndarray,
                  offsets: np.ndarray, regularization: float = 0.01) -> np.ndarray:
    """固定初始状态和偏置，对可辨识参数做线性最小二乘。
    
    Args:
        segments: 数据段列表
        initial_states: 各段初始状态
        basis: (44, 28) 可辨识基
        prior_raw: (56,) 先验模型
        offsets: (4, 3) 当前偏置
        regularization: 正则化系数
    
    Returns:
        z: (28,) 可辨识坐标
    """
    context = gd.Context(1)
    
    try:
        # 构造参考模型（用先验参数 + 当前偏置）
        reference_raw = prior_raw.copy()
        reference_raw[40:52] = offsets.reshape(-1)
        reference_model = ip.build_model(reference_raw)
        
        # 收集所有段的回归矩阵和残差
        Y_all = []
        tau_obs_all = []
        
        print("  构造回归矩阵...")
        for seg_idx, (segment, (q0, v0)) in enumerate(zip(segments, initial_states)):
            if seg_idx % 5 == 0:
                print(f"    段 {seg_idx + 1}/{len(segments)}")
            
            Y_seg = compute_regressor_matrix(context, reference_model, segment, q0, v0)
            tau_obs_seg = segment.tau_obs.reshape(-1)  # (steps*4,)
            
            Y_all.append(Y_seg)
            tau_obs_all.append(tau_obs_seg)
        
        Y = np.vstack(Y_all)  # (total_steps*4, 44)
        tau_obs = np.concatenate(tau_obs_all)  # (total_steps*4,)
        
        print(f"  回归矩阵形状：{Y.shape}")
        
        # 投影到可辨识子空间：Y_z = Y @ basis
        Y_z = Y @ basis  # (total_steps*4, 28)
        
        # 加权最小二乘（含正则化）
        # min ||Y_z @ z - tau_obs||^2 + lambda ||z - z_prior||^2
        prior_values = pb.inertial_from_raw(prior_raw)
        z_prior = basis.T @ prior_values  # (28,)
        
        # 构造增广系统
        n_data = Y_z.shape[0]
        reg_weight = np.sqrt(regularization * n_data)
        
        Y_aug = np.vstack([Y_z, reg_weight * np.eye(28)])
        tau_aug = np.concatenate([tau_obs, reg_weight * z_prior])
        
        # 求解
        z, residuals, rank, s = np.linalg.lstsq(Y_aug, tau_aug, rcond=None)
        
        print(f"  线性最小二乘：数据残差 {residuals[0]:.6e}")
        
        # 投影到物理可行域（确保质量为正、惯量正定）
        print("  投影到物理可行域...")
        values_estimated = prior_values + basis @ (z - basis.T @ prior_values)
        raw_physical, mismatch, cost = pb.physical_model(
            values_estimated, basis, prior_raw, offsets.reshape(-1), identifiable_weight=1e4
        )
        z_physical = pb.to_identifiable(raw_physical, basis)
        print(f"  物理投影：可辨识坐标偏差 {np.linalg.norm(z_physical - z):.6e}")
        
        return z_physical
    
    finally:
        context.close()


def two_stage_optimize(segments: List[Segment], prior_raw: np.ndarray,
                      basis: np.ndarray, max_outer_iter: int = 3,
                      regularization: float = 0.01) -> Tuple[np.ndarray, np.ndarray, List]:
    """两阶段优化：交替优化初始状态和参数。
    
    Returns:
        z: (28,) 可辨识坐标
        offsets: (4, 3) 偏置
        initial_states: 各段优化后的初始状态
    """
    # 初始化偏置（使用标称几何）
    offsets = pb.NOMINAL_OFFSETS.copy()
    
    # 初始化各段初始状态
    initial_states = [(seg.q_obs[0].copy(), seg.v_obs[0].copy()) for seg in segments]
    
    # 初始可辨识坐标（从先验）
    prior_values = pb.inertial_from_raw(prior_raw)
    z = basis.T @ prior_values
    
    # 确保初始模型物理合法
    print("初始化物理合法模型...")
    values_init = prior_values.copy()
    init_raw, _, _ = pb.physical_model(values_init, basis, prior_raw, offsets.reshape(-1))
    z = pb.to_identifiable(init_raw, basis)
    
    for outer_iter in range(max_outer_iter):
        print(f"\n=== 外循环 {outer_iter + 1}/{max_outer_iter} ===")
        
        # 步骤 1：固定参数，优化各段初始状态
        values_current = prior_values + basis @ (z - basis.T @ prior_values)
        current_raw, _, _ = pb.physical_model(
            values_current, basis, prior_raw, offsets.reshape(-1), identifiable_weight=1e4
        )
        current_model = ip.build_model(current_raw)
        context = gd.Context(1)
        
        try:
            print("步骤 1：优化各段初始状态...")
            for seg_idx, segment in enumerate(segments):
                q0, v0 = initial_states[seg_idx]
                q0_opt, v0_opt = optimize_segment_initial_state(
                    context, current_model, segment, q0, v0, max_iter=50
                )
                initial_states[seg_idx] = (q0_opt, v0_opt)
                
                if seg_idx % 3 == 0 or seg_idx == len(segments) - 1:
                    loss_seg = segment_loss(context, current_model, segment, q0_opt, v0_opt)
                    print(f"  段 {seg_idx + 1}/{len(segments)}, 损失: {loss_seg:.6e}")
        finally:
            context.close()
        
        # 步骤 2：固定初始状态和偏置，线性最小二乘参数
        print("\n步骤 2：线性最小二乘参数...")
        z = linear_ls_step(segments, initial_states, basis, prior_raw, offsets, regularization)
        
        # 计算当前总损失（使用物理投影确保合法性）
        values_current = prior_values + basis @ (z - basis.T @ prior_values)
        current_raw, _, _ = pb.physical_model(
            values_current, basis, prior_raw, offsets.reshape(-1), identifiable_weight=1e4
        )
        current_model = ip.build_model(current_raw)
        context = gd.Context(1)
        try:
            total_loss = 0.0
            for segment, (q0, v0) in zip(segments, initial_states):
                total_loss += segment_loss(context, current_model, segment, q0, v0)
            print(f"\n当前总损失：{total_loss / len(segments):.6e}")
        finally:
            context.close()
    
    return z, offsets, initial_states


def save_results(z: np.ndarray, offsets: np.ndarray, basis: np.ndarray,
                prior_raw: np.ndarray, segments: List[Segment],
                initial_states: List, save_path: Path):
    """保存结果。"""
    # 构造 56 维模型
    raw_model = pb.from_identifiable(z, basis, prior_raw, offsets.reshape(-1))
    
    # 保存参数
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_path,
        identifiable=z,
        offsets=offsets,
        model=raw_model,
        reduced=z  # 兼容旧评估脚本
    )
    print(f"\n参数已保存到 {save_path}")
    
    # 保存平滑轨迹
    context = gd.Context(1)
    try:
        model = ip.build_model(raw_model)
        all_q_smooth = []
        all_v_smooth = []
        
        for segment, (q0, v0) in zip(segments, initial_states):
            q_sim, v_sim = simulate_segment(context, model, segment, q0, v0)
            all_q_smooth.append(q_sim)
            all_v_smooth.append(v_sim)
        
        smooth_path = save_path.parent / f"map_smoothed_{save_path.stem}.npz"
        np.savez_compressed(
            smooth_path,
            q_smooth=np.array(all_q_smooth),
            v_smooth=np.array(all_v_smooth),
            initial_states=np.array([(q0, v0) for q0, v0 in initial_states])
        )
        print(f"平滑轨迹已保存到 {smooth_path}")
    finally:
        context.close()


def main():
    parser = argparse.ArgumentParser(description="MAP 参数辨识（新参数类）")
    parser.add_argument("--data", type=Path, default=Path("data/closed_loop_data_noisy.npz"))
    parser.add_argument("--window", type=int, default=150, help="窗口长度")
    parser.add_argument("--segments", type=int, default=10, help="使用的数据段数量")
    parser.add_argument("--outer-iter", type=int, default=3)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--save", type=Path, default=Path("data/identified_map.npz"))
    args = parser.parse_args()
    
    print("=== MAP 参数辨识（新 28 维可辨识参数类）===\n")
    
    # 计算可辨识基
    print("计算可辨识基...")
    basis, singular, rank = pb.compute_basis(count=120)
    print(f"可辨识维数：{rank}（惯性 {rank-4} + 阻尼 4）")
    print(f"奇异值范围：{singular[0]:.3e} 到 {singular[23]:.3e}")
    print(f"零空间最大：{singular[24]:.3e}\n")
    
    # 加载先验（优先使用 regressor_corrected51）
    prior_path = Path('data/regressor_corrected51.npz')
    if prior_path.exists():
        prior_data = np.load(prior_path)
        prior_raw = prior_data['model']
        print(f"先验模型：{prior_path.name}（regressor-ident EIV 修正结果）\n")
    else:
        prior_raw = pb.prior_model()
        print(f"先验模型：{Path('data/identified_analytic.npz').name}\n")
    
    # 加载数据
    segments = load_and_split_data(args.data, args.window, args.segments)
    
    # 优化
    print("\n开始两阶段优化...")
    z, offsets, initial_states = two_stage_optimize(
        segments, prior_raw, basis,
        max_outer_iter=args.outer_iter,
        regularization=args.regularization
    )
    
    # 保存结果
    save_results(z, offsets, basis, prior_raw, segments, initial_states, args.save)
    
    print("\n完成。请使用 eval_openloop.py 评估开环性能：")
    print(f"  python3 eval_openloop.py --model {args.save}")


if __name__ == '__main__':
    main()
