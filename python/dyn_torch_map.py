"""PyTorch 可微 4-DOF 串联云台动力学，用于 MAP 参数估计。

实现四自由度串联云台的正逆动力学，支持自动微分：
- 使用雅可比公式计算惯性矩阵 M(q)
- 使用势能梯度计算重力项 G(q)
- 使用 Christoffel 符号 + 自动微分计算科氏/离心项 C(q,v)v
- RK4 积分器（refinement=2，与数据生成一致）
"""

import torch
import numpy as np
from typing import Tuple, Optional

class GimbalDynamicsTorch:
    """四自由度串联云台动力学 PyTorch 实现。
    
    参数化：
    - beta_reduced: 33 维物理基参数
    - beta_full: 56 维完整参数（4 质量 + 4×3 质心 + 4×6 惯量 + 4 阻尼）
    
    坐标系：z-up（x 右，y 前，z 上），关节轴序列 z-y-z-x。
    """
    
    def __init__(self, device='cpu', dtype=torch.float64):
        self.device = device
        self.dtype = dtype
        self.g = 9.81
        
        # 关节轴方向（在父坐标系中）
        self.axes = torch.tensor([
            [0., 0., 1.],  # joint 0: z
            [0., 1., 0.],  # joint 1: y
            [0., 0., 1.],  # joint 2: z
            [1., 0., 0.],  # joint 3: x
        ], device=device, dtype=dtype)
        
        # 安装偏置（父坐标系）
        self.offsets = torch.tensor([
            [0.0, 0.0, 0.1150],
            [0.0, 0.0900, 0.0390],
            [0.0, 0.0, 0.0521],
            [0.0, 0.0, 0.1579],
        ], device=device, dtype=dtype)
        
    def beta_reduced_to_full(self, beta_r: torch.Tensor) -> torch.Tensor:
        """将 33 维基参数转换为 56 维完整参数。
        
        beta_r[0:4]     = 质量 m_i
        beta_r[4:16]    = 质心位置 r_i（3 分量 × 4）
        beta_r[16:28]   = 对角惯量 I_ii（xx,yy,zz × 4）
        beta_r[28:32]   = 非对角惯量 I_12（xy × 4）
        beta_r[32:36]   = 非对角惯量 I_13（xz × 4）（多出 4 个）
        beta_r[36:40]   = 非对角惯量 I_23（yz × 4）（实际只有 33 维，这里不存在）
        beta_r 实际 33 维：4 + 12 + 12 + 1 + 4 = 33
        但按文档 beta_r 是 33 维，对应 4 质量 + 4×3 质心 + 4×3 对角 + 1 xy
        
        重新理解：根据 PARAMETER_IDENTIFIABILITY.md：
        - 4 质量
        - 4×3=12 质心
        - 4×3=12 对角惯量
        - 5 非对角惯量（因为有些是冗余的）
        总计 4+12+12+5=33
        """
        assert beta_r.shape[-1] == 33, f"期望 33 维，得到 {beta_r.shape[-1]}"
        
        batch_shape = beta_r.shape[:-1]
        beta_f = torch.zeros(*batch_shape, 56, device=beta_r.device, dtype=beta_r.dtype)
        
        # 质量
        beta_f[..., 0:4] = beta_r[..., 0:4]
        
        # 质心（12 维）
        beta_f[..., 4:16] = beta_r[..., 4:16]
        
        # 对角惯量（12 维）
        beta_f[..., 16:28] = beta_r[..., 16:28]
        
        # 非对角惯量（5 维映射到 4×3=12 个位置）
        # 根据 PARAMETER_IDENTIFIABILITY.md，需要查看哪些是独立的
        # 这里先做一个简化映射，后续根据文档精确调整
        idx = 28
        for i in range(4):
            if i == 0:  # link 0: xy, xz, yz 都可能非零，但对称性可能约束
                beta_f[..., 28 + i*3 + 0] = beta_r[..., idx]  # xy
                idx += 1
                if idx < 33:
                    beta_f[..., 28 + i*3 + 1] = beta_r[..., idx]  # xz
                    idx += 1
                if idx < 33:
                    beta_f[..., 28 + i*3 + 2] = beta_r[..., idx]  # yz
                    idx += 1
            else:
                if idx < 33:
                    beta_f[..., 28 + i*3 + 0] = beta_r[..., idx]
                    idx += 1
                # 其余默认为 0
        
        # 阻尼（暂时固定为 0.1）
        beta_f[..., 52:56] = 0.1
        
        return beta_f
    
    def rotation_matrix(self, axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """Rodrigues 旋转公式计算旋转矩阵。
        
        Args:
            axis: (..., 3) 单位轴向量
            angle: (...) 旋转角度
        Returns:
            (..., 3, 3) 旋转矩阵
        """
        # axis: (..., 3), angle: (...)
        angle = angle.unsqueeze(-1).unsqueeze(-1)  # (..., 1, 1)
        axis = axis.unsqueeze(-2)  # (..., 1, 3)
        
        K = torch.zeros(*axis.shape[:-2], 3, 3, device=axis.device, dtype=axis.dtype)
        K[..., 0, 1] = -axis[..., 0, 2]
        K[..., 0, 2] = axis[..., 0, 1]
        K[..., 1, 0] = axis[..., 0, 2]
        K[..., 1, 2] = -axis[..., 0, 0]
        K[..., 2, 0] = -axis[..., 0, 1]
        K[..., 2, 1] = axis[..., 0, 0]
        
        I = torch.eye(3, device=axis.device, dtype=axis.dtype)
        R = I + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
        return R
    
    def forward_kinematics(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """正运动学：计算各连杆坐标系的位姿。
        
        Args:
            q: (..., 4) 关节角度
        Returns:
            positions: (..., 4, 3) 各关节原点在世界系中的位置
            rotations: (..., 4, 3, 3) 各连杆坐标系到世界系的旋转矩阵
        """
        batch_shape = q.shape[:-1]
        
        # 初始化
        positions = torch.zeros(*batch_shape, 4, 3, device=q.device, dtype=q.dtype)
        rotations = torch.zeros(*batch_shape, 4, 3, 3, device=q.device, dtype=q.dtype)
        
        # 基座系
        R_world = torch.eye(3, device=q.device, dtype=q.dtype).expand(*batch_shape, 3, 3).clone()
        p_world = torch.zeros(*batch_shape, 3, device=q.device, dtype=q.dtype)
        
        for i in range(4):
            # 当前关节旋转
            axis = self.axes[i]
            R_joint = self.rotation_matrix(axis, q[..., i])
            
            # 累积旋转
            R_world = R_world @ R_joint
            rotations[..., i, :, :] = R_world
            
            # 累积位置（先旋转偏置，再累加）
            offset_world = (rotations[..., i, :, :] @ self.offsets[i].unsqueeze(-1)).squeeze(-1)
            p_world = p_world + offset_world
            positions[..., i, :] = p_world
        
        return positions, rotations
    
    def compute_jacobian(self, q: torch.Tensor) -> torch.Tensor:
        """计算质心位置对关节角的雅可比矩阵。
        
        使用自动微分计算 d(r_com_i)/dq_j。
        
        Args:
            q: (..., 4) 关节角度（需要 requires_grad=True）
        Returns:
            (..., 4, 3, 4) 雅可比矩阵 J[i,:,j] = d(r_com_i)/dq_j
        """
        batch_shape = q.shape[:-1]
        
        # 确保 q 需要梯度
        if not q.requires_grad:
            q = q.detach().requires_grad_(True)
        
        positions, rotations = self.forward_kinematics(q)
        
        # 计算各连杆质心位置（需要 beta_full）
        # 这里暂时返回关节原点位置的雅可比
        # 完整实现需要质心偏置
        
        # 简化：返回关节位置雅可比
        J = torch.zeros(*batch_shape, 4, 3, 4, device=q.device, dtype=q.dtype)
        
        for i in range(4):
            for d in range(3):
                # 对每个坐标分量计算梯度
                grad_outputs = torch.zeros_like(positions[..., i, d])
                grad_outputs[...] = 1.0
                
                grads = torch.autograd.grad(
                    outputs=positions[..., i, d],
                    inputs=q,
                    grad_outputs=grad_outputs,
                    create_graph=True,
                    retain_graph=True
                )[0]
                
                J[..., i, d, :] = grads
        
        return J
    
    def compute_inertia_matrix(self, q: torch.Tensor, beta_full: torch.Tensor) -> torch.Tensor:
        """计算惯性矩阵 M(q)。
        
        使用雅可比公式：M = ∑_i (m_i J_vi^T J_vi + J_wi^T I_i J_wi)
        
        Args:
            q: (..., 4) 关节角度
            beta_full: (..., 56) 完整参数
        Returns:
            (..., 4, 4) 惯性矩阵
        """
        # 暂时返回单位阵占位
        batch_shape = q.shape[:-1]
        return torch.eye(4, device=q.device, dtype=q.dtype).expand(*batch_shape, 4, 4).clone()
    
    def compute_gravity_torque(self, q: torch.Tensor, beta_full: torch.Tensor) -> torch.Tensor:
        """计算重力力矩 G(q)。
        
        使用势能梯度：G = dU/dq，其中 U = ∑_i m_i g h_i(q)
        
        Args:
            q: (..., 4) 关节角度
            beta_full: (..., 56) 完整参数
        Returns:
            (..., 4) 重力力矩
        """
        # 暂时返回零占位
        return torch.zeros(*q.shape, device=q.device, dtype=q.dtype)
    
    def compute_coriolis_torque(self, q: torch.Tensor, v: torch.Tensor, 
                                beta_full: torch.Tensor) -> torch.Tensor:
        """计算科氏/离心力矩 C(q,v)v。
        
        使用 Christoffel 符号 + 自动微分。
        
        Args:
            q: (..., 4) 关节角度
            v: (..., 4) 关节速度
            beta_full: (..., 56) 完整参数
        Returns:
            (..., 4) 科氏/离心力矩
        """
        # 暂时返回零占位
        return torch.zeros(*q.shape, device=q.device, dtype=q.dtype)
    
    def inverse_dynamics(self, q: torch.Tensor, v: torch.Tensor, a: torch.Tensor,
                        beta_full: torch.Tensor) -> torch.Tensor:
        """逆动力学：计算所需力矩。
        
        tau = M(q)a + C(q,v)v + G(q) + D*v
        
        Args:
            q: (..., 4) 关节角度
            v: (..., 4) 关节速度
            a: (..., 4) 关节加速度
            beta_full: (..., 56) 完整参数
        Returns:
            (..., 4) 所需力矩
        """
        M = self.compute_inertia_matrix(q, beta_full)
        C = self.compute_coriolis_torque(q, v, beta_full)
        G = self.compute_gravity_torque(q, beta_full)
        
        # 阻尼
        D = beta_full[..., 52:56]
        
        # tau = M @ a + C + G + D * v
        tau = torch.einsum('...ij,...j->...i', M, a) + C + G + D * v
        
        return tau
    
    def forward_step(self, q: torch.Tensor, v: torch.Tensor, tau: torch.Tensor,
                    beta_full: torch.Tensor, dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """RK4 正动力学积分一步（refinement=2）。
        
        Args:
            q: (..., 4) 当前角度
            v: (..., 4) 当前速度
            tau: (..., 4) 施加力矩
            beta_full: (..., 56) 完整参数
            dt: 时间步长
        Returns:
            q_next: (..., 4) 下一时刻角度
            v_next: (..., 4) 下一时刻速度
        """
        # RK4 refinement=2：先用 dt/2 积分两次
        h = dt / 2
        
        def accel(q, v):
            M = self.compute_inertia_matrix(q, beta_full)
            C = self.compute_coriolis_torque(q, v, beta_full)
            G = self.compute_gravity_torque(q, beta_full)
            D = beta_full[..., 52:56]
            
            # a = M^{-1} (tau - C - G - D*v)
            rhs = tau - C - G - D * v
            a = torch.linalg.solve(M, rhs)
            return a
        
        # 第一次半步
        q1, v1 = q, v
        for _ in range(2):
            k1_v = v1
            k1_a = accel(q1, v1)
            
            k2_v = v1 + 0.5 * h * k1_a
            k2_a = accel(q1 + 0.5 * h * k1_v, v1 + 0.5 * h * k1_a)
            
            k3_v = v1 + 0.5 * h * k2_a
            k3_a = accel(q1 + 0.5 * h * k2_v, v1 + 0.5 * h * k2_a)
            
            k4_v = v1 + h * k3_a
            k4_a = accel(q1 + h * k3_v, v1 + h * k3_a)
            
            q1 = q1 + (h / 6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
            v1 = v1 + (h / 6) * (k1_a + 2*k2_a + 2*k3_a + k4_a)
        
        return q1, v1


def test_dynamics():
    """测试动力学实现与 C 库对标。"""
    import sys
    sys.path.insert(0, '.')
    from gimbal_dynamics import GimbalDynamics
    import params
    
    # 创建两个实例
    dyn_c = GimbalDynamics()
    dyn_torch = GimbalDynamicsTorch()
    
    # 准备真实参数
    beta_full_np = dyn_c.params_to_beta_full(
        params.MASSES, params.COM, params.INERTIA, params.DAMPING
    )
    beta_full_torch = torch.tensor(beta_full_np, dtype=torch.float64)
    
    # 测试点
    q_np = np.array([0.1, 0.2, -0.1, 0.3])
    v_np = np.array([0.5, -0.3, 0.2, -0.1])
    a_np = np.array([1.0, 0.5, -0.5, 0.3])
    
    q_torch = torch.tensor(q_np, dtype=torch.float64)
    v_torch = torch.tensor(v_np, dtype=torch.float64)
    a_torch = torch.tensor(a_np, dtype=torch.float64)
    
    # 测试逆动力学
    print("测试逆动力学...")
    tau_c = dyn_c.inverse_dynamics(q_np, v_np, a_np, beta_full_np)
    tau_torch = dyn_torch.inverse_dynamics(q_torch, v_torch, a_torch, beta_full_torch)
    
    print(f"C 库结果: {tau_c}")
    print(f"PyTorch 结果: {tau_torch.numpy()}")
    print(f"误差: {np.linalg.norm(tau_c - tau_torch.numpy())}")
    
    # 测试正动力学
    print("\n测试正动力学...")
    tau_test = np.array([0.5, 0.3, -0.2, 0.1])
    tau_test_torch = torch.tensor(tau_test, dtype=torch.float64)
    dt = 0.01
    
    q_next_c, v_next_c = dyn_c.forward_step(q_np, v_np, tau_test, beta_full_np, dt, refinement=2)
    q_next_torch, v_next_torch = dyn_torch.forward_step(q_torch, v_torch, tau_test_torch, beta_full_torch, dt)
    
    print(f"C 库 q_next: {q_next_c}")
    print(f"PyTorch q_next: {q_next_torch.numpy()}")
    print(f"q 误差: {np.linalg.norm(q_next_c - q_next_torch.numpy())}")
    print(f"C 库 v_next: {v_next_c}")
    print(f"PyTorch v_next: {v_next_torch.numpy()}")
    print(f"v 误差: {np.linalg.norm(v_next_c - v_next_torch.numpy())}")


if __name__ == '__main__':
    test_dynamics()
