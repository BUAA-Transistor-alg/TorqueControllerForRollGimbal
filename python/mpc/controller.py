"""滚动时域 MPC 控制器：求解在 C++ 端完成。

Python 只负责把当前状态、地形、底盘状态与未来目标角度交给 gimbal_dynamics.MpcSolver；
C++ 端完成代价组装、反向伴随梯度、投影 L-BFGS 与箱约束，返回本拍力矩。热启动解与上一拍
实际力矩由求解器内部维护，reset() 清除。

代价（K 为预测步数，损失按 1/K 取时间平均）：
  1/K Σ_k Σ_i [ wa_i (q_ki − q*_ki)^2 + wv_i v_ki^2
                + wt_i τ_ki^2 + wr_i (τ_ki − τ_{k−1,i})^2
                + wl_i hinge_i(q_ki)^2 ]
角速度目标恒为零，wv_i 只作为速度正则化。k = 0 的变化率项参照上一次采用的第一步力矩，
由求解器内部维护。hinge_i 为关节软限位二次铰链（关节 1 为无限位大 yaw，不设铰链；
其余关节按 gimbal_env 的硬限位向内收 0.07 rad）。
"""

from __future__ import annotations

import time

import numpy as np

import gimbal_dynamics as gd
import gimbal_env

DT = gimbal_env.DT
REFINEMENT = gimbal_env.REFINEMENT
TORQUE_LIMIT = 8.0
HORIZON = 20
LIMIT_MARGIN = 0.07
MAX_ITER = 30
TOLERANCE = 1e-4
THREADS = 4

LIMITS = gimbal_env.JOINT_LIMITS
LIMIT_LOWER = np.array([-np.inf if limits[0] is None else limits[0]
                        for limits in LIMITS])
LIMIT_UPPER = np.array([np.inf if limits[1] is None else limits[1]
                        for limits in LIMITS])


def default_weights() -> dict:
    """默认代价权重。速度项是零目标的正则化；限位项取高权重把轨迹推离挡块。"""
    return {
        "angle_weights": np.array([1.0, 1.0, 1.0, 1.0]),
        "velocity_weights": np.array([0.001, 0.001, 0.001, 0.001]),
        "torque_weights": np.array([0.001, 0.001, 0.001, 0.001]),
        "torque_rate_weights": np.array([0.01, 0.01, 0.01, 0.01]),
        "limit_weights": np.array([0.0, 200.0, 200.0, 200.0]),
        "limit_lower": LIMIT_LOWER,
        "limit_upper": LIMIT_UPPER,
        "limit_margin": np.full(4, LIMIT_MARGIN),
    }


class MpcController:
    """滚动时域力矩控制器：每个控制拍调用一次 C++ 求解器。

    solve() 返回本拍施加的力矩（4 维），求解器内部保留完整解用于下一刻热启动。
    """

    def __init__(self, model: gd.TcrgModel, horizon: int = HORIZON,
                 torque_limit: float = TORQUE_LIMIT, weights: dict | None = None,
                 max_iter: int = MAX_ITER, tolerance: float = TOLERANCE,
                 refinement: int = REFINEMENT, dt: float = DT,
                 thread_count: int = THREADS):
        self.model = model
        self.horizon = int(horizon)
        self.torque_limit = float(torque_limit)
        self.weights = default_weights() if weights is None else dict(weights)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.refinement = int(refinement)
        self.dt = float(dt)
        self.solver = gd.MpcSolver(
            model, horizon=self.horizon, step=self.dt, refinement=self.refinement,
            torque_limit=self.torque_limit, max_iter=self.max_iter,
            tolerance=self.tolerance, weights=self.weights,
            thread_count=int(thread_count))
        self.last_info = {}
        self._last_command = np.zeros(4)

    @property
    def last_torque(self) -> np.ndarray:
        return self._last_command.copy()

    def reset(self) -> None:
        """清空热启动解与上一拍力矩，供回合切换使用。"""
        self.solver.reset()
        self.last_info = {}
        self._last_command = np.zeros(4)

    def command(self, angle, velocity, target_angle,
                terrain: gd.TcrgTerrain, base: gd.TcrgBaseState) -> np.ndarray:
        """与 PID 控制器共用的调用接口，等价于 solve。"""
        return self.solve(angle, velocity, target_angle, terrain, base)

    def solve(self, angle, velocity, target_angle,
              terrain: gd.TcrgTerrain, base: gd.TcrgBaseState) -> np.ndarray:
        target_angle = np.asarray(target_angle, dtype=float)
        if target_angle.shape != (self.horizon, 4):
            raise ValueError(f"target_angle 形状必须为 ({self.horizon}, 4)")
        initial = gd.make_state(angle, velocity)
        started = time.perf_counter()
        command, info = self.solver.solve(initial, terrain, base, target_angle)
        info["solve_time"] = time.perf_counter() - started
        self.last_info = info
        self._last_command = np.asarray(command, dtype=float)
        return self._last_command

    def close(self) -> None:
        self.solver.close()
