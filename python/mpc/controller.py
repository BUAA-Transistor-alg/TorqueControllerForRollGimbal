"""滚动时域 MPC 控制器。

每个控制拍求解一个 K 步有限时域最优控制问题，决策变量为 K×4 个关节力矩，
箱约束为 ±torque_limit；求解器为 scipy L-BFGS-B。目标函数与梯度全部由
C 接口 tcrg_control_gradient 一次调用给出（反向伴随法），Python 侧不手算轨迹梯度。

代价构成（K 为预测步数，损失已按 1/K 做时间平均）：
  1/K Σ_k Σ_i [ wa_i (q_ki - q*_ki)^2 + wv_i (v_ki - v*_ki)^2
                + wt_i τ_ki^2 + wr_i (τ_ki - τ_{k-1,i})^2
                + wl_i hinge_i(q_ki)^2 ]
  + 1/K Σ_i [ waT_i (q_Ki - q*_{K-1,i})^2 + wvT_i (v_Ki - v*_{K-1,i})^2 ]
其中 τ_{-1,i} 为上一拍实际施加的力矩，hinge_i 为关节软限位二次铰链
（关节 1 为无限位大 yaw，不设铰链；其余关节按 gimbal_env 的硬限位向内收 0.07 rad）。
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

import gimbal_dynamics as gd
import gimbal_env

DT = gimbal_env.DT
REFINEMENT = gimbal_env.REFINEMENT
TORQUE_LIMIT = 8.0
HORIZON = 20
LIMIT_MARGIN = 0.07

LIMITS = gimbal_env.JOINT_LIMITS
LIMIT_LOWER = np.array([-np.inf if limits[0] is None else limits[0]
                        for limits in LIMITS])
LIMIT_UPPER = np.array([np.inf if limits[1] is None else limits[1]
                        for limits in LIMITS])


def default_weights() -> dict:
    """默认代价权重。终端权重相对逐步跟踪权重放大，限位项取高权重把轨迹推离挡块。"""
    return {
        "angle_weights": np.array([1.0, 1.0, 1.0, 1.0]),
        "velocity_weights": np.array([0.05, 0.05, 0.05, 0.05]),
        "torque_weights": np.array([0.001, 0.001, 0.001, 0.001]),
        "torque_rate_weights": np.array([0.01, 0.01, 0.01, 0.01]),
        "terminal_angle_weights": np.array([30.0, 30.0, 30.0, 30.0]),
        "terminal_velocity_weights": np.array([3.0, 3.0, 3.0, 3.0]),
        "limit_weights": np.array([0.0, 200.0, 200.0, 200.0]),
        "limit_lower": LIMIT_LOWER,
        "limit_upper": LIMIT_UPPER,
        "limit_margin": np.full(4, LIMIT_MARGIN),
    }


class MpcController:
    """滚动时域力矩控制器。

    solve() 返回本拍施加的力矩（4 维），并保留上一拍解用于下一刻热启动。
    """

    def __init__(self, context: gd.Context, model: gd.TcrgModel,
                 horizon: int = HORIZON, torque_limit: float = TORQUE_LIMIT,
                 weights: dict | None = None, max_iter: int = 30,
                 refinement: int = REFINEMENT, dt: float = DT):
        self.context = context
        self.model = model
        self.horizon = int(horizon)
        self.torque_limit = float(torque_limit)
        self.weights = default_weights() if weights is None else dict(weights)
        self.max_iter = int(max_iter)
        self.refinement = int(refinement)
        self.dt = float(dt)
        self._solution = None
        self._previous_torque = np.zeros(4)
        self.last_info = {}

    @property
    def last_torque(self) -> np.ndarray:
        return self._previous_torque.copy()

    def reset(self) -> None:
        """清空热启动解与上一拍力矩，供界面复位或回合切换使用。"""
        self._solution = None
        self._previous_torque = np.zeros(4)
        self.last_info = {}

    def _warm_start(self, initial: gd.TcrgState, terrain: gd.TcrgTerrain,
                    base: gd.TcrgBaseState) -> np.ndarray:
        """上一拍解左移一位、末位保持；首拍无解时用当前状态的偏置力矩铺满预测区间。"""
        if self._solution is not None:
            start = np.empty_like(self._solution)
            start[:-1] = self._solution[1:]
            start[-1] = self._solution[-1]
            return start
        bias = np.asarray(self.context.bias_torque(self.model, terrain, base, initial))
        return np.tile(bias, (self.horizon, 1))

    def command(self, angle, velocity, target_angle, target_velocity,
                terrain: gd.TcrgTerrain, base: gd.TcrgBaseState) -> np.ndarray:
        """与 PID 控制器共用的调用接口，等价于 solve。"""
        return self.solve(angle, velocity, target_angle, target_velocity, terrain, base)

    def solve(self, angle, velocity, target_angle, target_velocity,
              terrain: gd.TcrgTerrain, base: gd.TcrgBaseState) -> np.ndarray:
        angle = np.asarray(angle, dtype=float).reshape(4)
        velocity = np.asarray(velocity, dtype=float).reshape(4)
        target_angle = np.asarray(target_angle, dtype=float)
        target_velocity = np.asarray(target_velocity, dtype=float)
        if target_angle.shape != (self.horizon, 4):
            raise ValueError(f"target_angle 形状必须为 ({self.horizon}, 4)")
        if target_velocity.shape != (self.horizon, 4):
            raise ValueError(f"target_velocity 形状必须为 ({self.horizon}, 4)")
        initial = gd.make_state(angle, velocity)
        limit = self.torque_limit
        cache: dict = {}
        call_count = [0]

        def evaluate(flat: np.ndarray) -> dict:
            # fun 与 jac 共享缓存：同一迭代点只调用一次 C 接口
            if "flat" not in cache or not np.array_equal(cache["flat"], flat):
                call_count[0] += 1
                torques = np.asarray(flat, dtype=float).reshape(self.horizon, 4)
                loss, gradient, _ = self.context.control_gradient(
                    self.model, terrain, base, initial, torques,
                    target_angle=target_angle,
                    target_velocity=target_velocity,
                    previous_torque=self._previous_torque,
                    step=self.dt,
                    refinement=self.refinement,
                    **self.weights,
                )
                cache["flat"] = np.array(flat, dtype=float, copy=True)
                cache["loss"] = float(loss)
                cache["gradient"] = np.asarray(gradient, dtype=float).reshape(-1)
            return cache

        def fun(flat: np.ndarray) -> float:
            return evaluate(flat)["loss"]

        def jac(flat: np.ndarray) -> np.ndarray:
            return evaluate(flat)["gradient"]

        start = np.clip(self._warm_start(initial, terrain, base), -limit, limit)
        started = time.perf_counter()
        result = minimize(
            fun,
            start.reshape(-1),
            jac=jac,
            method="L-BFGS-B",
            bounds=[(-limit, limit)] * (self.horizon * 4),
            options={"maxiter": self.max_iter, "maxcor": 10, "ftol": 1e-12,
                     "gtol": 1e-6, "maxls": 30},
        )
        elapsed = time.perf_counter() - started
        solution = np.clip(np.asarray(result.x, dtype=float).reshape(self.horizon, 4),
                           -limit, limit)
        self._solution = solution
        command = solution[0].copy()
        self._previous_torque = command.copy()
        self.last_info = {
            "success": bool(result.success),
            "iterations": int(result.nit),
            "evaluations": int(result.nfev),
            "loss": float(result.fun),
            "solve_time": elapsed,
            "gradient_norm": float(np.linalg.norm(np.asarray(result.jac, dtype=float))),
            "calls": call_count[0],
        }
        return command
