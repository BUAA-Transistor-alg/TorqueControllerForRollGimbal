"""滚动时域 MPC 控制器：求解在 C++ 端完成。

Python 只负责把当前状态、地形、底盘状态与未来目标角度交给 gimbal_dynamics.MpcSolver；
C++ 端完成代价组装、反向伴随梯度、投影 L-BFGS 与箱约束，返回本拍力矩。箱约束为各轴
±params.AXIS_TORQUE_LIMIT，取实车通信子模组 McuDataPreprocessor 中四个力矩最大值的绝对值。
热启动解与上一拍实际力矩由求解器内部维护，reset() 清除。

代价（K 为预测步数，损失按 1/K 取时间平均）：
  1/K Σ_k Σ_i [ wa_i (q_ki − q*_ki)^2 + wv_i v_ki^2
                + wt_i τ_ki^2 + wr_i (τ_ki − τ_{k−1,i})^2
                + wl_i hinge_i(q_ki)^2 ]
角速度目标恒为零，wv_i 只作为速度正则化。k = 0 的变化率项参照上一次采用的第一步力矩，
由求解器内部维护。hinge_i 为关节软限位二次铰链（关节 1 为无限位大 yaw，不设铰链；
其余关节按 gimbal_env 的硬限位向内收 0.07 rad）。

控制器参数以命名集合给出：current 为界面与实车当前使用的参数，robust 为参数失真
鲁棒性实验调出的另一套参数；两套数值各自保留，默认构造仍使用本文件顶部的默认值。
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np

import gimbal_dynamics as gd
import gimbal_env
import params

DT = gimbal_env.DT
REFINEMENT = gimbal_env.REFINEMENT

# 各轴力矩上限（N·m），取实车通信子模组 McuDataPreprocessor 中四个力矩最大值的绝对值。
TORQUE_LIMIT = np.asarray(params.AXIS_TORQUE_LIMIT, dtype=float)

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
                 torque_limit: Sequence[float] = TORQUE_LIMIT,
                 weights: dict | None = None,
                 max_iter: int = MAX_ITER, tolerance: float = TOLERANCE,
                 refinement: int = REFINEMENT, dt: float = DT,
                 thread_count: int = THREADS):
        self.model = model
        self.horizon = int(horizon)
        self.torque_limit = np.asarray(torque_limit, dtype=float)
        if self.torque_limit.shape != (4,):
            raise ValueError("torque_limit 形状必须为 (4,)")
        if (not np.all(np.isfinite(self.torque_limit))
                or np.any(self.torque_limit <= 0.0)):
            raise ValueError("torque_limit 各分量必须是大于 0 的有限值")
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


# 控制器参数集合。current 为界面与实车当前使用的参数（与
# python/visualization/controllers.py 的 HORIZON=15、MAX_ITER=8 一致），robust 为
# 参数失真鲁棒性实验调出的另一套参数；两套数值各自保留，互不覆盖。
def current_parameters() -> dict:
    """当前控制器参数：预测 15 步、每拍最多迭代 8 次，代价权重取默认值。"""
    return {"horizon": 15, "max_iter": 8, "tolerance": TOLERANCE,
            "weights": default_weights()}


def robust_parameters() -> dict:
    """鲁棒参数：在全失真条件下不发散、不震荡且跟踪误差在容差内的另一套参数。

    与当前参数相比：预测步数由 15 增至 20（0.2 s），速度、力矩与力矩变化率权重分别
    提高到 0.01、0.02、1.5，迭代上限提高到 12。降低的闭环带宽与强的指令变化率惩罚
    换来了对参数失真、执行增益误差与 0~2 拍随机控制/观测时延的稳定裕度。
    """
    weights = default_weights()
    weights["velocity_weights"] = np.array([0.01, 0.01, 0.01, 0.01])
    weights["torque_weights"] = np.array([0.02, 0.02, 0.02, 0.02])
    weights["torque_rate_weights"] = np.array([1.5, 1.5, 1.5, 1.5])
    return {"horizon": 20, "max_iter": 12, "tolerance": TOLERANCE,
            "weights": weights}


def controller_parameters(name: str) -> dict:
    """按名称返回完整控制器参数：预测步数、迭代上限、收敛阈值与代价权重。"""
    sets = {"current": current_parameters, "robust": robust_parameters}
    if name not in sets:
        raise ValueError(f"未知的控制器参数集合: {name}")
    parameters = sets[name]()
    parameters["weights"] = {
        key: (value.copy() if isinstance(value, np.ndarray) else value)
        for key, value in parameters["weights"].items()}
    return parameters


def make_controller(model: gd.TcrgModel, parameter_set: str = "current",
                    torque_limit: Sequence[float] = TORQUE_LIMIT,
                    thread_count: int = THREADS) -> MpcController:
    """按参数集合名称构造控制器。"""
    parameters = controller_parameters(parameter_set)
    return MpcController(model, torque_limit=torque_limit,
                         thread_count=thread_count, **parameters)
