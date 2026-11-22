"""ctypes 绑定：调用 C 接口中的动力学前向仿真。

库文件按以下顺序查找：
  1. 环境变量 GIMBAL_DYNAMICS_LIB
  2. 仓库根目录下的 build、build-new、build-final、build-release 中的 libgimbal_dynamics.so
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import List, Sequence

import numpy as np

import params as default_params


class TcrgModel(ctypes.Structure):
    _fields_ = [
        ("mass", ctypes.c_double * 4),
        ("com", (ctypes.c_double * 3) * 4),
        ("inertia", (ctypes.c_double * 6) * 4),
        ("joint_offsets", (ctypes.c_double * 3) * 4),
        ("damping", ctypes.c_double * 4),
        ("gravity", ctypes.c_double),
    ]


class TcrgState(ctypes.Structure):
    _fields_ = [
        ("angle", ctypes.c_double * 4),
        ("velocity", ctypes.c_double * 4),
    ]


class TcrgBaseState(ctypes.Structure):
    _fields_ = [
        ("acceleration", ctypes.c_double * 3),
        ("yaw", ctypes.c_double),
        ("yaw_rate", ctypes.c_double),
        ("yaw_acceleration", ctypes.c_double),
    ]


class TcrgTerrain(ctypes.Structure):
    _fields_ = [
        ("slope_angle", ctypes.c_double),
        ("slope_direction", ctypes.c_double),
    ]


class TcrgLossSpec(ctypes.Structure):
    _fields_ = [
        ("angle_weights", ctypes.c_double * 4),
        ("velocity_weights", ctypes.c_double * 4),
        ("torque_weights", ctypes.c_double * 4),
        ("target_angle", ctypes.POINTER(ctypes.c_double)),
        ("target_velocity", ctypes.POINTER(ctypes.c_double)),
    ]


class TcrgControlLoss(ctypes.Structure):
    """控制量梯度使用的轨迹损失构成，字段顺序与 c_api.h 的 TcrgControlLoss 一致。

    损失（K 为预测步数）：
      L = 1/K Σ_k Σ_i [ wa_i (q_ki - q*_ki)^2 + wv_i (v_ki - v*_ki)^2
                        + wt_i τ_ki^2 + wr_i (τ_ki - τ_{k-1,i})^2
                        + wl_i hinge_i(q_ki)^2 ]
          + 1/K Σ_i [ waT_i (q_Ki - q*_{K-1,i})^2 + wvT_i (v_Ki - v*_{K-1,i})^2 ]
    hinge_i(q) = q - (upper_i - margin_i)（q 高于上软限时），
                 (lower_i + margin_i) - q（q 低于下软限时），否则为 0。
    τ_{-1,i} 取 previous_torque；previous_torque 为空指针时第 0 步跳过变化率项。
    """

    _fields_ = [
        ("angle_weights", ctypes.c_double * 4),
        ("velocity_weights", ctypes.c_double * 4),
        ("torque_weights", ctypes.c_double * 4),
        ("torque_rate_weights", ctypes.c_double * 4),
        ("terminal_angle_weights", ctypes.c_double * 4),
        ("terminal_velocity_weights", ctypes.c_double * 4),
        ("limit_weights", ctypes.c_double * 4),
        ("limit_lower", ctypes.c_double * 4),
        ("limit_upper", ctypes.c_double * 4),
        ("limit_margin", ctypes.c_double * 4),
        ("target_angle", ctypes.POINTER(ctypes.c_double)),
        ("target_velocity", ctypes.POINTER(ctypes.c_double)),
        ("previous_torque", ctypes.POINTER(ctypes.c_double)),
    ]


def _find_library() -> str:
    environment = os.environ.get("GIMBAL_DYNAMICS_LIB")
    if environment:
        return environment
    root = Path(__file__).resolve().parent.parent
    candidates = [
        root / "build" / "libgimbal_dynamics.so",
        root / "build-new" / "libgimbal_dynamics.so",
        root / "build-final" / "libgimbal_dynamics.so",
        root / "build-release" / "libgimbal_dynamics.so",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        "未找到 libgimbal_dynamics.so，请先构建 gimbal_dynamics_c 目标，"
        "或用 GIMBAL_DYNAMICS_LIB 指定库路径"
    )


_lib = ctypes.CDLL(_find_library())

_lib.tcrg_context_create.argtypes = [ctypes.c_int]
_lib.tcrg_context_create.restype = ctypes.c_void_p
_lib.tcrg_context_destroy.argtypes = [ctypes.c_void_p]
_lib.tcrg_context_destroy.restype = None
_lib.tcrg_forward_step.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(TcrgModel),
    ctypes.POINTER(TcrgTerrain),
    ctypes.POINTER(TcrgBaseState),
    ctypes.POINTER(TcrgState),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_double,
    ctypes.c_int,
    ctypes.POINTER(TcrgState),
]
_lib.tcrg_forward_step.restype = ctypes.c_int
_lib.tcrg_forward_simulate.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(TcrgModel),
    ctypes.POINTER(TcrgTerrain),
    ctypes.POINTER(TcrgBaseState),
    ctypes.POINTER(TcrgState),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.POINTER(TcrgState),
]
_lib.tcrg_forward_simulate.restype = ctypes.c_int
_lib.tcrg_bias_torque.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(TcrgModel),
    ctypes.POINTER(TcrgTerrain),
    ctypes.POINTER(TcrgBaseState),
    ctypes.POINTER(TcrgState),
    ctypes.POINTER(ctypes.c_double),
]
_lib.tcrg_bias_torque.restype = ctypes.c_int
_lib.tcrg_inverse_dynamics.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(TcrgModel),
    ctypes.POINTER(TcrgTerrain),
    ctypes.POINTER(TcrgBaseState),
    ctypes.POINTER(TcrgState),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
]
_lib.tcrg_inverse_dynamics.restype = ctypes.c_int
_lib.tcrg_parameter_gradient.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(TcrgModel),
    ctypes.POINTER(TcrgTerrain),
    ctypes.POINTER(TcrgBaseState),
    ctypes.POINTER(TcrgState),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.POINTER(TcrgLossSpec),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(TcrgState),
    ctypes.POINTER(ctypes.c_double),
]
_lib.tcrg_parameter_gradient.restype = ctypes.c_int
_lib.tcrg_inertial_basis.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(TcrgModel),
    ctypes.POINTER(TcrgTerrain),
    ctypes.POINTER(TcrgBaseState),
    ctypes.POINTER(TcrgState),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
]
_lib.tcrg_inertial_basis.restype = ctypes.c_int
_lib.tcrg_last_error.argtypes = []
_lib.tcrg_last_error.restype = ctypes.c_char_p

_control_gradient_bound = False


def _bind_control_gradient() -> None:
    """首次使用时绑定控制量梯度接口。

    库尚未导出 tcrg_control_gradient 时在此处直接抛出 AttributeError，
    不使用静默降级路径。
    """
    global _control_gradient_bound
    if _control_gradient_bound:
        return
    function = _lib.tcrg_control_gradient
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(TcrgModel),
        ctypes.POINTER(TcrgTerrain),
        ctypes.POINTER(TcrgBaseState),
        ctypes.POINTER(TcrgState),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.POINTER(TcrgControlLoss),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(TcrgState),
        ctypes.POINTER(ctypes.c_double),
    ]
    function.restype = ctypes.c_int
    _control_gradient_bound = True


def _weights(values, name: str, default: float = 0.0) -> List[float]:
    if values is None:
        return [default] * 4
    vector = [float(value) for value in values]
    if len(vector) != 4:
        raise ValueError(f"{name} 长度必须为 4")
    return vector


def _error() -> RuntimeError:
    message = _lib.tcrg_last_error()
    text = message.decode("utf-8") if message else "未知错误"
    return RuntimeError(text)


def make_model(
    masses: Sequence[float] | None = None,
    com: Sequence[Sequence[float]] | None = None,
    inertia: Sequence[Sequence[float]] | None = None,
    joint_offsets: Sequence[Sequence[float]] | None = None,
    damping: Sequence[float] | None = None,
    gravity: float | None = None,
) -> TcrgModel:
    model = TcrgModel()
    masses = default_params.MASSES if masses is None else masses
    com = default_params.COM if com is None else com
    inertia = default_params.INERTIA if inertia is None else inertia
    joint_offsets = default_params.JOINT_OFFSETS if joint_offsets is None else joint_offsets
    damping = default_params.DAMPING if damping is None else damping
    gravity = default_params.GRAVITY if gravity is None else gravity
    for i in range(4):
        model.mass[i] = float(masses[i])
        for c in range(3):
            model.com[i][c] = float(com[i][c])
            model.joint_offsets[i][c] = float(joint_offsets[i][c])
        for c in range(6):
            model.inertia[i][c] = float(inertia[i][c])
        model.damping[i] = float(damping[i])
    model.gravity = float(gravity)
    return model


def make_state(angle: Sequence[float] | None = None,
               velocity: Sequence[float] | None = None) -> TcrgState:
    state = TcrgState()
    angle = [0.0] * 4 if angle is None else angle
    velocity = [0.0] * 4 if velocity is None else velocity
    for i in range(4):
        state.angle[i] = float(angle[i])
        state.velocity[i] = float(velocity[i])
    return state


def make_base(acceleration: Sequence[float] | None = None,
              yaw: float = 0.0, yaw_rate: float = 0.0,
              yaw_acceleration: float = 0.0) -> TcrgBaseState:
    base = TcrgBaseState()
    acceleration = [0.0, 0.0, 0.0] if acceleration is None else acceleration
    for c in range(3):
        base.acceleration[c] = float(acceleration[c])
    base.yaw = float(yaw)
    base.yaw_rate = float(yaw_rate)
    base.yaw_acceleration = float(yaw_acceleration)
    return base


def make_terrain(slope_angle: float = 0.0, slope_direction: float = 0.0) -> TcrgTerrain:
    terrain = TcrgTerrain()
    terrain.slope_angle = float(slope_angle)
    terrain.slope_direction = float(slope_direction)
    return terrain


class Context:
    def __init__(self, thread_count: int = 1):
        self._context = _lib.tcrg_context_create(int(thread_count))
        if not self._context:
            raise _error()

    def close(self) -> None:
        if getattr(self, "_context", None):
            _lib.tcrg_context_destroy(self._context)
            self._context = None

    def __enter__(self) -> "Context":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def forward_step(
        self,
        model: TcrgModel,
        terrain: TcrgTerrain,
        base: TcrgBaseState,
        state: TcrgState,
        torque: Sequence[float],
        step: float = 0.004,
        refinement: int = 2,
    ) -> TcrgState:
        torque_array = (ctypes.c_double * 4)(*[float(t) for t in torque])
        next_state = TcrgState()
        result = _lib.tcrg_forward_step(
            self._context,
            ctypes.byref(model),
            ctypes.byref(terrain),
            ctypes.byref(base),
            ctypes.byref(state),
            torque_array,
            ctypes.c_double(float(step)),
            ctypes.c_int(int(refinement)),
            ctypes.byref(next_state),
        )
        if result != 0:
            raise _error()
        return next_state

    def bias_torque(self, model: TcrgModel, terrain: TcrgTerrain,
                    base: TcrgBaseState, state: TcrgState) -> List[float]:
        values = (ctypes.c_double * 4)()
        result = _lib.tcrg_bias_torque(
            self._context,
            ctypes.byref(model),
            ctypes.byref(terrain),
            ctypes.byref(base),
            ctypes.byref(state),
            values,
        )
        if result != 0:
            raise _error()
        return [values[i] for i in range(4)]

    def parameter_gradient(
        self,
        model: TcrgModel,
        terrain: TcrgTerrain,
        base: TcrgBaseState,
        initial: TcrgState,
        torques: Sequence[Sequence[float]],
        target_angle: Sequence[Sequence[float]] | None,
        target_velocity: Sequence[Sequence[float]] | None,
        angle_weights: Sequence[float],
        velocity_weights: Sequence[float],
        torque_weights: Sequence[float],
        step: float,
        refinement: int,
    ):
        count = len(torques)
        torque_array = (ctypes.c_double * (count * 4))(
            *[float(value) for row in torques for value in row]
        )
        angle_array = None
        velocity_array = None
        if target_angle is not None:
            angle_array = (ctypes.c_double * (count * 4))(
                *[float(value) for row in target_angle for value in row]
            )
        if target_velocity is not None:
            velocity_array = (ctypes.c_double * (count * 4))(
                *[float(value) for row in target_velocity for value in row]
            )
        loss_spec = TcrgLossSpec()
        for i in range(4):
            loss_spec.angle_weights[i] = float(angle_weights[i])
            loss_spec.velocity_weights[i] = float(velocity_weights[i])
            loss_spec.torque_weights[i] = float(torque_weights[i])
        if angle_array is not None:
            loss_spec.target_angle = ctypes.cast(angle_array, ctypes.POINTER(ctypes.c_double))
        if velocity_array is not None:
            loss_spec.target_velocity = ctypes.cast(velocity_array, ctypes.POINTER(ctypes.c_double))
        gradient_array = (ctypes.c_double * 56)()
        loss_value = ctypes.c_double()
        final_state = TcrgState()
        result = _lib.tcrg_parameter_gradient(
            self._context,
            ctypes.byref(model),
            ctypes.byref(terrain),
            ctypes.byref(base),
            ctypes.byref(initial),
            torque_array,
            ctypes.c_size_t(count),
            ctypes.c_double(float(step)),
            ctypes.c_int(int(refinement)),
            ctypes.byref(loss_spec),
            ctypes.byref(loss_value),
            ctypes.byref(final_state),
            gradient_array,
        )
        if result != 0:
            raise _error()
        return loss_value.value, [gradient_array[i] for i in range(56)], final_state

    def inertial_basis(self, model: TcrgModel, terrain: TcrgTerrain,
                       base: TcrgBaseState, state: TcrgState,
                       qdd: Sequence[float]) -> List[List[float]]:
        acceleration = (ctypes.c_double * 4)(*[float(value) for value in qdd])
        values = (ctypes.c_double * (44 * 4))()
        result = _lib.tcrg_inertial_basis(
            self._context,
            ctypes.byref(model),
            ctypes.byref(terrain),
            ctypes.byref(base),
            ctypes.byref(state),
            acceleration,
            values,
        )
        if result != 0:
            raise _error()
        return [[values[column * 4 + row] for row in range(4)] for column in range(44)]

    def inverse_dynamics(self, model: TcrgModel, terrain: TcrgTerrain,
                         base: TcrgBaseState, state: TcrgState,
                         qdd: Sequence[float]) -> List[float]:
        acceleration = (ctypes.c_double * 4)(*[float(value) for value in qdd])
        values = (ctypes.c_double * 4)()
        result = _lib.tcrg_inverse_dynamics(
            self._context,
            ctypes.byref(model),
            ctypes.byref(terrain),
            ctypes.byref(base),
            ctypes.byref(state),
            acceleration,
            values,
        )
        if result != 0:
            raise _error()
        return [values[i] for i in range(4)]

    def forward_simulate(
        self,
        model: TcrgModel,
        terrain: TcrgTerrain,
        base: TcrgBaseState,
        initial: TcrgState,
        torques: Sequence[Sequence[float]],
        step: float = 0.004,
        refinement: int = 2,
    ) -> TcrgState:
        count = len(torques)
        if count == 0:
            raise ValueError("torques 不能为空")
        flat = (ctypes.c_double * (count * 4))(*[float(t) for row in torques for t in row])
        final_state = TcrgState()
        result = _lib.tcrg_forward_simulate(
            self._context,
            ctypes.byref(model),
            ctypes.byref(terrain),
            ctypes.byref(base),
            ctypes.byref(initial),
            flat,
            ctypes.c_size_t(count),
            ctypes.c_double(float(step)),
            ctypes.c_int(int(refinement)),
            ctypes.byref(final_state),
        )
        if result != 0:
            raise _error()
        return final_state

    def control_gradient(
        self,
        model: TcrgModel,
        terrain: TcrgTerrain,
        base,
        initial: TcrgState,
        torques: Sequence[Sequence[float]],
        target_angle: Sequence[Sequence[float]] | None = None,
        target_velocity: Sequence[Sequence[float]] | None = None,
        angle_weights: Sequence[float] | None = None,
        velocity_weights: Sequence[float] | None = None,
        torque_weights: Sequence[float] | None = None,
        torque_rate_weights: Sequence[float] | None = None,
        terminal_angle_weights: Sequence[float] | None = None,
        terminal_velocity_weights: Sequence[float] | None = None,
        limit_weights: Sequence[float] | None = None,
        limit_lower: Sequence[float] | None = None,
        limit_upper: Sequence[float] | None = None,
        limit_margin: Sequence[float] | None = None,
        previous_torque: Sequence[float] | None = None,
        step: float = 0.01,
        refinement: int = 2,
    ):
        """一次调用同时返回轨迹损失、损失对控制量的梯度与末端状态。

        torques 形状为 (num_steps, 4)；target_angle/target_velocity 形状为 (num_steps, 4)，
        为空表示该组跟踪项不参与；base 为单个 TcrgBaseState（广播到整个预测区间）
        或长度为 num_steps 的序列；previous_torque 为上一拍实际施加的力矩。
        返回 (loss, gradient[num_steps, 4], final_state)。
        """
        _bind_control_gradient()
        count = len(torques)
        if count == 0:
            raise ValueError("torques 不能为空")
        torque_array = (ctypes.c_double * (count * 4))(
            *[float(value) for row in torques for value in row])
        angle_array = None
        velocity_array = None
        if target_angle is not None:
            angle_values = [float(value) for row in target_angle for value in row]
            if len(angle_values) != count * 4:
                raise ValueError("target_angle 的形状必须为 (num_steps, 4)")
            angle_array = (ctypes.c_double * (count * 4))(*angle_values)
        if target_velocity is not None:
            velocity_values = [float(value) for row in target_velocity for value in row]
            if len(velocity_values) != count * 4:
                raise ValueError("target_velocity 的形状必须为 (num_steps, 4)")
            velocity_array = (ctypes.c_double * (count * 4))(*velocity_values)
        if isinstance(base, TcrgBaseState):
            base_array = (TcrgBaseState * count)(*[base] * count)
        else:
            base_values = list(base)
            if len(base_values) != count:
                raise ValueError("base 序列长度必须等于 num_steps")
            base_array = (TcrgBaseState * count)(*base_values)
        loss_spec = TcrgControlLoss()
        weight_fields = (
            ("angle_weights", angle_weights),
            ("velocity_weights", velocity_weights),
            ("torque_weights", torque_weights),
            ("torque_rate_weights", torque_rate_weights),
            ("terminal_angle_weights", terminal_angle_weights),
            ("terminal_velocity_weights", terminal_velocity_weights),
            ("limit_weights", limit_weights),
        )
        for name, values in weight_fields:
            field = getattr(loss_spec, name)
            for joint, value in enumerate(_weights(values, name)):
                field[joint] = value
        for joint, value in enumerate(_weights(limit_lower, "limit_lower",
                                               float("-inf"))):
            loss_spec.limit_lower[joint] = value
        for joint, value in enumerate(_weights(limit_upper, "limit_upper",
                                               float("inf"))):
            loss_spec.limit_upper[joint] = value
        for joint, value in enumerate(_weights(limit_margin, "limit_margin")):
            loss_spec.limit_margin[joint] = value
        if angle_array is not None:
            loss_spec.target_angle = ctypes.cast(angle_array,
                                                 ctypes.POINTER(ctypes.c_double))
        if velocity_array is not None:
            loss_spec.target_velocity = ctypes.cast(velocity_array,
                                                    ctypes.POINTER(ctypes.c_double))
        previous_array = None
        if previous_torque is not None:
            previous_array = (ctypes.c_double * 4)(
                *[float(value) for value in previous_torque])
            loss_spec.previous_torque = ctypes.cast(previous_array,
                                                    ctypes.POINTER(ctypes.c_double))
        loss_value = ctypes.c_double()
        final_state = TcrgState()
        gradient_array = (ctypes.c_double * (count * 4))()
        result = _lib.tcrg_control_gradient(
            self._context,
            ctypes.byref(model),
            ctypes.byref(terrain),
            base_array,
            ctypes.byref(initial),
            torque_array,
            ctypes.c_size_t(count),
            ctypes.c_double(float(step)),
            ctypes.c_int(int(refinement)),
            ctypes.byref(loss_spec),
            ctypes.byref(loss_value),
            ctypes.byref(final_state),
            gradient_array,
        )
        if result != 0:
            raise _error()
        gradient = np.ctypeslib.as_array(gradient_array).copy().reshape(count, 4)
        return loss_value.value, gradient, final_state


def state_to_lists(state: TcrgState) -> tuple[List[float], List[float]]:
    return [state.angle[i] for i in range(4)], [state.velocity[i] for i in range(4)]
