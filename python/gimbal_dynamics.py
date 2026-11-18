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
_lib.tcrg_last_error.argtypes = []
_lib.tcrg_last_error.restype = ctypes.c_char_p


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


def state_to_lists(state: TcrgState) -> tuple[List[float], List[float]]:
    return [state.angle[i] for i in range(4)], [state.velocity[i] for i in range(4)]
