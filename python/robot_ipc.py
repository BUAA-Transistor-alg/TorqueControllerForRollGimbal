"""s2c ipc_client 的 ctypes 绑定：读取实车快照并写入四轴纯力矩。

库文件按以下顺序查找：
  1. 环境变量 GIMBAL_ROBOT_IPC_LIB
  2. 仓库根目录下的 build/librobot_ipc.so、build-new/librobot_ipc.so、
     build-final/librobot_ipc.so、build-release/librobot_ipc.so

找不到时抛出 FileNotFoundError，需要先构建 robot_ipc_c 目标。
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path


class TcrgRobotObservation(ctypes.Structure):
    """一帧实车观测，字段顺序与 robot_ipc.h 的 TcrgRobotObservation 一致。"""

    _fields_ = [
        ("angle", ctypes.c_double * 4),
        ("velocity", ctypes.c_double * 4),
        ("torque", ctypes.c_double * 4),
        ("slope_angle", ctypes.c_double),
        ("slope_direction", ctypes.c_double),
        ("base_yaw", ctypes.c_double),
        ("base_yaw_rate", ctypes.c_double),
        ("base_yaw_acceleration", ctypes.c_double),
        ("base_acceleration", ctypes.c_double * 3),
        ("server_time", ctypes.c_double),
        ("mcu_age", ctypes.c_double),
        ("imu_age", ctypes.c_double),
    ]


def _find_library() -> str:
    environment = os.environ.get("GIMBAL_ROBOT_IPC_LIB")
    if environment:
        return environment
    root = Path(__file__).resolve().parent.parent
    candidates = [
        root / "build" / "librobot_ipc.so",
        root / "build-new" / "librobot_ipc.so",
        root / "build-final" / "librobot_ipc.so",
        root / "build-release" / "librobot_ipc.so",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        "未找到 librobot_ipc.so，请先构建 robot_ipc_c 目标，"
        "或用 GIMBAL_ROBOT_IPC_LIB 指定库路径"
    )


_lib = ctypes.CDLL(_find_library())

_lib.tcrg_robot_ipc_create.argtypes = [ctypes.c_char_p, ctypes.c_double]
_lib.tcrg_robot_ipc_create.restype = ctypes.c_void_p
_lib.tcrg_robot_ipc_destroy.argtypes = [ctypes.c_void_p]
_lib.tcrg_robot_ipc_destroy.restype = None
_lib.tcrg_robot_ipc_reset.argtypes = [ctypes.c_void_p]
_lib.tcrg_robot_ipc_reset.restype = None
_lib.tcrg_robot_ipc_update.argtypes = [ctypes.c_void_p,
                                       ctypes.POINTER(TcrgRobotObservation)]
_lib.tcrg_robot_ipc_update.restype = ctypes.c_int
_lib.tcrg_robot_ipc_write_torque.argtypes = [ctypes.c_void_p,
                                             ctypes.POINTER(ctypes.c_double)]
_lib.tcrg_robot_ipc_write_torque.restype = ctypes.c_int
_lib.tcrg_robot_ipc_last_error.argtypes = []
_lib.tcrg_robot_ipc_last_error.restype = ctypes.c_char_p


def _error() -> RuntimeError:
    message = _lib.tcrg_robot_ipc_last_error()
    text = message.decode("utf-8") if message else "未知错误"
    return RuntimeError(text)


class Robot:
    """实车控制器句柄，配合 RealGimbalEnv 使用。"""

    def __init__(self, shm_name: str = "/s2c_serial_comm",
                 sample_timeout: float = 0.5):
        handle = _lib.tcrg_robot_ipc_create(shm_name.encode("utf-8"),
                                            ctypes.c_double(float(sample_timeout)))
        if not handle:
            raise _error()
        self._handle = handle

    def _require(self):
        if not self._handle:
            raise RuntimeError("Robot 已关闭")
        return self._handle

    def reset(self) -> None:
        """清除反解的连续量状态，下一次 update 重新开始解卷绕与角加速度差分。"""
        _lib.tcrg_robot_ipc_reset(self._require())

    def update(self) -> TcrgRobotObservation:
        """读取一帧观测并反解；失败时抛出 RuntimeError。"""
        observation = TcrgRobotObservation()
        result = _lib.tcrg_robot_ipc_update(self._require(), ctypes.byref(observation))
        if result != 0:
            raise _error()
        return observation

    def write_torque(self, torque) -> None:
        """写入四个轴的纯力矩控制，torque 为长度 4 的可迭代对象。"""
        values = [float(value) for value in torque]
        if len(values) != 4:
            raise ValueError("torque 长度必须为 4")
        command = (ctypes.c_double * 4)(*values)
        result = _lib.tcrg_robot_ipc_write_torque(self._require(), command)
        if result != 0:
            raise _error()

    def close(self) -> None:
        """释放句柄，重复调用不产生副作用。"""
        handle = getattr(self, "_handle", None)
        if not handle:
            return
        self._handle = None
        _lib.tcrg_robot_ipc_destroy(handle)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
