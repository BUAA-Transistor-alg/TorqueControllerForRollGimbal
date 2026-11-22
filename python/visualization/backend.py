"""仿真后端：与采集脚本相同的 Environment 接口，供界面与控制器使用。

界面只依赖 gimbal_env.Environment 的 reset / step / close 三个方法，观测是包含关节角、
关节角速度、坡度、坡向、底盘状态的字典。默认后端是 100 Hz（dt = 0.01 s）的 GimbalEnv；
换成真实世界控制时实现同一个接口并在创建 Viewer 时传入即可，控制器与界面不需要改动。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gimbal_env import DT, GimbalEnv


def make_simulation_env(dt: float = DT) -> GimbalEnv:
    """默认仿真后端；接口见 gimbal_env.Environment。"""
    return GimbalEnv(dt=dt, refinement=2)
