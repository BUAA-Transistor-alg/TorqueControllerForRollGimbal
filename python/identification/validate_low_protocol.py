"""低激励推荐协议的种子稳健性验证：同一协议换随机种子重新采集与拟合。

用法：python3 validate_low_protocol.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from acquisition import collect_low_excitation as collector
import eval_openloop as open_loop

CANDIDATES = {
    "E1": {"slow_amp": (0.45, 0.05, 0.08, 0.08), "slow_freq": (0.5, 0.5, 0.5, 0.5),
           "fast_amp": (0.031, 0.019, 0.025, 0.025), "fast_freq": (9.0, 9.0, 9.0, 9.0),
           "tilt": 10.0, "roll_center": -0.08, "static": 0.08},
}
