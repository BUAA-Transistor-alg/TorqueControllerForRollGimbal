"""控制器共用的滚动目标序列。

按键只按固定速度移动当前目标。每拍把当前目标压入序列尾端，最旧的元素被挤出：序列第一个
元素是当前拍使用的参考目标，其余元素是后续各拍的预测目标。除新压入的尾端外，已有元素
的数值不发生变化，因此预测目标是一批目标点随时间滚动，而不是由历史斜率外推得到。
"""

from __future__ import annotations

from collections import deque

import numpy as np


class TargetSequence:
    """长度固定为 steps + 1 的目标点滚动缓冲。"""

    def __init__(self, steps: int, target):
        self.steps = int(steps)
        if self.steps < 1:
            raise ValueError("预测步数必须是正整数")
        self._points = deque(maxlen=self.steps + 1)
        self.reset(target)

    def reset(self, target) -> None:
        """用当前目标填满序列，供回合开始与模式切换使用。"""
        initial = np.asarray(target, dtype=float)
        self._points = deque([initial.copy() for _ in range(self.steps + 1)],
                             maxlen=self.steps + 1)

    def push(self, target) -> None:
        """压入当前目标，最旧的元素被挤出。"""
        self._points.append(np.asarray(target, dtype=float).copy())

    @property
    def points(self) -> np.ndarray:
        """全部目标点，按时间从旧到新排列。"""
        return np.stack(self._points)

    @property
    def reference(self) -> np.ndarray:
        """当前拍使用的参考目标，即序列最旧的元素。"""
        return self._points[0].copy()

    @property
    def predictions(self) -> np.ndarray:
        """后续各拍的预测目标，最后一个元素是刚压入的当前目标。"""
        return self.points[1:]

    def relative_preview(self, dt: float):
        """返回参考目标、参考目标速度（每拍差值）与相对参考目标的预瞄偏移。"""
        points = self.points
        reference = points[0].copy()
        rate = (points[1] - reference) / float(dt)
        offsets = (points[1:] - reference[None, :]).reshape(-1)
        return reference, rate, offsets
