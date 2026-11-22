"""用 C 接口的伴随控制量梯度构造滚动时域 MPC 控制器。

模块：
  controller.py  MpcController
  scenarios.py   闭环评估场景与被控对象
  evaluate.py    PID 与 MPC 的闭环对比、指标与图
  check_gradient.py  控制量梯度的中心差分校验
"""
