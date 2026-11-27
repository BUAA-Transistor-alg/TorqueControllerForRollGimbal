"""滚动时域 MPC：求解器在 C++ 端，Python 只负责组装输入与调用。

模块：
  controller.py  MpcController（调用 gimbal_dynamics.MpcSolver）
  scenarios.py   闭环评估场景与被控对象
  evaluate.py    PID 与 MPC 的闭环对比、指标与图
  robustness.py  参数失真鲁棒性实验：参数、量测、执行增益与时延失真、包络扫描与报告
  check_gradient.py  控制量梯度接口的中心差分校验
"""
