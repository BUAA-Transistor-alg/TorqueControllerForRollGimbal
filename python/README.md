# python 端动力学仿真接口

本目录包含：

- `gimbal_dynamics.py`：通过 ctypes 调用 C 接口（`include/gimbal_dynamics/c_api.h`）的封装。
- `params.py`：由 SolidWorks 质量属性换算得到的四连杆参数，阻尼暂取 0.1 N·m·s/rad。
- `ui.py`：PyBullet 可视化与键盘交互，PyBullet 只负责显示，动力学由本项目库计算。

## 构建共享库

在仓库根目录执行：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target gimbal_dynamics_c -j
```

产物为 `build/libgimbal_dynamics.so`。也可以用 `build-new`、`build-final`、`build-release`，
或者用环境变量指定：

```bash
export GIMBAL_DYNAMICS_LIB=/path/to/libgimbal_dynamics.so
```

## 依赖

```bash
python3 -m pip install pybullet numpy
```

## 运行界面

```bash
cd python
python3 ui.py
```

键盘：

- `1` / `2`：关节 1（大 yaw）+1 / -1 N·m（力矩模式）或目标位置 + / -（PID 模式）
- `3` / `4`：关节 2（roll）
- `5` / `6`：关节 3（小 yaw）
- `7` / `8`：关节 4（pitch）
- `m`：在直接力矩模式与 PID 目标模式之间切换
- `r`：回到零位并清空 PID 状态
- `q` 或 `ESC`：退出

力矩模式：按住某个键时对应关节持续施加 ±1 N·m，松开后回到 0。

PID 模式：按住某个键时对应关节的位置目标以 `params.PID_TARGET_RATE`（默认 0.5 rad/s）
的速度移动，松开后目标停止；PID 根据位置误差计算力矩，并叠加由 C 接口
`tcrg_bias_torque` 给出的重力/底盘运动前馈。PID 增益、目标限幅、力矩限幅均在
`params.py` 中设置。切换模式时目标会重置为当前关节角，不会产生跳变。

启动时默认进入力矩模式；可以用 `--mode pid` 直接以 PID 模式启动，用
`--target-rate` 修改目标速度。

## 无界面自检

```bash
python3 ui.py --direct --steps 100
python3 ui.py --direct --steps 100 --torque 1.0
python3 ui.py --direct --steps 100 --mode pid --torque 1.0
```

`--direct` 模式没有键盘事件，`--torque` 在力矩模式下表示恒定力矩，在 PID 模式下表示
按住对应正方向键（例如 1.0 表示持续按住 `1`）。

## 直接调用前向仿真

```python
import gimbal_dynamics as gd

context = gd.Context(1)
model = gd.make_model()
terrain = gd.make_terrain()
base = gd.make_base()
state = gd.make_state()

# 推进 0.02 s，每个主步 0.004 s，内部 2 个子步
state = context.forward_simulate(
    model, terrain, base, state,
    torques=[[0.0, 0.0, 0.0, 0.0]] * 5,
    step=0.004, refinement=2,
)
print(gd.state_to_lists(state))
```

## 参数说明

`params.py` 中的惯量是绕质心的标准惯性张量（非对角项为 `-∫x_i x_j dm`），
坐标为本项目坐标轴（x 向右、y 向前、z 向上），单位为 kg·m²；质心相对各自关节原点。
阻尼 `DAMPING` 目前统一为 0.1，需要时可直接修改该列表。
