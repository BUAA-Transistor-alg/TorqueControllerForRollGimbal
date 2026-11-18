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

- `1` / `2`：关节 1（大 yaw）+1 / -1 N·m
- `3` / `4`：关节 2（roll）+1 / -1 N·m
- `5` / `6`：关节 3（小 yaw）+1 / -1 N·m
- `7` / `8`：关节 4（pitch）+1 / -1 N·m
- `r`：回到零位
- `q` 或 `ESC`：退出

按住某个键时对应关节持续施加 ±1 N·m，松开后该关节指令回到 0。

## 无界面自检

```bash
python3 ui.py --direct --steps 100
python3 ui.py --direct --steps 100 --torque 1.0
```

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
