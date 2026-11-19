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

PID 模式：按住某个键时对应关节的位置目标以 `params.PID_TARGET_RATE`（默认 1.5 rad/s）
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

## 限位数据采集与参数辨识

采集脚本 `collect_data.py` 只在数据采集时施加关节限位，不修改动力学模型：

- roll：`[-45°, +5°]`，小 yaw：`[-50°, +50°]`，pitch：`[-45°, +45°]`；大 yaw 不限制。
- 每段 3 s，300 个时间点，`dt = 0.01 s`。
- 段内底盘不移动、不旋转，仅通过坡角与坡向给出不超过 20° 的底盘倾角。
- 记录每个时间点的关节力矩、角度、角速度，以及底盘状态与初始状态。

```bash
cd python
python3 collect_data.py --segments 24
```

输出为 `python/data/identification_data.npz`。采集时先用逆动力学按参考轨迹生成力矩，再用同一组力矩推进仿真并检查实际关节角是否仍在限位内；超限的段会重新生成。

辨识脚本使用同一条逆动力学残差：

```bash
python3 identify.py --analysis                 # 真值残差、残差雅可比秩与零空间
python3 identify.py --fit --segments 8 --stride 4 --max-nfev 25
```

在 24 段数据上，真值处残差最大约 `8.5e-3 N·m`，主要来自 0.01 s 采样下用差分求角加速度的离散误差。残差雅可比在采样数据上的秩约为 `33 / 56`，零空间约 23 维；`o1.x`、`o1.y`、`o1.z`、`o2.z`、`B1.com_z`、`B1` 的五个非轴向惯量等方向完全不被这批静态底盘数据激发，B2、B3 的部分质心与惯量也只能以组合形式出现。

最小二乘拟合从真值附近扰动出发，8 段数据、25 次函数评估后残差最大约 `2.3e-3 N·m`，代价从 `8.1` 降到 `5.9e-4`；可辨识方向上的参数（如 B1.com_y、B3.com_z、B4 的部分惯量、d2）能收敛到真值附近，零空间方向上的参数则会停在初值附近，说明仅靠这批限位数据无法唯一确定全部 56 个参数。


## 基参数重参数化

`identify_base.py` 用结构回归矩阵构造 35 个基参数：

- 在丰富虚拟激励下（关节大范围、底盘平移加速度、偏航角速度/角加速度、不同坡角坡向）对逆动力学关于 56 个原始参数求数值雅可比；
- 对归一化雅可比做 SVD，秩为 `35 / 56`，取前 35 个右奇异向量组成 `U`；
- 基参数 `beta` 与原参数 `x` 的关系为：

```text
x(beta) = x_CAD + scale * U * (beta - beta0)
beta0 = U^T (x_CAD / scale)
```

其中 `x_CAD` 为 SolidWorks 换算得到的先验，`scale` 为各参数尺度；`U` 的零空间分量固定为先验值。这样 56 个原始参数全部写成 35 个基参数的线性函数。

```bash
python3 identify_base.py --export-mapping data/base_mapping.npz \
    --segments 8 --stride 4 --max-nfev 25
```

映射矩阵 `U`、`beta0`、`scale` 与参数名保存在 `python/data/base_mapping.npz` 中。

在这批限位数据上，基参数最小二乘从真值附近扰动出发，代价从 `1.07e-1` 降到 `5.62e-4`，最终残差最大约 `2.4e-3 N·m`；大多数原始参数误差在 1% 到 5% 量级，明显优于直接优化 56 个参数时不可辨识方向上的大幅漂移。剩余偏差来自数据只能激发 33 个方向（静态底盘），而结构基参数有 35 个方向，其中两个方向在这批数据中仍然不可辨识。

## 参数说明

`params.py` 中的惯量是绕质心的标准惯性张量（非对角项为 `-∫x_i x_j dm`），
坐标为本项目坐标轴（x 向右、y 向前、z 向上），单位为 kg·m²；质心相对各自关节原点。
阻尼 `DAMPING` 目前统一为 0.1，需要时可直接修改该列表。
