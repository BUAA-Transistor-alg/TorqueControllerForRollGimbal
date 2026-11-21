# python 端说明

全部命令在仓库根目录执行，数据集与图表位于根目录的 `data/`。

## 目录结构

```
python/
  gimbal_dynamics.py     ctypes 封装，调用 C 接口（include/gimbal_dynamics/c_api.h）
  params.py              SolidWorks 质量属性换算得到的四连杆参数，阻尼取 0.1 N·m·s/rad
  gimbal_env.py          仿真环境，提供 reset 与 step 接口
  acquisition/           数据采集：采集循环、协议生成、加噪与数据切片
  identification/        参数辨识：回归、评估、物理投影、协议与噪声实验
  visualization/         PyBullet 可视化与键盘交互
```

三个子目录中的脚本都能直接运行：`python3 python/<分组>/<脚本>.py`，脚本自己把
`python/` 加入模块搜索路径。

## 构建共享库

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target gimbal_dynamics_c -j
```

产物为 `build/libgimbal_dynamics.so`。也可以用 `build-new`、`build-final` 等目录，
或者用环境变量指定：

```bash
export GIMBAL_DYNAMICS_LIB=/path/to/libgimbal_dynamics.so
```

## 依赖

```bash
python3 -m pip install numpy scipy matplotlib pybullet
```

## 数据采集（python/acquisition）

- `collect_closed_loop.py`：标准协议的闭环 PID 采集（100 段，每段 3 s，300 点，dt = 0.01 s）。
- `collect_rich_data.py`：强激励闭环采集（更大参考幅值与增益范围）。
- `collect_low_excitation.py`：低激励协议，慢分量提供姿态行程、快分量提供加速度。
- `collect_base_motion.py`：带底盘晃动与滑移的采集。
- `collect_data.py`：按参考轨迹的逆动力学力矩推进，记录控制-响应序列。
- `add_noise.py`：给记录数据加高斯测量噪声。
- `make_strong_protocol.py`、`slice_dataset.py`、`merge_datasets.py`：协议生成与数据集切片、拼接。

限位：roll `[-45°, +5°]`，小 yaw `[-50°, +50°]`，pitch `[-45°, +45°]`，大 yaw 不限制；
段内底盘不移动、不旋转，只通过坡角与坡向给出不超过 20° 的倾角。

```bash
python3 python/acquisition/collect_closed_loop.py --segments 100
python3 python/acquisition/add_noise.py
```

## 仿真环境（python/gimbal_env.py）

```python
from gimbal_env import GimbalEnv

env = GimbalEnv()
observation = env.reset({"angle": [0.0] * 4, "velocity": [0.0] * 4,
                         "terrain": {"slope_angle": 0.1, "slope_direction": 0.3}})
observation = env.step({"torque": [0.0, 0.0, 0.0, 0.0]})
print(observation["angle"], observation["slope_angle"])
```

`reset` 返回实际采用的观测（超限位初值会被夹取），`step` 返回一步后的完整观测；
接口由抽象基类 `Environment` 声明，接入真实环境时实现同一套接口。

## 参数辨识（python/identification）

- `regressor_fit.py`、`regressor_full.py`：平滑算子、噪声传播、解析误差在变量修正与物理约束回归。
- `param_basis.py`：经验可辨识基（44 维参数空间中的 28 维可辨识子空间）。
- `combine_models.py`、`finalize_model.py`、`export_final.py`：合并、物理整理与导出。
- `eval_openloop.py`、`check_direction.py`、`leaderboard.py`：开环误差、方向一致率与候选排序。
- `estimate_geometry_linear.py`、`geo_robust_fit.py`：逐段几何量估计与几何偏差稳健辨识。
- `study_*.py`、`report_*.py`、`validate_*.py`：低激励协议、噪声尺度、几何偏差、底盘运动等实验与汇总。

```bash
python3 python/identification/regressor_full.py --data data/closed_loop_data_noisy.npz \
    --save data/identified_regressor.npz --mode corrected --width 51 --order 5 --stride 3
python3 python/identification/eval_openloop.py --identified data/identified_final.npz
```

参数定义、可辨识维数与物理可行条件见 `PARAMETER_IDENTIFIABILITY.md`，动力学约定见
`DYNAMICS.md`。

## 可视化（python/visualization）

```bash
python3 python/visualization/ui.py                      # 界面
python3 python/visualization/ui.py --direct --steps 100 # 无界面自检
```

键盘：`1`/`2` 关节 1（大 yaw），`3`/`4` 关节 2（roll），`5`/`6` 关节 3（小 yaw），
`7`/`8` 关节 4（pitch），`m` 在力矩模式与 PID 目标模式间切换，`r` 回零位，`q` 或 `ESC` 退出。
PyBullet 只负责显示，动力学由本项目库计算。

## 直接调用前向仿真

```python
import gimbal_dynamics as gd

context = gd.Context(1)
model = gd.make_model()
terrain = gd.make_terrain()
base = gd.make_base()
state = gd.make_state()

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
阻尼 `DAMPING` 目前统一为 0.1。
