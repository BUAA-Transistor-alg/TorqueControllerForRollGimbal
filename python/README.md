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


## 物理基参数重参数化

`identify_physical.py` 按 `PARAMETER_IDENTIFIABILITY.md` 的取法构造 35 个物理基参数：

- B1：`beta1 = I1_zz + m1(c1x^2 + c1y^2)`，`beta2 = m1 c1x`，`beta3 = m1 c1y`。
- B2、B3、B4：等效质量 `m_i`、一阶矩 `h_i = m_i c_i` 三个分量、空间惯量沿关节轴的投影 `a_i^T I_o,i a_i`、垂直于关节轴平面内的惯量差 `p_i^T I_o,i p_i - q_i^T I_o,i q_i`。
- 10 个独立安装偏置与 4 个阻尼。

反解到原始参数时选择保持正定的规范：B1 取 `c1z = 0`、小正 `I1_xx = I1_yy`、其余惯性积为 0；B2 到 B4 取横向惯量迹为规范值、横向交叉项为 0。`o1.z`、`o2.z` 固定为 0；辨识时默认把 `o1.x`、`o1.y` 也固定为 0，此时独立基参数为 33 个，可以用 `--no-fix-o1-xy` 恢复 35 个。

```bash
cd python
python3 identify_physical.py --segments 24 --stride 6 --max-nfev 12
```

在 24 段数据、每 6 个点取一个样本、固定 `o1.x = o1.y = 0` 的条件下，代价从 `8.90e-1` 降到 `2.59e-3`，最终残差最大约 `6.7e-3 N·m`，与真值处由离散差分造成的 `8.5e-3 N·m` 残差处于同一量级。被规范固定的 B1 非轴向惯量、`B1.com_z`、`o1.z`、`o2.z` 不对应真值；其余参数中 `B1.inertia_zz`、`B3.mass`、`B3.com_z`、`B4.com_y`、`d2` 等误差在 3% 到 10% 量级，B2 的 `com_z` 与 B3 的 `com_x` 误差较大，这是静态底盘数据对相应方向激励不足造成的。

## 参数灵敏度梯度下降

C 接口新增：

```c
int tcrg_parameter_gradient(const TcrgContext* context, const TcrgModel* model,
                            const TcrgTerrain* terrain, const TcrgBaseState* base,
                            const TcrgState* initial, const double* torque,
                            size_t num_steps, double step, int refinement,
                            const TcrgLossSpec* loss_spec, double* loss,
                            TcrgState* final_state, double gradient[56]);
```

它调用库里的参数灵敏度实现，返回轨迹损失对 56 个原始参数的梯度。库内参数按每个刚体连续 10 个排列，`gradient_fit.py` 用 `PERM` 把它置换到本项目分组布局。

重参数化梯度由链式法则给出：

```text
dL/dbeta = T^T dL/dx
T = dx/dbeta
```

`gradient_fit.py` 单独实现了 `T` 的解析表达式，并做两项差分校验：

```bash
python3 gradient_fit.py --check-jacobian
python3 gradient_fit.py --check-loss-gradient --steps 60
```

结果：

```text
映射雅可比最大绝对误差      2.88e-11
基参数损失梯度最大绝对误差  1.33e-06
```

课程学习优化：

```bash
python3 gradient_fit.py --optimize --segments 3 --iterations 4 \
    --stages 10,20,40 --learning-rate 0.001 --regularization 0.001
```

优化器使用 Adam，更新量为 `u = beta / scale`，带 L2 正则、梯度范数裁剪与回溯接受准则。短段课程的结果：

```text
步数 20：损失 6.58e-03 -> 2.17e-03
步数 40：损失 8.21e-03 -> 3.35e-03
```

在更长的段上，开环力矩序列对参数误差非常敏感：参数稍微偏离 CAD 值就会使 3 s 轨迹发散，长段损失迅速增大，回溯会拒绝这些更新。要完成全 3 s 的梯度优化，需要闭环数据或更强的轨迹稳定约束。

## 参数说明

`params.py` 中的惯量是绕质心的标准惯性张量（非对角项为 `-∫x_i x_j dm`），
坐标为本项目坐标轴（x 向右、y 向前、z 向上），单位为 kg·m²；质心相对各自关节原点。
阻尼 `DAMPING` 目前统一为 0.1，需要时可直接修改该列表。
