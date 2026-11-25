# 锁定 roll 轴的三自由度模型

本模型由 [DYNAMICS.md](DYNAMICS.md) 的四刚体串链在 roll 关节（原第 2 个关节）固定为给定角度 θr 后得到。原模型参数不变；B1 与 B2 合并成一个刚体，剩余广义坐标为

$$q=(\theta_1,\ \theta_3,\ \theta_4)^T,$$

顺序为大 yaw、小 yaw、pitch。力矩、阻尼、损失权重、目标与限位都按同一顺序取三个分量。

本模型只提供前向仿真与控制量梯度，不提供参数梯度。

## 1 合并参数

记 $F=R_y(\theta_r)$。合并刚体的坐标系取 B1 的原坐标系（原点 $p_1$，姿态 $R_1$）。合并质量、质心与绕质心的惯量为

$$m_{12}=m_1+m_2,$$

$$c_{12}=\frac{m_1c_1+m_2\left(o_2+Fc_2\right)}{m_{12}},$$

$$
I_{12}=I_1+FI_2F^T
+m_1\left[(d_1\cdot d_1)I_3-d_1d_1^T\right]
+m_2\left[(d_2\cdot d_2)I_3-d_2d_2^T\right],
$$

其中

$$d_1=c_1-c_{12},\qquad d_2=o_2+Fc_2-c_{12}.$$

当 $m_{12}=0$ 时取 $c_{12}=0$，两个平移项为零。

合并后的关节偏置为

$$o_3'=o_2+Fo_3.$$

固定旋转 $F$ 作用在 joint 3 的局部旋转之前。

## 2 运动学

$$
\begin{aligned}
R_1&=R_0R_z(\theta_1), & R_3&=R_1FR_z(\theta_3), & R_4&=R_3R_x(\theta_4),\\
p_1&=p_0+R_0o_1, & p_3&=p_1+R_1o_3', & p_4&=p_3+R_3o_4,\\
a_1&=R_0e_z, & a_3&=R_1Fe_z, & a_4&=R_3e_x,\\
g_{12}&=p_1+R_1c_{12}, & g_3&=p_3+R_3c_3, & g_4&=p_4+R_4c_4.
\end{aligned}
$$

三个剩余刚体依次为合并刚体、B3、B4；三个剩余关节依次为原 joint 1、joint 3、joint 4。

## 3 动力学

质量矩阵、偏置力与运动方程与 [DYNAMICS.md](DYNAMICS.md) 第 7 节形式相同，只是刚体数和关节数都为三：

$$
M(q)\ddot q+D\dot q+b(q,\dot q)=\tau,\qquad
\ddot q=M^{-1}\left[\tau-D\dot q-b\right],
$$

其中 $D=\mathrm{diag}(d_1,d_3,d_4)$，原 roll 阻尼 $d_2$ 不进入降维后的方程（被约束力矩吸收）。$b$ 按逆动力学递推、令 $\ddot q=0$ 求值；递推中的关节轴为 $a_1,a_3,a_4$，关节原点为 $p_1,p_3,p_4$。

## 4 接口

### C++

`include/gimbal_dynamics/roll_locked.hpp`：

- `merge_model(original, locked_angle)`：由原四自由度参数与锁定角计算合并后的三自由度参数。
- `forward_dynamics(...)`：单步加速度。
- `simulate(...)`：仅前向仿真，输出每个主步状态与损失。
- `simulate_and_gradient(...)`：前向仿真并返回损失对三个力矩的解析梯度。

### C

`include/gimbal_dynamics/c_api.h`：

- `tcrg_roll_locked_parameters`：输出合并后的质量、质心、惯量、偏置、阻尼与重力。
- `tcrg_roll_locked_forward_step`：单步前向。
- `tcrg_roll_locked_forward_simulate`：多步前向。
- `tcrg_roll_locked_control_gradient`：损失与对三个力矩的梯度。

状态、力矩与目标都按大 yaw、小 yaw、pitch 排列；固定 roll 角作为独立参数传入。

### Python

`python/gimbal_dynamics.py` 的 `Context`：

- `roll_locked_parameters(model, locked_angle)`
- `roll_locked_forward_step(...)`
- `roll_locked_forward_simulate(...)`
- `roll_locked_control_gradient(...)`

## 5 校验

`tests/roll_locked_check.cpp` 校验：

- 合并刚体动能与 B1+B2 动能一致；
- 前向加速度与原四自由度模型在锁定状态下的 3×3 子块解一致；
- 用完整四自由度质量矩阵逐 RK4 阶段计算保持 roll 加速度为零的准确力矩，把该力矩与剩余三个力矩一起送入完整模型，按同一 RK4 轨迹与三自由度模型逐主步对照；
- 控制量梯度与中心差分一致；
- 锁定角 0、$\pm\pi/2$、$\pi$ 下模型有效。

`tests/roll_locked_c_api_check.c` 校验 C 接口的参数、前向、梯度与错误返回；`tests/roll_locked_python_check.py` 校验 Python 绑定与中心差分梯度。

## 6 可视化界面

仿真环境 `GimbalEnv` 支持在运行中锁定/解锁 roll 轴（`supports_roll_lock = True`）：

- `lock_roll()`：记录当前 roll 实际角度作为锁定角；如果 roll 有角速度，按完全非弹性碰撞计算碰撞后的四维速度。用当前完整四自由度质量矩阵 $M$ 与 roll 方向 $e_2$ 求解
  $M(\dot q^{+}-\dot q^{-})=e_2\Lambda$，约束 $\dot q^{+}_2=0$，因此 $\dot q^{+}=\dot q^{-}-M^{-1}e_2\,\dot q^{-}_2/(M^{-1})_{22}$。这是带预设底盘时关节约束的广义动量冲量平衡：冲量沿 roll 轴，roll 相对速度归零，其余关节速度按完整质量矩阵耦合重新分配。
- 锁定期间 `step()` 使用本文件的三自由度模型推进，roll 角固定、roll 速度为零；`unlock_roll()` 用当前三自由度状态还原四维状态（roll 角为锁定角、roll 速度为零）。
- 可视化界面按 `l` 锁定/解锁；锁定后 roll 目标固定为锁定角，roll 按键被过滤，模式文字显示锁定状态。实机环境 `RealGimbalEnv` 的 `supports_roll_lock` 为 `False`，界面不提供该功能。
- `tests/roll_lock_ui_check.py` 校验冲量方向、动能不增加、锁定/解锁状态、Viewer 目标固定与按键过滤。
