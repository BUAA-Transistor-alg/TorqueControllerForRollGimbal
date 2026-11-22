# 交接说明（强化学习自适应控制器）

## 1. 项目与分支

仓库是四自由度串链云台（B0 底盘固定 → B1 大 yaw 绕 z → B2 roll 绕 y → B3 小 yaw 绕 z → B4 pitch 绕 x），
C 库负责动力学，python 端负责采集、辨识与强化学习。

- `main` 分支：参数辨识已经完成并整理过。最终模型 `data/identified_final.npz`（56 维原始参数），
  方法与结果见 `TASK_SUMMARY.md`，可辨识维数与物理可行条件见 `PARAMETER_IDENTIFIABILITY.md`，
  python 目录结构见 `docs/PYTHON.md`。
- `sac-adaptive` 分支：强化学习控制器（当前工作）。最新提交 `2c3437c`。

所有强化学习代码在 `python/rl/`，说明文档在 `docs/RL.md`（含各版本的完整实验记录）。

## 2. 目标

训练一个循环 SAC 控制器，让四个关节角跟踪给定目标曲线；先在固定参数下学会控制，再让参数每次重置都
大幅变化（并随机电机力矩系数），迫使控制器利用回合内的反馈更新上下文来在线适应；同时训练一个前向网络，
把收敛后的上下文翻译成动力学参数（28 维可辨识坐标）。对大 yaw 多圈转动、底盘运动、观测噪声与偏差、
超限目标、限位回避都有明确要求。

## 3. 现状：代码与产物

### 3.1 环境 `python/rl/tracking_env.py`

被控对象是真实模型（`gimbal_dynamics` 的 C 接口）；每个回合重置时采样参数、电机系数、底盘运动、
观测偏差与噪声实现、参考轨迹。

参数采样（`sample_parameters`）：质量、阻尼按 exp(U(-s, s)) 缩放；惯量按每个刚体一个标量缩放
exp(U(-min(s,1.2), min(s,1.2)))（保证张量正定）；质心偏移 U(-0.03, 0.03)·min(s,1) 米；
安装偏置 N(0, 0.03·min(s,1)) 弧度；采样后用逆动力学与随机力矩开环仿真做可行性检查。
第二阶段训练尺度 s = 1.0 到 1.5。

电机力矩系数：`scale_j = 1.25 * max_gravity_j + U(0, 2)` 牛米，上限 8；`1.25*max_gravity_j > 8` 时重采参数。
动作 `a ∈ [-1,1]`，实际力矩 `tau_j = a_j * scale_j`；大 yaw 无位置限位，其余关节有硬限位
（roll [-45°, 5°]，小 yaw ±50°，pitch ±45°），超限按刚性挡块夹回并清零速度。

参考轨迹：多正弦（每关节两个分量，0.2 到 1.2 rad/s）+ 0 到 3 个随机阶跃（幅度 ±0.35 rad 内随机）
+ 大 yaw 可能带持续漂移（0.1 到 0.4 rad/s）；有限位的关节参考被夹在限位外 0.35 rad 以内；阶跃与连续变化
都能把目标推出限位（实测约 13% 到 23% 的样本超限）。难度系数 `difficulty` 可缩放幅值、阶跃与漂移，
训练时按课程从 0.4 到 0.8 线性升到 1.0。

目标预瞄：每一步观测里有未来 20 个点（0.2 s）的目标位置，形式为相对当前目标的 wrap 角度，
叠加随时间增长的随机偏移（约 0.03 rad）与噪声（约 0.015 rad）。

底盘运动：偏航摆动（幅度 0 到 6 度、频率 0.5 到 3 rad/s）与水平线加速度（0 到 0.3 m/s²、0.3 到 2 rad/s），
动力学用真实底盘状态推进。

观测（`OBSERVATION_DIM = 126`，顺序见 `observation()`）：
sin/cos 关节角 8、角速度/2 4、sin/cos 参考角 8、wrap 误差/0.2 4、参考角速度/2 4、预瞄 80、
上一步动作 4、滤波角加速度/20 4、sin/cos 坡度与坡向 4、sin/cos 底盘偏航 2、偏航角速度/2 1、
底盘线加速度/0.3 3。所有观测量都带噪声与偏差：角度 0.003 rad + ±0.010 偏差，角速度 0.05 + ±0.020 + 3% 比例，
坡度 0.5° + ±1°，坡向 2° + ±5°，底盘偏航 2° + ±5°，偏航角速度 0.05 + 0.1，底盘加速度 0.02 + 0.05。
角加速度由一阶低通（系数 0.2）后的速度差分得到。

奖励（`step()`，逐项权重都是模块常量）：位置 `r^2/(1+r^2/9) + 1.5*|e|`（r = e/0.02，小误差饱和 + 大误差
不饱和）；速度 0.05×`r^2/(1+r^2/25)`；力矩 0.02×`a^2/4`；力矩平滑 0.02；限位裕度 100×进入裕度平方
（裕度 2°）；撞限位 50；超出限位幅度 500×幅度平方。限位惩罚的增长速度刻意超过跟踪收益的下降速度。

核心接口：`reset()` 返回观测；`step(action)` 返回（观测，奖励，done，info）；`info()` 给出真实误差、
撞限位次数、参数原始值、可辨识坐标与电机系数。环境还提供 `difficulty`、`noise_scale`、`bias_scale`
供评估扫描使用。

### 3.2 网络与 SAC `python/rl/networks.py`、`python/rl/sac.py`

- 上下文：门控线性递归 `h_t = a_t ⊙ h_{t-1} + (1-a_t) ⊙ W x_t`，`a_t = sigmoid(W_g x_t + b_g)`，
  隐状态 96 维，单个向量承载全部历史。
- 策略头：LayerNorm + 两层 256 隐层 + tanh 高斯；两个动作价值网络各带同样的递归编码器（两层 256）。
- SAC：gamma 0.99、软更新 0.005、学习率 3e-4、温度系数自动调节（目标熵 -4）、序列回放，
  窗口 32 步展开、批次 16、每迭代 48 次更新、8 个并行环境、回放容量 400 回合。
- 上下文正则（第二阶段）：稳定项（单位化上下文逐步变化平方和，权重 1.0，第 500 步后生效）；
  归一化熵项（各维方差相对均匀分布的比值，目标 0，系数对偶上升自动调节，上限 exp(4)）。
- 提取器（`ParameterExtractor`）：LayerNorm + 96→64→28，把回合末尾 128 步规范化上下文的平均映射到
  28 维可辨识坐标；损失是标准化坐标的均方误差。
  - 探针训练（`update_extractor_probe`）：每迭代用当前策略新跑 6 个回合写入容量 4096 的滚动数据集，
    旧的自动淘汰（不是固定缓冲区），每迭代在 256 批次上更新 8 步、学习率 3e-4、二阶权重衰减 1e-2。
  - 策略辅助（`actor_auxiliary`）：同一批新鲜窗口经编码器（带梯度）→ 提取器 → 损失，
    按系数 0.1 到 0.15 回传到策略编码器。

### 3.3 训练入口 `python/rl/train.py`

```bash
cd /home/huhu233/rm2027/TorqueControllerForRollGimbal

# 第一阶段：固定参数（800 步回合，约 260 次迭代）
OPENBLAS_NUM_THREADS=6 python3 -u python/rl/train.py --stage 1 --episode-steps 800 --envs 8 \
    --iterations 260 --updates 64 --batch 16 --window 32 --parameter-scale 0.0 \
    --difficulty-start 0.8 --resume data/rl_v4_stage1.pt \
    --save data/rl_v5_stage1.pt --log data/rl_v5_stage1.json

# 第二阶段：参数变化 + 上下文正则 + 流式提取器（1500 步回合）
OPENBLAS_NUM_THREADS=6 python3 -u python/rl/train.py --stage 2 --episode-steps 1500 --envs 8 \
    --iterations 300 --updates 48 --batch 16 --window 32 --parameter-scale 1.0 \
    --stability-weight 1.0 --stability-warmup 500 --context-entropy-target 0.0 \
    --context-entropy-weight 1.0 --extractor-weight 0.15 --extractor-episodes 6 \
    --extractor-steps 8 --extractor-batch 256 --extractor-capacity 4096 \
    --extractor-learning-rate 3e-4 --extractor-converged-length 128 --difficulty-start 1.0 \
    --resume data/rl_v7_stage2.pt --save data/rl_v7_stage2.pt --log data/rl_v7_stage2.json
```

每 20 次迭代存一次检查点，日志写到 `--log` 指定的 JSON（含各迭代误差、温度、稳定项、熵、探针与辅助损失）。

### 3.4 评估脚本

```bash
# 控制性能（分关节、限位内外、撞限位、阶跃响应、十个时段）
python3 python/rl/evaluate.py --policy data/rl_v7_stage2.pt --episodes 4 --episode-steps 1500 \
    --parameter-scale 1.0 --report data/rl_eval_v7_stage2.json

# 提取器留出评估（R^2、线性探针）
python3 python/rl/evaluate_extractor.py --policy data/rl_v7_stage2.pt --episodes 120 \
    --episode-steps 1200 --parameter-scale 1.0

# 可适应范围（参数尺度 / 噪声 / 偏差三组扫描）
python3 python/rl/study_range.py --policy data/rl_v7_stage2.pt --episodes 3 --episode-steps 1000 \
    --report data/rl_v7_range.json --sweep parameter

# 上下文消融（保留 vs 每步清零）
python3 python/rl/ablation_context.py --policy data/rl_v7_stage2.pt --episodes 3 \
    --episode-steps 1000 --parameter-scale 1.0
```

注意：评估脚本的 `--hidden-dim 96 --layer-dim 256 --extractor-layer-dim 64` 必须与检查点一致，
否则加载失败（策略会报形状不匹配，提取器会打印“结构变化，重新初始化”并随机初始化）。

### 3.5 各版本结果（1500 步评估，参数尺度 1.0 除非注明）

| 版本 | 主要改动 | 平均误差 | 限位内误差 | 分关节误差（大 yaw/roll/小 yaw/pitch） | 撞限位 |
|---|---|---|---|---|---|
| v4 | 126 维观测、预瞄、底盘运动、96/256 网络、长训练 | 0.1535 | 0.1337 | 0.505 / 0.059 / 0.022 / 0.028 | 0.0 |
| v5 | 不饱和位置代价、提取器（加权损失）、限位权重 30 | 0.0466 | 0.0460 | 0.090 / 0.037 / 0.018 / 0.041 | 25.5 |
| v6 | 流式提取器、限位 100/50、超限平方 500 | 0.0763 | 0.0773 | 0.131 / 0.060 / 0.051 / 0.062 | 28.5 |
| v7 | 提取器标准化损失、探针 3e-4 | 0.0714 | 0.0677 | 0.116 / 0.057 / 0.056 / 0.057 | 14.75 |

v7 的参数范围扫描（3 回合 × 1000 步）：限位内误差 0.0502 到 0.0509（尺度 0.5 到 2.5），撞限位 0。
v7 的上下文消融：保留上下文 0.1207 rad，每步清零 0.0811 rad（上下文没有正收益）。
v7 的提取器：留出回合 R² = -0.015，线性探针 R² = -0.52，探针损失停在 0.8 到 1.1（标准化均方误差）。

检查点与日志：`data/rl_v4_stage1.pt`（第一阶段最好，限位内 0.0218）、`data/rl_v5_stage2b.pt`
（跟踪最好 0.0466，但用的是旧的弱限位惩罚）、`data/rl_v7_stage2.pt`（当前最新，带强限位惩罚）。
对应的 `*.json` 是训练日志，`data/rl_v*_range.json` 与 `data/rl_eval_v*.json` 是评估结果。

### 3.6 图表

```bash
export MPLCONFIGDIR=data/.mplconfig
python3 python/rl/plot.py --mode training --logs 阶段一=data/rl_v5_stage1.json 阶段二=data/rl_v7_stage2.json --output data/rl_v7_training.png
python3 python/rl/plot.py --mode episode --archive data/rl_eval_v7_stage2.npz --output data/rl_v7_tracking.png
```

## 4. 已知问题（按优先级）

1. **提取器学不到参数**。标准化后探针损失停在约 1.0，即收敛上下文解释的参数方差约为 0；留出 R² -0.015；
   上下文消融显示控制上也没有正收益。诊断：观测角度偏差 ±0.010 rad 给跟踪误差设了约 0.01 rad 的地板，
   参数前馈的收益被地板淹没，强化学习没有动力去估计参数；上下文由目标预瞄与当前观测主导，参数信息瞬态。
   另有一个已修的数值错误：原来的 `1/std^2` 加权损失被低方差维度主导，数值上恒为 2e-4，提取器只在拟合
   几乎恒定的维度（这是 v5/v6 的“完美损失”假象）。
2. **跟踪与限位的权衡没调好**。v5（限位权重 30）跟踪最好 0.0466 但撞限位 25 次；v7（100/50/500）扫描里
   撞限位 0 但跟踪退化到 0.050 到 0.068。绝对值位置项让控制器更愿意贴近目标，需要重新分配两项权重。
3. **大 yaw 的漂移跟踪**。v5 的不饱和代价把分关节误差从 0.505 修到 0.090，v7 又回到 0.116（新惩罚让控制器
   更保守），还需要在两者之间取平衡。
4. **训练成本**。1500 步回合、8 环境、48 次更新的配置约 12 到 15 秒一迭代，300 次迭代约 1 小时；
   回合长度和网络再加大需要按小时计。

## 5. 接下来的任务（建议顺序）

1. **加动力学辅助头，强制上下文表示动力学**。这是最可能解决提取器问题的改动：
   - 在流式数据里同时记录每一步的（规范化上下文，动作，当前 sin/cos 角与角速度，下一拍的滤波角加速度）。
     环境观测里第 112 到 115 维就是滤波角加速度，第 108 到 111 维是上一步动作，可直接取用。
   - 新增 `DynamicsHead`：输入 concat(上下文, 动作, sin/cos 角, 角速度) → 输出下一拍角加速度（4 维），
     用标准化加速度的均方误差训练；把该损失以 0.05 到 0.2 的系数加进策略的辅助损失（梯度回传上下文）。
   - 预期：上下文必须编码质量、惯量、重力与电机系数才能预测加速度，参数提取器随后才有可学的信号。
   - 训练 200 到 400 次迭代后用 `evaluate_extractor.py` 看留出 R² 是否转正。
2. **降低观测角度偏差**（例如 ±0.010 改成 ±0.003，或用从大到小的课程）。偏差地板是“参数信息没有收益”
   的直接原因之一；降低后跟踪误差地板下降，参数前馈的收益才显现。
3. **重配跟踪与限位权重**。保留超限幅度平方项（它是把撞限位降到 0 的关键），把裕度项权重从 100 降到
   30 到 50、撞限位惩罚从 50 降到 20 到 30，抢回跟踪精度，再用 `study_range.py` 复核撞限位是否仍为 0。
4. **提取器备选方案**：
   - 对比学习：同一参数回合的上下文互为正样本、不同参数为负样本（InfoNCE），避免回归目标被噪声淹没；
   - 或者把提取器的输入从“收敛上下文平均”改成“回合内若干时刻的上下文拼起来”，提高可辨识性；
   - 或者纯线性提取器配强权重衰减，先确认参数信息是否线性可分。
5. **评估协议固定下来**：留出回合用固定种子集合（例如 `evaluate.py` 的 7000 到 7003、
   `evaluate_extractor.py` 的 21000 起 120 个回合），报告指标至少包含：分关节误差、限位内外误差、撞限位次数、
   阶跃峰值与恢复时间、参数尺度 0.5 到 2.5 的扫描、上下文消融、提取器留出 R²。

## 6. 环境与依赖

- torch 2.9.0（CPU）、numpy、scipy、matplotlib；18 核，建议 `OPENBLAS_NUM_THREADS=6`。
- 画图前设置 `MPLCONFIGDIR=data/.mplconfig`。
- C 库：`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j`，产物 `build/libgimbal_dynamics.so`。
- 数据与检查点都在仓库根目录的 `data/`（已被 `.gitignore` 忽略）；训练与评估命令都在仓库根目录执行。
