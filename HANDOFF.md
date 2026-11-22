# 交接说明（强化学习自适应控制器）

## 1 项目与分支

仓库是四自由度串链云台（B0 棋盘固定 → B1 大 yaw 绕 z → B2 roll 绕 y → B3 小 yaw 绕 z →
B4 pitch 绕 x），C 库负责动力学，python 端负责采集、辨识与强化学习。

- `main` 分支：参数辨识已经完成并整理过。最终模型 `data/identified_final.npz`（56 维原始参数），
  方法与结果见 `TASK_SUMMARY.md`，可辨识维数与物理可行条件见 `PARAMETER_IDENTIFIABILITY.md`，
  python 目录结构见 `docs/PYTHON.md`。
- `sac-adaptive` 分支：强化学习自适应控制器（当前工作）。

所有强化学习代码在 `python/rl/`，方法、结果、局限与复现命令见 `docs/RL.md`。

## 2 目标

训练一个循环控制器，让四个关节角跟踪带阶跃、漂移与超限目标的参考曲线；参数与电机力矩系数每回合
随机变化且回合内缓慢漂移，要求控制器利用回合内的反馈更新内部状态来在线适应，并且这个状态能够
在线估计当前的等效扰动。对大 yaw 多圈转动、底盘运动、观测噪声与偏差、超限目标、限位回避都有要求。

## 3 最终方案

控制器 = **标称模型结构化前馈 + 循环残差策略**：

```
a = clip(a_struct(观测) + 0.25 · tanh(μ(h_t) + σ·ε), -1, 1)
```

1. **结构化前馈 `a_struct`**：用设计参数（不是被控对象的随机参数）在当前测得状态下做逆动力学，
   得到"本应施加的动作"，并叠加比例反馈、预瞄前馈、限位制动与盒约束下的有界力矩分配；其中
   重力/惯量估计的偏差由**负载估计**（动作失配的一阶滑动平均）补偿。四组权重由真参数教师的动作
   岭回归标定（`python/rl/calibrate_structured.py`，当前值写在 `python/rl/teacher.py`）。
2. **快路 64 维门控线性递归**吃基础观测与当前标称物理特征，**慢路 64 维 GRU** 只吃动作失配与负载，
   按一阶滑动平均解析初始化并冻结，作为控制器固定的在线扰动估计器；负载头线性读出。
3. **残差策略**只训练两个 256 隐层，输出被限制在 ±0.25 动作单位；探索标准差固定（`exp(-2)`）。
4. **辅助监督头**（参数回归、响应回归、动力学预测、负载回归）把动力学信息压进状态，同时作为
   可读性探针；它们与残差头共享快路编码器，但慢路与负载头只由负载回归监督，避免被策略梯度改写。

训练：8 个并行环境、1500 步回合、32 次更新/迭代、参数尺度课程 0.8 → 1.75（前 35% 迭代）、
漂移幅度从 25% 迭代起升到 `drift_scale = 0.35, drift_motor = 0.3`；每 10 到 20 次迭代按固定种子
评估并保存"选择分"最低的检查点。最终策略取选择分最低的 `data/rl_distill_best.pt`。

## 4 关键实现位置

| 文件 | 职责 |
|---|---|
| `python/rl/tracking_env.py` | 环境：参数/电机系数/漂移采样、参考轨迹、142 维观测（含量测标称物理特征与负载）、奖励、限位与响应标签 |
| `python/rl/teacher.py` | 真参数教师（参数知识上限）与结构化自适应控制器（残差策略的前馈基线与蒸馏目标） |
| `python/rl/networks.py` | 快慢两路编码器、结构化前馈 + 循环残差策略、价值网络、四个辅助头 |
| `python/rl/sac.py` | 序列回放、SAC 更新、辅助监督、检查点读写 |
| `python/rl/train.py` | 训练入口（参数课程、漂移课程、预热、最佳检查点筛选） |
| `python/rl/calibrate_structured.py` | 结构化控制器权重标定 |
| `python/rl/evaluate.py` | 控制评估：尺度扫描、漂移与冻结配对、噪声/偏差放大、固定参数基线对照 |
| `python/rl/evaluate_reference.py` | 真参数教师与结构化控制器的参考评估 |
| `python/rl/evaluate_extractor.py` | 隐藏状态：参数/响应/负载回归 R²、慢路清零/冻结/交换干预、漂移跟踪 |
| `python/rl/deploy.py` | 部署：由状态与目标位置产生力矩，维护 0.2 s 目标缓冲与外推预瞄 |
| `python/visualization/backend.py` | 仿真后端：`gimbal_env.Environment` 接口，100 Hz |
| `python/visualization/controllers.py` | 界面控制器：直接力矩、PID 目标位置、策略网络目标位置 |
| `python/rl/plot.py` | 训练曲线、尺度扫描、回合跟踪图 |

## 5 结果（种子 7000–7007，8 回合 × 1500 步）

| 参数尺度 | 本策略限位内误差 | 固定参数专训基线 | 真参数教师 | 撞限位（平均/最大） |
|---:|---:|---:|---:|---:|
| 0.0 | **0.0151** | 0.0172 | 0.0132 | 0.00 / 0 |
| 0.5 | **0.0123** | 0.0144 | 0.0085 | 0.00 / 0 |
| 1.0 | **0.0163** | 0.0205 | 0.0099 | 0.00 / 0 |
| 1.5 | **0.0168** | 0.0210 | — | 0.00 / 0 |
| 2.0 | **0.0189** | 0.0216 | — | 0.00 / 0 |
| 2.5 | **0.0240** | 0.0250 | 0.0133 | 0.75 / 6 |

- 固定参数档误差是固定参数专训基线的 0.88 倍，比真参数教师高 14%（参数知识的上限）；参数尺度 0 到 2.0 的
  40 个评估回合全部零碰撞；尺度 2.5 只有 1 个回合出现 6 个碰撞步。
- 回合内漂移 0.35/0.3：限位内误差 0.0210，配对的冻结漂移对照 0.0202，比值 1.04。
- 观测噪声放大 4 倍 0.0178、系统偏差放大 3 倍 0.0201，均零碰撞。
- 隐藏状态：负载回归 R² 0.9999（漂移下 0.9997）；慢路每步清零后误差退化 3.41 倍；冻结 1.58 倍；
  持续交换 100 步 1.27 倍；对 32 维参数标签的线性探针 R² 0.19。

## 6 已知局限

1. **整体尺度不可观测**：质量、惯量、阻尼与电机系数同时乘同一个正数会得到逐位相同的仿真轨迹，
   绝对参数坐标不可学习。评估用的物理目标必须是尺度归一的（相对电机增益、动作单位的扰动与
   惯量、固定姿态保持力矩）。
2. **完整 28 维坐标无法从闭环轨迹高精度恢复**：带噪测量与未知电机系数下，固定名义基投影 R² 为负；
   本方案把状态监督目标放在"当前等效扰动"上（R² 0.999 量级），参数坐标只作为辅助诊断。
3. **参数尺度 2.5 的残余碰撞**：roll 行程只有 50°、上界 5°，极端参数下执行器接近饱和，负载估计
   的残差把关节缓慢推上挡块；把限位裕度与惩罚加强可以把高尺度碰撞清零，但会牺牲 10% 到 15% 的
   跟踪精度（实测 `data/rl_safety_best.pt`，未采用）。
4. **残差策略的收益有限**：结构化前馈已经把动作做到与真参数教师相差 0.03 量级，强化学习主要贡献
   是参数尺度 0.5 到 2.0 上 15% 到 25% 的跟踪提升与零碰撞；价值梯度如果直接改写编码器会破坏在线
   负载估计，所以编码器只由监督目标训练。

## 7 复现

见 `docs/RL.md` 第 6 节。最常用的两条：

```bash
cd /home/huhu233/rm2027/TorqueControllerForRollGimbal

# 训练
OPENBLAS_NUM_THREADS=8 python3 -u python/rl/train.py --iterations 280 --episode-steps 1500 \
    --envs 8 --updates 32 --batch 16 --window 32 --auxiliary-batch 8 --auxiliary-window 128 \
    --buffer-episodes 140 --actor-warmup 30 --parameter-scale-start 0.8 \
    --parameter-scale-final 1.75 --curriculum-fraction 0.35 --drift-final 0.35 \
    --drift-motor-final 0.3 --drift-start-iteration 0.25 --parameter-weight 0.2 \
    --response-weight 0.2 --load-weight 1.0 --dynamics-weight 0.2 --behavior-weight 0.05 \
    --learning-rate 5e-5 --head-learning-rate 3e-4 --residual-scale 0.25 --log-std-init -2.0 \
    --load-from-state --hide-load --evaluate-every 20 --evaluate-episodes 4 \
    --evaluate-scales 0.0,1.75 --save data/rl_adaptive.pt --save-best data/rl_adaptive_best.pt \
    --log data/rl_adaptive.json

# 评估
python3 python/rl/evaluate.py --policy data/rl_distill_best.pt --compare data/rl_fixed_best.pt \
    --episodes 8 --episode-steps 1500 --scales 0.0,0.5,1.0,1.5,2.0,2.5 \
    --drift-scales 0.0,0.2,0.35 --report data/rl_eval_adaptive.json
```

## 8 环境与依赖

- torch 2.9.0（CPU）、numpy、scipy、matplotlib；18 核，建议 `OPENBLAS_NUM_THREADS=8`。
- 画图前设置 `MPLCONFIGDIR=data/.mplconfig`。
- C 库：`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j`，
  产物 `build/libgimbal_dynamics.so`。
- 可视化界面 `python3 python/visualization/ui.py --mode rl` 进入策略模式（`m` 在三种模式间循环切换）。
- 数据与检查点都在仓库根目录的 `data/`（已被 `.gitignore` 忽略）；训练与评估命令都在仓库根目录执行。
- 关键检查点：`data/rl_distill_best.pt`（最终策略，另存 `data/rl_final.pt`）、`data/rl_fixed_best.pt`（固定参数专训基线）、
  `data/rl_adaptive_best.pt`（蒸馏前的自适应策略）、`data/rl_safety_best.pt`（保守版，未采用）。
