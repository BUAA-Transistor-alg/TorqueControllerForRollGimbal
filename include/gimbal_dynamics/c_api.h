#pragma once

// 动力学前向仿真的 C 接口。
// 供 ctypes、C 程序或其他语言调用，内部固定使用双精度后端。

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TcrgContext TcrgContext;

// 4 个刚体的惯性参数、关节安装偏置、粘性阻尼与重力。
// 惯量分量顺序：xx, yy, zz, xy, xz, yz，标准惯性张量记法，单位 kg·m^2。
typedef struct TcrgModel {
    double mass[4];
    double com[4][3];
    double inertia[4][6];
    double joint_offsets[4][3];
    double damping[4];
    double gravity;
} TcrgModel;

typedef struct TcrgState {
    double angle[4];
    double velocity[4];
} TcrgState;

typedef struct TcrgBaseState {
    double acceleration[3];
    double yaw;
    double yaw_rate;
    double yaw_acceleration;
} TcrgBaseState;

typedef struct TcrgTerrain {
    double slope_angle;
    double slope_direction;
} TcrgTerrain;

// 参数梯度使用的轨迹损失权重与目标轨迹（target 长度均为 num_steps*4）。
typedef struct TcrgLossSpec {
    double angle_weights[4];
    double velocity_weights[4];
    double torque_weights[4];
    const double* target_angle;
    const double* target_velocity;
} TcrgLossSpec;

// 控制量梯度使用的 MPC 代价配置：跟踪、力矩幅值、力矩变化率与关节软限位。
// target_angle / target_velocity 长度均为 num_steps*4；previous_torque 长度为 4。
// 三个指针均可为空：空目标按 0 处理，空 previous_torque 表示跳过 k = 0 的变化率项；
// previous_torque 非空时，它是上一次采用的第一步力矩，τ_0 与它的差距计入变化率项。
typedef struct TcrgControlLoss {
    double angle_weights[4];
    double velocity_weights[4];
    double torque_weights[4];
    double torque_rate_weights[4];
    double limit_weights[4];
    double limit_lower[4];
    double limit_upper[4];
    double limit_margin[4];
    const double* target_angle;
    const double* target_velocity;
    const double* previous_torque;
} TcrgControlLoss;

// MPC 求解器配置：时域、步长、约束、迭代上限、收敛阈值与完整代价权重。
// 各关节力矩箱约束为 ±torque_limit[i]，关节顺序：大 yaw、roll、小 yaw、pitch；限位区间由
// limit_lower / limit_upper / limit_margin 给出；权重为零的关节不参与限位项。
// 求解器内部维护上一次采用的第一步力矩，变化率项包含 τ_0 与它的差距。
typedef struct TcrgMpcConfig {
    size_t horizon;
    double step;
    int refinement;
    double torque_limit[4];
    int max_iter;
    double tolerance;
    double angle_weights[4];
    double velocity_weights[4];
    double torque_weights[4];
    double torque_rate_weights[4];
    double limit_weights[4];
    double limit_lower[4];
    double limit_upper[4];
    double limit_margin[4];
} TcrgMpcConfig;

// 单次 MPC 求解信息
typedef struct TcrgMpcInfo {
    double loss;
    int iterations;
    int evaluations;
    int converged;
    double projected_gradient;
} TcrgMpcInfo;

typedef struct TcrgMpc TcrgMpc;

// 创建/销毁上下文；thread_count 为参与前向仿真的总线程数（含调用线程），1 表示串行。
TcrgContext* tcrg_context_create(int thread_count);
void tcrg_context_destroy(TcrgContext* context);

// 前进一步主步。返回 0 表示成功，非 0 表示失败，可用 tcrg_last_error 获取原因。
int tcrg_forward_step(const TcrgContext* context, const TcrgModel* model,
                      const TcrgTerrain* terrain, const TcrgBaseState* base,
                      const TcrgState* current, const double torque[4], double step,
                      int refinement, TcrgState* next);

// 前进 num_steps 个主步，底盘状态与地形在整个区间保持不变。
// torque 的长度为 num_steps，每个主步一个力矩向量。
int tcrg_forward_simulate(const TcrgContext* context, const TcrgModel* model,
                          const TcrgTerrain* terrain, const TcrgBaseState* base,
                          const TcrgState* initial, const double* torque,
                          size_t num_steps, double step, int refinement,
                          TcrgState* final_state);

// 计算当前位形下使关节加速度为零所需的偏置力矩（含重力与底盘运动项，qdot 取零）。
// 用于 PID 模式的重力前馈；输出长度 4。
int tcrg_bias_torque(const TcrgContext* context, const TcrgModel* model,
                     const TcrgTerrain* terrain, const TcrgBaseState* base,
                     const TcrgState* state, double bias[4]);

// 逆动力学：给定状态与关节角加速度，返回所需关节力矩。
// torque = M(q) qdd + D qdot + b(q, qdot, base)，长度 4。
int tcrg_inverse_dynamics(const TcrgContext* context, const TcrgModel* model,
                          const TcrgTerrain* terrain, const TcrgBaseState* base,
                          const TcrgState* state, const double qdd[4], double torque[4]);

// 参数灵敏度：返回损失、损失对 56 个原始参数的梯度以及末端状态。
// torque 长度为 num_steps*4；target 由 loss_spec 指定。
int tcrg_parameter_gradient(const TcrgContext* context, const TcrgModel* model,
                            const TcrgTerrain* terrain, const TcrgBaseState* base,
                            const TcrgState* initial, const double* torque,
                            size_t num_steps, double step, int refinement,
                            const TcrgLossSpec* loss_spec, double* loss,
                            TcrgState* final_state, double gradient[56]);

// 控制量梯度：返回损失、损失对 num_steps*4 个关节力矩的解析梯度以及末端状态。
// base 为长度 num_steps 的数组（预测区间内逐步的底盘状态）；torque 长度 num_steps*4，
// gradient 长度 num_steps*4，gradient[k*4+i] = ∂L/∂τ_{k,i}。
int tcrg_control_gradient(const TcrgContext* context, const TcrgModel* model,
                          const TcrgTerrain* terrain, const TcrgBaseState* base,
                          const TcrgState* initial, const double* torque,
                          size_t num_steps, double step, int refinement,
                          const TcrgControlLoss* loss_spec, double* loss,
                          TcrgState* final_state, double* gradient);

// ---- 锁定 roll 轴（第 2 个关节）的三自由度模型 ----
//
// 原模型的 B1 与 B2 在 roll 角固定为 locked_angle 后合并；剩余状态与力矩的顺序为
// 大 yaw、小 yaw、pitch（对应原关节 1、3、4）。该模型只提供前向仿真与控制量梯度，
// 不提供参数梯度。

// 三自由度状态：三个关节角与三个关节角速度。
typedef struct TcrgRollLockedState {
    double angle[3];
    double velocity[3];
} TcrgRollLockedState;

// 合并后的模型参数：body 0 为 B1+B2，body 1 为 B3，body 2 为 B4。
// joint_offsets 依次为 o1、o3' = o2 + Ry(locked_angle) o3、o4；
// 关节 3 的固定旋转 Ry(locked_angle) 由 locked_angle 决定，不单独输出。
// 惯量分量顺序与 TcrgModel 相同：xx, yy, zz, xy, xz, yz。
typedef struct TcrgRollLockedModel {
    double mass[3];
    double com[3][3];
    double inertia[3][6];
    double joint_offsets[3][3];
    double damping[3];
    double gravity;
    double locked_angle;
} TcrgRollLockedModel;

// 控制量梯度使用的 MPC 代价：三个剩余关节的权重、目标与限位。
// target_angle / target_velocity 长度均为 num_steps*3；previous_torque 长度为 3。
// 三个指针均可为空：空目标按 0 处理，空 previous_torque 表示跳过 k = 0 的变化率项。
typedef struct TcrgRollLockedControlLoss {
    double angle_weights[3];
    double velocity_weights[3];
    double torque_weights[3];
    double torque_rate_weights[3];
    double limit_weights[3];
    double limit_lower[3];
    double limit_upper[3];
    double limit_margin[3];
    const double* target_angle;
    const double* target_velocity;
    const double* previous_torque;
} TcrgRollLockedControlLoss;

// 由原四自由度参数与固定 roll 角计算合并后的三自由度模型参数。
int tcrg_roll_locked_parameters(const TcrgContext* context, const TcrgModel* model,
                                double locked_angle, TcrgRollLockedModel* parameters);

// 前进一步主步；torque 长度 3。
int tcrg_roll_locked_forward_step(const TcrgContext* context, const TcrgModel* model,
                                  double locked_angle, const TcrgTerrain* terrain,
                                  const TcrgBaseState* base,
                                  const TcrgRollLockedState* current, const double torque[3],
                                  double step, int refinement, TcrgRollLockedState* next);

// 前进 num_steps 个主步；base 在全部主步中保持不变，torque 长度 num_steps*3。
int tcrg_roll_locked_forward_simulate(const TcrgContext* context, const TcrgModel* model,
                                      double locked_angle, const TcrgTerrain* terrain,
                                      const TcrgBaseState* base,
                                      const TcrgRollLockedState* initial, const double* torque,
                                      size_t num_steps, double step, int refinement,
                                      TcrgRollLockedState* final_state);

// 控制量梯度：返回损失、损失对 num_steps*3 个力矩的解析梯度与末端状态。
// base 为长度 num_steps 的数组；gradient 长度为 num_steps*3，
// gradient[k*3+i] = ∂L/∂τ_{k,i}。
int tcrg_roll_locked_control_gradient(const TcrgContext* context, const TcrgModel* model,
                                      double locked_angle, const TcrgTerrain* terrain,
                                      const TcrgBaseState* base,
                                      const TcrgRollLockedState* initial, const double* torque,
                                      size_t num_steps, double step, int refinement,
                                      const TcrgRollLockedControlLoss* loss_spec, double* loss,
                                      TcrgRollLockedState* final_state, double* gradient);

// 创建 MPC 求解器：model 在创建时固定，thread_count 为参与阶段 Jacobian 计算的线程数。
// 求解器内部维护热启动解与上一拍实际力矩，Python 或 C 调用方只需提供当前状态与目标。
TcrgMpc* tcrg_mpc_create(const TcrgModel* model, const TcrgMpcConfig* config,
                         int thread_count);
void tcrg_mpc_destroy(TcrgMpc* solver);
void tcrg_mpc_reset(TcrgMpc* solver);

// 求解本拍：target_angle 长度 horizon*4（第 k 行为预测第 k+1 拍的目标），
// target_velocity 可为空（为空时角速度目标为零）；torque 与 info 分别写回本拍力矩与求解信息。
int tcrg_mpc_solve(TcrgMpc* solver, const TcrgTerrain* terrain, const TcrgBaseState* base,
                   const TcrgState* initial, const double* target_angle,
                   const double* target_velocity, double torque[4], TcrgMpcInfo* info);

// 40 个惯性基参数（每个刚体 m、h、I_o 的 10 个分量）与 4 个阻尼参数的
// 逆动力学基矩阵。basis 的行顺序为 4 个关节力矩，列顺序为 44 个参数。
int tcrg_inertial_basis(const TcrgContext* context, const TcrgModel* model,
                        const TcrgTerrain* terrain, const TcrgBaseState* base,
                        const TcrgState* state, const double qdd[4], double basis[44][4]);

// 最近一次失败的说明；没有失败时为空字符串。
const char* tcrg_last_error(void);

#ifdef __cplusplus
}
#endif
