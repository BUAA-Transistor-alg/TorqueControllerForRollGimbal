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

// 控制量梯度使用的 MPC 代价配置：跟踪、力矩幅值、力矩变化率、关节软限位与终端项。
// target_angle / target_velocity 长度均为 num_steps*4；previous_torque 长度为 4。
// 三个指针均可为空：空目标按 0 处理，空 previous_torque 表示跳过 k = 0 的变化率项。
typedef struct TcrgControlLoss {
    double angle_weights[4];
    double velocity_weights[4];
    double torque_weights[4];
    double torque_rate_weights[4];
    double terminal_angle_weights[4];
    double terminal_velocity_weights[4];
    double limit_weights[4];
    double limit_lower[4];
    double limit_upper[4];
    double limit_margin[4];
    const double* target_angle;
    const double* target_velocity;
    const double* previous_torque;
} TcrgControlLoss;

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
