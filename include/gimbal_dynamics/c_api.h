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

// 最近一次失败的说明；没有失败时为空字符串。
const char* tcrg_last_error(void);

#ifdef __cplusplus
}
#endif
