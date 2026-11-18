#include <math.h>
#include <stdio.h>
#include <string.h>

#include "gimbal_dynamics/c_api.h"

static void fill_model(TcrgModel* model) {
    memset(model, 0, sizeof(*model));
    model->mass[0] = 4.808020281;
    model->mass[1] = 0.200507029;
    model->mass[2] = 0.291080894;
    model->mass[3] = 0.700391795;
    model->com[0][1] = -0.050450;
    model->com[0][2] = 0.060000;
    model->com[1][1] = -0.016802414;
    model->com[1][2] = 0.014380483;
    model->com[2][0] = -0.012253076;
    model->com[2][1] = -0.000151758;
    model->com[2][2] = 0.072121603;
    model->com[3][0] = 0.000003764;
    model->com[3][1] = 0.092850351;
    model->com[3][2] = -0.007125478;
    model->inertia[0][0] = 0.020383018;
    model->inertia[0][1] = 0.042323454;
    model->inertia[0][2] = 0.051167076;
    model->inertia[1][0] = 0.000174341;
    model->inertia[1][1] = 0.000148724;
    model->inertia[1][2] = 0.000129367;
    model->inertia[1][5] = -0.000000753;
    model->inertia[2][0] = 0.000544490;
    model->inertia[2][1] = 0.000669084;
    model->inertia[2][2] = 0.000187878;
    model->inertia[2][3] = 0.000001533;
    model->inertia[2][4] = 0.000023834;
    model->inertia[2][5] = 0.000000874;
    model->inertia[3][0] = 0.002783140;
    model->inertia[3][1] = 0.000824962;
    model->inertia[3][2] = 0.002335965;
    model->inertia[3][3] = 0.000000482;
    model->inertia[3][4] = 0.000000638;
    model->inertia[3][5] = 0.000367505;
    model->joint_offsets[0][2] = 0.1150;
    model->joint_offsets[1][1] = 0.0900;
    model->joint_offsets[1][2] = 0.0390;
    model->joint_offsets[2][2] = 0.0521;
    model->joint_offsets[3][2] = 0.1579;
    model->damping[0] = model->damping[1] = model->damping[2] = model->damping[3] = 0.1;
    model->gravity = 9.81;
}

int main(void) {
    TcrgModel model;
    fill_model(&model);
    TcrgTerrain terrain;
    terrain.slope_angle = 0.0;
    terrain.slope_direction = 0.0;
    TcrgBaseState base;
    memset(&base, 0, sizeof(base));
    TcrgState state;
    memset(&state, 0, sizeof(state));
    TcrgState next;
    double torque[4] = {0.0, 0.0, 0.0, 0.0};
    double torques[8] = {0.0};

    TcrgContext* context = tcrg_context_create(1);
    if (context == NULL) {
        printf("创建上下文失败: %s\n", tcrg_last_error());
        return 1;
    }
    if (tcrg_forward_step(context, &model, &terrain, &base, &state, torque, 0.004, 2, &next) != 0) {
        printf("单步失败: %s\n", tcrg_last_error());
        tcrg_context_destroy(context);
        return 1;
    }
    for (int i = 0; i < 4; ++i) {
        if (!isfinite(next.angle[i]) || !isfinite(next.velocity[i])) {
            printf("单步结果不是有限值\n");
            tcrg_context_destroy(context);
            return 1;
        }
    }
    if (tcrg_forward_simulate(context, &model, &terrain, &base, &state, torques, 2, 0.004, 2,
                              &next) != 0) {
        printf("多步失败: %s\n", tcrg_last_error());
        tcrg_context_destroy(context);
        return 1;
    }
    if (!isfinite(next.angle[3]) || !isfinite(next.velocity[3])) {
        printf("多步结果不是有限值\n");
        tcrg_context_destroy(context);
        return 1;
    }
    memset(&state, 0, sizeof(state));
    double bias[4] = {0.0, 0.0, 0.0, 0.0};
    if (tcrg_bias_torque(context, &model, &terrain, &base, &state, bias) != 0) {
        printf("偏置力矩失败: %s\n", tcrg_last_error());
        tcrg_context_destroy(context);
        return 1;
    }
    for (int i = 0; i < 4; ++i) {
        if (!isfinite(bias[i])) {
            printf("偏置力矩不是有限值\n");
            tcrg_context_destroy(context);
            return 1;
        }
    }
    tcrg_context_destroy(context);
    printf("C 接口校验通过\n");
    return 0;
}
