// C 接口冒烟校验：锁定 roll 轴三自由度模型的参数、前向仿真、控制量梯度与错误返回。
#include <math.h>
#include <stdio.h>

#include "gimbal_dynamics/c_api.h"

static int finite_vector(const double* values, int count) {
    for (int i = 0; i < count; ++i) {
        if (!isfinite(values[i])) {
            return 0;
        }
    }
    return 1;
}

static void fill_model(TcrgModel* model) {
    const double masses[4] = {4.808020281, 0.200507029, 0.291080894, 0.700391795};
    const double com[4][3] = {
        {0.0, -0.050450, 0.060000},
        {0.0, -0.016802414, 0.014380483},
        {-0.012253076, -0.000151758, 0.072121603},
        {0.000003764, 0.092850351, -0.007125478},
    };
    const double inertia[4][6] = {
        {0.020383018, 0.042323454, 0.051167076, 0.0, 0.0, 0.0},
        {0.000174341, 0.000148724, 0.000129367, 0.0, 0.0, -0.000000753},
        {0.000544490, 0.000669084, 0.000187878, 0.000001533, 0.000023834, 0.000000874},
        {0.002783140, 0.000824962, 0.002335965, 0.000000482, 0.000000638, 0.000367505},
    };
    const double offsets[4][3] = {
        {0.0, 0.0, 0.1150},
        {0.0, 0.0900, 0.0390},
        {0.0, 0.0, 0.0521},
        {0.0, 0.0, 0.1579},
    };
    for (int body = 0; body < 4; ++body) {
        model->mass[body] = masses[body];
        for (int component = 0; component < 3; ++component) {
            model->com[body][component] = com[body][component];
            model->joint_offsets[body][component] = offsets[body][component];
        }
        for (int component = 0; component < 6; ++component) {
            model->inertia[body][component] = inertia[body][component];
        }
        model->damping[body] = 0.01;
    }
    model->gravity = 9.81;
}

int main(void) {
    TcrgContext* context = tcrg_context_create(1);
    if (context == NULL) {
        printf("上下文创建失败：%s\n", tcrg_last_error());
        return 1;
    }
    TcrgModel model;
    fill_model(&model);
    const TcrgTerrain terrain = {0.05, 0.7};
    const TcrgBaseState base = {{0.1, 0.2, 0.0}, 0.3, 0.5, 0.2};
    const TcrgRollLockedState initial = {{0.1, 0.2, 0.3}, {0.0, 0.0, 0.0}};
    const double locked_angle = 0.4;

    TcrgRollLockedModel parameters;
    if (tcrg_roll_locked_parameters(context, &model, locked_angle, &parameters) != 0) {
        printf("合并参数失败：%s\n", tcrg_last_error());
        tcrg_context_destroy(context);
        return 1;
    }
    const double expected_mass = model.mass[0] + model.mass[1];
    if (fabs(parameters.mass[0] - expected_mass) > 1e-12) {
        printf("合并质量不正确\n");
        tcrg_context_destroy(context);
        return 1;
    }

    const double torque[3] = {0.1, 0.2, 0.3};
    TcrgRollLockedState next;
    if (tcrg_roll_locked_forward_step(context, &model, locked_angle, &terrain, &base, &initial,
                                      torque, 0.01, 2, &next) != 0) {
        printf("单步前向失败：%s\n", tcrg_last_error());
        tcrg_context_destroy(context);
        return 1;
    }
    if (!finite_vector(next.angle, 3) || !finite_vector(next.velocity, 3)) {
        printf("单步状态包含非有限值\n");
        tcrg_context_destroy(context);
        return 1;
    }

    const size_t steps = 5;
    double torques[15];
    double targets_angle[15];
    double targets_velocity[15];
    for (size_t k = 0; k < steps; ++k) {
        for (int joint = 0; joint < 3; ++joint) {
            torques[k * 3 + joint] = 0.1 * (double)(joint + 1) - 0.05 * (double)k;
            targets_angle[k * 3 + joint] = 0.01 * (double)(k + joint);
            targets_velocity[k * 3 + joint] = 0.02 * (double)(joint - 1);
        }
    }
    TcrgRollLockedState final_state;
    if (tcrg_roll_locked_forward_simulate(context, &model, locked_angle, &terrain, &base,
                                          &initial, torques, steps, 0.01, 2,
                                          &final_state) != 0) {
        printf("前向仿真失败：%s\n", tcrg_last_error());
        tcrg_context_destroy(context);
        return 1;
    }
    if (!finite_vector(final_state.angle, 3) || !finite_vector(final_state.velocity, 3)) {
        printf("仿真末状态包含非有限值\n");
        tcrg_context_destroy(context);
        return 1;
    }

    TcrgRollLockedControlLoss loss_spec;
    for (int joint = 0; joint < 3; ++joint) {
        loss_spec.angle_weights[joint] = 1.0 + 0.5 * (double)joint;
        loss_spec.velocity_weights[joint] = 0.3 + 0.1 * (double)joint;
        loss_spec.torque_weights[joint] = 0.05;
        loss_spec.torque_rate_weights[joint] = 0.4 + 0.1 * (double)joint;
        loss_spec.limit_weights[joint] = 1.0;
        loss_spec.limit_lower[joint] = -0.2;
        loss_spec.limit_upper[joint] = 0.2;
        loss_spec.limit_margin[joint] = 0.01;
    }
    loss_spec.target_angle = targets_angle;
    loss_spec.target_velocity = targets_velocity;
    loss_spec.previous_torque = NULL;
    double loss = 0.0;
    double gradient[15];
    if (tcrg_roll_locked_control_gradient(context, &model, locked_angle, &terrain, &base,
                                          &initial, torques, steps, 0.01, 2, &loss_spec,
                                          &loss, &final_state, gradient) != 0) {
        printf("控制量梯度失败：%s\n", tcrg_last_error());
        tcrg_context_destroy(context);
        return 1;
    }
    if (!isfinite(loss) || !finite_vector(gradient, 15)) {
        printf("损失或梯度包含非有限值\n");
        tcrg_context_destroy(context);
        return 1;
    }

    if (tcrg_roll_locked_forward_step(NULL, &model, locked_angle, &terrain, &base, &initial,
                                      torque, 0.01, 2, &next) != 1) {
        printf("空上下文没有返回参数错误\n");
        tcrg_context_destroy(context);
        return 1;
    }
    if (tcrg_roll_locked_forward_simulate(context, &model, locked_angle, &terrain, &base,
                                          &initial, torques, 0, 0.01, 2, &final_state) != 1) {
        printf("零步数没有返回参数错误\n");
        tcrg_context_destroy(context);
        return 1;
    }

    tcrg_context_destroy(context);
    printf("C 接口校验通过\n");
    return 0;
}
