// 校验：控制量梯度与参数灵敏度对有限差分一致；前向损失、末状态在三种处理器间一致；
// 单精度与双精度结果接近；参数布局与编号正确。
#include <cmath>
#include <cstdio>
#include <vector>

#include "gimbal_dynamics/handlers.hpp"
#include "gimbal_dynamics/parameters.hpp"
#include "support.hpp"

using namespace tcrg::gimbal_dynamics;

namespace {

template <typename Real>
Real forward_loss(ForwardSimulation<Real>& handler, const support::Problem<Real>& problem) {
    ForwardOutputT<Real> output;
    handler.run(support::forward_input(problem), output);
    return output.loss;
}

// 扩展损失算例：终端跟踪、关节软限位二次铰链、力矩变化率全部激活。
// 限位设置使 4 个关节在轨迹起点都位于铰链的二次增长区间，其中关节 1 从下侧越界。
struct ExtendedProblem {
    support::Problem<double> problem;
    std::vector<math::Vector<double, 4>> previous_torque;

    void build() {
        problem = support::make_problem<double>(8);
        problem.spec.refinement = 2;
        problem.loss.torque_rate_weights = math::Vector<double, 4>::from(0.7, 0.4, 0.6, 0.5);
        problem.loss.terminal_angle_weights = math::Vector<double, 4>::from(3.0, 2.0, 2.5, 3.5);
        problem.loss.terminal_velocity_weights = math::Vector<double, 4>::from(1.5, 1.0, 1.2, 0.8);
        problem.loss.limit_weights = math::Vector<double, 4>::from(2.0, 2.5, 1.5, 2.0);
        problem.loss.limit_lower = math::Vector<double, 4>::from(-0.50, -0.08, -0.20, -0.20);
        problem.loss.limit_upper = math::Vector<double, 4>::from(0.15, 0.50, 0.14, 0.03);
        problem.loss.limit_margin = math::Vector<double, 4>::from(0.02, 0.01, 0.01, 0.005);
        previous_torque.resize(1);
        previous_torque[0] = math::Vector<double, 4>::from(0.10, -0.08, 0.02, -0.01);
        problem.loss.previous_torque = previous_torque.data();
    }
};

bool parameter_layout() {
    ModelParameters model = ModelParameters::demo();
    const Real mass = parameters::value(model, 0);
    if (mass != model.bodies[0].mass) {
        return false;
    }
    parameters::set_value(model, 0, Real(1.5));
    if (model.bodies[0].mass != Real(1.5)) {
        return false;
    }
    for (int i = 0; i < parameters::kParameterCount; ++i) {
        if (parameters::names()[static_cast<std::size_t>(i)] == nullptr) {
            return false;
        }
    }
    return true;
}

bool check_double() {
    auto problem = support::make_problem<double>(6);
    problem.spec.refinement = 1;
    ControlGradient<double> control(2);
    ControlGradientOutputT<double> output;
    control.run(support::control_input(problem), output);

    ForwardSimulation<double> forward(1);
    ForwardOutputT<double> forward_output;
    forward.run(support::forward_input(problem), forward_output);

    ParameterSensitivity<double> sensitivity(1);
    ParameterSensitivityOutputT<double> sensitivity_output;
    sensitivity.run(support::parameter_input(problem), sensitivity_output);
    if (sensitivity_output.loss != output.loss || forward_output.loss != output.loss) {
        printf("三种处理器的前向损失不一致\n");
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        if (sensitivity_output.final_state.angle[i] != forward_output.step_end.back().angle[i] ||
            forward_output.step_end.back().angle[i] != output.final_state.angle[i]) {
            printf("三种处理器的末状态不一致\n");
            return false;
        }
    }

    const double epsilon = 1e-6;
    double torque_error = 0.0;
    for (std::size_t k = 0; k < problem.num_steps; ++k) {
        for (int i = 0; i < 4; ++i) {
            support::Problem<double> plus = problem;
            support::Problem<double> minus = problem;
            plus.torque[k][i] += epsilon;
            minus.torque[k][i] -= epsilon;
            const double difference = (forward_loss(forward, plus) - forward_loss(forward, minus)) /
                                      (2.0 * epsilon);
            torque_error = std::max(
                torque_error, std::fabs(difference - static_cast<double>(output.gradient[k][i])) /
                                  std::max(1.0, std::fabs(static_cast<double>(output.gradient[k][i]))));
        }
    }
    printf("控制量梯度与中心差分最大相对偏差 = %.3e\n", torque_error);

    double parameter_error = 0.0;
    for (int index = 0; index < parameters::kParameterCount; ++index) {
        support::Problem<double> plus = problem;
        support::Problem<double> minus = problem;
        const Real value = parameters::value(plus.model, index);
        parameters::set_value(plus.model, index, value + Real(epsilon));
        parameters::set_value(minus.model, index, value - Real(epsilon));
        ParameterSensitivityOutputT<double> plus_output;
        ParameterSensitivityOutputT<double> minus_output;
        ParameterSensitivity<double> plus_handler(1);
        ParameterSensitivity<double> minus_handler(1);
        plus_handler.run(support::parameter_input(plus), plus_output);
        minus_handler.run(support::parameter_input(minus), minus_output);
        const double difference =
            (static_cast<double>(plus_output.loss) - static_cast<double>(minus_output.loss)) /
            (2.0 * epsilon);
        parameter_error = std::max(
            parameter_error,
            std::fabs(difference - static_cast<double>(sensitivity_output.gradient[index])) /
                std::max(1.0, std::fabs(static_cast<double>(sensitivity_output.gradient[index]))));
    }
    printf("参数灵敏度与中心差分最大相对偏差 = %.3e\n", parameter_error);

    if (!(torque_error < 2e-7) || !(parameter_error < 1e-5)) {
        return false;
    }
    return parameter_layout();
}

bool check_extended_loss() {
    ExtendedProblem extended;
    extended.build();
    const support::Problem<double>& problem = extended.problem;

    ControlGradient<double> control(2);
    ControlGradientOutputT<double> control_output;
    control.run(support::control_input(problem), control_output);

    ForwardSimulation<double> forward(1);
    ForwardOutputT<double> forward_output;
    forward.run(support::forward_input(problem), forward_output);

    ParameterSensitivity<double> sensitivity(1);
    ParameterSensitivityOutputT<double> sensitivity_output;
    sensitivity.run(support::parameter_input(problem), sensitivity_output);
    if (sensitivity_output.loss != control_output.loss || forward_output.loss != control_output.loss) {
        printf("扩展损失下三种处理器的前向损失不一致\n");
        return false;
    }

    // 确认限位铰链在算例中确实被激活，且上下两侧都有覆盖
    std::size_t upper_active = 0;
    std::size_t lower_active = 0;
    for (const simulation::StateT<double>& state : forward_output.step_end) {
        for (int i = 0; i < 4; ++i) {
            const double upper = problem.loss.limit_upper[i] - problem.loss.limit_margin[i];
            const double lower = problem.loss.limit_lower[i] + problem.loss.limit_margin[i];
            if (state.angle[i] > upper) {
                ++upper_active;
            } else if (state.angle[i] < lower) {
                ++lower_active;
            }
        }
    }
    printf("扩展损失铰链激活数：上侧 %zu，下侧 %zu（状态总数 %zu）\n", upper_active, lower_active,
           problem.num_steps * 4);
    if (upper_active == 0 || lower_active == 0) {
        printf("限位铰链没有同时覆盖上下两侧\n");
        return false;
    }

    const double epsilon = 1e-6;
    double torque_error = 0.0;
    for (std::size_t k = 0; k < problem.num_steps; ++k) {
        for (int i = 0; i < 4; ++i) {
            support::Problem<double> plus = problem;
            support::Problem<double> minus = problem;
            plus.torque[k][i] += epsilon;
            minus.torque[k][i] -= epsilon;
            const double difference = (forward_loss(forward, plus) - forward_loss(forward, minus)) /
                                      (2.0 * epsilon);
            torque_error =
                std::max(torque_error,
                         std::fabs(difference - control_output.gradient[k][i]) /
                             std::max(1.0, std::fabs(control_output.gradient[k][i])));
        }
    }
    printf("扩展损失控制量梯度与中心差分最大相对偏差 = %.3e\n", torque_error);

    double parameter_error = 0.0;
    for (int index = 0; index < parameters::kParameterCount; ++index) {
        support::Problem<double> plus = problem;
        support::Problem<double> minus = problem;
        const double value = parameters::value(plus.model, index);
        parameters::set_value(plus.model, index, value + epsilon);
        parameters::set_value(minus.model, index, value - epsilon);
        ParameterSensitivityOutputT<double> plus_output;
        ParameterSensitivityOutputT<double> minus_output;
        ParameterSensitivity<double> plus_handler(1);
        ParameterSensitivity<double> minus_handler(1);
        plus_handler.run(support::parameter_input(plus), plus_output);
        minus_handler.run(support::parameter_input(minus), minus_output);
        const double difference =
            (static_cast<double>(plus_output.loss) - static_cast<double>(minus_output.loss)) /
            (2.0 * epsilon);
        parameter_error = std::max(
            parameter_error,
            std::fabs(difference - static_cast<double>(sensitivity_output.gradient[index])) /
                std::max(1.0, std::fabs(static_cast<double>(sensitivity_output.gradient[index]))));
    }
    printf("扩展损失参数灵敏度与中心差分最大相对偏差 = %.3e\n", parameter_error);

    if (!(torque_error < 2e-7) || !(parameter_error < 1e-5)) {
        return false;
    }
    return true;
}

bool check_extended_batch() {
    ExtendedProblem extended;
    extended.build();
    support::Problem<double> single = extended.problem;
    // 批量路径不携带 previous_torque，单样本对照同样置空
    single.loss.previous_torque = nullptr;
    ControlGradient<double> single_handler(1);
    ControlGradientOutputT<double> single_output;
    single_handler.run(support::control_input(single), single_output);

    const std::size_t batch = 3;
    const std::size_t steps = single.num_steps;
    std::vector<simulation::StateT<double>> initial(batch, single.initial);
    std::vector<BaseStateT<double>> base(batch * steps);
    std::vector<math::Vector<double, 4>> torque(batch * steps);
    std::vector<TerrainT<double>> terrains(batch, single.terrain);
    std::vector<math::Vector<double, 4>> target_angle(batch * steps);
    std::vector<math::Vector<double, 4>> target_velocity(batch * steps);
    for (std::size_t s = 0; s < batch; ++s) {
        for (std::size_t k = 0; k < steps; ++k) {
            base[s * steps + k] = single.base[k];
            torque[s * steps + k] = single.torque[k];
            target_angle[s * steps + k] = single.target_angle[k];
            target_velocity[s * steps + k] = single.target_velocity[k];
        }
    }
    ControlGradientBatchInputT<double> input;
    input.model = single.model;
    input.terrain = terrains.data();
    input.initial = initial.data();
    input.base = base.data();
    input.torque = torque.data();
    input.num_steps = steps;
    input.spec = single.spec;
    input.loss_weights.angle_weights = single.loss.angle_weights;
    input.loss_weights.velocity_weights = single.loss.velocity_weights;
    input.loss_weights.torque_weights = single.loss.torque_weights;
    input.loss_weights.torque_rate_weights = single.loss.torque_rate_weights;
    input.loss_weights.terminal_angle_weights = single.loss.terminal_angle_weights;
    input.loss_weights.terminal_velocity_weights = single.loss.terminal_velocity_weights;
    input.loss_weights.limit_weights = single.loss.limit_weights;
    input.loss_weights.limit_lower = single.loss.limit_lower;
    input.loss_weights.limit_upper = single.loss.limit_upper;
    input.loss_weights.limit_margin = single.loss.limit_margin;
    input.target_angle = target_angle.data();
    input.target_velocity = target_velocity.data();
    ControlGradient<double> batch_handler(2);
    ControlGradientBatchOutputT<double> batch_output;
    batch_handler.runBatch(batch, input, batch_output);

    const double loss_deviation =
        std::fabs(batch_output.mean_loss - single_output.loss) /
        std::max(1.0, std::fabs(single_output.loss));
    printf("批量 LossWeights 路径与单样本损失相对偏差 = %.3e\n", loss_deviation);
    if (!(loss_deviation < 1e-12)) {
        return false;
    }
    for (std::size_t s = 0; s < batch; ++s) {
        for (std::size_t k = 0; k < steps; ++k) {
            for (int i = 0; i < 4; ++i) {
                if (batch_output.gradient[s * steps + k][i] != single_output.gradient[k][i]) {
                    printf("批量控制量梯度与单样本不一致\n");
                    return false;
                }
            }
        }
    }
    printf("批量 LossWeights 路径逐样本控制量梯度与单样本逐位一致\n");
    return true;
}

bool check_float() {
    auto problem = support::make_problem<float>(20);
    ControlGradient<float> handler(4);
    ControlGradientOutputT<float> output;
    handler.run(support::control_input(problem), output);
    auto problem_double = support::make_problem<double>(20);
    ControlGradient<double> double_handler(4);
    ControlGradientOutputT<double> double_output;
    double_handler.run(support::control_input(problem_double), double_output);
    const double error = std::fabs(static_cast<double>(output.loss) -
                                   static_cast<double>(double_output.loss)) /
                         std::fabs(static_cast<double>(double_output.loss));
    printf("单精度与双精度前向损失相对偏差 = %.3e\n", error);
    return error < 1e-5 && std::isfinite(static_cast<double>(output.loss));
}

}  // namespace

int main() {
    if (!check_double()) {
        printf("校验失败\n");
        return 1;
    }
    if (!check_extended_loss()) {
        printf("校验失败\n");
        return 1;
    }
    if (!check_extended_batch()) {
        printf("校验失败\n");
        return 1;
    }
    if (!check_float()) {
        printf("校验失败\n");
        return 1;
    }
    printf("校验通过\n");
    return 0;
}
