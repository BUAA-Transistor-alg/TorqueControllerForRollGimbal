// 校验：控制量梯度与参数灵敏度对有限差分一致；前向损失、末状态在三种处理器间一致；
// 单精度与双精度结果接近；参数布局与编号正确。
#include <cmath>
#include <cstdio>
#include <vector>

#include "gimbal_dynamics/handlers.hpp"
#include "gimbal_dynamics/parameters.hpp"
#include "support.hpp"

using namespace gimbal;

namespace {

template <typename Real>
Real forward_loss(ForwardSimulation<Real>& handler, const support::Problem<Real>& problem) {
    ForwardOutputT<Real> output;
    handler.run(support::forward_input(problem), output);
    return output.loss;
}

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
    if (!check_float()) {
        printf("校验失败\n");
        return 1;
    }
    printf("校验通过\n");
    return 0;
}
