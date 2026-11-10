// 校验：处理器线程数由构造参数决定，缺省时使用默认值；批量大小由每次调用指定；
// 结果与串行一致；单精度与双精度结果接近。
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "gimbal_dynamics/handlers.hpp"
#include "support.hpp"

using namespace tcrg::gimbal_dynamics;

int main() {
    auto problem = support::make_problem<double>(20);
    auto input = support::control_input(problem);

    ControlGradient<double> serial(1);
    ControlGradientOutputT<double> serial_output;
    serial.run(input, serial_output);

    ControlGradient<double> parallel(8);
    ControlGradientOutputT<double> parallel_output;
    parallel.run(input, parallel_output);
    if (serial_output.loss != parallel_output.loss) {
        printf("不同线程数损失不一致\n");
        return 1;
    }
    for (std::size_t k = 0; k < problem.num_steps; ++k) {
        for (int i = 0; i < 4; ++i) {
            if (serial_output.gradient[k][i] != parallel_output.gradient[k][i]) {
                printf("不同线程数梯度不一致\n");
                return 1;
            }
        }
    }

    ControlGradient<double> default_control;
    ControlGradientOutputT<double> default_control_output;
    default_control.run(input, default_control_output);
    if (default_control_output.loss != parallel_output.loss) {
        printf("默认线程数与 8 线程结果不一致\n");
        return 1;
    }

    ForwardSimulation<double> default_forward;
    ForwardSimulation<double> explicit_forward(1);
    ForwardOutputT<double> default_forward_output;
    ForwardOutputT<double> explicit_forward_output;
    default_forward.run(support::forward_input(problem), default_forward_output);
    explicit_forward.run(support::forward_input(problem), explicit_forward_output);
    if (default_forward_output.loss != explicit_forward_output.loss) {
        printf("前向仿真默认线程数结果不一致\n");
        return 1;
    }

    ParameterSensitivity<double> default_parameter;
    ParameterSensitivityOutputT<double> default_parameter_output;
    default_parameter.run(support::parameter_input(problem), default_parameter_output);
    ParameterSensitivity<double> explicit_parameter(8);
    ParameterSensitivityOutputT<double> explicit_parameter_output;
    explicit_parameter.run(support::parameter_input(problem), explicit_parameter_output);
    if (default_parameter_output.loss != explicit_parameter_output.loss) {
        printf("参数灵敏度默认线程数结果不一致\n");
        return 1;
    }

    const std::size_t batch = 3;
    ControlGradientBatchInputT<double> batch_input;
    batch_input.model = problem.model;
    std::vector<simulation::StateT<double>> initial(batch, problem.initial);
    std::vector<BaseStateT<double>> base(batch * problem.num_steps);
    std::vector<math::Vector<double, 4>> torque(batch * problem.num_steps);
    std::vector<TerrainT<double>> terrain_batch(batch, problem.terrain);
    std::vector<math::Vector<double, 4>> target_angle(batch * problem.num_steps);
    std::vector<math::Vector<double, 4>> target_velocity(batch * problem.num_steps);
    for (std::size_t s = 0; s < batch; ++s) {
        for (std::size_t k = 0; k < problem.num_steps; ++k) {
            base[s * problem.num_steps + k] = problem.base[k];
            torque[s * problem.num_steps + k] = problem.torque[k];
            target_angle[s * problem.num_steps + k] = problem.target_angle[k];
            target_velocity[s * problem.num_steps + k] = problem.target_velocity[k];
        }
    }
    batch_input.initial = initial.data();
    batch_input.base = base.data();
    batch_input.torque = torque.data();
    batch_input.num_steps = problem.num_steps;
    batch_input.spec = problem.spec;
    batch_input.terrain = terrain_batch.data();
    batch_input.loss_weights.angle_weights = problem.loss.angle_weights;
    batch_input.loss_weights.velocity_weights = problem.loss.velocity_weights;
    batch_input.loss_weights.torque_weights = problem.loss.torque_weights;
    batch_input.target_angle = target_angle.data();
    batch_input.target_velocity = target_velocity.data();
    ControlGradient<double> batch_handler(8);
    ControlGradientBatchOutputT<double> batch_output;
    batch_handler.runBatch(batch, batch_input, batch_output);
    for (std::size_t s = 0; s < batch; ++s) {
        for (std::size_t k = 0; k < problem.num_steps; ++k) {
            for (int i = 0; i < 4; ++i) {
                if (batch_output.gradient[s * problem.num_steps + k][i] !=
                    serial_output.gradient[k][i]) {
                    printf("批量结果与单样本不一致\n");
                    return 1;
                }
            }
        }
    }

    ParameterSensitivity<double> sensitivity_handler(8);
    ParameterSensitivityOutputT<double> single_sensitivity;
    sensitivity_handler.run(support::parameter_input(problem), single_sensitivity);
    ParameterSensitivityBatchInputT<double> parameter_batch_input;
    parameter_batch_input.model = problem.model;
    parameter_batch_input.initial = initial.data();
    parameter_batch_input.base = base.data();
    parameter_batch_input.torque = torque.data();
    parameter_batch_input.num_steps = problem.num_steps;
    parameter_batch_input.spec = problem.spec;
    parameter_batch_input.terrain = terrain_batch.data();
    parameter_batch_input.loss_weights.angle_weights = problem.loss.angle_weights;
    parameter_batch_input.loss_weights.velocity_weights = problem.loss.velocity_weights;
    parameter_batch_input.loss_weights.torque_weights = problem.loss.torque_weights;
    parameter_batch_input.target_angle = target_angle.data();
    parameter_batch_input.target_velocity = target_velocity.data();
    ParameterSensitivityBatchOutputT<double> batch_sensitivity;
    sensitivity_handler.runBatch(batch, parameter_batch_input, batch_sensitivity);
    bool parameter_batch_ok = std::fabs(batch_sensitivity.mean_loss - single_sensitivity.loss) <
                              1e-12 * std::max(1.0, std::fabs(single_sensitivity.loss));
    for (int i = 0; i < parameters::kParameterCount; ++i) {
        parameter_batch_ok =
            parameter_batch_ok &&
            std::fabs(batch_sensitivity.gradient[i] - single_sensitivity.gradient[i]) <
                1e-12 * std::max(1.0, std::fabs(single_sensitivity.gradient[i]));
    }
    if (!parameter_batch_ok) {
        printf("参数灵敏度批量平均值与单样本不一致\n");
        return 1;
    }

    auto problem_float = support::make_problem<float>(20);
    ControlGradient<float> float_handler(8);
    ControlGradientOutputT<float> float_output;
    float_handler.run(support::control_input(problem_float), float_output);
    const double loss_error =
        std::fabs(static_cast<double>(float_output.loss) - static_cast<double>(serial_output.loss)) /
        std::fabs(static_cast<double>(serial_output.loss));
    double gradient_error = 0.0;
    for (std::size_t k = 0; k < problem.num_steps; ++k) {
        for (int i = 0; i < 4; ++i) {
            gradient_error = std::max(
                gradient_error,
                std::fabs(static_cast<double>(float_output.gradient[k][i]) -
                          static_cast<double>(serial_output.gradient[k][i])) /
                    std::max(1.0, std::fabs(static_cast<double>(serial_output.gradient[k][i]))));
        }
    }
    printf("单精度与双精度：损失相对偏差 %.3e，梯度最大相对偏差 %.3e\n", loss_error, gradient_error);
    if (!(loss_error < 1e-5) || !(gradient_error < 1e-4)) {
        printf("精度切换校验失败\n");
        return 1;
    }
    printf("线程数与批量配置校验通过\n");
    return 0;
}
