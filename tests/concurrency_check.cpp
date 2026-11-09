// 校验：多个处理器实例从不同外部线程并发调用时结果与串行一致；
// 批量调用与单样本调用逐位一致。
#include <array>
#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>

#include "gimbal_dynamics/handlers.hpp"
#include "support.hpp"

using namespace gimbal;

namespace {

bool same_gradient(const std::vector<math::Vector<double, 4>>& a,
                   const std::vector<math::Vector<double, 4>>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t k = 0; k < a.size(); ++k) {
        for (int i = 0; i < 4; ++i) {
            if (a[k][i] != b[k][i]) {
                return false;
            }
        }
    }
    return true;
}

bool same_sensitivity(const math::Vector<double, parameters::kParameterCount>& a,
                      const math::Vector<double, parameters::kParameterCount>& b) {
    for (int i = 0; i < parameters::kParameterCount; ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    auto problem = support::make_problem<double>(12);
    auto input = support::control_input(problem);
    auto parameter_input = support::parameter_input(problem);

    ControlGradient<double> reference(1, 1);
    ControlGradientOutputT<double> reference_output;
    reference.run(input, reference_output);
    ParameterSensitivity<double> reference_parameter(1, 1);
    ParameterSensitivityOutputT<double> reference_parameter_output;
    reference_parameter.run(parameter_input, reference_parameter_output);

    constexpr int kThreads = 3;
    std::array<bool, kThreads> control_ok{};
    std::array<bool, kThreads> parameter_ok{};
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            ControlGradient<double> handler(2, 1);
            ControlGradientOutputT<double> output;
            handler.run(input, output);
            control_ok[t] = output.loss == reference_output.loss &&
                            same_gradient(output.gradient, reference_output.gradient);
            ParameterSensitivity<double> sensitivity(2, 1);
            ParameterSensitivityOutputT<double> sensitivity_output;
            sensitivity.run(parameter_input, sensitivity_output);
            parameter_ok[t] = sensitivity_output.loss == reference_parameter_output.loss &&
                              same_sensitivity(sensitivity_output.gradient,
                                               reference_parameter_output.gradient);
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    for (int t = 0; t < kThreads; ++t) {
        if (!control_ok[t] || !parameter_ok[t]) {
            printf("并发调用结果与串行不一致\n");
            return 1;
        }
    }

    const std::size_t batch = 8;
    std::vector<simulation::StateT<double>> initial(batch, problem.initial);
    std::vector<BaseStateT<double>> base(batch * problem.num_steps);
    std::vector<math::Vector<double, 4>> torque(batch * problem.num_steps);
    std::vector<simulation::LossSpecT<double>> losses(batch, problem.loss);
    for (std::size_t s = 0; s < batch; ++s) {
        for (std::size_t k = 0; k < problem.num_steps; ++k) {
            base[s * problem.num_steps + k] = problem.base[k];
            torque[s * problem.num_steps + k] = problem.torque[k];
        }
    }
    ControlGradientBatchInputT<double> batch_input;
    batch_input.model = problem.model;
    batch_input.terrain = problem.terrain;
    batch_input.initial = initial.data();
    batch_input.base = base.data();
    batch_input.torque = torque.data();
    batch_input.num_steps = problem.num_steps;
    batch_input.spec = problem.spec;
    batch_input.loss = losses.data();
    ControlGradient<double> batch_handler(4, batch);
    ControlGradientBatchOutputT<double> batch_output;
    batch_handler.runBatch(batch, batch_input, batch_output);
    std::vector<math::Vector<double, 4>> sample_gradient(problem.num_steps);
    for (std::size_t s = 0; s < batch; ++s) {
        for (std::size_t k = 0; k < problem.num_steps; ++k) {
            sample_gradient[k] = batch_output.gradient[s * problem.num_steps + k];
        }
        if (!same_gradient(sample_gradient, reference_output.gradient)) {
            printf("批量结果与单样本结果不一致\n");
            return 1;
        }
    }
    printf("并发与批量一致性校验通过\n");
    return 0;
}
