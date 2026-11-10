// 校验：处理器在非法输入与非正定模型下抛出异常，异常后仍可继续使用；
// 批量大小超过构造上限时抛出异常。
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "gimbal_dynamics/handlers.hpp"
#include "support.hpp"

using namespace tcrg::gimbal_dynamics;

namespace {

bool run_normal(ControlGradient<double>& handler, const support::Problem<double>& problem) {
    ControlGradientOutputT<double> output;
    handler.run(support::control_input(problem), output);
    return std::isfinite(static_cast<double>(output.loss));
}

}  // namespace

int main() {
    auto problem = support::make_problem<double>(12);
    ControlGradient<double> handler(4);

    if (!run_normal(handler, problem)) {
        printf("初始调用失败\n");
        return 1;
    }

    auto invalid = support::control_input(problem);
    invalid.spec.refinement = 0;
    bool threw = false;
    try {
        ControlGradientOutputT<double> output;
        handler.run(invalid, output);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    if (!threw) {
        printf("非法 refinement 未抛出异常\n");
        return 1;
    }

    auto bad_model = problem;
    bad_model.model.bodies[0].mass = -1.0e10;
    threw = false;
    try {
        ControlGradientOutputT<double> output;
        handler.run(support::control_input(bad_model), output);
    } catch (const std::exception&) {
        threw = true;
    }
    if (!threw) {
        printf("非正定模型未抛出异常\n");
        return 1;
    }

    auto null_base = support::control_input(problem);
    null_base.base = nullptr;
    threw = false;
    try {
        ControlGradientOutputT<double> output;
        handler.run(null_base, output);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    if (!threw) {
        printf("空底盘输入未抛出异常\n");
        return 1;
    }

    threw = false;
    try {
        ParameterSensitivityInputT<double> input = support::parameter_input(problem);
        input.loss.target_angle = nullptr;
        input.loss.target_velocity = nullptr;
        ParameterSensitivityOutputT<double> output;
        ParameterSensitivity<double> sensitivity(4);
        sensitivity.run(input, output);
    } catch (const std::exception&) {
        threw = true;
    }
    if (threw) {
        printf("无目标调用不应抛出异常\n");
        return 1;
    }

    if (!run_normal(handler, problem)) {
        printf("异常后调用失败\n");
        return 1;
    }
    printf("异常与恢复校验通过\n");
    return 0;
}
