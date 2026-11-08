// 校验两个可求导仿真模块：
//   1) gimbal::control_gradient  ：对每步 4 个关节力矩反向求梯度（离散伴随）
//   2) gimbal::parameter_gradient：对 56 个模型参数前向求灵敏度（前向模式）
// 检查项：
//   * 两个模块的前向结果（损失、逐主步末状态、阶段状态与阶段底盘量）逐位一致；
//   * 两类解析梯度与手工中心差分一致（多组步数 / 细化倍数 / 主步长）。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "gimbal_dynamics/control_gradient.hpp"
#include "gimbal_dynamics/parameter_gradient.hpp"
#include "gimbal_dynamics/simulation.hpp"

using namespace gimbal;

namespace {

struct Config {
    std::size_t num_steps;
    int refinement;
    double step;
    double torque_scale;
};

struct Inputs {
    ModelParameters model;
    Terrain terrain;
    simulation::State initial;
    std::vector<BaseState> base;
    std::vector<Eigen::Vector4d> torque;
    simulation::TrajectorySpec spec;
    simulation::LossSpec loss;
    std::vector<Eigen::Vector4d> target_angle;
    std::vector<Eigen::Vector4d> target_velocity;
};

Inputs make_inputs(const Config& config) {
    Inputs inputs;
    inputs.model = ModelParameters::demo();
    inputs.terrain.slope_angle = 0.05;
    inputs.terrain.slope_direction = 0.7;
    inputs.initial.angle = Eigen::Vector4d(0.20, -0.10, 0.15, 0.05);
    inputs.initial.velocity = Eigen::Vector4d(0.30, -0.20, 0.10, -0.05);
    inputs.spec.step = config.step;
    inputs.spec.refinement = config.refinement;

    const std::size_t num_steps = config.num_steps;
    inputs.base.resize(num_steps);
    inputs.torque.resize(num_steps);
    inputs.target_angle.resize(num_steps);
    inputs.target_velocity.resize(num_steps);
    for (std::size_t k = 0; k < num_steps; ++k) {
        const double time = static_cast<double>(k) * config.step;
        BaseState& base = inputs.base[k];
        base.yaw = 0.2 + 0.4 * time;
        base.yaw_rate = 0.4;
        base.yaw_acceleration = 0.3;
        base.acceleration = Eigen::Vector3d(0.2, 0.15, 0.05);
        inputs.torque[k] =
            config.torque_scale *
            Eigen::Vector4d(0.20 + 0.01 * static_cast<double>(k),
                            -0.12 + 0.008 * static_cast<double>(k), 0.06, -0.04);
        for (int i = 0; i < 4; ++i) {
            inputs.target_angle[k][i] = 0.10 * std::sin(0.5 * time + 0.7 * i);
            inputs.target_velocity[k][i] = 0.20 * std::cos(0.3 * time + 0.4 * i);
        }
    }
    inputs.loss.angle_weights = Eigen::Vector4d(1.0, 2.0, 0.8, 1.5);
    inputs.loss.velocity_weights = Eigen::Vector4d(0.3, 0.5, 0.2, 0.4);
    inputs.loss.torque_weights = Eigen::Vector4d(0.05, 0.05, 0.03, 0.03);
    inputs.loss.target_angle = inputs.target_angle.data();
    inputs.loss.target_velocity = inputs.target_velocity.data();
    return inputs;
}

// 参数的典型量级，用于选取中心差分的步长（参数值为 0 时也能给出合适步长）
double parameter_characteristic(int index) {
    if (index < parameters::kBodyCount * parameters::kBodyParameterCount) {
        const int slot = index % parameters::kBodyParameterCount;
        if (slot == 0) {
            return 0.5;
        }
        if (slot <= 3) {
            return 0.02;
        }
        return 1e-3;
    }
    if (index < parameters::kParameterCount - parameters::kDampingParameterCount) {
        return 0.05;
    }
    return 0.01;
}

double parameter_step(const ModelParameters& model, int index) {
    const double value = parameters::value(model, index);
    return 1e-5 * std::max(std::fabs(value), parameter_characteristic(index));
}

bool check_forward_consistency(const Inputs& inputs, bool verbose) {
    control_gradient::Workspace control_workspace;
    parameter_gradient::Workspace parameter_workspace;
    simulation::Trace reference_trace;
    simulation::State control_final;
    simulation::State parameter_final;
    const std::size_t num_steps = inputs.base.size();

    const double control_loss = control_gradient::simulateAndGradient(
        inputs.model, inputs.terrain, inputs.initial, inputs.base.data(), inputs.torque.data(),
        num_steps, inputs.spec, inputs.loss, control_workspace, nullptr, &control_final);
    const double parameter_loss = parameter_gradient::simulateAndSensitivity(
        inputs.model, inputs.terrain, inputs.initial, inputs.base.data(), inputs.torque.data(),
        num_steps, inputs.spec, inputs.loss, parameter_workspace, nullptr, &parameter_final);
    const double reference_loss = simulation::simulate(
        inputs.model, inputs.terrain, inputs.initial, inputs.base.data(), inputs.torque.data(),
        num_steps, inputs.spec, inputs.loss, reference_trace);

    bool identical = control_loss == parameter_loss && parameter_loss == reference_loss;
    for (std::size_t k = 0; k < num_steps; ++k) {
        const simulation::State& a = control_workspace.trace.step_end[k];
        const simulation::State& b = parameter_workspace.trace.step_end[k];
        const simulation::State& c = reference_trace.step_end[k];
        for (int i = 0; i < 4; ++i) {
            identical = identical && a.angle[i] == b.angle[i] && b.angle[i] == c.angle[i];
            identical = identical &&
                        a.velocity[i] == b.velocity[i] && b.velocity[i] == c.velocity[i];
        }
    }
    const std::size_t stage_count =
        num_steps * static_cast<std::size_t>(inputs.spec.refinement) * 4;
    for (std::size_t index = 0; index < stage_count; ++index) {
        const simulation::State& a = control_workspace.trace.stage_state[index];
        const simulation::State& b = parameter_workspace.trace.stage_state[index];
        for (int i = 0; i < 4; ++i) {
            identical = identical && a.angle[i] == b.angle[i] && a.velocity[i] == b.velocity[i];
        }
        const BaseState& base_a = control_workspace.trace.stage_base[index];
        const BaseState& base_b = parameter_workspace.trace.stage_base[index];
        identical = identical && base_a.yaw == base_b.yaw &&
                    base_a.yaw_rate == base_b.yaw_rate &&
                    base_a.yaw_acceleration == base_b.yaw_acceleration &&
                    base_a.acceleration == base_b.acceleration;
    }
    for (int i = 0; i < 4; ++i) {
        identical = identical && control_final.angle[i] == parameter_final.angle[i] &&
                    control_final.angle[i] == reference_trace.step_end.back().angle[i] &&
                    control_final.velocity[i] == parameter_final.velocity[i] &&
                    control_final.velocity[i] == reference_trace.step_end.back().velocity[i];
    }

    if (verbose) {
        printf("前向结果一致性：\n");
        printf("  control_gradient  损失 = %.17g\n", control_loss);
        printf("  parameter_gradient 损失 = %.17g\n", parameter_loss);
        printf("  simulation         损失 = %.17g\n", reference_loss);
        printf("  损失、逐主步末状态、阶段状态、阶段底盘量、末状态 %s\n",
               identical ? "逐位一致" : "存在差异");
    }
    return identical;
}

bool nearly_equal(double actual, double expected) {
    return std::fabs(actual - expected) <= 1e-12 * std::max(1.0, std::fabs(expected));
}

// 校验参数编号与模型字段的对应关系
bool check_parameter_layout() {
    bool ok = true;
    ModelParameters model = ModelParameters::demo();

    for (int index = 0; index < parameters::kParameterCount; ++index) {
        const double original_value = parameters::value(model, index);
        parameters::set_value(model, index, original_value + 0.25);
        ok = ok && nearly_equal(parameters::value(model, index), original_value + 0.25);
        parameters::set_value(model, index, original_value);
        ok = ok && nearly_equal(parameters::value(model, index), original_value);
    }

    // 代表编号与模型字段直接对照
    parameters::set_value(model, 0, 1.5);
    ok = ok && nearly_equal(model.bodies[0].mass, 1.5);
    parameters::set_value(model, 1, 0.03);
    ok = ok && nearly_equal(model.bodies[0].com[0], 0.03);
    parameters::set_value(model, 3, 0.04);
    ok = ok && nearly_equal(model.bodies[0].com[2], 0.04);
    parameters::set_value(model, 4, 0.002);
    ok = ok && nearly_equal(model.bodies[0].inertia(0, 0), 0.002);
    parameters::set_value(model, 8, 0.0007);
    ok = ok && nearly_equal(model.bodies[0].inertia(0, 2), 0.0007) &&
         nearly_equal(model.bodies[0].inertia(2, 0), 0.0007);
    parameters::set_value(model, 10, 0.8);
    ok = ok && nearly_equal(model.bodies[1].mass, 0.8);
    parameters::set_value(model, 40, 0.11);
    ok = ok && nearly_equal(model.joint_offsets[0][0], 0.11);
    parameters::set_value(model, 42, 0.12);
    ok = ok && nearly_equal(model.joint_offsets[0][2], 0.12);
    parameters::set_value(model, 52, 0.02);
    ok = ok && nearly_equal(model.damping[0], 0.02);
    parameters::set_value(model, 55, 0.03);
    ok = ok && nearly_equal(model.damping[3], 0.03);

    for (int index = 0; index < parameters::kParameterCount; ++index) {
        ok = ok && parameters::names()[index] != nullptr;
    }
    return ok;
}

double check_torque_gradient(const Inputs& inputs, bool verbose) {
    control_gradient::Workspace workspace;
    const std::size_t num_steps = inputs.base.size();
    std::vector<Eigen::Vector4d> gradient(num_steps);
    const double loss = control_gradient::simulateAndGradient(
        inputs.model, inputs.terrain, inputs.initial, inputs.base.data(), inputs.torque.data(),
        num_steps, inputs.spec, inputs.loss, workspace, gradient.data(), nullptr);

    double max_relative = 0.0;
    if (verbose) {
        printf("\n力矩梯度（离散伴随，反向模式）与中心差分对比，损失 = %.12g\n", loss);
        printf("%3s %2s | %18s | %18s | %10s\n", "步", "分量", "解析梯度", "中心差分",
               "相对误差");
    }
    for (std::size_t k = 0; k < num_steps; ++k) {
        for (int i = 0; i < 4; ++i) {
            const double step = 1e-6 * std::max(1.0, std::fabs(inputs.torque[k][i]));
            std::vector<Eigen::Vector4d> plus = inputs.torque;
            std::vector<Eigen::Vector4d> minus = inputs.torque;
            plus[k][i] += step;
            minus[k][i] -= step;
            simulation::Trace trace_plus;
            simulation::Trace trace_minus;
            const double loss_plus = simulation::simulate(
                inputs.model, inputs.terrain, inputs.initial, inputs.base.data(), plus.data(),
                num_steps, inputs.spec, inputs.loss, trace_plus);
            const double loss_minus = simulation::simulate(
                inputs.model, inputs.terrain, inputs.initial, inputs.base.data(), minus.data(),
                num_steps, inputs.spec, inputs.loss, trace_minus);
            const double finite_difference = (loss_plus - loss_minus) / (2.0 * step);
            const double relative = std::fabs(finite_difference - gradient[k][i]) /
                                    std::max(1.0, std::fabs(finite_difference));
            max_relative = std::max(max_relative, relative);
            if (verbose) {
                printf("%3zu %2d | %+.12e | %+.12e | %.3e\n", k, i, gradient[k][i],
                       finite_difference, relative);
            }
        }
    }
    return max_relative;
}

double check_parameter_gradient(const Inputs& inputs, bool verbose) {
    parameter_gradient::Workspace workspace;
    const std::size_t num_steps = inputs.base.size();
    parameter_gradient::Gradient gradient;
    const double loss = parameter_gradient::simulateAndSensitivity(
        inputs.model, inputs.terrain, inputs.initial, inputs.base.data(), inputs.torque.data(),
        num_steps, inputs.spec, inputs.loss, workspace, &gradient, nullptr);

    double max_relative = 0.0;
    if (verbose) {
        printf("\n模型参数灵敏度（前向灵敏度，前向模式）与中心差分对比，损失 = %.12g\n", loss);
        printf("%2s %-16s | %18s | %18s | %10s\n", "编号", "参数名", "解析梯度", "中心差分",
               "相对误差");
    }
    for (int index = 0; index < parameters::kParameterCount; ++index) {
        const double value = parameters::value(inputs.model, index);
        const double step = parameter_step(inputs.model, index);
        ModelParameters plus = inputs.model;
        ModelParameters minus = inputs.model;
        parameters::set_value(plus, index, value + step);
        parameters::set_value(minus, index, value - step);
        simulation::Trace trace_plus;
        simulation::Trace trace_minus;
        const double loss_plus =
            simulation::simulate(plus, inputs.terrain, inputs.initial, inputs.base.data(),
                                 inputs.torque.data(), num_steps, inputs.spec, inputs.loss,
                                 trace_plus);
        const double loss_minus =
            simulation::simulate(minus, inputs.terrain, inputs.initial, inputs.base.data(),
                                 inputs.torque.data(), num_steps, inputs.spec, inputs.loss,
                                 trace_minus);
        const double finite_difference = (loss_plus - loss_minus) / (2.0 * step);
        const double relative = std::fabs(finite_difference - gradient[index]) /
                                std::max(1.0, std::fabs(finite_difference));
        max_relative = std::max(max_relative, relative);
        if (verbose) {
            printf("%2d %-16s | %+.12e | %+.12e | %.3e\n", index, parameters::names()[index],
                   gradient[index], finite_difference, relative);
        }
    }
    return max_relative;
}

struct CaseResult {
    bool forward_identical;
    double torque_relative;
    double parameter_relative;
};

CaseResult run_case(const Config& config, bool verbose) {
    const Inputs inputs = make_inputs(config);
    CaseResult result;
    result.forward_identical = check_forward_consistency(inputs, verbose);
    result.torque_relative = check_torque_gradient(inputs, verbose);
    result.parameter_relative = check_parameter_gradient(inputs, verbose);
    return result;
}

}  // namespace

int main() {
    const bool layout_ok = check_parameter_layout();
    printf("参数编号与模型字段对应关系：%s\n", layout_ok ? "一致" : "不一致");

    const Config main_config{8, 2, 0.01, 1.0};
    const CaseResult main_result = run_case(main_config, true);

    const Config extra_configs[] = {
        {1, 1, 0.01, 1.0},
        {5, 4, 0.02, 0.5},
        {12, 3, 0.005, 2.0},
    };
    constexpr int kExtraCount = 3;
    CaseResult extra_results[kExtraCount];
    for (int c = 0; c < kExtraCount; ++c) {
        extra_results[c] = run_case(extra_configs[c], false);
    }

    printf("\n其余工况（逐步：步数 / 细化倍数 / 主步长 / 力矩缩放）：\n");
    printf("%3s %2s %6s %6s | %8s | %12s | %12s\n", "步数", "细化", "步长", "缩放",
           "前向一致", "力矩最大误差", "参数最大误差");
    for (int c = 0; c < kExtraCount; ++c) {
        printf("%3zu %2d %6.3f %6.2f | %8s | %12.3e | %12.3e\n", extra_configs[c].num_steps,
               extra_configs[c].refinement, extra_configs[c].step, extra_configs[c].torque_scale,
               extra_results[c].forward_identical ? "是" : "否",
               extra_results[c].torque_relative, extra_results[c].parameter_relative);
    }

    double max_torque = main_result.torque_relative;
    double max_parameter = main_result.parameter_relative;
    bool all_forward_identical = main_result.forward_identical;
    for (int c = 0; c < kExtraCount; ++c) {
        max_torque = std::max(max_torque, extra_results[c].torque_relative);
        max_parameter = std::max(max_parameter, extra_results[c].parameter_relative);
        all_forward_identical = all_forward_identical && extra_results[c].forward_identical;
    }

    printf("\n全部工况最大相对误差：力矩梯度 %.3e，参数灵敏度 %.3e\n", max_torque,
           max_parameter);
    const bool ok =
        layout_ok && all_forward_identical && max_torque < 1e-6 && max_parameter < 1e-5;
    printf("%s\n", ok ? "校验通过" : "校验失败");
    return ok ? 0 : 1;
}
