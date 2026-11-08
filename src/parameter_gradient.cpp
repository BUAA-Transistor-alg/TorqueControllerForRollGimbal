#include "gimbal_dynamics/parameter_gradient.hpp"

#include <stdexcept>

#include "model_evaluation.hpp"
#include "simulation_core.hpp"
#include "stage_data.hpp"
#include "stage_jacobians.hpp"

namespace gimbal {
namespace parameter_gradient {

void Workspace::resize(std::size_t num_steps, int refinement) {
    trace.resize(num_steps, refinement);
}

void Workspace::release() {
    trace.release();
}

double simulateAndSensitivity(const ModelParameters& model, const Terrain& terrain,
                              const simulation::State& initial, const BaseState* base,
                              const math::Vector4d* torque, std::size_t num_steps,
                              const simulation::TrajectorySpec& spec,
                              const simulation::LossSpec& loss, Workspace& workspace,
                              Gradient* gradient, simulation::State* final_state) {
    simulation::validate(spec, num_steps);
    workspace.resize(num_steps, spec.refinement);
    const double h = spec.step / static_cast<double>(spec.refinement);
    const detail::ScalarModelParameters<double> scalar_model = detail::convert_model<double>(model);

    // 单遍完成前向仿真、阶段 Jacobian 与灵敏度传播；
    // 每个主步结束时用当前灵敏度矩阵把步末损失梯度累加进总梯度。
    math::Matrix<double, 8, parameters::kParameterCount> sensitivity =
        math::Matrix<double, 8, parameters::kParameterCount>::zero();
    Gradient total = Gradient::zero();
    detail::StageJacobians stage[4];
    detail::ParameterJacobian stage_parameter[4];
    const auto stage_visitor = [&](std::size_t stage_index, const detail::StageData& data) {
        const std::size_t substep = stage_index / 4;
        const int t = static_cast<int>(stage_index % 4);
        stage[t].state = math::Matrix8d::zero();
        stage_parameter[t] = detail::evaluate_parameter_jacobian(data, scalar_model, stage[t].state);
        if (t == 3) {
            math::Matrix8d propagation;
            detail::ParameterJacobian parameter_jacobian;
            detail::assemble_parameter_substep(stage, stage_parameter, h, propagation,
                                               parameter_jacobian);
            sensitivity = propagation * sensitivity + parameter_jacobian;
            (void)substep;
        }
    };
    const auto step_visitor = [&](std::size_t k, const simulation::State& step_end_state) {
        total += transpose(sensitivity) *
                 simulation::step_loss_gradient(loss, num_steps, k, step_end_state);
    };
    const double loss_value = detail::simulate_trajectory(
        model, terrain, initial, base, torque, num_steps, spec, loss, workspace.trace, false,
        stage_visitor, step_visitor);
    if (final_state != nullptr) {
        *final_state = workspace.trace.step_end.back();
    }
    if (gradient != nullptr) {
        *gradient = total;
    }
    return loss_value;
}

}  // namespace parameter_gradient
}  // namespace gimbal
