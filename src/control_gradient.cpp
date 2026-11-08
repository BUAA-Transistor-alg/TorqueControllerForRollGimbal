#include "gimbal_dynamics/control_gradient.hpp"

#include <stdexcept>

#include "model_evaluation.hpp"
#include "simulation_core.hpp"
#include "stage_data.hpp"
#include "stage_jacobians.hpp"

namespace gimbal {
namespace control_gradient {

void Workspace::resize(std::size_t num_steps, int refinement) {
    trace.resize(num_steps, refinement);
    const std::size_t substeps = num_steps * static_cast<std::size_t>(refinement);
    propagation.resize(substeps);
    input_jacobian.resize(substeps);
}

void Workspace::release() {
    trace.release();
    propagation.clear();
    input_jacobian.clear();
}

Real simulateAndGradient(const ModelParameters& model, const Terrain& terrain,
                           const simulation::State& initial, const BaseState* base,
                           const math::Vector4d* torque, std::size_t num_steps,
                           const simulation::TrajectorySpec& spec,
                           const simulation::LossSpec& loss, Workspace& workspace,
                           math::Vector4d* gradient, simulation::State* final_state) {
    simulation::validate(spec, num_steps);
    workspace.resize(num_steps, spec.refinement);
    const std::size_t refinement = static_cast<std::size_t>(spec.refinement);
    const Real h = spec.step / static_cast<Real>(spec.refinement);
    const detail::ScalarModelParameters<Real> scalar_model = detail::convert_model<Real>(model);

    // 单遍完成前向仿真与阶段 Jacobian：每个子步结束时把 W 与 Ψ 写入工作区
    detail::StageJacobians stage[4];
    const auto stage_visitor = [&](std::size_t stage_index, const detail::StageData& data) {
        const std::size_t substep = stage_index / 4;
        const int t = static_cast<int>(stage_index % 4);
        stage[t] = detail::evaluate_stage_jacobians(data, scalar_model);
        if (t == 3) {
            detail::assemble_substep(stage, h, workspace.propagation[substep],
                                     workspace.input_jacobian[substep]);
        }
    };
    const auto step_visitor = [](std::size_t, const simulation::State&) {};
    const Real loss_value = detail::simulate_trajectory(
        model, terrain, initial, base, torque, num_steps, spec, loss, workspace.trace, false,
        stage_visitor, step_visitor);
    if (final_state != nullptr) {
        *final_state = workspace.trace.step_end.back();
    }
    if (gradient == nullptr) {
        return loss_value;
    }

    // 反向伴随：协态按子步逆序传播，输入梯度在同一主步内累加
    math::Vector8d adjoint = math::Vector8d::zero();
    const Real torque_scale = 1.0 / static_cast<Real>(num_steps);
    for (std::size_t k = num_steps; k-- > 0;) {
        adjoint += simulation::step_loss_gradient(loss, num_steps, k, workspace.trace.step_end[k]);
        math::Vector4d accumulated = math::Vector4d::zero();
        for (std::size_t s = refinement; s-- > 0;) {
            const std::size_t substep = k * refinement + s;
            accumulated += transpose(workspace.input_jacobian[substep]) * adjoint;
            adjoint = transpose(workspace.propagation[substep]) * adjoint;
        }
        gradient[k] =
            accumulated + elementwise(loss.torque_weights, torque[k]) * torque_scale;
    }
    return loss_value;
}

}  // namespace control_gradient
}  // namespace gimbal
