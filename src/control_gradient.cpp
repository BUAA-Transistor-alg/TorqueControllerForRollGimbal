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

double simulateAndGradient(const ModelParameters& model, const Terrain& terrain,
                           const simulation::State& initial, const BaseState* base,
                           const Eigen::Vector4d* torque, std::size_t num_steps,
                           const simulation::TrajectorySpec& spec,
                           const simulation::LossSpec& loss, Workspace& workspace,
                           Eigen::Vector4d* gradient, simulation::State* final_state) {
    simulation::validate(spec, num_steps);
    if (base == nullptr) {
        throw std::invalid_argument("base 不能为空");
    }
    if (torque == nullptr) {
        throw std::invalid_argument("torque 不能为空");
    }
    workspace.resize(num_steps, spec.refinement);
    const std::size_t refinement_count = static_cast<std::size_t>(spec.refinement);
    std::vector<detail::StageData> stage_cache(
        num_steps * refinement_count * 4);
    const double loss_value = detail::simulate_trajectory(
        model, terrain, initial, base, torque, num_steps, spec, loss, workspace.trace,
        &stage_cache);
    if (final_state != nullptr) {
        *final_state = workspace.trace.step_end.back();
    }
    if (gradient == nullptr) {
        return loss_value;
    }

    const double h = spec.step / static_cast<double>(spec.refinement);
    const std::size_t refinement = static_cast<std::size_t>(spec.refinement);
    const detail::ScalarModelParameters<double> scalar_model =
        detail::convert_model<double>(model);

    // 由前向仿真已缓存的阶段中间量直接组装整子步的 W 与 Ψ
    for (std::size_t k = 0; k < num_steps; ++k) {
        for (std::size_t s = 0; s < refinement; ++s) {
            const std::size_t substep = k * refinement + s;
            detail::StageJacobians stage[4];
            for (int t = 0; t < 4; ++t) {
                const std::size_t index = substep * 4 + static_cast<std::size_t>(t);
                stage[t] = detail::evaluate_stage_jacobians(stage_cache[index], scalar_model);
            }
            detail::assemble_substep(stage, h, workspace.propagation[substep],
                                     workspace.input_jacobian[substep]);
        }
    }

    // 反向扫描：协态按子步逆序传播，输入梯度在同一主步内累加
    Eigen::Matrix<double, 8, 1> adjoint = Eigen::Matrix<double, 8, 1>::Zero();
    const double torque_scale = 1.0 / static_cast<double>(num_steps);
    for (std::size_t k = num_steps; k-- > 0;) {
        adjoint += simulation::step_loss_gradient(loss, num_steps, k, workspace.trace.step_end[k]);
        Eigen::Vector4d accumulated = Eigen::Vector4d::Zero();
        for (std::size_t s = refinement; s-- > 0;) {
            const std::size_t substep = k * refinement + s;
            accumulated += workspace.input_jacobian[substep].transpose() * adjoint;
            adjoint = workspace.propagation[substep].transpose() * adjoint;
        }
        gradient[k] =
            accumulated + torque_scale * loss.torque_weights.cwiseProduct(torque[k]);
    }
    return loss_value;
}

}  // namespace control_gradient
}  // namespace gimbal
