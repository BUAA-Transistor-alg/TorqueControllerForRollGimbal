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
                              const Eigen::Vector4d* torque, std::size_t num_steps,
                              const simulation::TrajectorySpec& spec,
                              const simulation::LossSpec& loss, Workspace& workspace,
                              Gradient* gradient, simulation::State* final_state) {
    simulation::validate(spec, num_steps);
    if (base == nullptr) {
        throw std::invalid_argument("base 不能为空");
    }
    if (torque == nullptr) {
        throw std::invalid_argument("torque 不能为空");
    }
    workspace.resize(num_steps, spec.refinement);
    const std::size_t refinement_count = static_cast<std::size_t>(spec.refinement);
    std::vector<detail::StageData> stage_cache(num_steps * refinement_count * 4);
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
    Eigen::Matrix<double, 8, parameters::kParameterCount> sensitivity =
        Eigen::Matrix<double, 8, parameters::kParameterCount>::Zero();
    Gradient total = Gradient::Zero();

    // 由前向仿真已缓存的阶段中间量同时得到状态 Jacobian 与参数 Jacobian
    for (std::size_t k = 0; k < num_steps; ++k) {
        for (std::size_t s = 0; s < refinement; ++s) {
            const std::size_t substep = k * refinement + s;
            detail::StageJacobians stage[4];
            detail::ParameterJacobian stage_parameter[4];
            for (int t = 0; t < 4; ++t) {
                const std::size_t index = substep * 4 + static_cast<std::size_t>(t);
                stage[t].state = Eigen::Matrix<double, 8, 8>::Zero();
                stage_parameter[t] = detail::evaluate_parameter_jacobian(
                    stage_cache[index], scalar_model, stage[t].state);
            }
            Eigen::Matrix<double, 8, 8> propagation;
            detail::ParameterJacobian parameter_jacobian;
            detail::assemble_parameter_substep(stage, stage_parameter, h, propagation,
                                               parameter_jacobian);
            sensitivity = propagation * sensitivity + parameter_jacobian;
        }
        total += sensitivity.transpose() *
                 simulation::step_loss_gradient(loss, num_steps, k, workspace.trace.step_end[k]);
    }
    *gradient = total;
    return loss_value;
}

}  // namespace parameter_gradient
}  // namespace gimbal
