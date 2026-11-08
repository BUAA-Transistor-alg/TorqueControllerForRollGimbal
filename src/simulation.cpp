#include "gimbal_dynamics/simulation.hpp"

#include <stdexcept>

#include "model_evaluation.hpp"
#include "simulation_core.hpp"
#include "stage_data.hpp"

namespace gimbal {
namespace simulation {

void Trace::resize(std::size_t steps, int refine) {
    num_steps = steps;
    refinement = refine;
    step_end.resize(steps);
    stage_state.resize(steps * static_cast<std::size_t>(refine) * 4);
    stage_base.resize(steps * static_cast<std::size_t>(refine) * 4);
}

void Trace::release() {
    num_steps = 0;
    refinement = 0;
    step_end.clear();
    stage_state.clear();
    stage_base.clear();
}

void validate(const TrajectorySpec& spec, std::size_t num_steps) {
    if (!(spec.step > 0.0)) {
        throw std::invalid_argument("主步长必须大于 0");
    }
    if (spec.refinement < 1) {
        throw std::invalid_argument("细化倍数必须大于等于 1");
    }
    if (num_steps < 1) {
        throw std::invalid_argument("步数必须大于等于 1");
    }
}

math::Vector8d state_derivative(const ModelParameters& model, const Terrain& terrain,
                                const BaseState& base, const State& state,
                                const math::Vector4d& torque) {
    const detail::ScalarModelParameters<Real> scalar_model = detail::convert_model<Real>(model);
    const detail::ScalarTerrain<Real> scalar_terrain = detail::convert_terrain<Real>(terrain);
    return detail::evaluate_derivative<Real>(scalar_model, scalar_terrain, base,
                                               detail::pack_state(state), torque);
}

Real step_loss(const LossSpec& loss, std::size_t num_steps, std::size_t step,
                 const State& state, const math::Vector4d& torque) {
    const Real scale = 1.0 / (2.0 * static_cast<Real>(num_steps));
    Real total = 0.0;
    for (int i = 0; i < 4; ++i) {
        const Real angle_target =
            (loss.target_angle != nullptr) ? loss.target_angle[step][i] : 0.0;
        const Real velocity_target =
            (loss.target_velocity != nullptr) ? loss.target_velocity[step][i] : 0.0;
        const Real angle_error = state.angle[i] - angle_target;
        const Real velocity_error = state.velocity[i] - velocity_target;
        total += loss.angle_weights[i] * angle_error * angle_error +
                 loss.velocity_weights[i] * velocity_error * velocity_error +
                 loss.torque_weights[i] * torque[i] * torque[i];
    }
    return scale * total;
}

math::Vector8d step_loss_gradient(const LossSpec& loss, std::size_t num_steps,
                                  std::size_t step, const State& state) {
    const Real scale = 1.0 / static_cast<Real>(num_steps);
    math::Vector8d gradient;
    for (int i = 0; i < 4; ++i) {
        const Real angle_target =
            (loss.target_angle != nullptr) ? loss.target_angle[step][i] : 0.0;
        const Real velocity_target =
            (loss.target_velocity != nullptr) ? loss.target_velocity[step][i] : 0.0;
        gradient[i] = scale * loss.angle_weights[i] * (state.angle[i] - angle_target);
        gradient[4 + i] =
            scale * loss.velocity_weights[i] * (state.velocity[i] - velocity_target);
    }
    return gradient;
}

Real simulate(const ModelParameters& model, const Terrain& terrain, const State& initial,
                const BaseState* base, const math::Vector4d* torque, std::size_t num_steps,
                const TrajectorySpec& spec, const LossSpec& loss, Trace& trace) {
    return detail::simulate_trajectory(
        model, terrain, initial, base, torque, num_steps, spec, loss, trace, true,
        [](std::size_t, const detail::StageData&) {},
        [](std::size_t, const State&) {});
}

}  // namespace simulation
}  // namespace gimbal
