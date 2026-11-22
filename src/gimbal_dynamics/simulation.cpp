#include "gimbal_dynamics/simulation.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "gimbal_dynamics/model_evaluation.hpp"
#include "gimbal_dynamics/simulation_core.hpp"
#include "gimbal_dynamics/stage_data.hpp"

namespace tcrg::gimbal_dynamics {
namespace simulation {
namespace {

// 关节软限位的铰链值：越出 [lower + margin, upper - margin] 时返回有符号越界量
Real limit_hinge(Real angle, Real lower, Real upper, Real margin) {
    if (angle > upper - margin) {
        return angle - (upper - margin);
    }
    if (angle < lower + margin) {
        return angle - (lower + margin);
    }
    return Real(0);
}

}  // namespace

void validate(const TrajectorySpec& spec, std::size_t num_steps) {
    if (!(spec.step > 0.0) || !std::isfinite(spec.step)) {
        throw std::invalid_argument("主步长必须是大于 0 的有限值");
    }
    if (spec.refinement < 1) {
        throw std::invalid_argument("细化倍数必须大于等于 1");
    }
    if (num_steps < 1) {
        throw std::invalid_argument("步数必须大于等于 1");
    }
    const std::size_t refinement = static_cast<std::size_t>(spec.refinement);
    if (refinement > std::numeric_limits<std::size_t>::max() / 4 ||
        num_steps > std::numeric_limits<std::size_t>::max() / (4 * refinement)) {
        throw std::invalid_argument("步数与细化倍数过大，缓冲区尺寸溢出");
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
                 const State& state, const math::Vector4d& torque,
                 const math::Vector4d* previous_torque) {
    const Real scale = 1.0 / static_cast<Real>(num_steps);
    Real total = 0.0;
    for (int i = 0; i < 4; ++i) {
        const Real angle_target =
            (loss.target_angle != nullptr) ? loss.target_angle[step][i] : 0.0;
        const Real velocity_target =
            (loss.target_velocity != nullptr) ? loss.target_velocity[step][i] : 0.0;
        const Real angle_error = state.angle[i] - angle_target;
        const Real velocity_error = state.velocity[i] - velocity_target;
        const Real hinge = limit_hinge(state.angle[i], loss.limit_lower[i], loss.limit_upper[i],
                                       loss.limit_margin[i]);
        total += loss.angle_weights[i] * angle_error * angle_error +
                 loss.velocity_weights[i] * velocity_error * velocity_error +
                 loss.torque_weights[i] * torque[i] * torque[i] +
                 loss.limit_weights[i] * hinge * hinge;
        if (previous_torque != nullptr) {
            const Real torque_rate = torque[i] - (*previous_torque)[i];
            total += loss.torque_rate_weights[i] * torque_rate * torque_rate;
        }
    }
    if (step + 1 == num_steps) {
        for (int i = 0; i < 4; ++i) {
            const Real angle_target =
                (loss.target_angle != nullptr) ? loss.target_angle[step][i] : 0.0;
            const Real velocity_target =
                (loss.target_velocity != nullptr) ? loss.target_velocity[step][i] : 0.0;
            const Real angle_error = state.angle[i] - angle_target;
            const Real velocity_error = state.velocity[i] - velocity_target;
            total += loss.terminal_angle_weights[i] * angle_error * angle_error +
                     loss.terminal_velocity_weights[i] * velocity_error * velocity_error;
        }
    }
    return scale * total;
}

math::Vector8d step_loss_gradient(const LossSpec& loss, std::size_t num_steps,
                                  std::size_t step, const State& state) {
    const Real scale = 2.0 / static_cast<Real>(num_steps);
    const bool terminal = (step + 1 == num_steps);
    math::Vector8d gradient;
    for (int i = 0; i < 4; ++i) {
        const Real angle_target =
            (loss.target_angle != nullptr) ? loss.target_angle[step][i] : 0.0;
        const Real velocity_target =
            (loss.target_velocity != nullptr) ? loss.target_velocity[step][i] : 0.0;
        const Real angle_error = state.angle[i] - angle_target;
        const Real velocity_error = state.velocity[i] - velocity_target;
        const Real hinge = limit_hinge(state.angle[i], loss.limit_lower[i], loss.limit_upper[i],
                                       loss.limit_margin[i]);
        Real angle_gradient = loss.angle_weights[i] * angle_error + loss.limit_weights[i] * hinge;
        Real velocity_gradient = loss.velocity_weights[i] * velocity_error;
        if (terminal) {
            angle_gradient += loss.terminal_angle_weights[i] * angle_error;
            velocity_gradient += loss.terminal_velocity_weights[i] * velocity_error;
        }
        gradient[i] = scale * angle_gradient;
        gradient[4 + i] = scale * velocity_gradient;
    }
    return gradient;
}

Real simulate(const ModelParameters& model, const Terrain& terrain, const State& initial,
                const BaseState* base, const math::Vector4d* torque, std::size_t num_steps,
                const TrajectorySpec& spec, const LossSpec& loss, Trace& trace) {
    return detail::simulate_forward(model, terrain, initial, base, torque, num_steps, spec,
                                    loss, trace);
}

}  // namespace simulation
}  // namespace tcrg::gimbal_dynamics
