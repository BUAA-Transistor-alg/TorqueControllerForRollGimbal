#include "gimbal_dynamics/simulation.hpp"

#include <stdexcept>
#include <vector>

#include "model_evaluation.hpp"
#include "simulation_core.hpp"
#include "stage_data.hpp"

namespace gimbal {
namespace simulation {
namespace {

// 主步内底盘量按等角加速度外推，线加速度保持不变
BaseState advance_base(const BaseState& base, double time) {
    BaseState result = base;
    result.yaw = base.yaw + base.yaw_rate * time + 0.5 * base.yaw_acceleration * time * time;
    result.yaw_rate = base.yaw_rate + base.yaw_acceleration * time;
    return result;
}

detail::StateVector<double> pack(const State& state) {
    detail::StateVector<double> result;
    for (int i = 0; i < 4; ++i) {
        result[i] = state.angle[i];
        result[4 + i] = state.velocity[i];
    }
    return result;
}

State unpack(const detail::StateVector<double>& state) {
    State result;
    for (int i = 0; i < 4; ++i) {
        result.angle[i] = state[i];
        result.velocity[i] = state[4 + i];
    }
    return result;
}

}  // namespace

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
    const detail::ScalarModelParameters<double> scalar_model = detail::convert_model<double>(model);
    const detail::ScalarTerrain<double> scalar_terrain = detail::convert_terrain<double>(terrain);
    return detail::evaluate_derivative<double>(scalar_model, scalar_terrain, base, pack(state),
                                               torque);
}

double step_loss(const LossSpec& loss, std::size_t num_steps, std::size_t step,
                 const State& state, const math::Vector4d& torque) {
    const double scale = 1.0 / (2.0 * static_cast<double>(num_steps));
    double total = 0.0;
    for (int i = 0; i < 4; ++i) {
        const double angle_target =
            (loss.target_angle != nullptr) ? loss.target_angle[step][i] : 0.0;
        const double velocity_target =
            (loss.target_velocity != nullptr) ? loss.target_velocity[step][i] : 0.0;
        const double angle_error = state.angle[i] - angle_target;
        const double velocity_error = state.velocity[i] - velocity_target;
        total += loss.angle_weights[i] * angle_error * angle_error +
                 loss.velocity_weights[i] * velocity_error * velocity_error +
                 loss.torque_weights[i] * torque[i] * torque[i];
    }
    return scale * total;
}

math::Vector8d step_loss_gradient(const LossSpec& loss, std::size_t num_steps,
                                  std::size_t step, const State& state) {
    const double scale = 1.0 / static_cast<double>(num_steps);
    math::Vector8d gradient;
    for (int i = 0; i < 4; ++i) {
        const double angle_target =
            (loss.target_angle != nullptr) ? loss.target_angle[step][i] : 0.0;
        const double velocity_target =
            (loss.target_velocity != nullptr) ? loss.target_velocity[step][i] : 0.0;
        gradient[i] = scale * loss.angle_weights[i] * (state.angle[i] - angle_target);
        gradient[4 + i] =
            scale * loss.velocity_weights[i] * (state.velocity[i] - velocity_target);
    }
    return gradient;
}

double simulate(const ModelParameters& model, const Terrain& terrain, const State& initial,
                const BaseState* base, const math::Vector4d* torque, std::size_t num_steps,
                const TrajectorySpec& spec, const LossSpec& loss, Trace& trace) {
    return detail::simulate_trajectory(model, terrain, initial, base, torque, num_steps, spec,
                                       loss, trace, nullptr);
}

}  // namespace simulation

namespace detail {

double simulate_trajectory(const ModelParameters& model, const Terrain& terrain,
                           const simulation::State& initial, const BaseState* base,
                           const math::Vector4d* torque, std::size_t num_steps,
                           const simulation::TrajectorySpec& spec,
                           const simulation::LossSpec& loss, simulation::Trace& trace,
                           std::vector<StageData>* stage_cache) {
    simulation::validate(spec, num_steps);
    if (base == nullptr) {
        throw std::invalid_argument("base 不能为空");
    }
    if (torque == nullptr) {
        throw std::invalid_argument("torque 不能为空");
    }
    trace.resize(num_steps, spec.refinement);
    const std::size_t substep_count = num_steps * static_cast<std::size_t>(spec.refinement);
    if (stage_cache != nullptr) {
        stage_cache->resize(substep_count * 4);
    }

    const ScalarModelParameters<double> scalar_model = convert_model<double>(model);
    const ScalarTerrain<double> scalar_terrain = convert_terrain<double>(terrain);
    const double h = spec.step / static_cast<double>(spec.refinement);
    StateVector<double> state = simulation::pack(initial);
    double total_loss = 0.0;

    for (std::size_t k = 0; k < num_steps; ++k) {
        const math::Vector4d torque_k = torque[k];
        for (int s = 0; s < spec.refinement; ++s) {
            const double substep_time = static_cast<double>(s) * h;
            const std::size_t record = (k * static_cast<std::size_t>(spec.refinement) +
                                        static_cast<std::size_t>(s)) * 4;

            const BaseState stage_input[4] = {
                simulation::advance_base(base[k], substep_time),
                simulation::advance_base(base[k], substep_time + 0.5 * h),
                simulation::advance_base(base[k], substep_time + 0.5 * h),
                simulation::advance_base(base[k], substep_time + h),
            };

            StateVector<double> stage_state[4];
            StateVector<double> stage_derivative[4];
            stage_state[0] = state;
            for (int t = 0; t < 4; ++t) {
                if (t > 0) {
                    const double advance = (t == 3) ? h : 0.5 * h;
                    stage_state[t] = stage_state[0] + stage_derivative[t - 1] * advance;
                }
                if (stage_cache != nullptr) {
                    StageData& data = (*stage_cache)[record + static_cast<std::size_t>(t)];
                    evaluate_stage_data(scalar_model, scalar_terrain, stage_input[t],
                                        simulation::unpack(stage_state[t]), torque_k, data);
                    for (int i = 0; i < 4; ++i) {
                        stage_derivative[t][i] = stage_state[t][4 + i];
                        stage_derivative[t][4 + i] = data.acceleration[i];
                    }
                } else {
                    stage_derivative[t] = evaluate_derivative<double>(
                        scalar_model, scalar_terrain, stage_input[t], stage_state[t], torque_k);
                }
                trace.stage_state[record + static_cast<std::size_t>(t)] =
                    simulation::unpack(stage_state[t]);
                trace.stage_base[record + static_cast<std::size_t>(t)] = stage_input[t];
            }

            state = stage_state[0] +
                    (stage_derivative[0] + stage_derivative[1] * 2.0 +
                     stage_derivative[2] * 2.0 + stage_derivative[3]) * (h / 6.0);
        }
        trace.step_end[k] = simulation::unpack(state);
        total_loss += simulation::step_loss(loss, num_steps, k, trace.step_end[k], torque_k);
    }
    return total_loss;
}

}  // namespace detail
}  // namespace gimbal
