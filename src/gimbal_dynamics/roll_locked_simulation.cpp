#include "gimbal_dynamics/roll_locked.hpp"

#include <array>
#include <cmath>
#include <stdexcept>

#include "gimbal_dynamics/roll_locked_internal.hpp"

namespace tcrg::gimbal_dynamics {
namespace roll_locked {
namespace {

BaseState advance_base(const BaseState& base, Real time) {
    BaseState result = base;
    result.yaw = base.yaw + base.yaw_rate * time +
                 Real(0.5) * base.yaw_acceleration * time * time;
    result.yaw_rate = base.yaw_rate + base.yaw_acceleration * time;
    return result;
}

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

namespace detail {

Real step_loss(const LossSpecT& loss, std::size_t num_steps, std::size_t step,
               const StateT& state, const Vector3& torque, const Vector3* previous_torque) {
    const Real scale = Real(1) / static_cast<Real>(num_steps);
    const Vector3 target_angle =
        (loss.target_angle != nullptr) ? loss.target_angle[step] : Vector3::zero();
    const Vector3 target_velocity =
        (loss.target_velocity != nullptr) ? loss.target_velocity[step] : Vector3::zero();
    Real total = Real(0);
    for (int joint = 0; joint < 3; ++joint) {
        const Real angle_error = state.angle[joint] - target_angle[joint];
        const Real velocity_error = state.velocity[joint] - target_velocity[joint];
        const Real hinge = limit_hinge(state.angle[joint], loss.limit_lower[joint],
                                       loss.limit_upper[joint], loss.limit_margin[joint]);
        total += loss.angle_weights[joint] * angle_error * angle_error +
                 loss.velocity_weights[joint] * velocity_error * velocity_error +
                 loss.torque_weights[joint] * torque[joint] * torque[joint] +
                 loss.limit_weights[joint] * hinge * hinge;
        if (previous_torque != nullptr) {
            const Real torque_rate = torque[joint] - (*previous_torque)[joint];
            total += loss.torque_rate_weights[joint] * torque_rate * torque_rate;
        }
    }
    return scale * total;
}

Vector6 step_loss_gradient(const LossSpecT& loss, std::size_t num_steps, std::size_t step,
                           const StateT& state) {
    const Real scale = Real(2) / static_cast<Real>(num_steps);
    const Vector3 target_angle =
        (loss.target_angle != nullptr) ? loss.target_angle[step] : Vector3::zero();
    const Vector3 target_velocity =
        (loss.target_velocity != nullptr) ? loss.target_velocity[step] : Vector3::zero();
    Vector6 gradient = Vector6::zero();
    for (int joint = 0; joint < 3; ++joint) {
        const Real angle_error = state.angle[joint] - target_angle[joint];
        const Real velocity_error = state.velocity[joint] - target_velocity[joint];
        const Real hinge = limit_hinge(state.angle[joint], loss.limit_lower[joint],
                                       loss.limit_upper[joint], loss.limit_margin[joint]);
        gradient[joint] =
            scale * (loss.angle_weights[joint] * angle_error +
                     loss.limit_weights[joint] * hinge);
        gradient[3 + joint] = scale * loss.velocity_weights[joint] * velocity_error;
    }
    return gradient;
}

}  // namespace detail

Real simulate(const ForwardInputT& input, ForwardOutputT& output) {
    if (input.base == nullptr || input.torque == nullptr) {
        throw std::invalid_argument("底盘状态与力矩不能为空");
    }
    tcrg::gimbal_dynamics::simulation::validate(input.spec, input.num_steps);
    const ModelT model = merge_model(input.model, input.locked_angle);
    const std::size_t refinement = static_cast<std::size_t>(input.spec.refinement);
    const Real h = input.spec.step / static_cast<Real>(refinement);
    output.step_end.resize(input.num_steps);
    output.stage_state.resize(input.num_steps * refinement * 4);
    output.stage_base.resize(input.num_steps * refinement * 4);

    StateT state = input.initial;
    Real total_loss = Real(0);
    for (std::size_t step = 0; step < input.num_steps; ++step) {
        const Vector3 torque = input.torque[step];
        for (std::size_t substep = 0; substep < refinement; ++substep) {
            const Real substep_time = static_cast<Real>(substep) * h;
            std::array<BaseState, 4> stage_base = {
                advance_base(input.base[step], substep_time),
                advance_base(input.base[step], substep_time + Real(0.5) * h),
                advance_base(input.base[step], substep_time + Real(0.5) * h),
                advance_base(input.base[step], substep_time + h),
            };
            std::array<StateT, 4> stage_state;
            std::array<StateT, 4> derivative;
            stage_state[0] = state;
            for (int stage = 0; stage < 4; ++stage) {
                if (stage > 0) {
                    const Real advance = (stage == 3) ? h : Real(0.5) * h;
                    stage_state[stage].angle =
                        stage_state[0].angle + derivative[stage - 1].angle * advance;
                    stage_state[stage].velocity =
                        stage_state[0].velocity + derivative[stage - 1].velocity * advance;
                }
                const Vector3 acceleration =
                    forward_dynamics(model, input.terrain, stage_base[stage],
                                     stage_state[stage].angle, stage_state[stage].velocity, torque);
                derivative[stage].angle = stage_state[stage].velocity;
                derivative[stage].velocity = acceleration;
                const std::size_t index =
                    (step * refinement + substep) * 4 + static_cast<std::size_t>(stage);
                output.stage_state[index] = stage_state[stage];
                output.stage_base[index] = stage_base[stage];
            }
            state.angle =
                stage_state[0].angle +
                (derivative[0].angle + derivative[1].angle * Real(2) +
                 derivative[2].angle * Real(2) + derivative[3].angle) *
                    (h / Real(6));
            state.velocity =
                stage_state[0].velocity +
                (derivative[0].velocity + derivative[1].velocity * Real(2) +
                 derivative[2].velocity * Real(2) + derivative[3].velocity) *
                    (h / Real(6));
        }
        output.step_end[step] = state;
        const Vector3* previous =
            (step == 0) ? input.loss.previous_torque : &input.torque[step - 1];
        total_loss += detail::step_loss(input.loss, input.num_steps, step, state, torque, previous);
    }
    output.loss = total_loss;
    return total_loss;
}

}  // namespace roll_locked
}  // namespace tcrg::gimbal_dynamics
