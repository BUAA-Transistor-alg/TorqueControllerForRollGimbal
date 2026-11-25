#include "gimbal_dynamics/roll_locked_internal.hpp"

#include <array>
#include <stdexcept>
#include <vector>

namespace tcrg::gimbal_dynamics {
namespace roll_locked {
namespace detail {
namespace {

Matrix3 skew_matrix(const Vector3& vector) {
    Matrix3 matrix = Matrix3::zero();
    matrix(0, 1) = -vector[2];
    matrix(0, 2) = vector[1];
    matrix(1, 0) = vector[2];
    matrix(1, 2) = -vector[0];
    matrix(2, 0) = -vector[1];
    matrix(2, 1) = vector[0];
    return matrix;
}

void evaluate_chain_direction(const StageData& stage, const ModelT& model, int joint,
                              std::array<Matrix3, 3>& delta_rotation,
                              std::array<Vector3, 3>& delta_origin,
                              std::array<Vector3, 3>& delta_axis,
                              std::array<Vector3, 3>& delta_com,
                              std::array<Vector3, 3>& delta_offset) {
    const ChainT& chain = stage.chain;
    const Vector3& axis = chain.axis[joint];
    const Matrix3 skew = skew_matrix(axis);
    for (int body = 0; body < 3; ++body) {
        delta_rotation[body] = Matrix3::zero();
        delta_origin[body] = Vector3::zero();
        delta_axis[body] = Vector3::zero();
        delta_com[body] = Vector3::zero();
        delta_offset[body] = Vector3::zero();
    }
    for (int body = 0; body < 3; ++body) {
        if (body >= joint) {
            delta_rotation[body] = skew * chain.rotation[body];
            delta_origin[body] =
                math::cross(axis, chain.joint_origin[body] - chain.joint_origin[joint]);
            delta_com[body] = delta_origin[body] + delta_rotation[body] * model.bodies[body].com;
        }
        if (body > joint) {
            delta_axis[body] = math::cross(axis, chain.axis[body]);
        }
        if (body > 0) {
            delta_offset[body] = delta_origin[body] - delta_origin[body - 1];
        }
    }
}

Vector3 evaluate_bias_direction(const StageData& stage, const ModelT& model,
                                const std::array<Matrix3, 3>& delta_rotation,
                                const std::array<Vector3, 3>& delta_origin,
                                const std::array<Vector3, 3>& delta_axis,
                                const std::array<Vector3, 3>& delta_com,
                                const std::array<Vector3, 3>& delta_offset,
                                const std::array<Real, 3>& delta_velocity) {
    const ChainT& chain = stage.chain;
    const BiasInternals& internals = stage.bias_internals;
    Vector3 previous_omega = chain.normal * stage.base.yaw_rate;
    Vector3 previous_alpha = chain.normal * stage.base.yaw_acceleration;
    Vector3 previous_delta_omega = Vector3::zero();
    Vector3 previous_delta_alpha = Vector3::zero();
    Vector3 previous_delta_origin_acceleration = Vector3::zero();

    std::array<Vector3, 3> delta_omega{};
    std::array<Vector3, 3> delta_alpha{};
    std::array<Vector3, 3> delta_force{};
    std::array<Vector3, 3> delta_couple{};
    for (int body = 0; body < 3; ++body) {
        const Vector3& offset = chain.offset[body];
        const Vector3& axis = chain.axis[body];
        const Real velocity = stage.velocity[body];
        const Vector3 omega = internals.omega[body];
        const Vector3 alpha = internals.alpha[body];

        const Vector3 delta_omega_body =
            previous_delta_omega + delta_axis[body] * velocity + axis * delta_velocity[body];
        const Vector3 delta_alpha_body =
            previous_delta_alpha +
            math::cross(previous_omega, delta_axis[body] * velocity +
                                            axis * delta_velocity[body]) +
            math::cross(previous_delta_omega, axis * velocity);
        const Vector3 delta_origin_acceleration =
            previous_delta_origin_acceleration +
            math::cross(previous_delta_alpha, offset) +
            math::cross(previous_alpha, delta_offset[body]) +
            math::cross(previous_delta_omega, math::cross(previous_omega, offset)) +
            math::cross(previous_omega, math::cross(previous_delta_omega, offset)) +
            math::cross(previous_omega, math::cross(previous_omega, delta_offset[body]));

        const Vector3 com_relative = chain.rotation[body] * model.bodies[body].com;
        const Vector3 delta_com_relative = delta_rotation[body] * model.bodies[body].com;
        const Vector3 delta_com_acceleration =
            delta_origin_acceleration + math::cross(delta_alpha_body, com_relative) +
            math::cross(alpha, delta_com_relative) +
            math::cross(delta_omega_body, math::cross(omega, com_relative)) +
            math::cross(omega, math::cross(delta_omega_body, com_relative)) +
            math::cross(omega, math::cross(omega, delta_com_relative));

        const Matrix3& inertia_world = stage.inertia_world[body];
        const Matrix3 delta_inertia_world =
            delta_rotation[body] * model.bodies[body].inertia *
                math::transpose(chain.rotation[body]) +
            chain.rotation[body] * model.bodies[body].inertia *
                math::transpose(delta_rotation[body]);
        delta_force[body] = delta_com_acceleration * model.bodies[body].mass;
        delta_couple[body] =
            delta_inertia_world * alpha + inertia_world * delta_alpha_body +
            math::cross(delta_omega_body, inertia_world * omega) +
            math::cross(omega, delta_inertia_world * omega + inertia_world * delta_omega_body);

        delta_omega[body] = delta_omega_body;
        delta_alpha[body] = delta_alpha_body;
        previous_omega = omega;
        previous_alpha = alpha;
        previous_delta_omega = delta_omega_body;
        previous_delta_alpha = delta_alpha_body;
        previous_delta_origin_acceleration = delta_origin_acceleration;
    }

    Vector3 delta_bias = Vector3::zero();
    Vector3 child_delta_force = Vector3::zero();
    Vector3 child_delta_moment = Vector3::zero();
    for (int body = 2; body >= 0; --body) {
        const Vector3& g = chain.com[body];
        const Vector3& p = chain.joint_origin[body];
        Vector3 delta_moment =
            delta_couple[body] + math::cross(delta_com[body] - delta_origin[body],
                                             internals.force[body]) +
            math::cross(g - p, delta_force[body]);
        if (body < 2) {
            delta_moment +=
                child_delta_moment +
                math::cross(delta_origin[body + 1] - delta_origin[body],
                            internals.joint_force[body + 1]) +
                math::cross(chain.joint_origin[body + 1] - chain.joint_origin[body],
                            child_delta_force);
        }
        delta_bias[body] = math::dot(delta_axis[body], internals.joint_moment[body]) +
                           math::dot(chain.axis[body], delta_moment);
        child_delta_force = delta_force[body] + child_delta_force;
        child_delta_moment = delta_moment;
    }
    return delta_bias;
}

Matrix3 evaluate_mass_direction(const StageData& stage, const ModelT& model,
                                const std::array<Matrix3, 3>& delta_rotation,
                                const std::array<Vector3, 3>& delta_origin,
                                const std::array<Vector3, 3>& delta_axis,
                                const std::array<Vector3, 3>& delta_com) {
    Matrix3 delta_mass = Matrix3::zero();
    for (int body = 0; body < 3; ++body) {
        Matrix3 delta_velocity_jacobian = Matrix3::zero();
        Matrix3 delta_angular_jacobian = Matrix3::zero();
        for (int joint = 0; joint <= body; ++joint) {
            const Vector3 column =
                math::cross(delta_axis[joint], stage.chain.com[body] - stage.chain.joint_origin[joint]) +
                math::cross(stage.chain.axis[joint], delta_com[body] - delta_origin[joint]);
            for (int row = 0; row < 3; ++row) {
                delta_velocity_jacobian(row, joint) = column[row];
                delta_angular_jacobian(row, joint) = delta_axis[joint][row];
            }
        }
        const Matrix3& velocity_jacobian = stage.velocity_jacobian[body];
        const Matrix3& angular_jacobian = stage.angular_jacobian[body];
        const Matrix3& inertia_world = stage.inertia_world[body];
        const Matrix3 delta_inertia_world =
            delta_rotation[body] * model.bodies[body].inertia *
                math::transpose(stage.chain.rotation[body]) +
            stage.chain.rotation[body] * model.bodies[body].inertia *
                math::transpose(delta_rotation[body]);
        delta_mass +=
            (math::transpose(delta_velocity_jacobian) * velocity_jacobian +
             math::transpose(velocity_jacobian) * delta_velocity_jacobian) *
            model.bodies[body].mass;
        delta_mass += math::transpose(delta_angular_jacobian) * inertia_world * angular_jacobian;
        delta_mass += math::transpose(angular_jacobian) * delta_inertia_world * angular_jacobian;
        delta_mass += math::transpose(angular_jacobian) * inertia_world * delta_angular_jacobian;
    }
    return delta_mass;
}

}  // namespace

StageJacobians evaluate_stage_jacobians(const StageData& stage, const ModelT& model) {
    StageJacobians result;
    for (int joint = 0; joint < 3; ++joint) {
        result.state(joint, 3 + joint) = Real(1);
    }

    for (int direction = 0; direction < 3; ++direction) {
        std::array<Matrix3, 3> delta_rotation{};
        std::array<Vector3, 3> delta_origin{};
        std::array<Vector3, 3> delta_axis{};
        std::array<Vector3, 3> delta_com{};
        std::array<Vector3, 3> delta_offset{};
        std::array<Real, 3> delta_velocity = {Real(0), Real(0), Real(0)};
        evaluate_chain_direction(stage, model, direction, delta_rotation, delta_origin,
                                 delta_axis, delta_com, delta_offset);
        const Matrix3 delta_mass =
            evaluate_mass_direction(stage, model, delta_rotation, delta_origin, delta_axis,
                                    delta_com);
        const Vector3 delta_bias =
            evaluate_bias_direction(stage, model, delta_rotation, delta_origin, delta_axis,
                                    delta_com, delta_offset, delta_velocity);
        const Vector3 right_hand_side = -(delta_bias + delta_mass * stage.acceleration);
        const Vector3 column = stage.factor.solve(right_hand_side);
        for (int row = 0; row < 3; ++row) {
            result.state(3 + row, direction) = column[row];
        }

        std::array<Matrix3, 3> zero_rotation{};
        std::array<Vector3, 3> zero_origin{};
        std::array<Vector3, 3> zero_axis{};
        std::array<Vector3, 3> zero_com{};
        std::array<Vector3, 3> zero_offset{};
        std::array<Real, 3> delta_speed = {Real(0), Real(0), Real(0)};
        delta_speed[direction] = Real(1);
        const Vector3 velocity_bias =
            evaluate_bias_direction(stage, model, zero_rotation, zero_origin, zero_axis,
                                    zero_com, zero_offset, delta_speed);
        Vector3 velocity_right_hand_side = -(velocity_bias);
        velocity_right_hand_side[direction] -= model.damping[direction];
        const Vector3 velocity_column = stage.factor.solve(velocity_right_hand_side);
        for (int row = 0; row < 3; ++row) {
            result.state(3 + row, 3 + direction) = velocity_column[row];
        }

        const Vector3 input_column = stage.factor.solve(Vector3::unit(direction));
        for (int row = 0; row < 3; ++row) {
            result.input(3 + row, direction) = input_column[row];
        }
    }
    return result;
}

void assemble_substep(const StageJacobians stage[4], Real h, Matrix6& propagation,
                      Matrix63& input_jacobian) {
    const Matrix6 identity = Matrix6::identity();
    const Matrix6 a2 = identity + stage[0].state * (Real(0.5) * h);
    const Matrix6 j2a2 = stage[1].state * a2;
    const Matrix6 a3 = identity + j2a2 * (Real(0.5) * h);
    const Matrix6 j3a3 = stage[2].state * a3;
    const Matrix6 a4 = identity + j3a3 * h;
    const Matrix6 j4a4 = stage[3].state * a4;
    propagation =
        identity + (stage[0].state + j2a2 * Real(2) + j3a3 * Real(2) + j4a4) * (h / Real(6));

    const Matrix63 b1 = stage[0].input;
    const Matrix63 b2 = stage[1].input + stage[1].state * b1 * (Real(0.5) * h);
    const Matrix63 b3 = stage[2].input + stage[2].state * b2 * (Real(0.5) * h);
    const Matrix63 b4 = stage[3].input + stage[3].state * b3 * h;
    input_jacobian =
        (b1 + b2 * Real(2) + b3 * Real(2) + b4) * (h / Real(6));
}

}  // namespace detail

namespace {

BaseState advance_base(const BaseState& base, Real time) {
    BaseState result = base;
    result.yaw = base.yaw + base.yaw_rate * time +
                 Real(0.5) * base.yaw_acceleration * time * time;
    result.yaw_rate = base.yaw_rate + base.yaw_acceleration * time;
    return result;
}

}  // namespace

Real simulate_and_gradient(const GradientInputT& input, GradientOutputT& output) {
    if (input.base == nullptr || input.torque == nullptr) {
        throw std::invalid_argument("底盘状态与力矩不能为空");
    }
    tcrg::gimbal_dynamics::simulation::validate(input.spec, input.num_steps);
    const ModelT model = merge_model(input.model, input.locked_angle);
    const std::size_t refinement = static_cast<std::size_t>(input.spec.refinement);
    const std::size_t substep_count = input.num_steps * refinement;
    const Real h = input.spec.step / static_cast<Real>(refinement);

    std::vector<detail::StageInput> stage_inputs(substep_count * 4);
    std::vector<StateT> step_end(input.num_steps);
    StateT state = input.initial;
    Real total_loss = Real(0);
    for (std::size_t step = 0; step < input.num_steps; ++step) {
        const detail::Vector3 torque = input.torque[step];
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
                const detail::Vector3 acceleration =
                    forward_dynamics(model, input.terrain, stage_base[stage],
                                     stage_state[stage].angle, stage_state[stage].velocity, torque);
                derivative[stage].angle = stage_state[stage].velocity;
                derivative[stage].velocity = acceleration;
                const std::size_t index =
                    (step * refinement + substep) * 4 + static_cast<std::size_t>(stage);
                detail::StageInput& recorded = stage_inputs[index];
                recorded.base = stage_base[stage];
                recorded.angle = stage_state[stage].angle;
                recorded.velocity = stage_state[stage].velocity;
                recorded.torque = torque;
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
        step_end[step] = state;
        const detail::Vector3* previous =
            (step == 0) ? input.loss.previous_torque : &input.torque[step - 1];
        total_loss += detail::step_loss(input.loss, input.num_steps, step, state, torque, previous);
    }

    std::vector<detail::Matrix6> propagation(substep_count);
    std::vector<detail::Matrix63> input_jacobian(substep_count);
    for (std::size_t substep = 0; substep < substep_count; ++substep) {
        detail::StageJacobians jacobians[4];
        for (int stage_index = 0; stage_index < 4; ++stage_index) {
            const detail::StageInput& recorded =
                stage_inputs[substep * 4 + static_cast<std::size_t>(stage_index)];
            StateT stage_state;
            stage_state.angle = recorded.angle;
            stage_state.velocity = recorded.velocity;
            detail::StageData data;
            detail::evaluate_stage_data(model, input.terrain, recorded.base, stage_state,
                                        recorded.torque, data);
            jacobians[stage_index] = detail::evaluate_stage_jacobians(data, model);
        }
        detail::assemble_substep(jacobians, h, propagation[substep], input_jacobian[substep]);
    }

    output.gradient.resize(input.num_steps);
    detail::Vector6 adjoint = detail::Vector6::zero();
    const Real torque_scale = Real(2) / static_cast<Real>(input.num_steps);
    for (std::size_t step = input.num_steps; step-- > 0;) {
        adjoint += detail::step_loss_gradient(input.loss, input.num_steps, step, step_end[step]);
        detail::Vector3 accumulated = detail::Vector3::zero();
        for (std::size_t substep_index = refinement; substep_index-- > 0;) {
            const std::size_t index = step * refinement + substep_index;
            accumulated += math::transpose(input_jacobian[index]) * adjoint;
            adjoint = math::transpose(propagation[index]) * adjoint;
        }
        const detail::Vector3* previous =
            (step == 0) ? input.loss.previous_torque : &input.torque[step - 1];
        detail::Vector3 torque_gradient =
            math::elementwise(input.loss.torque_weights, input.torque[step]);
        if (previous != nullptr) {
            torque_gradient += math::elementwise(input.loss.torque_rate_weights,
                                                 input.torque[step] - *previous);
        }
        if (step + 1 < input.num_steps) {
            torque_gradient -= math::elementwise(input.loss.torque_rate_weights,
                                                 input.torque[step + 1] - input.torque[step]);
        }
        output.gradient[step] = accumulated + torque_gradient * torque_scale;
    }
    output.loss = total_loss;
    output.final_state = step_end.back();
    return total_loss;
}

}  // namespace roll_locked
}  // namespace tcrg::gimbal_dynamics
