#pragma once

// 内部：前向积分循环，带可选的阶段访问回调。
//
// 公开的 simulation::simulate 与两个梯度模块共用这一份积分循环：
//   * 公开仿真只记录轨迹；
//   * 控制量梯度模块在阶段回调里计算状态 Jacobian 与输入 Jacobian，
//     并在每个子步结束时组装传播子，随后做反向伴随；
//   * 参数灵敏度模块在阶段回调里计算状态 Jacobian 与参数 Jacobian，
//     每个子步结束时更新灵敏度矩阵，每个主步结束时累加损失梯度。
// 单遍完成前向与 Jacobian 计算，阶段中间量不落盘。

#include <cstddef>
#include <stdexcept>

#include "gimbal_dynamics/dynamics_model.hpp"
#include "gimbal_dynamics/math_types.hpp"
#include "gimbal_dynamics/simulation.hpp"
#include "gimbal_dynamics/stage_data.hpp"

namespace tcrg::gimbal_dynamics {
namespace detail {

// 主步内底盘量按等角加速度外推，线加速度保持不变
inline BaseState advance_base(const BaseState& base, Real time) {
    BaseState result = base;
    result.yaw = base.yaw + base.yaw_rate * time + 0.5 * base.yaw_acceleration * time * time;
    result.yaw_rate = base.yaw_rate + base.yaw_acceleration * time;
    return result;
}

inline math::Vector8d pack_state(const simulation::State& state) {
    math::Vector8d result;
    for (int i = 0; i < 4; ++i) {
        result[i] = state.angle[i];
        result[4 + i] = state.velocity[i];
    }
    return result;
}

inline simulation::State unpack_state(const math::Vector8d& state) {
    simulation::State result;
    for (int i = 0; i < 4; ++i) {
        result.angle[i] = state[i];
        result.velocity[i] = state[4 + i];
    }
    return result;
}

// 纯前向仿真：不构造阶段中间量结构，只求加速度并记录轨迹
inline Real simulate_forward(const ModelParameters& model, const Terrain& terrain,
                             const simulation::State& initial, const BaseState* base,
                             const math::Vector4d* torque, std::size_t num_steps,
                             const simulation::TrajectorySpec& spec,
                             const simulation::LossSpec& loss, simulation::Trace& trace) {
    simulation::validate(spec, num_steps);
    if (base == nullptr) {
        throw std::invalid_argument("base 不能为空");
    }
    if (torque == nullptr) {
        throw std::invalid_argument("torque 不能为空");
    }
    trace.resize(num_steps, spec.refinement);
    const ScalarModelParameters<Real> scalar_model = convert_model<Real>(model);
    const TerrainFrame<Real> terrain_frame =
        evaluate_terrain_frame(convert_terrain<Real>(terrain));
    const std::size_t refinement = static_cast<std::size_t>(spec.refinement);
    const Real h = spec.step / static_cast<Real>(spec.refinement);
    math::Vector8d state = pack_state(initial);
    Real total_loss = 0.0;

    for (std::size_t k = 0; k < num_steps; ++k) {
        const math::Vector4d torque_k = torque[k];
        for (std::size_t s = 0; s < refinement; ++s) {
            const Real substep_time = static_cast<Real>(s) * h;
            const BaseState stage_input[4] = {
                advance_base(base[k], substep_time),
                advance_base(base[k], substep_time + 0.5 * h),
                advance_base(base[k], substep_time + 0.5 * h),
                advance_base(base[k], substep_time + h),
            };
            math::Vector8d stage_state[4];
            math::Vector8d stage_derivative[4];
            stage_state[0] = state;
            for (int t = 0; t < 4; ++t) {
                if (t > 0) {
                    const Real advance = (t == 3) ? h : 0.5 * h;
                    stage_state[t] = stage_state[0] + stage_derivative[t - 1] * advance;
                }
                math::Vector4d angle;
                math::Vector4d velocity;
                for (int i = 0; i < 4; ++i) {
                    angle[i] = stage_state[t][i];
                    velocity[i] = stage_state[t][4 + i];
                }
                const math::Vector4d acceleration = evaluate_acceleration<Real>(
                    scalar_model, terrain_frame, stage_input[t], angle, velocity, torque_k);
                for (int i = 0; i < 4; ++i) {
                    stage_derivative[t][i] = stage_state[t][4 + i];
                    stage_derivative[t][4 + i] = acceleration[i];
                }
                const std::size_t stage_index =
                    (k * refinement + s) * 4 + static_cast<std::size_t>(t);
                trace.stage_state[stage_index] = unpack_state(stage_state[t]);
                trace.stage_base[stage_index] = stage_input[t];
            }
            state = stage_state[0] + (stage_derivative[0] + stage_derivative[1] * 2.0 +
                                      stage_derivative[2] * 2.0 + stage_derivative[3]) *
                                         (h / 6.0);
        }
        trace.step_end[k] = unpack_state(state);
        const math::Vector4d* previous_torque =
            (k == 0) ? loss.previous_torque : &torque[k - 1];
        total_loss += simulation::step_loss(loss, num_steps, k, trace.step_end[k], torque_k,
                                            previous_torque);
    }
    return total_loss;
}

// 紧凑阶段输入：流水线生产者只发布这些量，工作者据此重算阶段中间量
struct StageInput {
    BaseState base;
    math::Vector4d angle;
    math::Vector4d qdot;
    math::Vector4d torque;
};

// 前向积分并把每个阶段的输入交给回调；阶段中间量由消费者按需重算，
// 避免生产者写入完整 StageData 带来的缓存一致性流量。
template <typename StageVisitor, typename StepVisitor>
Real simulate_trajectory_inputs(const ModelParameters& model, const Terrain& terrain,
                                const simulation::State& initial, const BaseState* base,
                                const math::Vector4d* torque, std::size_t num_steps,
                                const simulation::TrajectorySpec& spec,
                                const simulation::LossSpec& loss, simulation::Trace& trace,
                                StageInput* stage_inputs, StageVisitor&& stage_visitor,
                                StepVisitor&& step_visitor) {
    simulation::validate(spec, num_steps);
    if (base == nullptr) {
        throw std::invalid_argument("base 不能为空");
    }
    if (torque == nullptr) {
        throw std::invalid_argument("torque 不能为空");
    }
    trace.resize(num_steps, spec.refinement);
    const ScalarModelParameters<Real> scalar_model = convert_model<Real>(model);
    const TerrainFrame<Real> terrain_frame =
        evaluate_terrain_frame(convert_terrain<Real>(terrain));
    const std::size_t refinement = static_cast<std::size_t>(spec.refinement);
    const Real h = spec.step / static_cast<Real>(spec.refinement);
    math::Vector8d state = pack_state(initial);
    Real total_loss = 0.0;

    for (std::size_t k = 0; k < num_steps; ++k) {
        const math::Vector4d torque_k = torque[k];
        for (std::size_t s = 0; s < refinement; ++s) {
            const Real substep_time = static_cast<Real>(s) * h;
            const BaseState stage_base[4] = {
                advance_base(base[k], substep_time),
                advance_base(base[k], substep_time + 0.5 * h),
                advance_base(base[k], substep_time + 0.5 * h),
                advance_base(base[k], substep_time + h),
            };
            math::Vector8d stage_state[4];
            math::Vector8d stage_derivative[4];
            stage_state[0] = state;
            for (int t = 0; t < 4; ++t) {
                if (t > 0) {
                    const Real advance = (t == 3) ? h : 0.5 * h;
                    stage_state[t] = stage_state[0] + stage_derivative[t - 1] * advance;
                }
                math::Vector4d angle;
                math::Vector4d velocity;
                for (int i = 0; i < 4; ++i) {
                    angle[i] = stage_state[t][i];
                    velocity[i] = stage_state[t][4 + i];
                }
                const math::Vector4d acceleration = evaluate_acceleration<Real>(
                    scalar_model, terrain_frame, stage_base[t], angle, velocity, torque_k);
                for (int i = 0; i < 4; ++i) {
                    stage_derivative[t][i] = stage_state[t][4 + i];
                    stage_derivative[t][4 + i] = acceleration[i];
                }
                const std::size_t stage_index =
                    (k * refinement + s) * 4 + static_cast<std::size_t>(t);
                StageInput& input = stage_inputs[stage_index];
                input.base = stage_base[t];
                input.angle = angle;
                input.qdot = velocity;
                input.torque = torque_k;
                stage_visitor(stage_index, input);
                trace.stage_state[stage_index] = unpack_state(stage_state[t]);
                trace.stage_base[stage_index] = stage_base[t];
            }
            state = stage_state[0] + (stage_derivative[0] + stage_derivative[1] * 2.0 +
                                      stage_derivative[2] * 2.0 + stage_derivative[3]) *
                                         (h / 6.0);
        }
        trace.step_end[k] = unpack_state(state);
        const math::Vector4d* previous_torque =
            (k == 0) ? loss.previous_torque : &torque[k - 1];
        total_loss += simulation::step_loss(loss, num_steps, k, trace.step_end[k], torque_k,
                                            previous_torque);
        step_visitor(k, trace.step_end[k]);
    }
    return total_loss;
}

// 阶段回调签名：void(std::size_t stage_index, const StageData& data)
// 主步回调签名：void(std::size_t step_index, const simulation::State& step_end_state)
// stage_buffer 非空时阶段中间量就地写入该缓冲区，供流水线并行处理。
template <typename StageVisitor, typename StepVisitor>
Real simulate_trajectory(const ModelParameters& model, const Terrain& terrain,
                         const simulation::State& initial, const BaseState* base,
                         const math::Vector4d* torque, std::size_t num_steps,
                         const simulation::TrajectorySpec& spec,
                         const simulation::LossSpec& loss, simulation::Trace& trace,
                         bool record_stages, StageData* stage_buffer,
                         StageVisitor&& stage_visitor, StepVisitor&& step_visitor) {
    simulation::validate(spec, num_steps);
    if (base == nullptr) {
        throw std::invalid_argument("base 不能为空");
    }
    if (torque == nullptr) {
        throw std::invalid_argument("torque 不能为空");
    }
    trace.resize(num_steps, spec.refinement);
    const ScalarModelParameters<Real> scalar_model = convert_model<Real>(model);
    const ScalarTerrain<Real> scalar_terrain = convert_terrain<Real>(terrain);
    const TerrainFrame<Real> terrain_frame = evaluate_terrain_frame(scalar_terrain);
    const std::size_t refinement = static_cast<std::size_t>(spec.refinement);
    const Real h = spec.step / static_cast<Real>(spec.refinement);
    math::Vector8d state = pack_state(initial);
    Real total_loss = 0.0;
    StageData local_data;

    for (std::size_t k = 0; k < num_steps; ++k) {
        const math::Vector4d torque_k = torque[k];
        for (std::size_t s = 0; s < refinement; ++s) {
            const Real substep_time = static_cast<Real>(s) * h;
            const BaseState stage_input[4] = {
                advance_base(base[k], substep_time),
                advance_base(base[k], substep_time + 0.5 * h),
                advance_base(base[k], substep_time + 0.5 * h),
                advance_base(base[k], substep_time + h),
            };
            math::Vector8d stage_state[4];
            math::Vector8d stage_derivative[4];
            stage_state[0] = state;
            for (int t = 0; t < 4; ++t) {
                if (t > 0) {
                    const Real advance = (t == 3) ? h : 0.5 * h;
                    stage_state[t] = stage_state[0] + stage_derivative[t - 1] * advance;
                }
                const std::size_t stage_index =
                    (k * refinement + s) * 4 + static_cast<std::size_t>(t);
                StageData& data =
                    (stage_buffer != nullptr) ? stage_buffer[stage_index] : local_data;
                evaluate_stage_data(scalar_model, terrain_frame, stage_input[t],
                                    unpack_state(stage_state[t]), torque_k, data);
                for (int i = 0; i < 4; ++i) {
                    stage_derivative[t][i] = stage_state[t][4 + i];
                    stage_derivative[t][4 + i] = data.acceleration[i];
                }
                stage_visitor(stage_index, data);
                if (record_stages) {
                    trace.stage_state[stage_index] = unpack_state(stage_state[t]);
                    trace.stage_base[stage_index] = stage_input[t];
                }
            }
            state = stage_state[0] + (stage_derivative[0] + stage_derivative[1] * 2.0 +
                                      stage_derivative[2] * 2.0 + stage_derivative[3]) *
                                         (h / 6.0);
        }
        trace.step_end[k] = unpack_state(state);
        const math::Vector4d* previous_torque =
            (k == 0) ? loss.previous_torque : &torque[k - 1];
        total_loss += simulation::step_loss(loss, num_steps, k, trace.step_end[k], torque_k,
                                            previous_torque);
        step_visitor(k, trace.step_end[k]);
    }
    return total_loss;
}

}  // namespace detail
}  // namespace tcrg::gimbal_dynamics
