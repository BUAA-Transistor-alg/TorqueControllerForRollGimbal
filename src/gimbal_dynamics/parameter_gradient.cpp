#include "gimbal_dynamics/parameter_gradient.hpp"

#include <algorithm>
#include <atomic>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "gimbal_dynamics/model_evaluation.hpp"
#include "gimbal_dynamics/simulation_core.hpp"
#include "gimbal_dynamics/stage_data.hpp"
#include "gimbal_dynamics/stage_jacobians.hpp"
#include "gimbal_dynamics/thread_pool.hpp"

namespace tcrg::gimbal_dynamics {
namespace parameter_gradient {
namespace {

// 单样本流水线使用的阶段输入缓冲（每线程一份，跨调用复用）
thread_local std::vector<detail::StageInput> stage_input_buffer;
thread_local std::vector<math::Matrix8d> substep_propagation_buffer;
thread_local std::vector<detail::ParameterJacobian> substep_parameter_buffer;
thread_local std::unique_ptr<std::atomic<bool>[]> substep_ready_buffer;
thread_local std::size_t substep_ready_capacity = 0;

void spin_pause() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#else
    std::this_thread::yield();
#endif
}

// 串行实现：单样本单线程，批量调用与无工作线程时使用
Real simulateAndSensitivitySerial(const ModelParameters& model, const Terrain& terrain,
                                  const simulation::State& initial, const BaseState* base,
                                  const math::Vector4d* torque, std::size_t num_steps,
                                  const simulation::TrajectorySpec& spec,
                                  const simulation::LossSpec& loss, Workspace& workspace,
                                  Gradient* gradient, simulation::State* final_state) {
    simulation::validate(spec, num_steps);
    workspace.resize(num_steps, spec.refinement);
    const Real h = spec.step / static_cast<Real>(spec.refinement);
    const detail::ScalarModelParameters<Real> scalar_model = detail::convert_model<Real>(model);

    math::Matrix<Real, 8, parameters::kParameterCount> sensitivity =
        math::Matrix<Real, 8, parameters::kParameterCount>::zero();
    Gradient total = Gradient::zero();
    detail::StageJacobians stage[4];
    detail::ParameterJacobian stage_parameter[4];
    const auto stage_visitor = [&](std::size_t stage_index, const detail::StageData& data) {
        const int t = static_cast<int>(stage_index % 4);
        stage[t].state = math::Matrix<Real, 8, 8>::zero();
        stage_parameter[t] = detail::evaluate_parameter_jacobian(data, scalar_model, stage[t].state);
        if (t == 3) {
            math::Matrix<Real, 8, 8> propagation;
            detail::ParameterJacobian parameter_jacobian;
            detail::assemble_parameter_substep(stage, stage_parameter, h, propagation,
                                               parameter_jacobian);
            detail::apply_sensitivity(propagation, sensitivity, parameter_jacobian, sensitivity);
        }
    };
    const auto step_visitor = [&](std::size_t k, const simulation::State& step_end_state) {
        total += transpose(sensitivity) *
                 simulation::step_loss_gradient(loss, num_steps, k, step_end_state);
    };
    const Real loss_value = detail::simulate_trajectory(
        model, terrain, initial, base, torque, num_steps, spec, loss, workspace.trace, true,
        nullptr, stage_visitor, step_visitor);
    if (final_state != nullptr) {
        *final_state = workspace.trace.step_end.back();
    }
    if (gradient != nullptr) {
        *gradient = total;
    }
    return loss_value;
}

// 流水线实现：前向积分在调用线程，阶段 Jacobian 由线程池并行
Real simulateAndSensitivityThreaded(const ModelParameters& model, const Terrain& terrain,
                                    const simulation::State& initial, const BaseState* base,
                                    const math::Vector4d* torque, std::size_t num_steps,
                                    const simulation::TrajectorySpec& spec,
                                    const simulation::LossSpec& loss, Workspace& workspace,
                                    Gradient* gradient, simulation::State* final_state,
                                    detail::ThreadPool& pool) {
    simulation::validate(spec, num_steps);
    workspace.resize(num_steps, spec.refinement);
    const std::size_t refinement = static_cast<std::size_t>(spec.refinement);
    const std::size_t substep_count = num_steps * refinement;
    const std::size_t stage_count = substep_count * 4;
    const Real h = spec.step / static_cast<Real>(spec.refinement);
    stage_input_buffer.resize(stage_count);
    detail::StageInput* const stage_inputs = stage_input_buffer.data();
    substep_propagation_buffer.resize(substep_count);
    substep_parameter_buffer.resize(substep_count);
    math::Matrix8d* const substep_propagations = substep_propagation_buffer.data();
    detail::ParameterJacobian* const substep_parameters = substep_parameter_buffer.data();
    if (substep_ready_capacity < substep_count) {
        substep_ready_buffer.reset(new std::atomic<bool>[substep_count]);
        substep_ready_capacity = substep_count;
    }
    std::atomic<bool>* const substep_ready = substep_ready_buffer.get();
    for (std::size_t substep = 0; substep < substep_count; ++substep) {
        substep_ready[substep].store(false, std::memory_order_relaxed);
    }
    const detail::ScalarModelParameters<Real> scalar_model = detail::convert_model<Real>(model);
    const detail::TerrainFrame<Real> terrain_frame =
        detail::evaluate_terrain_frame(detail::convert_terrain<Real>(terrain));

    // 一个子步作为一个任务：工作线程重算阶段中间量、求四个阶段 Jacobian、组装 W 与 P 并标记就绪
    auto pipeline = pool.start_pipeline(
        substep_count, [stage_inputs, substep_ready, substep_propagations, substep_parameters, h,
                        &scalar_model, &terrain_frame](std::size_t substep) {
            detail::StageJacobians stage[4];
            detail::ParameterJacobian stage_parameter[4];
            for (int t = 0; t < 4; ++t) {
                const detail::StageInput& input =
                    stage_inputs[substep * 4 + static_cast<std::size_t>(t)];
                simulation::State state;
                state.angle = input.angle;
                state.velocity = input.qdot;
                detail::StageData data;
                detail::evaluate_stage_data(scalar_model, terrain_frame, input.base, state,
                                            input.torque, data);
                stage[t].state = math::Matrix<Real, 8, 8>::zero();
                stage_parameter[t] = detail::evaluate_parameter_jacobian(data, scalar_model,
                                                                        stage[t].state);
            }
            detail::assemble_parameter_substep(stage, stage_parameter, h,
                                               substep_propagations[substep],
                                               substep_parameters[substep]);
            substep_ready[substep].store(true, std::memory_order_release);
        });
    const auto stage_visitor = [&](std::size_t stage_index, const detail::StageInput&) {
        // 每个子步的四个阶段一起发布，减少生产者与工作线程之间的缓存行争用
        if ((stage_index & 3U) == 3U) {
            pipeline.publish(stage_index / 4);
        }
    };
    const auto step_visitor = [](std::size_t, const simulation::State&) {};
    Real loss_value;
    try {
        loss_value = detail::simulate_trajectory_inputs(
            model, terrain, initial, base, torque, num_steps, spec, loss, workspace.trace,
            stage_inputs, stage_visitor, step_visitor);
    } catch (...) {
        pipeline.abort();
        throw;
    }

    // 子步按顺序就绪后立即推进灵敏度，与剩余工作线程的计算重叠
    math::Matrix<Real, 8, parameters::kParameterCount> sensitivity =
        math::Matrix<Real, 8, parameters::kParameterCount>::zero();
    Gradient total = Gradient::zero();
    for (std::size_t k = 0; k < num_steps; ++k) {
        for (std::size_t s = 0; s < refinement; ++s) {
            const std::size_t substep = k * refinement + s;
            while (!substep_ready[substep].load(std::memory_order_acquire)) {
                spin_pause();
            }
            detail::apply_sensitivity(substep_propagations[substep], sensitivity,
                                      substep_parameters[substep], sensitivity);
        }
        total += transpose(sensitivity) *
                 simulation::step_loss_gradient(loss, num_steps, k, workspace.trace.step_end[k]);
    }
    pipeline.finish();
    if (final_state != nullptr) {
        *final_state = workspace.trace.step_end.back();
    }
    if (gradient != nullptr) {
        *gradient = total;
    }
    return loss_value;
}

}  // namespace

void Workspace::resize(std::size_t num_steps, int refinement) {
    trace.resize(num_steps, refinement);
}

void Workspace::release() {
    trace.release();
}

void BatchWorkspace::resize(std::size_t batch, const detail::ThreadPool& pool) {
    (void)batch;
    const std::size_t participants = static_cast<std::size_t>(pool.participant_count());
    if (workspaces.size() != participants) {
        workspaces.resize(participants);
    }
}

void BatchWorkspace::release() {
    workspaces.clear();
}

Real simulateAndSensitivity(const ModelParameters& model, const Terrain& terrain,
                            const simulation::State& initial, const BaseState* base,
                            const math::Vector4d* torque, std::size_t num_steps,
                            const simulation::TrajectorySpec& spec,
                            const simulation::LossSpec& loss, Workspace& workspace,
                            Gradient* gradient, simulation::State* final_state,
                            detail::ThreadPool* pool) {
    if (pool != nullptr && pool->worker_count() > 0) {
        return simulateAndSensitivityThreaded(model, terrain, initial, base, torque, num_steps,
                                              spec, loss, workspace, gradient, final_state, *pool);
    }
    return simulateAndSensitivitySerial(model, terrain, initial, base, torque, num_steps, spec,
                                        loss, workspace, gradient, final_state);
}

Real simulateAndSensitivityBatch(const ModelParameters& model, const Terrain& terrain,
                                 const simulation::State* initial, const BaseState* base,
                                 const math::Vector4d* torque, std::size_t batch,
                                 std::size_t num_steps, const simulation::TrajectorySpec& spec,
                                 const simulation::LossSpec* loss, BatchWorkspace& workspace,
                                 Gradient* gradient, simulation::State* final_state,
                                 detail::ThreadPool& pool) {
    if (batch == 0) {
        return Real(0);
    }
    if (initial == nullptr || base == nullptr || torque == nullptr || loss == nullptr) {
        throw std::invalid_argument("批量输入不能为空");
    }
    simulation::validate(spec, num_steps);
    workspace.resize(batch, pool);
    std::vector<Real> loss_values(batch);
    std::vector<Gradient> sample_gradients(batch);
    const std::size_t participants = static_cast<std::size_t>(pool.participant_count());
    const std::size_t chunk = std::max<std::size_t>(1, batch / participants);
    pool.parallel_dynamic(
        batch, [&](int participant, std::size_t sample) {
            Workspace& work = workspace.workspaces[static_cast<std::size_t>(participant)];
            const std::size_t offset = sample * num_steps;
            loss_values[sample] = simulateAndSensitivitySerial(
                model, terrain, initial[sample], base + offset, torque + offset, num_steps, spec,
                loss[sample], work, (gradient != nullptr) ? &sample_gradients[sample] : nullptr,
                (final_state != nullptr) ? final_state + sample : nullptr);
        },
        chunk);
    Real total = Real(0);
    Gradient accumulated = Gradient::zero();
    for (std::size_t sample = 0; sample < batch; ++sample) {
        total += loss_values[sample];
        accumulated += sample_gradients[sample];
    }
    if (gradient != nullptr) {
        *gradient = accumulated / static_cast<Real>(batch);
    }
    return total / static_cast<Real>(batch);
}

}  // namespace parameter_gradient
}  // namespace tcrg::gimbal_dynamics
