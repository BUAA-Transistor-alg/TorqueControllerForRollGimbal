#include "gimbal_dynamics/parameter_gradient.hpp"

#include <stdexcept>
#include <vector>

#include "model_evaluation.hpp"
#include "simulation_core.hpp"
#include "stage_data.hpp"
#include "stage_jacobians.hpp"
#include "thread_pool.hpp"

namespace gimbal {
namespace parameter_gradient {
namespace {

// 单样本流水线使用的阶段缓冲（每线程一份，跨调用复用）
thread_local std::vector<detail::StageData> stage_data_buffer;
thread_local std::vector<detail::StageJacobians> stage_jacobian_buffer;
thread_local std::vector<detail::ParameterJacobian> stage_parameter_buffer;
thread_local std::vector<math::Matrix8d> substep_propagation_buffer;
thread_local std::vector<detail::ParameterJacobian> substep_parameter_buffer;

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
        model, terrain, initial, base, torque, num_steps, spec, loss, workspace.trace, false,
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
                                    Gradient* gradient, simulation::State* final_state) {
    simulation::validate(spec, num_steps);
    workspace.resize(num_steps, spec.refinement);
    const std::size_t refinement = static_cast<std::size_t>(spec.refinement);
    const std::size_t stage_count = num_steps * refinement * 4;
    const Real h = spec.step / static_cast<Real>(spec.refinement);
    stage_data_buffer.resize(stage_count);
    stage_jacobian_buffer.resize(stage_count);
    stage_parameter_buffer.resize(stage_count);
    detail::StageData* const stage_data = stage_data_buffer.data();
    detail::StageJacobians* const stage_jacobians = stage_jacobian_buffer.data();
    detail::ParameterJacobian* const stage_parameters = stage_parameter_buffer.data();
    const detail::ScalarModelParameters<Real> scalar_model = detail::convert_model<Real>(model);

    auto pipeline = detail::ThreadPool::pool().start_pipeline(
        stage_count, [stage_data, stage_jacobians, stage_parameters, &scalar_model](
                         std::size_t index) {
            stage_jacobians[index].state = math::Matrix<Real, 8, 8>::zero();
            stage_parameters[index] = detail::evaluate_parameter_jacobian(
                stage_data[index], scalar_model, stage_jacobians[index].state);
        });
    const auto stage_visitor = [&](std::size_t stage_index, const detail::StageData&) {
        pipeline.publish(stage_index);
    };
    const auto step_visitor = [](std::size_t, const simulation::State&) {};
    const Real loss_value = detail::simulate_trajectory(
        model, terrain, initial, base, torque, num_steps, spec, loss, workspace.trace, false,
        stage_data, stage_visitor, step_visitor);
    pipeline.finish();

    // 子步之间相互独立，组装阶段按子步并行
    const std::size_t substep_count = num_steps * refinement;
    substep_propagation_buffer.resize(substep_count);
    substep_parameter_buffer.resize(substep_count);
    math::Matrix8d* const substep_propagations = substep_propagation_buffer.data();
    detail::ParameterJacobian* const substep_parameters = substep_parameter_buffer.data();
    detail::ThreadPool::pool().parallel_for(
        substep_count, [stage_jacobians, stage_parameters, substep_propagations,
                        substep_parameters, h](int, std::size_t begin, std::size_t end) {
            detail::StageJacobians stage[4];
            detail::ParameterJacobian stage_parameter[4];
            for (std::size_t substep = begin; substep < end; ++substep) {
                for (int t = 0; t < 4; ++t) {
                    stage[t] = stage_jacobians[substep * 4 + static_cast<std::size_t>(t)];
                    stage_parameter[t] =
                        stage_parameters[substep * 4 + static_cast<std::size_t>(t)];
                }
                detail::assemble_parameter_substep(stage, stage_parameter, h,
                                                   substep_propagations[substep],
                                                   substep_parameters[substep]);
            }
        });

    math::Matrix<Real, 8, parameters::kParameterCount> sensitivity =
        math::Matrix<Real, 8, parameters::kParameterCount>::zero();
    Gradient total = Gradient::zero();
    for (std::size_t k = 0; k < num_steps; ++k) {
        for (std::size_t s = 0; s < refinement; ++s) {
            const std::size_t substep = k * refinement + s;
            detail::apply_sensitivity(substep_propagations[substep], sensitivity,
                                      substep_parameters[substep], sensitivity);
        }
        total += transpose(sensitivity) *
                 simulation::step_loss_gradient(loss, num_steps, k, workspace.trace.step_end[k]);
    }
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

void BatchWorkspace::resize(std::size_t batch) {
    (void)batch;
    const std::size_t participants =
        static_cast<std::size_t>(detail::ThreadPool::batch_pool().participant_count());
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
                            Gradient* gradient, simulation::State* final_state) {
    if (detail::ThreadPool::pool().worker_count() > 0) {
        return simulateAndSensitivityThreaded(model, terrain, initial, base, torque, num_steps,
                                              spec, loss, workspace, gradient, final_state);
    }
    return simulateAndSensitivitySerial(model, terrain, initial, base, torque, num_steps, spec,
                                        loss, workspace, gradient, final_state);
}

Real simulateAndSensitivityBatch(const ModelParameters& model, const Terrain& terrain,
                                 const simulation::State* initial, const BaseState* base,
                                 const math::Vector4d* torque, std::size_t batch,
                                 std::size_t num_steps, const simulation::TrajectorySpec& spec,
                                 const simulation::LossSpec* loss, BatchWorkspace& workspace,
                                 Gradient* gradient, simulation::State* final_state) {
    if (batch == 0) {
        return Real(0);
    }
    if (initial == nullptr || base == nullptr || torque == nullptr || loss == nullptr) {
        throw std::invalid_argument("批量输入不能为空");
    }
    workspace.resize(batch);
    std::vector<Real> loss_values(batch);
    detail::ThreadPool::batch_pool().parallel_dynamic(
        batch, [&](int participant, std::size_t sample) {
            Workspace& work = workspace.workspaces[static_cast<std::size_t>(participant)];
            const std::size_t offset = sample * num_steps;
            loss_values[sample] = simulateAndSensitivitySerial(
                model, terrain, initial[sample], base + offset, torque + offset, num_steps, spec,
                loss[sample], work, (gradient != nullptr) ? gradient + sample : nullptr,
                (final_state != nullptr) ? final_state + sample : nullptr);
        });
    Real total = Real(0);
    for (std::size_t sample = 0; sample < batch; ++sample) {
        total += loss_values[sample];
    }
    return total / static_cast<Real>(batch);
}

}  // namespace parameter_gradient
}  // namespace gimbal
