#include "gimbal_dynamics/control_gradient.hpp"

#include <stdexcept>
#include <vector>

#include "model_evaluation.hpp"
#include "simulation_core.hpp"
#include "stage_data.hpp"
#include "stage_jacobians.hpp"
#include "thread_pool.hpp"

namespace gimbal {
namespace control_gradient {
namespace {

// 单样本流水线使用的阶段缓冲（每线程一份，跨调用复用）
thread_local std::vector<detail::StageData> stage_data_buffer;
thread_local std::vector<detail::StageJacobians> stage_jacobian_buffer;

// 反向伴随扫描
void reversePass(const simulation::LossSpec& loss, const math::Vector4d* torque,
                 std::size_t num_steps, std::size_t refinement, Workspace& workspace,
                 math::Vector4d* gradient) {
    math::Vector8d adjoint = math::Vector8d::zero();
    const Real torque_scale = Real(1) / static_cast<Real>(num_steps);
    for (std::size_t k = num_steps; k-- > 0;) {
        adjoint += simulation::step_loss_gradient(loss, num_steps, k, workspace.trace.step_end[k]);
        math::Vector4d accumulated = math::Vector4d::zero();
        for (std::size_t s = refinement; s-- > 0;) {
            const std::size_t substep = k * refinement + s;
            accumulated += transpose(workspace.input_jacobian[substep]) * adjoint;
            adjoint = transpose(workspace.propagation[substep]) * adjoint;
        }
        gradient[k] = accumulated + elementwise(loss.torque_weights, torque[k]) * torque_scale;
    }
}

// 串行实现：单样本单线程，批量调用与无工作线程时使用
Real simulateAndGradientSerial(const ModelParameters& model, const Terrain& terrain,
                               const simulation::State& initial, const BaseState* base,
                               const math::Vector4d* torque, std::size_t num_steps,
                               const simulation::TrajectorySpec& spec,
                               const simulation::LossSpec& loss, Workspace& workspace,
                               math::Vector4d* gradient, simulation::State* final_state) {
    simulation::validate(spec, num_steps);
    workspace.resize(num_steps, spec.refinement);
    const std::size_t refinement = static_cast<std::size_t>(spec.refinement);
    const Real h = spec.step / static_cast<Real>(spec.refinement);
    const detail::ScalarModelParameters<Real> scalar_model = detail::convert_model<Real>(model);

    detail::StageJacobians stage[4];
    const auto stage_visitor = [&](std::size_t stage_index, const detail::StageData& data) {
        const std::size_t substep = stage_index / 4;
        const int t = static_cast<int>(stage_index % 4);
        stage[t] = detail::evaluate_stage_jacobians(data, scalar_model);
        if (t == 3) {
            detail::assemble_substep(stage, h, workspace.propagation[substep],
                                     workspace.input_jacobian[substep]);
        }
    };
    const auto step_visitor = [](std::size_t, const simulation::State&) {};
    const Real loss_value = detail::simulate_trajectory(
        model, terrain, initial, base, torque, num_steps, spec, loss, workspace.trace, false,
        nullptr, stage_visitor, step_visitor);
    if (final_state != nullptr) {
        *final_state = workspace.trace.step_end.back();
    }
    if (gradient != nullptr) {
        reversePass(loss, torque, num_steps, refinement, workspace, gradient);
    }
    return loss_value;
}

// 流水线实现：前向积分在调用线程，阶段 Jacobian 由线程池并行
Real simulateAndGradientThreaded(const ModelParameters& model, const Terrain& terrain,
                                 const simulation::State& initial, const BaseState* base,
                                 const math::Vector4d* torque, std::size_t num_steps,
                                 const simulation::TrajectorySpec& spec,
                                 const simulation::LossSpec& loss, Workspace& workspace,
                                 math::Vector4d* gradient, simulation::State* final_state) {
    simulation::validate(spec, num_steps);
    workspace.resize(num_steps, spec.refinement);
    const std::size_t refinement = static_cast<std::size_t>(spec.refinement);
    const std::size_t stage_count = num_steps * refinement * 4;
    const Real h = spec.step / static_cast<Real>(spec.refinement);
    stage_data_buffer.resize(stage_count);
    stage_jacobian_buffer.resize(stage_count);
    detail::StageData* const stage_data = stage_data_buffer.data();
    detail::StageJacobians* const stage_jacobians = stage_jacobian_buffer.data();
    const detail::ScalarModelParameters<Real> scalar_model = detail::convert_model<Real>(model);

    auto pipeline = detail::ThreadPool::pool().start_pipeline(
        stage_count, [stage_data, stage_jacobians, &scalar_model](std::size_t index) {
            stage_jacobians[index] =
                detail::evaluate_stage_jacobians(stage_data[index], scalar_model);
        });
    const auto stage_visitor = [&](std::size_t stage_index, const detail::StageData&) {
        // 每个子步的四个阶段一起发布，减少生产者与工作线程之间的缓存行争用
        if ((stage_index & 3U) == 3U) {
            pipeline.publish(stage_index);
        }
    };
    const auto step_visitor = [](std::size_t, const simulation::State&) {};
    const Real loss_value = detail::simulate_trajectory(
        model, terrain, initial, base, torque, num_steps, spec, loss, workspace.trace, false,
        stage_data, stage_visitor, step_visitor);
    pipeline.finish();

    // 子步之间相互独立，组装阶段按子步并行
    const std::size_t substep_count = num_steps * refinement;
    math::Matrix8d* const propagations = workspace.propagation.data();
    math::Matrix84d* const input_jacobians = workspace.input_jacobian.data();
    detail::ThreadPool::pool().parallel_for(
        substep_count, [stage_jacobians, propagations, input_jacobians, h](
                           int, std::size_t begin, std::size_t end) {
            detail::StageJacobians stage[4];
            for (std::size_t substep = begin; substep < end; ++substep) {
                for (int t = 0; t < 4; ++t) {
                    stage[t] = stage_jacobians[substep * 4 + static_cast<std::size_t>(t)];
                }
                detail::assemble_substep(stage, h, propagations[substep], input_jacobians[substep]);
            }
        });
    if (final_state != nullptr) {
        *final_state = workspace.trace.step_end.back();
    }
    if (gradient != nullptr) {
        reversePass(loss, torque, num_steps, refinement, workspace, gradient);
    }
    return loss_value;
}

}  // namespace

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

Real simulateAndGradient(const ModelParameters& model, const Terrain& terrain,
                         const simulation::State& initial, const BaseState* base,
                         const math::Vector4d* torque, std::size_t num_steps,
                         const simulation::TrajectorySpec& spec,
                         const simulation::LossSpec& loss, Workspace& workspace,
                         math::Vector4d* gradient, simulation::State* final_state) {
    if (detail::ThreadPool::pool().worker_count() > 0) {
        return simulateAndGradientThreaded(model, terrain, initial, base, torque, num_steps,
                                           spec, loss, workspace, gradient, final_state);
    }
    return simulateAndGradientSerial(model, terrain, initial, base, torque, num_steps, spec, loss,
                                     workspace, gradient, final_state);
}

Real simulateAndGradientBatch(const ModelParameters& model, const Terrain& terrain,
                              const simulation::State* initial, const BaseState* base,
                              const math::Vector4d* torque, std::size_t batch,
                              std::size_t num_steps, const simulation::TrajectorySpec& spec,
                              const simulation::LossSpec* loss, BatchWorkspace& workspace,
                              math::Vector4d* gradient, simulation::State* final_state) {
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
            loss_values[sample] = simulateAndGradientSerial(
                model, terrain, initial[sample], base + offset, torque + offset, num_steps, spec,
                loss[sample], work, (gradient != nullptr) ? gradient + offset : nullptr,
                (final_state != nullptr) ? final_state + sample : nullptr);
        });
    Real total = Real(0);
    for (std::size_t sample = 0; sample < batch; ++sample) {
        total += loss_values[sample];
    }
    return total / static_cast<Real>(batch);
}

}  // namespace control_gradient
}  // namespace gimbal
