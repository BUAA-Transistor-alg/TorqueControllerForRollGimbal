#pragma once

// 后端内部：控制量梯度的实现（按构建精度编译）。
// 公开使用方式见 gimbal_dynamics/handlers.hpp 的处理器模板。

#include <cstddef>
#include <vector>

#include "gimbal_dynamics/math_types.hpp"
#include "gimbal_dynamics/simulation.hpp"

namespace tcrg::gimbal_dynamics {
namespace detail {
class ThreadPool;
}

namespace control_gradient {

// 调用方分配一次、可跨多次调用复用的缓冲
struct Workspace {
    simulation::Trace trace;
    std::vector<math::Matrix8d> propagation;
    std::vector<math::Matrix84d> input_jacobian;

    void resize(std::size_t num_steps, int refinement);
    void release();
};

// 批量工作区：每个参与者线程一份单样本工作区
struct BatchWorkspace {
    std::vector<Workspace> workspaces;

    void resize(std::size_t batch, const detail::ThreadPool& pool);
    void release();
};

Real simulateAndGradient(const ModelParameters& model, const Terrain& terrain,
                           const simulation::State& initial, const BaseState* base,
                           const math::Vector4d* torque, std::size_t num_steps,
                           const simulation::TrajectorySpec& spec,
                           const simulation::LossSpec& loss, Workspace& workspace,
                           math::Vector4d* gradient, simulation::State* final_state,
                           detail::ThreadPool* pool);

Real simulateAndGradientBatch(const ModelParameters& model, const Terrain& terrain,
                                const simulation::State* initial, const BaseState* base,
                                const math::Vector4d* torque, std::size_t batch,
                                std::size_t num_steps, const simulation::TrajectorySpec& spec,
                                const simulation::LossSpec* loss, BatchWorkspace& workspace,
                                math::Vector4d* gradient, simulation::State* final_state,
                                detail::ThreadPool& pool);

}  // namespace control_gradient
}  // namespace tcrg::gimbal_dynamics
