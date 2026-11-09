#pragma once

// 后端内部：参数灵敏度的实现（按构建精度编译）。
// 公开使用方式见 gimbal_dynamics/handlers.hpp 的处理器模板。

#include <cstddef>
#include <vector>

#include "gimbal_dynamics/math_types.hpp"
#include "gimbal_dynamics/parameters.hpp"
#include "gimbal_dynamics/simulation.hpp"

namespace gimbal {
namespace detail {
class ThreadPool;
}

namespace parameter_gradient {

using Gradient = math::Vector<Real, parameters::kParameterCount>;

struct Workspace {
    simulation::Trace trace;

    void resize(std::size_t num_steps, int refinement);
    void release();
};

struct BatchWorkspace {
    std::vector<Workspace> workspaces;

    void resize(std::size_t batch, const detail::ThreadPool& pool);
    void release();
};

Real simulateAndSensitivity(const ModelParameters& model, const Terrain& terrain,
                              const simulation::State& initial, const BaseState* base,
                              const math::Vector4d* torque, std::size_t num_steps,
                              const simulation::TrajectorySpec& spec,
                              const simulation::LossSpec& loss, Workspace& workspace,
                              Gradient* gradient, simulation::State* final_state,
                              detail::ThreadPool* pool);

Real simulateAndSensitivityBatch(const ModelParameters& model, const Terrain& terrain,
                                   const simulation::State* initial, const BaseState* base,
                                   const math::Vector4d* torque, std::size_t batch,
                                   std::size_t num_steps, const simulation::TrajectorySpec& spec,
                                   const simulation::LossSpec* loss, BatchWorkspace& workspace,
                                   Gradient* gradient, simulation::State* final_state,
                                   detail::ThreadPool& pool);

}  // namespace parameter_gradient
}  // namespace gimbal
