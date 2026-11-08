#pragma once

// 内部：带可选阶段中间量缓存的前向仿真。
//
// 公开的 simulation::simulate 与本函数共用同一份积分循环。
// 两个梯度模块传入 stage_cache，从而直接复用前向计算已求出的
// 链、质量矩阵、偏置力与加速度，导数阶段不再重复求值这些量。

#include <cstddef>
#include <vector>

#include "gimbal_dynamics/dynamics_model.hpp"
#include "gimbal_dynamics/math_types.hpp"
#include "gimbal_dynamics/simulation.hpp"
#include "stage_data.hpp"

namespace gimbal {
namespace detail {

// stage_cache 非空时写入 num_steps * refinement * 4 个阶段中间量
double simulate_trajectory(const ModelParameters& model, const Terrain& terrain,
                           const simulation::State& initial, const BaseState* base,
                           const math::Vector4d* torque, std::size_t num_steps,
                           const simulation::TrajectorySpec& spec,
                           const simulation::LossSpec& loss, simulation::Trace& trace,
                           std::vector<StageData>* stage_cache);

}  // namespace detail
}  // namespace gimbal
