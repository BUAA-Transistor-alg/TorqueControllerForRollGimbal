#pragma once

// 内部：单个 RK4 阶段的正演中间量。
//
// 导数阶段对每个阶段只求值一次本结构（链、质量矩阵、偏置力、加速度），
// 状态 Jacobian、输入 Jacobian 与参数 Jacobian 全部由它派生，
// 不再对每个求导方向重算整条链、质量矩阵与偏置力。

#include "gimbal_dynamics/dynamics_model.hpp"
#include "gimbal_dynamics/math_types.hpp"
#include "gimbal_dynamics/simulation.hpp"
#include "model_evaluation.hpp"

namespace gimbal {
namespace detail {

// 只保留导数路径实际读取的字段：前向阶段的解与偏置力中间量；
// 质量矩阵、偏置力与右端项按局部变量参与求解，不再保存。
struct StageData {
    BaseState base;
    math::Vector4d angle;
    math::Vector4d qdot;
    Chain<Real> chain;
    MassInternals<Real> mass_internals;
    BiasInternals<Real> bias_internals;
    math::Cholesky4 factor;
    math::Vector4d acceleration;
};

// 由阶段状态与力矩求值全部正演中间量（就地写入，避免大对象拷贝）
void evaluate_stage_data(const ScalarModelParameters<Real>& model,
                         const TerrainFrame<Real>& frame, const BaseState& base,
                         const simulation::State& state, const math::Vector4d& torque,
                         StageData& out);

}  // namespace detail
}  // namespace gimbal
