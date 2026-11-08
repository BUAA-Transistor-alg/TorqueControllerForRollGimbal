#pragma once

// 内部：单个 RK4 阶段的正演中间量。
//
// 导数阶段对每个阶段只求值一次本结构（链、质量矩阵、偏置力、加速度），
// 状态 Jacobian、输入 Jacobian 与参数 Jacobian 全部由它派生，
// 不再对每个求导方向重算整条链、质量矩阵与偏置力。

#include <Eigen/Core>

#include "gimbal_dynamics/dynamics_model.hpp"
#include "gimbal_dynamics/simulation.hpp"
#include "model_evaluation.hpp"

namespace gimbal {
namespace detail {

struct StageData {
    BaseState base;
    Vector4<double> angle;
    Vector4<double> qdot;
    Vector4<double> torque;
    Chain<double> chain;
    MassInternals<double> mass_internals;
    BiasInternals<double> bias_internals;
    Matrix4<double> mass;
    Vector4<double> bias;
    Vector4<double> right_hand_side;
    Vector4<double> acceleration;
};

// 由阶段状态与力矩求值全部正演中间量（就地写入，避免大对象拷贝）
void evaluate_stage_data(const ScalarModelParameters<double>& model,
                         const ScalarTerrain<double>& terrain, const BaseState& base,
                         const simulation::State& state, const Vector4<double>& torque,
                         StageData& out);

}  // namespace detail
}  // namespace gimbal
