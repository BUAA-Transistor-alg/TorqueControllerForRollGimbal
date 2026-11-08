#pragma once

// 内部：阶段 Jacobian 的求值接口与整子步组装。
//
// 每个 RK4 阶段只求值一次 StageData（链、质量矩阵、偏置力、加速度），
// 状态 Jacobian、输入 Jacobian 与参数 Jacobian 全部由它派生：
//   * 链的方向导数用解析公式（旋转导数、偏置/原点/质心平移），不需要三角函数；
//   * 偏置力的方向导数按方向类型分别传播（q、q̇、质心、关节偏置各自走短路径）；
//   * 质量与惯量参数的方向导数有闭式表达，不占方向；
//   * 最后统一用 M^{-1}(∂r − ∂M q̈) 组装成 ∂q̈。
// 这套实现与逐列复步长求值数学等价，梯度结果与逐列复步长的差异在 1e-14 以内。

#include "gimbal_dynamics/dynamics_model.hpp"
#include "gimbal_dynamics/math_types.hpp"
#include "gimbal_dynamics/parameters.hpp"
#include "model_evaluation.hpp"
#include "stage_data.hpp"

namespace gimbal {
namespace detail {

// 单阶段的状态 Jacobian ∂f/∂y 与输入 Jacobian ∂f/∂τ，y = [q; q̇]
struct StageJacobians {
    math::Matrix8d state = math::Matrix8d::zero();
    math::Matrix84d input = math::Matrix84d::zero();
};

StageJacobians evaluate_stage_jacobians(const StageData& stage,
                                        const ScalarModelParameters<double>& model);

// 单阶段的参数 Jacobian ∂f/∂p（8×56），同时给出状态 Jacobian
using ParameterJacobian = math::Matrix<double, 8, parameters::kParameterCount>;
ParameterJacobian evaluate_parameter_jacobian(const StageData& stage,
                                              const ScalarModelParameters<double>& model,
                                              math::Matrix8d& state_jacobian);

// 整子步传播子 W = ∂y'/∂y 与输入 Jacobian Ψ = ∂y'/∂u（u 在四个阶段中相同）
void assemble_substep(const StageJacobians stage[4], double h, math::Matrix8d& propagation,
                      math::Matrix84d& input_jacobian);

// 整子步参数 Jacobian P = ∂y'/∂p，同时给出 W
void assemble_parameter_substep(const StageJacobians stage[4],
                                const ParameterJacobian stage_parameter[4], double h,
                                math::Matrix8d& propagation,
                                ParameterJacobian& parameter_jacobian);

}  // namespace detail
}  // namespace gimbal
