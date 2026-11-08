#pragma once

// 对控制量求梯度的仿真：每步 4 个关节电机力矩 τ 是决策量，用离散伴随（反向模式）
// 求标量轨迹损失的解析梯度 dL/dτ。
//
// 与 gimbal::parameter_gradient 的关系：
//   * 本模块对 τ 求导，反向传播 8 维协态，代价与步数成线性、与决策量维数无关；
//   * 参数模块对 56 个模型参数求导，用前向灵敏度传播；
//   * 两个模块的前向计算都由 gimbal::simulation::simulate 完成，轨迹与损失逐位一致。
//
// 反向传播使用正向记录的每个 RK4 子步的四个阶段 Jacobian 组装出的整子步传播子：
//   W = ∂y'/∂y = I + h/6 (J1 + 2 J2 A2 + 2 J3 A3 + J4 A4)
//   A2 = I + h/2 J1，A3 = I + h/2 J2 A2，A4 = I + h J3 A3
//   Ψ = ∂y'/∂τ = h/6 (u1 + 2 u2 + 2 u3 + u4)，u1 = Ju1，u2 = Ju2 + h/2 Jx2 u1，依此类推
// 其中 Jx 与 Ju 为单阶段的状态 Jacobian 与输入 Jacobian，由该阶段一次求值出的
// 中间量（链、质量矩阵、偏置力）解析派生，见 src/stage_derivatives.cpp。

#include <cstddef>
#include <vector>

#include "gimbal_dynamics/math_types.hpp"
#include "gimbal_dynamics/simulation.hpp"

namespace gimbal {
namespace control_gradient {

// 调用方分配一次、可跨多次调用复用的缓冲
struct Workspace {
    simulation::Trace trace;                     // 前向轨迹记录
    std::vector<math::Matrix8d> propagation;     // 每子步 W
    std::vector<math::Matrix84d> input_jacobian; // 每子步 Ψ

    void resize(std::size_t num_steps, int refinement);
    void release();
};

// 前向仿真 + 反向伴随。返回损失值。
//   gradient     长度 num_steps，第 k 个元素为 dL/dτ_k；可为 nullptr（只做前向）
//   final_state  最后一个主步结束后的状态；可为 nullptr
// base 与 torque 的长度均为 num_steps。
double simulateAndGradient(const ModelParameters& model, const Terrain& terrain,
                           const simulation::State& initial, const BaseState* base,
                           const math::Vector4d* torque, std::size_t num_steps,
                           const simulation::TrajectorySpec& spec,
                           const simulation::LossSpec& loss, Workspace& workspace,
                           math::Vector4d* gradient, simulation::State* final_state);

}  // namespace control_gradient
}  // namespace gimbal
