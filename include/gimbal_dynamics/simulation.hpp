#pragma once

// 定步长前向仿真核心：4 个关节的 8 维状态、RK4 细化积分、标量轨迹损失。
//
// 对控制量求梯度的模块（gimbal::control_gradient）与对模型参数求灵敏度的模块
// （gimbal::parameter_gradient）都调用本文件的 simulate() 完成前向计算，
// 因此两者的前向轨迹、阶段状态与损失逐位一致。

#include <cstddef>
#include <vector>

#include "gimbal_dynamics/dynamics_model.hpp"
#include "gimbal_dynamics/math_types.hpp"

namespace gimbal {
namespace simulation {

// 8 维状态：4 个关节角与 4 个关节角速度
struct State {
    math::Vector4d angle = math::Vector4d::zero();
    math::Vector4d velocity = math::Vector4d::zero();
};

// 标量轨迹损失（时间均值形式，与参考工程一致）：
//   L = 1/(2K) Σ_k [ Σ_i wa_i (q_i - q*_i)^2 + Σ_i wv_i (v_i - v*_i)^2
//                    + Σ_i wt_i τ_{k,i}^2 ]
// 角与角速度取每个主步末的状态；目标序列为 nullptr 时目标恒为 0。
struct LossSpec {
    math::Vector4d angle_weights = math::Vector4d::zero();
    math::Vector4d velocity_weights = math::Vector4d::zero();
    math::Vector4d torque_weights = math::Vector4d::zero();
    const math::Vector4d* target_angle = nullptr;     // 长度 num_steps
    const math::Vector4d* target_velocity = nullptr;  // 长度 num_steps
};

// 定步长设置：每个主步内做 refinement 次经典 RK4
struct TrajectorySpec {
    Real step = 0.0;
    int refinement = 1;
};

// 前向轨迹记录：每个主步末状态，以及每个 RK4 子步四个阶段的状态与底盘量
struct Trace {
    std::size_t num_steps = 0;
    int refinement = 0;
    std::vector<State> step_end;        // 长度 K
    std::vector<State> stage_state;     // 长度 K*refinement*4
    std::vector<BaseState> stage_base;  // 长度 K*refinement*4

    void resize(std::size_t steps, int refine);
    void release();
};

// 配置校验。step <= 0 或 refinement < 1 时抛出 std::invalid_argument。
void validate(const TrajectorySpec& spec, std::size_t num_steps);

// 单步状态导数 y' = f(y, u)，y = [q; q̇]，u = τ。用于与 forward_dynamics 对照。
math::Vector8d state_derivative(const ModelParameters& model, const Terrain& terrain,
                                const BaseState& base, const State& state,
                                const math::Vector4d& torque);

// 单步损失（已含 1/K 缩放）
Real step_loss(const LossSpec& loss, std::size_t num_steps, std::size_t step,
                 const State& state, const math::Vector4d& torque);

// 单步损失对状态的梯度，顺序为 4 个关节角、4 个关节角速度
math::Vector8d step_loss_gradient(const LossSpec& loss, std::size_t num_steps,
                                  std::size_t step, const State& state);

// 前向积分。base 与 torque 的长度均为 num_steps；返回总损失并写满 trace。
// 底盘量在第 k 个主步内按等角加速度外推：
//   ψ(τ) = ψ_k + ψ̇_k τ + 1/2 ψ̈_k τ²，ψ̇(τ) = ψ̇_k + ψ̈_k τ，a0 保持不变。
Real simulate(const ModelParameters& model, const Terrain& terrain, const State& initial,
                const BaseState* base, const math::Vector4d* torque, std::size_t num_steps,
                const TrajectorySpec& spec, const LossSpec& loss, Trace& trace);

}  // namespace simulation
}  // namespace gimbal
