#pragma once

// 锁定 roll 轴（原模型第 2 个关节）的三自由度动力学模型。
//
// 原四刚体串链中，B1 与 B2 在 roll 角固定为 locked_angle 后合并成一个刚体；
// 剩余广义坐标为 (θ1, θ3, θ4)，顺序为大 yaw、小 yaw、pitch。
// 本模块只提供前向仿真与控制量（三个剩余力矩）梯度，不提供参数梯度。

#include <array>
#include <cstddef>
#include <vector>

#include "gimbal_dynamics/dynamics_model.hpp"
#include "gimbal_dynamics/math_types.hpp"
#include "gimbal_dynamics/simulation.hpp"

namespace tcrg::gimbal_dynamics {
namespace roll_locked {

using Vector3 = math::Vector<Real, 3>;
using Matrix3 = math::Matrix<Real, 3, 3>;

// 合并后的三自由度模型参数。
// bodies[0] 为 B1 与 B2 的合并刚体（坐标系取 B1），bodies[1] 为 B3，bodies[2] 为 B4。
// joint_offsets 依次为 o1（B0 中）、o3' = o2 + F o3（B1 中）、o4（B3 中）。
// fixed_rotations[i] 是第 i 个关节局部旋转之前的固定旋转，依次为 I、F = Ry(locked_angle)、I。
struct ModelT {
    std::array<BodyParameters, 3> bodies{};
    std::array<Vector3, 3> joint_offsets{
        Vector3::zero(), Vector3::zero(), Vector3::zero()};
    Vector3 damping = Vector3::zero();
    Real gravity = Real(9.81);
    std::array<Matrix3, 3> fixed_rotations{
        Matrix3::identity(), Matrix3::identity(), Matrix3::identity()};
    Real locked_angle = Real(0);
};

// 由原四自由度参数与固定 roll 角计算合并后的三自由度参数。
// 原参数不合法、locked_angle 非有限或合并后的惯量不是正定时抛出 std::invalid_argument。
ModelT merge_model(const ModelParameters& original, Real locked_angle);

// 三自由度状态：三个关节角与三个关节角速度。
struct StateT {
    Vector3 angle = Vector3::zero();
    Vector3 velocity = Vector3::zero();
};

// 控制量梯度使用的轨迹损失。权重、限位与目标均为三个剩余关节，
// 指针为空时按零目标和零阻尼力矩处理。
struct LossSpecT {
    Vector3 angle_weights = Vector3::zero();
    Vector3 velocity_weights = Vector3::zero();
    Vector3 torque_weights = Vector3::zero();
    Vector3 torque_rate_weights = Vector3::zero();
    Vector3 limit_weights = Vector3::zero();
    Vector3 limit_lower = Vector3::zero();
    Vector3 limit_upper = Vector3::zero();
    Vector3 limit_margin = Vector3::zero();
    const Vector3* target_angle = nullptr;     // num_steps
    const Vector3* target_velocity = nullptr;  // num_steps
    const Vector3* previous_torque = nullptr;  // 长度为 1；为空时第 0 步跳过变化率项
};

struct ForwardInputT {
    ModelParameters model;
    Real locked_angle = Real(0);
    Terrain terrain;
    StateT initial;
    const BaseState* base = nullptr;          // num_steps
    const Vector3* torque = nullptr;          // num_steps
    std::size_t num_steps = 0;
    simulation::TrajectorySpecT<Real> spec;
    LossSpecT loss;
};

struct ForwardOutputT {
    Real loss = Real(0);
    std::vector<StateT> step_end;              // num_steps
    std::vector<StateT> stage_state;           // num_steps * refinement * 4
    std::vector<BaseState> stage_base;         // num_steps * refinement * 4
};

using GradientInputT = ForwardInputT;

struct GradientOutputT {
    Real loss = Real(0);
    StateT final_state;
    std::vector<Vector3> gradient;             // num_steps
};

// 单步前向动力学：q̈ = M^{-1}(τ - D q̇ - b)。
Vector3 forward_dynamics(const ModelT& model, const Terrain& terrain, const BaseState& base,
                         const Vector3& angle, const Vector3& velocity, const Vector3& torque);

// 仅前向仿真；同时记录每个主步的损失。
Real simulate(const ForwardInputT& input, ForwardOutputT& output);

// 前向仿真并计算损失对 num_steps 个三自由度力矩的解析梯度（反向伴随）。
Real simulate_and_gradient(const GradientInputT& input, GradientOutputT& output);

}  // namespace roll_locked
}  // namespace tcrg::gimbal_dynamics
