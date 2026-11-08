#pragma once

#include <array>

#include "gimbal_dynamics/math_types.hpp"

namespace gimbal {

// 单个刚体 B1..B4 的惯性参数
struct BodyParameters {
    Real mass = 0.0;
    math::Vector3d com = math::Vector3d::zero();      // 质心相对刚体坐标系原点
    math::Matrix3d inertia = math::Matrix3d::zero();  // 绕质心，在刚体轴上
};

// 串链模型参数。B0 为底盘，运动由外部给定。
struct ModelParameters {
    std::array<BodyParameters, 4> bodies{};
    std::array<math::Vector3d, 4> joint_offsets = {math::Vector3d::zero(),
                                                   math::Vector3d::zero(),
                                                   math::Vector3d::zero(),
                                                   math::Vector3d::zero()};  // o_1..o_4
    math::Vector4d damping = math::Vector4d::zero();  // d_1..d_4
    Real gravity = 9.81;

    // 示例参数，实际使用时替换为实测值
    static ModelParameters demo();
};

// 地面坡角 λ 与坡向 γ
struct Terrain {
    Real slope_angle = 0.0;
    Real slope_direction = 0.0;
};

// 底盘运动输入。底盘位置与线速度不进入关节动力学。
struct BaseState {
    math::Vector3d acceleration = math::Vector3d::zero();  // 参考点线加速度 a0
    Real yaw = 0.0;                                      // ψ
    Real yaw_rate = 0.0;                                 // ψ̇
    Real yaw_acceleration = 0.0;                         // ψ̈
};

// 正向动力学：M(q) q̈ + D q̇ + b = τ，返回 q̈
math::Vector4d forward_dynamics(const ModelParameters& model, const Terrain& terrain,
                                const BaseState& base, const math::Vector4d& q,
                                const math::Vector4d& qdot, const math::Vector4d& torque);

}  // namespace gimbal
