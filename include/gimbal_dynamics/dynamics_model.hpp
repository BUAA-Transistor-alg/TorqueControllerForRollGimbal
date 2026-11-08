#pragma once

#include <array>

#include <Eigen/Core>

namespace gimbal {

// 单个刚体 B1..B4 的惯性参数
struct BodyParameters {
    double mass = 0.0;
    Eigen::Vector3d com = Eigen::Vector3d::Zero();      // 质心相对刚体坐标系原点
    Eigen::Matrix3d inertia = Eigen::Matrix3d::Zero();  // 绕质心，在刚体轴上
};

// 串链模型参数。B0 为底盘，运动由外部给定。
struct ModelParameters {
    std::array<BodyParameters, 4> bodies{};
    std::array<Eigen::Vector3d, 4> joint_offsets = {Eigen::Vector3d::Zero(),
                                                    Eigen::Vector3d::Zero(),
                                                    Eigen::Vector3d::Zero(),
                                                    Eigen::Vector3d::Zero()};  // o_1..o_4
    Eigen::Vector4d damping = Eigen::Vector4d::Zero();  // d_1..d_4
    double gravity = 9.81;

    // 示例参数，实际使用时替换为实测值
    static ModelParameters demo();
};

// 地面坡角 λ 与坡向 γ
struct Terrain {
    double slope_angle = 0.0;
    double slope_direction = 0.0;
};

// 底盘运动输入。底盘位置与线速度不进入关节动力学。
struct BaseState {
    Eigen::Vector3d acceleration = Eigen::Vector3d::Zero();  // 参考点线加速度 a0
    double yaw = 0.0;                                        // ψ
    double yaw_rate = 0.0;                                   // ψ̇
    double yaw_acceleration = 0.0;                           // ψ̈
};

// 正向动力学：M(q) q̈ + D q̇ + b = τ，返回 q̈
Eigen::Vector4d forward_dynamics(const ModelParameters& model, const Terrain& terrain,
                                 const BaseState& base, const Eigen::Vector4d& q,
                                 const Eigen::Vector4d& qdot,
                                 const Eigen::Vector4d& torque);

}  // namespace gimbal
