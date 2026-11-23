#include "robot_ipc/chassis_pose.hpp"

#include <cmath>

namespace tcrg {
namespace robot {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;

math::Matrix3d rotation_x(double angle) {
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    math::Matrix3d matrix = math::Matrix3d::identity();
    matrix(1, 1) = cosine;
    matrix(1, 2) = -sine;
    matrix(2, 1) = sine;
    matrix(2, 2) = cosine;
    return matrix;
}

math::Matrix3d rotation_y(double angle) {
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    math::Matrix3d matrix = math::Matrix3d::identity();
    matrix(0, 0) = cosine;
    matrix(0, 2) = sine;
    matrix(2, 0) = -sine;
    matrix(2, 2) = cosine;
    return matrix;
}

math::Matrix3d rotation_z(double angle) {
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    math::Matrix3d matrix = math::Matrix3d::identity();
    matrix(0, 0) = cosine;
    matrix(0, 1) = -sine;
    matrix(1, 0) = sine;
    matrix(1, 1) = cosine;
    return matrix;
}

math::Vector3d column(const math::Matrix3d& matrix, int index) {
    return math::Vector3d::from(matrix(0, index), matrix(1, index), matrix(2, index));
}

double clamp(double value, double low, double high) {
    return value < low ? low : (value > high ? high : value);
}

}  // namespace

math::Matrix3d euler_zxy_to_rotation(double yaw, double pitch, double roll) {
    return rotation_z(yaw) * rotation_x(pitch) * rotation_y(roll);
}

void rotation_to_euler_zxy(const math::Matrix3d& rotation, double& yaw, double& pitch,
                           double& roll) {
    pitch = std::asin(clamp(rotation(2, 1), -1.0, 1.0));
    const double cosine = std::cos(pitch);
    if (std::fabs(cosine) > 1.0e-6) {
        yaw = std::atan2(-rotation(0, 1), rotation(1, 1));
        roll = std::atan2(-rotation(2, 0), rotation(2, 2));
    } else {
        // 万向节锁：pitch = ±π/2 时 yaw 与 roll 只有组合可观测，取 roll = 0
        roll = 0.0;
        yaw = std::atan2(rotation(1, 0), rotation(0, 0));
    }
}

ChassisPose solve_chassis_pose(const ImuMeasurement& imu, const JointMeasurement& joints) {
    const math::Matrix3d rotation_imu =
        euler_zxy_to_rotation(imu.euler_yaw, imu.euler_pitch, imu.euler_roll);
    // R_0 = R_4 Rx(-q4) Rz(-q3) Ry(-q2) Rz(-q1)
    const math::Matrix3d chassis = rotation_imu * rotation_x(-joints.angle[3]) *
                                   rotation_z(-joints.angle[2]) * rotation_y(-joints.angle[1]) *
                                   rotation_z(-joints.angle[0]);

    ChassisPose pose;
    const math::Vector3d normal = column(chassis, 2);
    // 用 atan2 求坡角：acos 在法向接近竖直时会把舍入误差放大，atan2 在 λ 接近 0 与 π 时都稳定
    pose.slope_angle =
        std::atan2(std::sqrt(normal[0] * normal[0] + normal[1] * normal[1]), normal[2]);
    pose.slope_direction = std::atan2(normal[1], normal[0]);
    const math::Matrix3d ground = rotation_z(pose.slope_direction) *
                                  rotation_y(pose.slope_angle) *
                                  rotation_z(-pose.slope_direction);
    const math::Matrix3d planar = math::transpose(ground) * chassis;
    pose.yaw = std::atan2(planar(1, 0), planar(0, 0));

    // 头部角速度转到世界系后减去各关节轴的世界方向乘关节角速度
    const math::Vector3d head_omega =
        rotation_imu * math::Vector3d::from(imu.gx, imu.gy, imu.gz);
    const math::Matrix3d rotation1 = chassis * rotation_z(joints.angle[0]);
    const math::Matrix3d rotation2 = rotation1 * rotation_y(joints.angle[1]);
    const math::Matrix3d rotation3 = rotation2 * rotation_z(joints.angle[2]);
    const math::Vector3d joint_omega =
        normal * joints.velocity[0] + column(rotation1, 1) * joints.velocity[1] +
        column(rotation2, 2) * joints.velocity[2] + column(rotation3, 0) * joints.velocity[3];
    pose.yaw_rate = math::dot(head_omega - joint_omega, normal);
    pose.yaw_acceleration = 0.0;
    return pose;
}

void PoseTracker::reset() {
    initialized_ = false;
    yaw_turn_ = 0.0;
    last_yaw_ = 0.0;
    last_yaw_rate_ = 0.0;
}

ChassisPose PoseTracker::update(const ImuMeasurement& imu, const JointMeasurement& joints,
                                double elapsed) {
    ChassisPose pose = solve_chassis_pose(imu, joints);
    if (!initialized_) {
        initialized_ = true;
        last_yaw_ = pose.yaw;
        last_yaw_rate_ = pose.yaw_rate;
        pose.yaw_acceleration = 0.0;
        return pose;
    }
    // 以上一帧输出为参考把整圈跳变吸收进 yaw_turn_，输出连续的多圈 yaw
    const double increment = (pose.yaw + yaw_turn_) - last_yaw_;
    if (increment > kPi) {
        yaw_turn_ -= kTwoPi;
    } else if (increment < -kPi) {
        yaw_turn_ += kTwoPi;
    }
    pose.yaw += yaw_turn_;
    pose.yaw_acceleration =
        elapsed > 0.0 ? (pose.yaw_rate - last_yaw_rate_) / elapsed : 0.0;
    last_yaw_ = pose.yaw;
    last_yaw_rate_ = pose.yaw_rate;
    return pose;
}

}  // namespace robot
}  // namespace tcrg
