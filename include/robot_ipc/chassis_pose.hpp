#pragma once

// 由头部 IMU 与四个关节的测量反解底盘姿态、坡面参数与角速度。
//
// 链条与 DYNAMICS.md 一致：B0 为底盘，B1..B4 依次为大 yaw、roll、小 yaw、pitch 刚体：
//   R_4 = R_0 Rz(q1) Ry(q2) Rz(q3) Rx(q4)
//   R_0 = R_G(λ, γ) Rz(ψ)，R_G(λ, γ) = Rz(γ) Ry(λ) Rz(-γ)，n = R_G e_z
// IMU 固定在 pitch 节点（B4）上，欧拉角为 B4 的世界系姿态，ZXY 约定：
//   R_imu = Rz(yaw) Rx(pitch) Ry(roll)
// 陀螺为 B4 本体系的角速度。IMU 各轴与 B4 本体各轴同向，四个关节的零位与正方向与
// 动力学模型一致。以上坐标定义与本工程的动力学模型一致。
//
// 底盘角速度只有绕地面法向的分量：把 ω_4 减去各关节轴的世界方向乘关节角速度得到 ω_0，
// 再投影到法向得到 ψ̇。底盘参考点的线加速度由导航数据换算，不在此处处理。

#include "gimbal_dynamics/math_types.hpp"

namespace tcrg {
namespace robot {

namespace math = tcrg::gimbal_dynamics::math;

// 头部 IMU 的一帧测量
struct ImuMeasurement {
    double euler_yaw = 0.0;
    double euler_pitch = 0.0;
    double euler_roll = 0.0;
    double gx = 0.0;
    double gy = 0.0;
    double gz = 0.0;
};

// 四个关节的测量，顺序与链条一致：大 yaw、roll、小 yaw、pitch
struct JointMeasurement {
    double angle[4] = {0.0, 0.0, 0.0, 0.0};
    double velocity[4] = {0.0, 0.0, 0.0, 0.0};
};

// 底盘位姿与坡面参数：R_0 = R_G(λ, γ) Rz(ψ)
struct ChassisPose {
    double slope_angle = 0.0;       // λ，地面法向与世界 z 的夹角，取值 [0, π]
    double slope_direction = 0.0;   // γ，坡向，atan2(n_y, n_x)
    double yaw = 0.0;               // ψ，绕地面法向的方位角
    double yaw_rate = 0.0;          // ψ̇，绕地面法向的角速度
    double yaw_acceleration = 0.0;  // ψ̈，相邻两帧角速度差分得到
};

// ZXY 欧拉角与旋转矩阵互转：R = Rz(yaw) Rx(pitch) Ry(roll)
math::Matrix3d euler_zxy_to_rotation(double yaw, double pitch, double roll);
void rotation_to_euler_zxy(const math::Matrix3d& rotation, double& yaw, double& pitch, double& roll);

// 单帧反解：yaw 落在 (-π, π]，yaw_acceleration 置零
ChassisPose solve_chassis_pose(const ImuMeasurement& imu, const JointMeasurement& joints);

// 连续反解：yaw 解卷绕成多圈连续量，yaw_acceleration 由相邻两帧角速度与 elapsed 差分
class PoseTracker {
public:
    void reset();

    // elapsed 为与上一次 update 的实际间隔（秒），首次调用忽略
    ChassisPose update(const ImuMeasurement& imu, const JointMeasurement& joints, double elapsed);

private:
    bool initialized_ = false;
    double yaw_turn_ = 0.0;      // 累计圈数偏移（2π 的整数倍）
    double last_yaw_ = 0.0;      // 上一帧输出的多圈连续 yaw
    double last_yaw_rate_ = 0.0; // 上一帧的 yaw 角速度
};

}  // namespace robot
}  // namespace tcrg
