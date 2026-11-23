// 校验：chassis_pose.hpp 的 ZXY 欧拉角换算、底盘姿态反解、yaw 连续量、yaw 角速度与角加速度。
//
// IMU 输入全部由本文件手写的旋转矩阵独立构造：R_4 = R_0 Rz(q1) Ry(q2) Rz(q3) Rx(q4)，
// R_0 = Rz(gamma) Ry(lambda) Rz(-gamma) Rz(psi)，与 DYNAMICS.md 第 3、6 节一致。
#include <cmath>
#include <cstdio>
#include <random>

#include "robot_ipc/chassis_pose.hpp"

namespace {

using tcrg::gimbal_dynamics::math::Matrix3d;
using tcrg::gimbal_dynamics::math::Vector3d;
using tcrg::robot::ChassisPose;
using tcrg::robot::ImuMeasurement;
using tcrg::robot::JointMeasurement;
using tcrg::robot::PoseTracker;

constexpr double kPi = 3.14159265358979323846264338327950288;

// 手写的 3×3 旋转矩阵
struct Mat3 {
    double m[3][3];
};

Mat3 multiply(const Mat3& lhs, const Mat3& rhs) {
    Mat3 result{};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            double sum = 0.0;
            for (int k = 0; k < 3; ++k) {
                sum += lhs.m[row][k] * rhs.m[k][column];
            }
            result.m[row][column] = sum;
        }
    }
    return result;
}

Mat3 rotation_x(double angle) {
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    Mat3 result{};
    result.m[0][0] = 1.0;
    result.m[1][1] = cosine;
    result.m[1][2] = -sine;
    result.m[2][1] = sine;
    result.m[2][2] = cosine;
    return result;
}

Mat3 rotation_y(double angle) {
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    Mat3 result{};
    result.m[0][0] = cosine;
    result.m[0][2] = sine;
    result.m[1][1] = 1.0;
    result.m[2][0] = -sine;
    result.m[2][2] = cosine;
    return result;
}

Mat3 rotation_z(double angle) {
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    Mat3 result{};
    result.m[0][0] = cosine;
    result.m[0][1] = -sine;
    result.m[1][0] = sine;
    result.m[1][1] = cosine;
    result.m[2][2] = 1.0;
    return result;
}

// ZXY 欧拉角矩阵 R = Rz(yaw) Rx(pitch) Ry(roll) 的九个元素，按公式硬编码
Mat3 rotation_zxy(double yaw, double pitch, double roll) {
    const double cy = std::cos(yaw);
    const double sy = std::sin(yaw);
    const double cp = std::cos(pitch);
    const double sp = std::sin(pitch);
    const double cr = std::cos(roll);
    const double sr = std::sin(roll);
    Mat3 result{};
    result.m[0][0] = cy * cr - sy * sp * sr;
    result.m[0][1] = -sy * cp;
    result.m[0][2] = cy * sr + sy * sp * cr;
    result.m[1][0] = sy * cr + cy * sp * sr;
    result.m[1][1] = cy * cp;
    result.m[1][2] = sy * sr - cy * sp * cr;
    result.m[2][0] = -cp * sr;
    result.m[2][1] = sp;
    result.m[2][2] = cp * cr;
    return result;
}

Mat3 transpose(const Mat3& value) {
    Mat3 result{};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            result.m[row][column] = value.m[column][row];
        }
    }
    return result;
}

Vector3d apply(const Mat3& rotation, const Vector3d& vector) {
    Vector3d result = Vector3d::zero();
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            result[row] += rotation.m[row][column] * vector[column];
        }
    }
    return result;
}

Vector3d column(const Mat3& rotation, int index) {
    return Vector3d::from(rotation.m[0][index], rotation.m[1][index], rotation.m[2][index]);
}

Matrix3d to_matrix(const Mat3& value) {
    Matrix3d result = Matrix3d::zero();
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            result(row, column) = value.m[row][column];
        }
    }
    return result;
}

Mat3 to_mat3(const Matrix3d& value) {
    Mat3 result{};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            result.m[row][column] = value(row, column);
        }
    }
    return result;
}

// 包裹到 (-pi, pi]
double wrap_angle(double angle) {
    angle = std::fmod(angle, 2.0 * kPi);
    if (angle <= -kPi) {
        angle += 2.0 * kPi;
    }
    if (angle > kPi) {
        angle -= 2.0 * kPi;
    }
    return angle;
}

// 单项检查：累计最大偏差，失败时打印该项最差的期望值与实际值
struct Compare {
    const char* name;
    double tolerance;
    bool wrap = false;
    bool failed = false;
    double worst_error = 0.0;
    double expected = 0.0;
    double actual = 0.0;
    int row = -1;
    int column = -1;

    void add(double expected_value, double actual_value, int row_index = -1, int column_index = -1) {
        const double difference = expected_value - actual_value;
        const double error = wrap ? std::fabs(wrap_angle(difference)) : std::fabs(difference);
        if (!std::isfinite(error) || error > tolerance) {
            failed = true;
        }
        if (!(error <= worst_error)) {
            worst_error = error;
            expected = expected_value;
            actual = actual_value;
            row = row_index;
            column = column_index;
        }
    }

    bool report() const {
        if (!failed) {
            return true;
        }
        if (row >= 0) {
            printf("[失败] %s（第 %d 行第 %d 列）：期望 %.17g，实际 %.17g，偏差 %.6e，容差 %.1e\n", name,
                   row, column, expected, actual, worst_error, tolerance);
        } else {
            printf("[失败] %s：期望 %.17g，实际 %.17g，偏差 %.6e，容差 %.1e\n", name, expected, actual,
                   worst_error, tolerance);
        }
        return false;
    }
};

// 按 DYNAMICS.md 第 6 节构造链条
struct Chain {
    Mat3 ground;
    Mat3 chassis;
    Mat3 joint1;
    Mat3 joint2;
    Mat3 joint3;
    Mat3 head;
    Vector3d normal;
};

Chain build_chain(double slope_angle, double slope_direction, double yaw,
                  const double joint_angle[4]) {
    const Mat3 direction = rotation_z(slope_direction);
    Chain chain;
    chain.ground = multiply(multiply(direction, rotation_y(slope_angle)), rotation_z(-slope_direction));
    chain.chassis = multiply(chain.ground, rotation_z(yaw));
    chain.joint1 = multiply(chain.chassis, rotation_z(joint_angle[0]));
    chain.joint2 = multiply(chain.joint1, rotation_y(joint_angle[1]));
    chain.joint3 = multiply(chain.joint2, rotation_z(joint_angle[2]));
    chain.head = multiply(chain.joint3, rotation_x(joint_angle[3]));
    chain.normal = column(chain.ground, 2);
    return chain;
}

// 由 R_4 得到 IMU 的 ZXY 欧拉角；pitch 在 ±pi/2 附近时参数化退化，返回 false 供重抽样
bool fill_imu_angles(const Mat3& head, ImuMeasurement& imu) {
    tcrg::robot::rotation_to_euler_zxy(to_matrix(head), imu.euler_yaw, imu.euler_pitch,
                                       imu.euler_roll);
    return std::fabs(std::cos(imu.euler_pitch)) > 1.0e-3;
}

// 检查 1：euler_zxy_to_rotation 的九个元素与手写公式一致，rotation_to_euler_zxy 能转回原角
bool check_euler_conversion() {
    std::mt19937 rng(20261123u);
    std::uniform_real_distribution<double> angle_distribution(-kPi, kPi);
    std::uniform_real_distribution<double> pitch_distribution(-1.2, 1.2);

    Compare matrix{"euler_zxy_to_rotation 的元素", 1.0e-12};
    Compare back_yaw{"rotation_to_euler_zxy 的 yaw", 1.0e-12, true};
    Compare back_pitch{"rotation_to_euler_zxy 的 pitch", 1.0e-12};
    Compare back_roll{"rotation_to_euler_zxy 的 roll", 1.0e-12, true};

    for (int trial = 0; trial < 2000; ++trial) {
        const double yaw = angle_distribution(rng);
        const double pitch = pitch_distribution(rng);
        const double roll = angle_distribution(rng);
        const Mat3 expected = rotation_zxy(yaw, pitch, roll);
        const Matrix3d actual_matrix = tcrg::robot::euler_zxy_to_rotation(yaw, pitch, roll);
        const Mat3 actual = to_mat3(actual_matrix);
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                matrix.add(expected.m[row][column], actual.m[row][column], row, column);
            }
        }
        double round_yaw = 0.0;
        double round_pitch = 0.0;
        double round_roll = 0.0;
        tcrg::robot::rotation_to_euler_zxy(actual_matrix, round_yaw, round_pitch, round_roll);
        back_yaw.add(yaw, round_yaw);
        back_pitch.add(pitch, round_pitch);
        back_roll.add(roll, round_roll);
    }

    bool ok = matrix.report();
    ok = back_yaw.report() && ok;
    ok = back_pitch.report() && ok;
    ok = back_roll.report() && ok;
    return ok;
}

// 检查 2：由手写旋转构造 R_4 与 IMU 欧拉角，反解 lambda、gamma、psi
bool check_chain_pose() {
    std::mt19937 rng(20261124u);
    std::uniform_real_distribution<double> slope_angle_distribution(0.05, 0.6);
    std::uniform_real_distribution<double> angle_distribution(-kPi, kPi);
    std::uniform_real_distribution<double> joint1_distribution(-kPi, kPi);
    std::uniform_real_distribution<double> joint2_distribution(-0.78, 0.08);
    std::uniform_real_distribution<double> joint3_distribution(-0.87, 0.87);
    std::uniform_real_distribution<double> joint4_distribution(-0.78, 0.78);

    Compare slope_angle{"solve_chassis_pose 的 slope_angle", 1.0e-9};
    Compare slope_direction{"solve_chassis_pose 的 slope_direction", 1.0e-9, true};
    Compare yaw{"solve_chassis_pose 的 yaw", 1.0e-9, true};

    int accepted = 0;
    while (accepted < 300) {
        const double lambda = slope_angle_distribution(rng);
        const double gamma = angle_distribution(rng);
        const double psi = angle_distribution(rng);
        double joint_angle[4];
        joint_angle[0] = joint1_distribution(rng);
        joint_angle[1] = joint2_distribution(rng);
        joint_angle[2] = joint3_distribution(rng);
        joint_angle[3] = joint4_distribution(rng);

        const Chain chain = build_chain(lambda, gamma, psi, joint_angle);
        ImuMeasurement imu;
        if (!fill_imu_angles(chain.head, imu)) {
            continue;
        }
        JointMeasurement joints;
        for (int joint = 0; joint < 4; ++joint) {
            joints.angle[joint] = joint_angle[joint];
        }

        const ChassisPose pose = tcrg::robot::solve_chassis_pose(imu, joints);
        slope_angle.add(lambda, pose.slope_angle);
        slope_direction.add(gamma, pose.slope_direction);
        yaw.add(psi, pose.yaw);
        ++accepted;
    }

    bool ok = slope_angle.report();
    ok = slope_direction.report() && ok;
    ok = yaw.report() && ok;
    return ok;
}

// 检查 2 的补充：lambda = 0 时 gamma 无定义，只检查 slope_angle 与 yaw
bool check_flat_slope() {
    std::mt19937 rng(20261126u);
    std::uniform_real_distribution<double> angle_distribution(-kPi, kPi);
    std::uniform_real_distribution<double> joint1_distribution(-kPi, kPi);
    std::uniform_real_distribution<double> joint2_distribution(-0.78, 0.08);
    std::uniform_real_distribution<double> joint3_distribution(-0.87, 0.87);
    std::uniform_real_distribution<double> joint4_distribution(-0.78, 0.78);

    Compare slope_angle{"lambda = 0 时 solve_chassis_pose 的 slope_angle", 1.0e-9};
    Compare yaw{"lambda = 0 时 solve_chassis_pose 的 yaw", 1.0e-9, true};

    int accepted = 0;
    while (accepted < 200) {
        const double psi = angle_distribution(rng);
        double joint_angle[4];
        joint_angle[0] = joint1_distribution(rng);
        joint_angle[1] = joint2_distribution(rng);
        joint_angle[2] = joint3_distribution(rng);
        joint_angle[3] = joint4_distribution(rng);

        const Chain chain = build_chain(0.0, 0.0, psi, joint_angle);
        ImuMeasurement imu;
        if (!fill_imu_angles(chain.head, imu)) {
            continue;
        }
        JointMeasurement joints;
        for (int joint = 0; joint < 4; ++joint) {
            joints.angle[joint] = joint_angle[joint];
        }

        const ChassisPose pose = tcrg::robot::solve_chassis_pose(imu, joints);
        slope_angle.add(0.0, pose.slope_angle);
        yaw.add(psi, pose.yaw);
        ++accepted;
    }

    bool ok = slope_angle.report();
    ok = yaw.report() && ok;
    return ok;
}

// 检查 3：底盘 yaw 单调跨过 ±pi 时 PoseTracker 输出连续多圈量
bool check_yaw_continuity() {
    const double joint_angle[4] = {0.3, 0.2, -0.4, 0.1};
    const double slope_angle = 0.15;
    const double slope_direction = 0.5;
    const double yaw_rate = 2.0;
    const double elapsed = 0.01;
    const double step = yaw_rate * elapsed;

    Compare yaw{"PoseTracker.update 的连续 yaw", 1.0e-9};
    Compare increment{"PoseTracker.update 的相邻帧 yaw 增量", 1.0e-9};
    Compare first_acceleration{"PoseTracker.update 首帧的 yaw_acceleration", 1.0e-15};

    PoseTracker tracker;
    tracker.reset();
    double previous = 0.0;
    bool initialized = false;
    for (int frame = 0; frame <= 325; ++frame) {
        const double true_yaw = -3.0 + step * frame;
        const Chain chain = build_chain(slope_angle, slope_direction, true_yaw, joint_angle);
        ImuMeasurement imu;
        if (!fill_imu_angles(chain.head, imu)) {
            printf("[失败] PoseTracker.update 的连续 yaw：构造 IMU 欧拉角时 pitch 落在 ±pi/2 附近\n");
            return false;
        }
        const Vector3d omega = chain.normal * yaw_rate;
        const Vector3d gyro = apply(transpose(chain.head), omega);
        imu.gx = gyro[0];
        imu.gy = gyro[1];
        imu.gz = gyro[2];

        JointMeasurement joints;
        for (int joint = 0; joint < 4; ++joint) {
            joints.angle[joint] = joint_angle[joint];
        }

        const ChassisPose pose = tracker.update(imu, joints, elapsed);
        yaw.add(true_yaw, pose.yaw);
        if (!initialized) {
            initialized = true;
            first_acceleration.add(0.0, pose.yaw_acceleration);
        } else {
            increment.add(step, pose.yaw - previous);
        }
        previous = pose.yaw;
    }

    bool ok = yaw.report();
    ok = increment.report() && ok;
    ok = first_acceleration.report() && ok;
    return ok;
}

// 检查 4：按 omega_4 公式构造 B4 本体系陀螺输入，反解出的 yaw_rate 应等于 psi_dot
bool check_yaw_rate() {
    std::mt19937 rng(20261127u);
    std::uniform_real_distribution<double> slope_angle_distribution(0.05, 0.6);
    std::uniform_real_distribution<double> angle_distribution(-kPi, kPi);
    std::uniform_real_distribution<double> joint1_distribution(-kPi, kPi);
    std::uniform_real_distribution<double> joint2_distribution(-0.78, 0.08);
    std::uniform_real_distribution<double> joint3_distribution(-0.87, 0.87);
    std::uniform_real_distribution<double> joint4_distribution(-0.78, 0.78);
    std::uniform_real_distribution<double> rate_distribution(-3.0, 3.0);

    Compare yaw_rate{"solve_chassis_pose 的 yaw_rate", 1.0e-9};

    int accepted = 0;
    while (accepted < 300) {
        const double lambda = slope_angle_distribution(rng);
        const double gamma = angle_distribution(rng);
        const double psi = angle_distribution(rng);
        const double psi_dot = rate_distribution(rng);
        double joint_angle[4];
        double joint_velocity[4];
        joint_angle[0] = joint1_distribution(rng);
        joint_angle[1] = joint2_distribution(rng);
        joint_angle[2] = joint3_distribution(rng);
        joint_angle[3] = joint4_distribution(rng);
        for (int joint = 0; joint < 4; ++joint) {
            joint_velocity[joint] = rate_distribution(rng);
        }

        const Chain chain = build_chain(lambda, gamma, psi, joint_angle);
        ImuMeasurement imu;
        if (!fill_imu_angles(chain.head, imu)) {
            continue;
        }

        Vector3d omega = chain.normal * psi_dot;
        omega += chain.normal * joint_velocity[0];
        omega += column(chain.joint1, 1) * joint_velocity[1];
        omega += column(chain.joint2, 2) * joint_velocity[2];
        omega += column(chain.joint3, 0) * joint_velocity[3];
        const Vector3d gyro = apply(transpose(chain.head), omega);
        imu.gx = gyro[0];
        imu.gy = gyro[1];
        imu.gz = gyro[2];

        JointMeasurement joints;
        for (int joint = 0; joint < 4; ++joint) {
            joints.angle[joint] = joint_angle[joint];
            joints.velocity[joint] = joint_velocity[joint];
        }

        const ChassisPose pose = tcrg::robot::solve_chassis_pose(imu, joints);
        yaw_rate.add(psi_dot, pose.yaw_rate);
        ++accepted;
    }

    return yaw_rate.report();
}

// 检查 5：psi_dot 在两次 update 之间线性变化时，yaw_acceleration 应等于 psi_ddot
bool check_yaw_acceleration() {
    const double joint_angle[4] = {0.2, 0.15, -0.3, 0.1};
    const double slope_angle = 0.25;
    const double slope_direction = 0.8;
    const double yaw_acceleration = 2.5;
    const double start_time = 0.4;
    const double elapsed = 0.01;
    const double start_yaw = 0.2;

    auto make_imu = [&](double time, ImuMeasurement& imu) {
        const double yaw = start_yaw + 0.5 * yaw_acceleration * time * time;
        const double yaw_rate = yaw_acceleration * time;
        const Chain chain = build_chain(slope_angle, slope_direction, yaw, joint_angle);
        if (!fill_imu_angles(chain.head, imu)) {
            return false;
        }
        const Vector3d omega = chain.normal * yaw_rate;
        const Vector3d gyro = apply(transpose(chain.head), omega);
        imu.gx = gyro[0];
        imu.gy = gyro[1];
        imu.gz = gyro[2];
        return true;
    };

    ImuMeasurement first_imu;
    ImuMeasurement second_imu;
    if (!make_imu(start_time, first_imu) || !make_imu(start_time + elapsed, second_imu)) {
        printf("[失败] PoseTracker.update 的 yaw_acceleration：构造 IMU 欧拉角时 pitch 落在 ±pi/2 附近\n");
        return false;
    }

    JointMeasurement joints;
    for (int joint = 0; joint < 4; ++joint) {
        joints.angle[joint] = joint_angle[joint];
    }

    PoseTracker tracker;
    tracker.reset();
    tracker.update(first_imu, joints, 0.0);
    const ChassisPose pose = tracker.update(second_imu, joints, elapsed);

    const double error =
        std::fabs(pose.yaw_acceleration - yaw_acceleration) / std::fabs(yaw_acceleration);
    if (!std::isfinite(error) || error >= 1.0e-6) {
        printf("[失败] PoseTracker.update 的 yaw_acceleration：期望 %.17g，实际 %.17g，相对偏差 %.6e，"
               "容差 1e-06\n",
               yaw_acceleration, pose.yaw_acceleration, error);
        return false;
    }
    return true;
}

}  // namespace

int main() {
    bool ok = true;
    if (!check_euler_conversion()) {
        ok = false;
    }
    if (!check_chain_pose()) {
        ok = false;
    }
    if (!check_flat_slope()) {
        ok = false;
    }
    if (!check_yaw_continuity()) {
        ok = false;
    }
    if (!check_yaw_rate()) {
        ok = false;
    }
    if (!check_yaw_acceleration()) {
        ok = false;
    }
    if (!ok) {
        printf("校验失败\n");
        return 1;
    }
    printf("校验通过\n");
    return 0;
}
