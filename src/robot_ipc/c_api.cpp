#include "robot_ipc/robot_ipc.h"

#include <chrono>
#include <cmath>
#include <exception>
#include <string>

#include "robot_client.hpp"

namespace {

thread_local std::string g_last_error;

void set_error(const std::string& message) { g_last_error = message; }

bool is_finite(double value) { return std::isfinite(value) == 1; }

bool all_finite(const double* values, int count) {
    for (int index = 0; index < count; ++index) {
        if (!is_finite(values[index])) {
            return false;
        }
    }
    return true;
}

}  // namespace

struct TcrgRobot {
    TcrgRobot(const std::string& name, double timeout)
        : client(name), sample_timeout(timeout) {}

    tcrg::robot::RobotClient client;
    tcrg::robot::PoseTracker tracker;
    double sample_timeout = 0.5;
    bool has_update = false;
    double last_server_time = 0.0;
};

extern "C" {

TcrgRobot* tcrg_robot_ipc_create(const char* shm_name, double sample_timeout) {
    if (shm_name == nullptr) {
        set_error("共享内存名字为空");
        return nullptr;
    }
    if (!(sample_timeout > 0.0)) {
        set_error("数据超时阈值必须为正数");
        return nullptr;
    }
    try {
        return new TcrgRobot(shm_name, sample_timeout);
    } catch (const std::exception& error) {
        set_error(std::string("创建实车句柄失败：") + error.what());
        return nullptr;
    }
}

void tcrg_robot_ipc_destroy(TcrgRobot* robot) { delete robot; }

void tcrg_robot_ipc_reset(TcrgRobot* robot) {
    if (robot == nullptr) {
        return;
    }
    robot->tracker.reset();
    robot->has_update = false;
    robot->last_server_time = 0.0;
}

int tcrg_robot_ipc_update(TcrgRobot* robot, TcrgRobotObservation* observation) {
    if (robot == nullptr || observation == nullptr) {
        set_error("参数为空");
        return 1;
    }
    tcrg::robot::RobotSnapshot snapshot;
    if (!robot->client.read(snapshot)) {
        set_error("IPC 读取失败：串口通信服务端未运行或尚未发布快照");
        return 2;
    }
    const double now = snapshot.server_time;
    if (!(now > 0.0)) {
        set_error("服务端快照时间戳无效");
        return 3;
    }
    // steady_clock 在 Linux 上是全系统一致的 CLOCK_MONOTONIC：快照时间戳落后于当前时刻
    // 说明服务端已经不再发布，避免在冻结的数据上继续控制
    const double current_time =
        std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
            .count();
    if (current_time - now > robot->sample_timeout) {
        set_error("服务端快照超时：串口通信服务端可能已停止发布");
        return 4;
    }
    if (!(snapshot.mcu_receive_time > 0.0) ||
        now - snapshot.mcu_receive_time > robot->sample_timeout) {
        set_error("电控数据超时：未收到电控数据或最近一帧超过允许延迟");
        return 5;
    }
    if (!(snapshot.imu_receive_time > 0.0) ||
        now - snapshot.imu_receive_time > robot->sample_timeout) {
        set_error("IMU 数据超时：未收到 IMU 数据或最近一帧超过允许延迟");
        return 6;
    }
    if (!all_finite(snapshot.joints.angle, 4) ||
        !all_finite(snapshot.joints.velocity, 4) || !all_finite(snapshot.torque, 4)) {
        set_error("电控数据包含非有限值");
        return 7;
    }
    if (!is_finite(snapshot.imu.euler_yaw) || !is_finite(snapshot.imu.euler_pitch) ||
        !is_finite(snapshot.imu.euler_roll) || !is_finite(snapshot.imu.gx) ||
        !is_finite(snapshot.imu.gy) || !is_finite(snapshot.imu.gz)) {
        set_error("IMU 数据包含非有限值");
        return 8;
    }

    const double elapsed = robot->has_update ? now - robot->last_server_time : 0.0;
    const tcrg::robot::ChassisPose pose =
        robot->tracker.update(snapshot.imu, snapshot.joints, elapsed);

    for (int joint = 0; joint < 4; ++joint) {
        observation->angle[joint] = snapshot.joints.angle[joint];
        observation->velocity[joint] = snapshot.joints.velocity[joint];
        observation->torque[joint] = snapshot.torque[joint];
    }
    observation->slope_angle = pose.slope_angle;
    observation->slope_direction = pose.slope_direction;
    observation->base_yaw = pose.yaw;
    observation->base_yaw_rate = pose.yaw_rate;
    observation->base_yaw_acceleration = pose.yaw_acceleration;
    observation->base_acceleration[0] = 0.0;
    observation->base_acceleration[1] = 0.0;
    observation->base_acceleration[2] = 0.0;
    observation->server_time = now;
    observation->mcu_age = now - snapshot.mcu_receive_time;
    observation->imu_age = now - snapshot.imu_receive_time;

    robot->has_update = true;
    robot->last_server_time = now;
    return 0;
}

int tcrg_robot_ipc_write_torque(TcrgRobot* robot, const double torque[4]) {
    if (robot == nullptr || torque == nullptr) {
        set_error("参数为空");
        return 1;
    }
    if (!all_finite(torque, 4)) {
        set_error("力矩包含非有限值");
        return 2;
    }
    if (!robot->client.write_torque(torque)) {
        set_error("IPC 写入失败：串口通信服务端未运行");
        return 3;
    }
    return 0;
}

const char* tcrg_robot_ipc_last_error(void) { return g_last_error.c_str(); }

}  // extern "C"
