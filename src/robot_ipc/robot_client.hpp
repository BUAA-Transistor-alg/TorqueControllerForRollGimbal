#pragma once

// 实车 IPC 客户端：只使用子模组 Sentry2Communicator 的 ipc_client 中的 SerialCommClient，
// 只调用 vis_get_all_info 与 vis_write，不调用导航部分。
// 本头文件依赖子模组的公开头文件，属于库内部实现。

#include <string>

#include "SerialCommClient.hpp"
#include "serial_comm/SerialCommTypes.hpp"

#include "robot_ipc/chassis_pose.hpp"

namespace tcrg {
namespace robot {

// 一次完整快照里环境需要的测量量
struct RobotSnapshot {
    JointMeasurement joints;
    ImuMeasurement imu;
    double torque[4] = {0.0, 0.0, 0.0, 0.0};
    double server_time = 0.0;
    double mcu_receive_time = 0.0;
    double imu_receive_time = 0.0;
};

class RobotClient {
public:
    explicit RobotClient(const std::string& shm_name);

    // 读取一帧快照；服务端未就绪时返回 false
    bool read(RobotSnapshot& snapshot);

    // 写入四个轴的纯力矩；火控与视觉向导航通信回写最近一次读到的快照值
    bool write_torque(const double torque[4]);

private:
    s2c::serial_comm::SerialCommClient client_;
    s2c::serial_comm::AllInfo last_info_{};
    bool has_info_ = false;
};

}  // namespace robot
}  // namespace tcrg
