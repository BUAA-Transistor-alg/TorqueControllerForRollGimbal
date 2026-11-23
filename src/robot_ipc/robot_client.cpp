#include "robot_client.hpp"

namespace tcrg {
namespace robot {

RobotClient::RobotClient(const std::string& shm_name) : client_(shm_name) {}

bool RobotClient::read(RobotSnapshot& snapshot) {
    s2c::serial_comm::AllInfo info{};
    if (!client_.vis_get_all_info(info)) {
        return false;
    }
    last_info_ = info;
    has_info_ = true;

    const s2c::com::mcu::ReceivePacket& mcu = info.mcu_data;
    snapshot.joints.angle[0] = mcu.yaw_big_position;
    snapshot.joints.angle[1] = mcu.roll_position;
    snapshot.joints.angle[2] = mcu.yaw_small_position;
    snapshot.joints.angle[3] = mcu.pitch_position;
    snapshot.joints.velocity[0] = mcu.yaw_big_velocity;
    snapshot.joints.velocity[1] = mcu.roll_velocity;
    snapshot.joints.velocity[2] = mcu.yaw_small_velocity;
    snapshot.joints.velocity[3] = mcu.pitch_velocity;
    snapshot.torque[0] = mcu.yaw_big_torque;
    snapshot.torque[1] = mcu.roll_torque;
    snapshot.torque[2] = mcu.yaw_small_torque;
    snapshot.torque[3] = mcu.pitch_torque;

    const s2c::com::imu::ReceivePacket& imu = info.imu_data;
    snapshot.imu.euler_yaw = imu.euler_yaw;
    snapshot.imu.euler_pitch = imu.euler_pitch;
    snapshot.imu.euler_roll = imu.euler_roll;
    snapshot.imu.gx = imu.gx;
    snapshot.imu.gy = imu.gy;
    snapshot.imu.gz = imu.gz;

    snapshot.server_time = info.server_info.timestamp;
    snapshot.mcu_receive_time = info.server_info.last_mcu_receive_time;
    snapshot.imu_receive_time = info.server_info.last_imu_receive_time;
    return true;
}

bool RobotClient::write_torque(const double torque[4]) {
    s2c::serial_comm::VisWrite write{};
    if (has_info_) {
        write.serial.fire_control = last_info_.send_data.fire_control;
        write.vision_to_nav = last_info_.vis_to_nav;
    }

    s2c::serial_comm::AxisControlData& axes = write.serial.axes;
    axes.yaw_big_mode = s2c::com::mcu::AXIS_MODE_TORQUE_ONLY;
    axes.yaw_big_target_position = 0.0;
    axes.yaw_big_target_velocity = 0.0f;
    axes.yaw_big_torque = static_cast<float>(torque[0]);
    axes.roll_mode = s2c::com::mcu::AXIS_MODE_TORQUE_ONLY;
    axes.roll_target_position = 0.0f;
    axes.roll_target_velocity = 0.0f;
    axes.roll_torque = static_cast<float>(torque[1]);
    axes.yaw_small_mode = s2c::com::mcu::AXIS_MODE_TORQUE_ONLY;
    axes.yaw_small_target_position = 0.0f;
    axes.yaw_small_target_velocity = 0.0f;
    axes.yaw_small_torque = static_cast<float>(torque[2]);
    axes.pitch_mode = s2c::com::mcu::AXIS_MODE_TORQUE_ONLY;
    axes.pitch_target_position = 0.0f;
    axes.pitch_target_velocity = 0.0f;
    axes.pitch_torque = static_cast<float>(torque[3]);
    return client_.vis_write(write);
}

}  // namespace robot
}  // namespace tcrg
