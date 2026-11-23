#pragma once

// 实车环境的 C 接口：由子模组 Sentry2Communicator 的 ipc_client 读写实车控制器，
// 并把头部 IMU 与四个关节的测量反解成底盘位姿、坡面参数与角速度。
// 只使用 ipc_client 的 vis_get_all_info 与 vis_write，不使用导航部分。
//
// 返回值为 0 表示成功，非 0 表示失败，失败原因用 tcrg_robot_ipc_last_error 读取。

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TcrgRobot TcrgRobot;

// 一帧实车观测：关节状态、实际控制力矩、底盘位姿与时间戳
typedef struct TcrgRobotObservation {
    double angle[4];                // 大 yaw、roll、小 yaw、pitch，rad
    double velocity[4];             // rad/s
    double torque[4];               // 电控实际使用的控制力矩，N·m
    double slope_angle;             // λ，rad
    double slope_direction;         // γ，rad
    double base_yaw;                // ψ，多圈连续，rad
    double base_yaw_rate;           // ψ̇，rad/s
    double base_yaw_acceleration;   // ψ̈，rad/s²，首帧为 0
    double base_acceleration[3];    // 底盘参考点线加速度，暂置为 0
    double server_time;             // 服务端快照时间戳，steady_clock 秒
    double mcu_age;                 // 读取时刻距上一次电控包的时间，s
    double imu_age;                 // 读取时刻距上一次 IMU 包的时间，s
} TcrgRobotObservation;

// 创建实车句柄。shm_name 为共享内存名字（以 / 开头）；server 未启动不影响创建，
// 由 tcrg_robot_ipc_update 报告。sample_timeout 为电控与 IMU 数据的最大允许延迟（秒）。
TcrgRobot* tcrg_robot_ipc_create(const char* shm_name, double sample_timeout);
void tcrg_robot_ipc_destroy(TcrgRobot* robot);

// 清除反解的连续量状态，下一次 update 重新开始解卷绕与角加速度差分
void tcrg_robot_ipc_reset(TcrgRobot* robot);

// 读取一帧观测并反解；服务端未就绪、数据超时或包含非有限值时返回非 0
int tcrg_robot_ipc_update(TcrgRobot* robot, TcrgRobotObservation* observation);

// 写入四个轴的纯力矩控制；其余字段回写最近一次读到的快照值
int tcrg_robot_ipc_write_torque(TcrgRobot* robot, const double torque[4]);

// 最近一次失败的说明
const char* tcrg_robot_ipc_last_error(void);

#ifdef __cplusplus
}
#endif
