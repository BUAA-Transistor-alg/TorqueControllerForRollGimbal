// 校验：robot_ipc.h 在共享内存名字非法与服务端缺席时的失败路径，以及句柄生命周期。
//
// 程序不依赖本机是否运行串口通信服务端：用到的共享内存名字永不创建。
#include <cstdio>

#include "robot_ipc/robot_ipc.h"

namespace {

constexpr double kSampleTimeout = 0.05;

bool has_text(const char* text) { return text != nullptr && text[0] != '\0'; }

// 检查 1：不以 / 开头的共享内存名字必须被拒绝，并留下错误说明
bool check_rejected_name(const char* shm_name) {
    bool ok = true;
    TcrgRobot* robot = tcrg_robot_ipc_create(shm_name, kSampleTimeout);
    if (robot != nullptr) {
        printf("[失败] tcrg_robot_ipc_create(\"%s\")：期望返回 NULL，实际返回非 NULL\n", shm_name);
        tcrg_robot_ipc_destroy(robot);
        ok = false;
    }
    const char* error = tcrg_robot_ipc_last_error();
    if (!has_text(error)) {
        printf("[失败] 共享内存名字 \"%s\" 被拒绝后 tcrg_robot_ipc_last_error：期望非空文本，实际 %s\n",
               shm_name, error == nullptr ? "NULL" : "空字符串");
        ok = false;
    }
    return ok;
}

// 检查 2：服务端缺席时创建仍成功，更新与写入失败，重置不崩溃，销毁一次
bool check_absent_server() {
    bool ok = true;
    const char* shm_name = "/tcrg_robot_ipc_check_absent";

    TcrgRobot* robot = tcrg_robot_ipc_create(shm_name, kSampleTimeout);
    if (robot == nullptr) {
        printf("[失败] tcrg_robot_ipc_create(\"%s\")：期望返回非 NULL，实际返回 NULL，错误为 %s\n",
               shm_name, tcrg_robot_ipc_last_error());
        return false;
    }

    TcrgRobotObservation observation{};
    if (tcrg_robot_ipc_update(robot, &observation) == 0) {
        printf("[失败] tcrg_robot_ipc_update：服务端缺席时期望返回非 0，实际返回 0\n");
        ok = false;
    }
    const char* update_error = tcrg_robot_ipc_last_error();
    if (!has_text(update_error)) {
        printf("[失败] 服务端缺席时 tcrg_robot_ipc_update 失败后 tcrg_robot_ipc_last_error：期望非空文本，"
               "实际 %s\n",
               update_error == nullptr ? "NULL" : "空字符串");
        ok = false;
    }

    const double torque[4] = {0.0, 0.0, 0.0, 0.0};
    if (tcrg_robot_ipc_write_torque(robot, torque) == 0) {
        printf("[失败] tcrg_robot_ipc_write_torque：服务端缺席时期望返回非 0，实际返回 0\n");
        ok = false;
    }

    tcrg_robot_ipc_reset(robot);
    tcrg_robot_ipc_destroy(robot);
    return ok;
}

}  // namespace

int main() {
    bool ok = true;
    if (!check_rejected_name("")) {
        ok = false;
    }
    if (!check_rejected_name("tcrg_robot_ipc_check_relative")) {
        ok = false;
    }
    if (!check_absent_server()) {
        ok = false;
    }
    if (!ok) {
        printf("校验失败\n");
        return 1;
    }
    printf("校验通过\n");
    return 0;
}
