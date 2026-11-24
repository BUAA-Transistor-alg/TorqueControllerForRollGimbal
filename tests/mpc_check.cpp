// 校验：C++ MPC 求解器的箱约束、损失一致性、迭代下降、热启动与复位行为。
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "gimbal_dynamics/handlers.hpp"
#include "gimbal_dynamics/mpc_solver.hpp"
#include "support.hpp"

using namespace tcrg::gimbal_dynamics;

namespace {

mpc::ConfigT<double> make_config(std::size_t horizon, int max_iter) {
    mpc::ConfigT<double> config;
    config.horizon = horizon;
    config.step = 0.01;
    config.refinement = 2;
    config.torque_limit = math::Vector<double, 4>::from(5.0, 4.0, 3.0, 2.0);
    config.max_iter = max_iter;
    config.tolerance = 1e-4;
    config.loss.angle_weights = math::Vector<double, 4>::from(1.0, 1.0, 1.0, 1.0);
    config.loss.velocity_weights = math::Vector<double, 4>::from(0.001, 0.001, 0.001, 0.001);
    config.loss.torque_weights = math::Vector<double, 4>::from(0.001, 0.001, 0.001, 0.001);
    config.loss.torque_rate_weights = math::Vector<double, 4>::from(0.01, 0.01, 0.01, 0.01);
    config.loss.limit_weights = math::Vector<double, 4>::from(0.0, 200.0, 200.0, 200.0);
    config.loss.limit_lower = math::Vector<double, 4>::from(
        -1e9, -0.5, -0.6, -0.5);
    config.loss.limit_upper = math::Vector<double, 4>::from(1e9, 0.2, 0.6, 0.5);
    config.loss.limit_margin = math::Vector<double, 4>::from(0.07, 0.07, 0.07, 0.07);
    return config;
}

bool finite_and_bounded(const math::Vector<double, 4>& torque,
                        const math::Vector<double, 4>& limit) {
    for (int joint = 0; joint < 4; ++joint) {
        if (!std::isfinite(torque[joint]) ||
            std::fabs(torque[joint]) > limit[joint] + 1e-9) {
            return false;
        }
    }
    return true;
}

double evaluate_loss(const support::Problem<double>& problem,
                     const std::vector<math::Vector<double, 4>>& torques,
                     const mpc::ConfigT<double>& config,
                     const math::Vector<double, 4>* previous_torque) {
    ControlGradient<double> handler(1);
    ControlGradientInputT<double> input;
    input.model = problem.model;
    input.terrain = problem.terrain;
    input.initial = problem.initial;
    std::vector<BaseStateT<double>> bases(problem.num_steps, problem.base[0]);
    input.base = bases.data();
    input.torque = torques.data();
    input.num_steps = problem.num_steps;
    input.spec.step = config.step;
    input.spec.refinement = config.refinement;
    simulation::LossSpecT<double> loss;
    loss.angle_weights = config.loss.angle_weights;
    loss.velocity_weights = config.loss.velocity_weights;
    loss.torque_weights = config.loss.torque_weights;
    loss.torque_rate_weights = config.loss.torque_rate_weights;
    loss.limit_weights = config.loss.limit_weights;
    loss.limit_lower = config.loss.limit_lower;
    loss.limit_upper = config.loss.limit_upper;
    loss.limit_margin = config.loss.limit_margin;
    loss.target_angle = problem.target_angle.data();
    loss.target_velocity = problem.target_velocity.data();
    loss.previous_torque = previous_torque;
    input.loss = loss;
    ControlGradientOutputT<double> output;
    handler.run(input, output);
    return output.loss;
}

bool check_solver() {
    const std::size_t horizon = 10;
    support::Problem<double> problem = support::make_problem<double>(horizon);
    const mpc::ConfigT<double> config = make_config(horizon, 40);
    const math::Vector<double, 4> zero_previous = math::Vector<double, 4>::zero();

    // 一阶求解器与四十步求解器从同一冷启动出发，四十步不应更差
    mpc::SolverT<double> short_solver(1, problem.model, make_config(horizon, 1));
    mpc::InfoT<double> short_info;
    const math::Vector<double, 4> short_command =
        short_solver.solve(problem.initial, problem.terrain, problem.base[0],
                           problem.target_angle.data(), problem.target_velocity.data(),
                           &short_info);
    if (!finite_and_bounded(short_command, config.torque_limit)) {
        printf("单步求解结果不是有限值或越界\n");
        return false;
    }
    mpc::SolverT<double> solver(1, problem.model, config);
    mpc::InfoT<double> info;
    const math::Vector<double, 4> command =
        solver.solve(problem.initial, problem.terrain, problem.base[0],
                     problem.target_angle.data(), problem.target_velocity.data(), &info);
    if (!finite_and_bounded(command, config.torque_limit)) {
        printf("求解结果不是有限值或越界\n");
        return false;
    }
    if (info.loss > short_info.loss + 1e-9) {
        printf("迭代没有降低损失: %.6e -> %.6e\n", short_info.loss, info.loss);
        return false;
    }
    if (!std::isfinite(info.projected_gradient) || info.projected_gradient < 0.0) {
        printf("投影梯度不合理\n");
        return false;
    }
    if (info.evaluations <= 0 || info.iterations <= 0) {
        printf("求解信息不合理\n");
        return false;
    }
    if (solver.solution().size() != horizon ||
        solver.solution()[0][0] != command[0]) {
        printf("解序列与返回力矩不一致\n");
        return false;
    }

    // 用独立处理器在同一条解上重新求损失，必须与求解器报告的损失一致
    const double checked = evaluate_loss(problem, solver.solution(), config, &zero_previous);
    const double deviation =
        std::fabs(checked - info.loss) / std::max(1.0, std::fabs(checked));
    if (!(deviation < 1e-12)) {
        printf("求解器损失与独立求值不一致: %.6e vs %.6e\n", checked, info.loss);
        return false;
    }

    // 热启动：同一状态与目标连续求解，结果仍须可行
    mpc::InfoT<double> warm_info;
    const math::Vector<double, 4> warm_command =
        solver.solve(problem.initial, problem.terrain, problem.base[0],
                     problem.target_angle.data(), problem.target_velocity.data(), &warm_info);
    if (!finite_and_bounded(warm_command, config.torque_limit) ||
        !std::isfinite(warm_info.loss)) {
        printf("热启动求解结果不合理\n");
        return false;
    }
    // 热启动时 k = 0 的变化率项参照第一次采用的力矩，独立求值必须与求解器报告一致
    const double warm_checked = evaluate_loss(problem, solver.solution(), config, &command);
    const double warm_deviation =
        std::fabs(warm_checked - warm_info.loss) / std::max(1.0, std::fabs(warm_checked));
    if (!(warm_deviation < 1e-12)) {
        printf("热启动损失与独立求值不一致: %.6e vs %.6e\n", warm_checked, warm_info.loss);
        return false;
    }

    // 复位后重新冷启动，结果应与第一次一致
    solver.reset();
    mpc::InfoT<double> restarted_info;
    const math::Vector<double, 4> restarted_command =
        solver.solve(problem.initial, problem.terrain, problem.base[0],
                     problem.target_angle.data(), problem.target_velocity.data(),
                     &restarted_info);
    if (std::fabs(restarted_info.loss - info.loss) > 1e-9 * std::max(1.0, std::fabs(info.loss))) {
        printf("复位后的求解结果不一致\n");
        return false;
    }
    if (!finite_and_bounded(restarted_command, config.torque_limit)) {
        printf("复位后的力矩不合理\n");
        return false;
    }
    return true;
}

// 默认各轴力矩上限取实车通信子模组 McuDataPreprocessor 中四个力矩最大值的绝对值。
bool check_default_limits() {
    const mpc::ConfigT<double> config;
    const double expected[4] = {6.561, 6.640, 2.000, 3.992};
    for (int joint = 0; joint < 4; ++joint) {
        if (std::fabs(config.torque_limit[joint] - expected[joint]) > 1e-12) {
            printf("关节 %d 的默认力矩上限与标定值不一致\n", joint);
            return false;
        }
    }
    return true;
}

// 各轴箱约束独立生效：第一个关节放宽，其余三轴收紧，求解结果必须落在各自上限内。
bool check_axis_limits() {
    const std::size_t horizon = 10;
    support::Problem<double> problem = support::make_problem<double>(horizon);
    // 目标偏离初值较远，迫使力矩饱和到箱约束边界
    for (std::size_t k = 0; k < horizon; ++k) {
        problem.target_angle[k] = math::Vector<double, 4>::from(0.8, -0.8, 0.8, -0.8);
        problem.target_velocity[k] = math::Vector<double, 4>::zero();
    }
    mpc::ConfigT<double> config = make_config(horizon, 40);
    config.torque_limit = math::Vector<double, 4>::from(0.5, 0.005, 0.005, 0.005);
    mpc::SolverT<double> solver(1, problem.model, config);
    mpc::InfoT<double> info;
    const math::Vector<double, 4> command =
        solver.solve(problem.initial, problem.terrain, problem.base[0],
                     problem.target_angle.data(), problem.target_velocity.data(), &info);
    if (!finite_and_bounded(command, config.torque_limit)) {
        printf("本拍力矩没有落在各轴上限内\n");
        return false;
    }
    for (const math::Vector<double, 4>& step : solver.solution()) {
        if (!finite_and_bounded(step, config.torque_limit)) {
            printf("预测区间的力矩没有落在各轴上限内\n");
            return false;
        }
    }
    // 收紧的关节必须确实被自身上限约束住，否则无法区分各轴上限
    for (int joint = 1; joint < 4; ++joint) {
        if (!(std::fabs(command[joint]) > 0.5 * config.torque_limit[joint])) {
            printf("关节 %d 没有受到自身箱约束作用\n", joint);
            return false;
        }
    }
    return true;
}

// 任一轴的力矩上限非法时构造求解器必须报错。
bool check_invalid_limit() {
    support::Problem<double> problem = support::make_problem<double>(5);
    mpc::ConfigT<double> config = make_config(5, 1);
    config.torque_limit[2] = 0.0;
    bool thrown = false;
    try {
        mpc::SolverT<double> solver(1, problem.model, config);
    } catch (const std::invalid_argument&) {
        thrown = true;
    }
    if (!thrown) {
        printf("非法轴力矩上限没有报错\n");
        return false;
    }
    return true;
}

}  // namespace

int main() {
    if (!check_default_limits() || !check_solver() || !check_axis_limits() ||
        !check_invalid_limit()) {
        printf("MPC 求解器校验失败\n");
        return 1;
    }
    printf("MPC 求解器校验通过\n");
    return 0;
}
