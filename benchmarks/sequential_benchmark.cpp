// 连续调用基准：K=20、refinement=2、step=0.01。
// 每次调用的初始状态与力矩来自上一次调用的输出（末状态与梯度），
// 相邻调用之间存在数据依赖，测量单线程下平均每次调用的用时。

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "gimbal_dynamics/control_gradient.hpp"
#include "gimbal_dynamics/parameter_gradient.hpp"
#include "gimbal_dynamics/simulation.hpp"

using gimbal::Real;

namespace {

constexpr std::size_t kNumSteps = 20;
constexpr int kRefinement = 2;
constexpr Real kStep = 0.01;
constexpr int kCalls = 200;

struct Problem {
    gimbal::ModelParameters model;
    gimbal::Terrain terrain;
    gimbal::simulation::State initial;
    std::vector<gimbal::BaseState> base;
    std::vector<gimbal::math::Vector4d> torque;
    std::vector<gimbal::math::Vector4d> target_angle;
    std::vector<gimbal::math::Vector4d> target_velocity;
    gimbal::simulation::LossSpec loss;
    gimbal::simulation::TrajectorySpec spec;
};

Problem make_problem() {
    Problem problem;
    problem.model = gimbal::ModelParameters::demo();
    problem.terrain.slope_angle = 0.05;
    problem.terrain.slope_direction = 0.7;
    problem.initial.angle = gimbal::math::Vector4d::from(0.20, -0.10, 0.15, 0.05);
    problem.initial.velocity = gimbal::math::Vector4d::from(0.30, -0.20, 0.10, -0.05);
    problem.base.resize(kNumSteps);
    problem.torque.resize(kNumSteps);
    problem.target_angle.resize(kNumSteps);
    problem.target_velocity.resize(kNumSteps);
    for (std::size_t k = 0; k < kNumSteps; ++k) {
        const Real step_time = static_cast<Real>(k) * kStep;
        gimbal::BaseState& base = problem.base[k];
        base.yaw = 0.2 + 0.4 * step_time;
        base.yaw_rate = 0.4;
        base.yaw_acceleration = 0.3;
        base.acceleration[0] = 0.2;
        base.acceleration[1] = 0.15;
        base.acceleration[2] = 0.05;
        problem.torque[k] = gimbal::math::Vector4d::from(0.20, -0.12, 0.06, -0.04);
        for (int i = 0; i < 4; ++i) {
            problem.target_angle[k][i] = 0.10 * std::sin(0.5 * step_time + 0.7 * i);
            problem.target_velocity[k][i] = 0.20 * std::cos(0.3 * step_time + 0.4 * i);
        }
    }
    problem.loss.angle_weights = gimbal::math::Vector4d::from(1.0, 2.0, 0.8, 1.5);
    problem.loss.velocity_weights = gimbal::math::Vector4d::from(0.3, 0.5, 0.2, 0.4);
    problem.loss.torque_weights = gimbal::math::Vector4d::from(0.05, 0.05, 0.03, 0.03);
    problem.loss.target_angle = problem.target_angle.data();
    problem.loss.target_velocity = problem.target_velocity.data();
    problem.spec.step = kStep;
    problem.spec.refinement = kRefinement;
    return problem;
}

template <typename Function>
Real measure_average(Function&& function) {
    const auto begin = std::chrono::steady_clock::now();
    for (int call = 0; call < kCalls; ++call) {
        function();
    }
    const auto end = std::chrono::steady_clock::now();
    const Real total = std::chrono::duration<Real, std::micro>(end - begin).count();
    return total / static_cast<Real>(kCalls);
}

}  // namespace

int main() {
    Problem problem = make_problem();

    // 前向仿真：连续调用，初始状态取上一次的末状态
    {
        gimbal::simulation::Trace trace;
        gimbal::simulation::State state = problem.initial;
        Real checksum = 0.0;
        Real best = 1e9;
        for (int batch = 0; batch < 5; ++batch) {
            best = std::min(best, measure_average([&] {
                checksum += gimbal::simulation::simulate(
                    problem.model, problem.terrain, state, problem.base.data(),
                    problem.torque.data(), kNumSteps, problem.spec, problem.loss, trace);
                state = trace.step_end.back();
            }));
        }
        printf("前向仿真        平均每次调用 %8.2f us（校验值 %.6f）\n", best, checksum);
    }

    // 控制量梯度：每次调用用上一次的梯度更新力矩，末状态作为下一次初始状态
    {
        gimbal::control_gradient::Workspace workspace;
        std::vector<gimbal::math::Vector4d> gradient(kNumSteps);
        Real checksum = 0.0;
        Real best = 1e9;
        for (int batch = 0; batch < 5; ++batch) {
            gimbal::simulation::State state = problem.initial;
            std::vector<gimbal::math::Vector4d> torque = problem.torque;
            best = std::min(best, measure_average([&] {
                checksum += gimbal::control_gradient::simulateAndGradient(
                    problem.model, problem.terrain, state, problem.base.data(), torque.data(),
                    kNumSteps, problem.spec, problem.loss, workspace, gradient.data(), &state);
                for (std::size_t k = 0; k < kNumSteps; ++k) {
                    for (int i = 0; i < 4; ++i) {
                        torque[k][i] += 1e-6 * gradient[k][i];
                    }
                }
            }));
        }
        printf("控制量梯度模块  平均每次调用 %8.2f us（校验值 %.6f）\n", best, checksum);
    }

    // 参数灵敏度：同样用上一次的梯度更新力矩
    {
        gimbal::parameter_gradient::Workspace workspace;
        gimbal::parameter_gradient::Gradient gradient;
        Real checksum = 0.0;
        Real best = 1e9;
        for (int batch = 0; batch < 5; ++batch) {
            gimbal::simulation::State state = problem.initial;
            std::vector<gimbal::math::Vector4d> torque = problem.torque;
            best = std::min(best, measure_average([&] {
                checksum += gimbal::parameter_gradient::simulateAndSensitivity(
                    problem.model, problem.terrain, state, problem.base.data(), torque.data(),
                    kNumSteps, problem.spec, problem.loss, workspace, &gradient, &state);
                for (std::size_t k = 0; k < kNumSteps; ++k) {
                    for (int i = 0; i < 4; ++i) {
                        torque[k][i] += 1e-8 * gradient[10 + i];
                    }
                }
            }));
        }
        printf("参数灵敏度模块  平均每次调用 %8.2f us（校验值 %.6f）\n", best, checksum);
    }
    return 0;
}
