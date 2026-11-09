// 连续调用基准：K=20、refinement=2、step=0.01。
// 单样本模式与批量模式都让相邻调用之间存在数据依赖（上一次的末状态与梯度作为下一次输入），
// 单线程测量在批量模式下按参与者线程并行，统计平均每次调用与平均每个样本的用时。

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "gimbal_dynamics/control_gradient.hpp"
#include "gimbal_dynamics/parameter_gradient.hpp"
#include "gimbal_dynamics/simulation.hpp"

using gimbal::Real;

namespace {

constexpr std::size_t kNumSteps = 20;
constexpr int kRefinement = 2;
constexpr Real kStep = 0.01;
constexpr int kCalls = 100;

struct Problem {
    gimbal::ModelParameters model;
    gimbal::Terrain terrain;
    std::vector<gimbal::BaseState> base;
    std::vector<gimbal::math::Vector4d> torque;
    std::vector<gimbal::math::Vector4d> target_angle;
    std::vector<gimbal::math::Vector4d> target_velocity;
    gimbal::simulation::LossSpec loss;
    gimbal::simulation::TrajectorySpec spec;
    gimbal::simulation::State initial;
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

void run_single(const Problem& problem) {
    {
        gimbal::simulation::Trace trace;
        gimbal::simulation::State state = problem.initial;
        Real checksum = 0;
        Real best = 1e9;
        for (int batch = 0; batch < 4; ++batch) {
            state = problem.initial;
            best = std::min(best, measure_average([&] {
                checksum += gimbal::simulation::simulate(
                    problem.model, problem.terrain, state, problem.base.data(),
                    problem.torque.data(), kNumSteps, problem.spec, problem.loss, trace);
                state = trace.step_end.back();
            }));
        }
        std::printf("单样本前向仿真      平均每次调用 %8.2f us（校验值 %.6f）\n", best,
                    static_cast<double>(checksum));
    }
    {
        gimbal::control_gradient::Workspace workspace;
        std::vector<gimbal::math::Vector4d> torque(kNumSteps);
        std::vector<gimbal::math::Vector4d> gradient(kNumSteps);
        Real checksum = 0;
        Real best = 1e9;
        for (int batch = 0; batch < 4; ++batch) {
            gimbal::simulation::State state = problem.initial;
            torque = problem.torque;
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
        std::printf("单样本控制量梯度    平均每次调用 %8.2f us（校验值 %.6f）\n", best,
                    static_cast<double>(checksum));
    }
    {
        gimbal::parameter_gradient::Workspace workspace;
        gimbal::parameter_gradient::Gradient gradient;
        std::vector<gimbal::math::Vector4d> torque(kNumSteps);
        Real checksum = 0;
        Real best = 1e9;
        for (int batch = 0; batch < 4; ++batch) {
            gimbal::simulation::State state = problem.initial;
            torque = problem.torque;
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
        std::printf("单样本参数灵敏度    平均每次调用 %8.2f us（校验值 %.6f）\n", best,
                    static_cast<double>(checksum));
    }
}

void run_batch(const Problem& problem, std::size_t batch) {
    const std::size_t total = batch * kNumSteps;
    {
        gimbal::control_gradient::BatchWorkspace workspace;
        std::vector<gimbal::simulation::State> initial(batch, problem.initial);
        std::vector<gimbal::simulation::State> final_state(batch);
        std::vector<gimbal::BaseState> base(total);
        std::vector<gimbal::math::Vector4d> torque(total);
        std::vector<gimbal::math::Vector4d> gradient(total);
        std::vector<gimbal::simulation::LossSpec> losses(batch, problem.loss);
        for (std::size_t s = 0; s < batch; ++s) {
            for (std::size_t k = 0; k < kNumSteps; ++k) {
                base[s * kNumSteps + k] = problem.base[k];
                torque[s * kNumSteps + k] = problem.torque[k];
            }
        }
        Real checksum = 0;
        Real best = 1e9;
        for (int round = 0; round < 4; ++round) {
            for (std::size_t s = 0; s < batch; ++s) {
                initial[s] = problem.initial;
                for (std::size_t k = 0; k < kNumSteps; ++k) {
                    torque[s * kNumSteps + k] = problem.torque[k];
                }
            }
            best = std::min(best, measure_average([&] {
                checksum += gimbal::control_gradient::simulateAndGradientBatch(
                    problem.model, problem.terrain, initial.data(), base.data(), torque.data(),
                    batch, kNumSteps, problem.spec, losses.data(), workspace, gradient.data(),
                    final_state.data());
                for (std::size_t s = 0; s < batch; ++s) {
                    initial[s] = final_state[s];
                    for (std::size_t k = 0; k < kNumSteps; ++k) {
                        for (int i = 0; i < 4; ++i) {
                            torque[s * kNumSteps + k][i] +=
                                1e-6 * gradient[s * kNumSteps + k][i];
                        }
                    }
                }
            }));
        }
        std::printf("批量 %3zu 控制量梯度  每次调用 %9.2f us，平均每个样本 %7.3f us（校验值 %.6f）\n",
                    batch, best, best / static_cast<Real>(batch),
                    static_cast<double>(checksum));
    }
    {
        gimbal::parameter_gradient::BatchWorkspace workspace;
        std::vector<gimbal::simulation::State> initial(batch, problem.initial);
        std::vector<gimbal::simulation::State> final_state(batch);
        std::vector<gimbal::BaseState> base(total);
        std::vector<gimbal::math::Vector4d> torque(total);
        std::vector<gimbal::parameter_gradient::Gradient> gradients(batch);
        std::vector<gimbal::simulation::LossSpec> losses(batch, problem.loss);
        for (std::size_t s = 0; s < batch; ++s) {
            for (std::size_t k = 0; k < kNumSteps; ++k) {
                base[s * kNumSteps + k] = problem.base[k];
                torque[s * kNumSteps + k] = problem.torque[k];
            }
        }
        Real checksum = 0;
        Real best = 1e9;
        for (int round = 0; round < 4; ++round) {
            for (std::size_t s = 0; s < batch; ++s) {
                initial[s] = problem.initial;
                for (std::size_t k = 0; k < kNumSteps; ++k) {
                    torque[s * kNumSteps + k] = problem.torque[k];
                }
            }
            best = std::min(best, measure_average([&] {
                checksum += gimbal::parameter_gradient::simulateAndSensitivityBatch(
                    problem.model, problem.terrain, initial.data(), base.data(), torque.data(),
                    batch, kNumSteps, problem.spec, losses.data(), workspace, gradients.data(),
                    final_state.data());
                for (std::size_t s = 0; s < batch; ++s) {
                    initial[s] = final_state[s];
                    for (std::size_t k = 0; k < kNumSteps; ++k) {
                        for (int i = 0; i < 4; ++i) {
                            torque[s * kNumSteps + k][i] +=
                                1e-8 * gradients[s][10 + i];
                        }
                    }
                }
            }));
        }
        std::printf("批量 %3zu 参数灵敏度  每次调用 %9.2f us，平均每个样本 %7.3f us（校验值 %.6f）\n",
                    batch, best, best / static_cast<Real>(batch),
                    static_cast<double>(checksum));
    }
}

}  // namespace

int main(int argc, char** argv) {
    const Problem problem = make_problem();
    const bool single_only = (argc > 1 && std::string(argv[1]) == "single");
    const bool batch_only = (argc > 1 && std::string(argv[1]) == "batch");
    if (!batch_only) {
        run_single(problem);
    }
    if (!single_only) {
        for (std::size_t batch : {8U, 32U, 128U}) {
            run_batch(problem, batch);
        }
    }
    return 0;
}
