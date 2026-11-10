// 连续多次调用基准：K=20、refinement=2、step=0.01，前后两次调用具有数据依赖。
// 用法：sequential_benchmark [single|batch]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "gimbal_dynamics/handlers.hpp"
#include "support.hpp"

using namespace tcrg::gimbal_dynamics;

namespace {

constexpr std::size_t kNumSteps = 20;
constexpr int kThreads = 8;

double now_microseconds() {
    return std::chrono::duration<double, std::micro>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

struct SingleState {
    std::vector<BaseStateT<double>> base;
    std::vector<math::Vector<double, 4>> torque;
    std::vector<math::Vector<double, 4>> target_angle;
    std::vector<math::Vector<double, 4>> target_velocity;
};

SingleState build_state(double offset) {
    SingleState state;
    state.base.resize(kNumSteps);
    state.torque.resize(kNumSteps);
    state.target_angle.resize(kNumSteps);
    state.target_velocity.resize(kNumSteps);
    for (std::size_t k = 0; k < kNumSteps; ++k) {
        const double t = static_cast<double>(k) * 0.01;
        BaseStateT<double>& base = state.base[k];
        base.yaw = 0.2 + 0.4 * t;
        base.yaw_rate = 0.4;
        base.yaw_acceleration = 0.3;
        base.acceleration[0] = 0.2;
        base.acceleration[1] = 0.15;
        base.acceleration[2] = 0.05;
        state.torque[k] =
            math::Vector<double, 4>::from(0.20 + offset, -0.12, 0.06, -0.04);
        for (int i = 0; i < 4; ++i) {
            state.target_angle[k][i] = 0.10 * std::sin(0.5 * t + 0.7 * i);
            state.target_velocity[k][i] = 0.20 * std::cos(0.3 * t + 0.4 * i);
        }
    }
    return state;
}

simulation::LossSpecT<double> build_loss(const SingleState& state) {
    simulation::LossSpecT<double> loss;
    loss.angle_weights = math::Vector<double, 4>::from(1.0, 2.0, 0.8, 1.5);
    loss.velocity_weights = math::Vector<double, 4>::from(0.3, 0.5, 0.2, 0.4);
    loss.torque_weights = math::Vector<double, 4>::from(0.05, 0.05, 0.03, 0.03);
    loss.target_angle = state.target_angle.data();
    loss.target_velocity = state.target_velocity.data();
    return loss;
}

template <typename Function>
double run_rounds(Function&& function, int rounds, int calls) {
    double best = 1e100;
    for (int round = 0; round < rounds; ++round) {
        const double start = now_microseconds();
        for (int call = 0; call < calls; ++call) {
            function();
        }
        const double elapsed = now_microseconds() - start;
        best = std::min(best, elapsed / static_cast<double>(calls));
    }
    return best;
}

void benchmark_single() {
    ControlGradient<double> gradient_handler(kThreads);
    ParameterSensitivity<double> sensitivity_handler(kThreads);
    ForwardSimulation<double> forward_handler(kThreads);
    const double offset = 1e-3;
    SingleState state = build_state(offset);
    simulation::LossSpecT<double> loss = build_loss(state);
    simulation::StateT<double> initial =
        support::make_problem<double>(kNumSteps).initial;

    auto run_gradient = [&] {
        ControlGradientInputT<double> input;
        input.model = ModelParametersT<double>::demo();
        input.terrain.slope_angle = 0.05;
        input.terrain.slope_direction = 0.7;
        input.initial = initial;
        input.base = state.base.data();
        input.torque = state.torque.data();
        input.num_steps = kNumSteps;
        input.spec.step = 0.01;
        input.spec.refinement = 2;
        input.loss = loss;
        ControlGradientOutputT<double> output;
        gradient_handler.run(input, output);
        initial = output.final_state;
        for (std::size_t k = 0; k < kNumSteps; ++k) {
            for (int i = 0; i < 4; ++i) {
                state.torque[k][i] += 1e-8 * output.gradient[k][i];
                state.torque[k][i] = std::max(-0.4, std::min(0.4, state.torque[k][i]));
            }
        }
    };
    auto run_sensitivity = [&] {
        ParameterSensitivityInputT<double> input;
        input.model = ModelParametersT<double>::demo();
        input.terrain.slope_angle = 0.05;
        input.terrain.slope_direction = 0.7;
        input.initial = initial;
        input.base = state.base.data();
        input.torque = state.torque.data();
        input.num_steps = kNumSteps;
        input.spec.step = 0.01;
        input.spec.refinement = 2;
        input.loss = loss;
        ParameterSensitivityOutputT<double> output;
        sensitivity_handler.run(input, output);
        initial = output.final_state;
        for (std::size_t k = 0; k < kNumSteps; ++k) {
            for (int i = 0; i < 4; ++i) {
                state.torque[k][i] += 1e-8 * static_cast<double>(i + 1);
            }
        }
    };
    auto run_forward = [&] {
        ForwardInputT<double> input;
        input.model = ModelParametersT<double>::demo();
        input.terrain.slope_angle = 0.05;
        input.terrain.slope_direction = 0.7;
        input.initial = initial;
        input.base = state.base.data();
        input.torque = state.torque.data();
        input.num_steps = kNumSteps;
        input.spec.step = 0.01;
        input.spec.refinement = 2;
        input.loss = loss;
        ForwardOutputT<double> output;
        forward_handler.run(input, output);
        initial = output.step_end.back();
    };

    const double forward_time = run_rounds(run_forward, 4, 100);
    const double gradient_time = run_rounds(run_gradient, 4, 100);
    const double sensitivity_time = run_rounds(run_sensitivity, 4, 100);
    printf("单样本前向仿真   平均每次调用 %8.2f us\n", forward_time);
    printf("单样本控制量梯度 平均每次调用 %8.2f us\n", gradient_time);
    printf("单样本参数灵敏度 平均每次调用 %8.2f us\n", sensitivity_time);
}

struct BatchState {
    std::vector<simulation::StateT<double>> initial;
    std::vector<BaseStateT<double>> base;
    std::vector<math::Vector<double, 4>> torque;
    std::vector<simulation::LossSpecT<double>> losses;
    std::vector<math::Vector<double, 4>> target_angle;
    std::vector<math::Vector<double, 4>> target_velocity;
};

BatchState build_batch(std::size_t batch) {
    BatchState state;
    auto single = support::make_problem<double>(kNumSteps);
    state.initial.resize(batch, single.initial);
    state.base.resize(batch * kNumSteps);
    state.torque.resize(batch * kNumSteps);
    state.target_angle = single.target_angle;
    state.target_velocity = single.target_velocity;
    state.losses.resize(batch, single.loss);
    for (std::size_t s = 0; s < batch; ++s) {
        for (std::size_t k = 0; k < kNumSteps; ++k) {
            const double t = static_cast<double>(k) * 0.01;
            BaseStateT<double>& base = state.base[s * kNumSteps + k];
            base.yaw = 0.2 + 0.4 * t;
            base.yaw_rate = 0.4;
            base.yaw_acceleration = 0.3;
            base.acceleration[0] = 0.2;
            base.acceleration[1] = 0.15;
            base.acceleration[2] = 0.05;
            state.torque[s * kNumSteps + k] =
                math::Vector<double, 4>::from(0.20, -0.12, 0.06, -0.04);
        }
    }
    return state;
}

void benchmark_batch(std::size_t batch) {
    ControlGradient<double> gradient_handler(kThreads);
    ParameterSensitivity<double> sensitivity_handler(kThreads);
    BatchState state = build_batch(batch);

    ControlGradientBatchInputT<double> gradient_input;
    gradient_input.model = ModelParametersT<double>::demo();
    gradient_input.terrain.slope_angle = 0.05;
    gradient_input.terrain.slope_direction = 0.7;
    gradient_input.initial = state.initial.data();
    gradient_input.base = state.base.data();
    gradient_input.torque = state.torque.data();
    gradient_input.num_steps = kNumSteps;
    gradient_input.spec.step = 0.01;
    gradient_input.spec.refinement = 2;
    gradient_input.loss = state.losses.data();
    ControlGradientBatchOutputT<double> gradient_output;
    auto run_gradient = [&] {
        gradient_handler.runBatch(batch, gradient_input, gradient_output);
        for (std::size_t s = 0; s < batch; ++s) {
            for (std::size_t k = 0; k < kNumSteps; ++k) {
                state.initial[s] = gradient_output.final_state[s];
                for (int i = 0; i < 4; ++i) {
                    double& value = state.torque[s * kNumSteps + k][i];
                    value += 1e-8 * gradient_output.gradient[s * kNumSteps + k][i];
                    value = std::max(-0.4, std::min(0.4, value));
                }
            }
        }
        gradient_input.torque = state.torque.data();
        gradient_input.initial = state.initial.data();
    };

    ParameterSensitivityBatchInputT<double> sensitivity_input;
    sensitivity_input.model = ModelParametersT<double>::demo();
    sensitivity_input.terrain.slope_angle = 0.05;
    sensitivity_input.terrain.slope_direction = 0.7;
    sensitivity_input.initial = state.initial.data();
    sensitivity_input.base = state.base.data();
    sensitivity_input.torque = state.torque.data();
    sensitivity_input.num_steps = kNumSteps;
    sensitivity_input.spec.step = 0.01;
    sensitivity_input.spec.refinement = 2;
    sensitivity_input.loss = state.losses.data();
    ParameterSensitivityBatchOutputT<double> sensitivity_output;
    auto run_sensitivity = [&] {
        sensitivity_handler.runBatch(batch, sensitivity_input, sensitivity_output);
        for (std::size_t s = 0; s < batch; ++s) {
            state.initial[s] = sensitivity_output.final_state[s];
            for (std::size_t k = 0; k < kNumSteps; ++k) {
                for (int i = 0; i < 4; ++i) {
                    double& value = state.torque[s * kNumSteps + k][i];
                    value += 1e-8 * static_cast<double>(i + 1);
                    value = std::max(-0.4, std::min(0.4, value));
                }
            }
        }
        sensitivity_input.initial = state.initial.data();
    };

    const double gradient_time = run_rounds(run_gradient, 3, 30);
    const double sensitivity_time = run_rounds(run_sensitivity, 3, 30);
    printf("批量 %3zu 控制量梯度 每次调用 %9.2f us，平均每个样本 %8.2f us（校验值 %.6f）\n",
           batch, gradient_time, gradient_time / static_cast<double>(batch),
           static_cast<double>(gradient_output.mean_loss));
    printf("批量 %3zu 参数灵敏度 每次调用 %9.2f us，平均每个样本 %8.2f us（校验值 %.6f）\n",
           batch, sensitivity_time, sensitivity_time / static_cast<double>(batch),
           static_cast<double>(sensitivity_output.mean_loss));
}

}  // namespace

int main(int argc, char** argv) {
    const bool single = argc < 2 || std::strcmp(argv[1], "single") == 0;
    const bool batch = argc < 2 || std::strcmp(argv[1], "batch") == 0;
    if (single) {
        benchmark_single();
    }
    if (batch) {
        benchmark_batch(8);
        benchmark_batch(32);
        benchmark_batch(128);
        benchmark_batch(256);
    }
    return 0;
}
