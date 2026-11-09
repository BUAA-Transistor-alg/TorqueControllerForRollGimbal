// 并发调用检查：单样本与批量接口共用线程池时的串行化、结果一致性与无死锁。
#include <atomic>
#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>

#include "gimbal_dynamics/control_gradient.hpp"
#include "gimbal_dynamics/parameter_gradient.hpp"
#include "gimbal_dynamics/simulation.hpp"

namespace {

using gimbal::Real;

struct Inputs {
    gimbal::ModelParameters model;
    gimbal::Terrain terrain;
    gimbal::simulation::State initial;
    std::vector<gimbal::BaseState> base;
    std::vector<gimbal::math::Vector4d> torque;
    gimbal::simulation::LossSpec loss;
    gimbal::simulation::TrajectorySpec spec;
    std::vector<gimbal::math::Vector4d> target_angle;
    std::vector<gimbal::math::Vector4d> target_velocity;
    std::size_t num_steps = 20;
};

Inputs make_inputs() {
    Inputs inputs;
    inputs.model = gimbal::ModelParameters::demo();
    inputs.terrain.slope_angle = 0.05;
    inputs.terrain.slope_direction = 0.7;
    inputs.initial.angle = gimbal::math::Vector4d::from(0.20, -0.10, 0.15, 0.05);
    inputs.initial.velocity = gimbal::math::Vector4d::from(0.30, -0.20, 0.10, -0.05);
    inputs.spec.step = 0.01;
    inputs.spec.refinement = 2;
    inputs.base.resize(inputs.num_steps);
    inputs.torque.resize(inputs.num_steps);
    inputs.target_angle.resize(inputs.num_steps);
    inputs.target_velocity.resize(inputs.num_steps);
    for (std::size_t k = 0; k < inputs.num_steps; ++k) {
        const Real t = static_cast<Real>(k) * inputs.spec.step;
        gimbal::BaseState& base = inputs.base[k];
        base.yaw = 0.2 + 0.4 * t;
        base.yaw_rate = 0.4;
        base.yaw_acceleration = 0.3;
        base.acceleration[0] = 0.2;
        base.acceleration[1] = 0.15;
        base.acceleration[2] = 0.05;
        inputs.torque[k] = gimbal::math::Vector4d::from(0.20, -0.12, 0.06, -0.04);
        for (int i = 0; i < 4; ++i) {
            inputs.target_angle[k][i] = 0.10 * std::sin(0.5 * t + 0.7 * i);
            inputs.target_velocity[k][i] = 0.20 * std::cos(0.3 * t + 0.4 * i);
        }
    }
    inputs.loss.angle_weights = gimbal::math::Vector4d::from(1.0, 2.0, 0.8, 1.5);
    inputs.loss.velocity_weights = gimbal::math::Vector4d::from(0.3, 0.5, 0.2, 0.4);
    inputs.loss.torque_weights = gimbal::math::Vector4d::from(0.05, 0.05, 0.03, 0.03);
    inputs.loss.target_angle = inputs.target_angle.data();
    inputs.loss.target_velocity = inputs.target_velocity.data();
    return inputs;
}

}  // namespace

int main() {
    const Inputs inputs = make_inputs();
    gimbal::control_gradient::Workspace reference_control;
    std::vector<gimbal::math::Vector4d> reference_gradient(inputs.num_steps);
    gimbal::parameter_gradient::Workspace reference_parameter;
    gimbal::parameter_gradient::Gradient reference_sensitivity;
    gimbal::control_gradient::simulateAndGradient(
        inputs.model, inputs.terrain, inputs.initial, inputs.base.data(), inputs.torque.data(),
        inputs.num_steps, inputs.spec, inputs.loss, reference_control, reference_gradient.data(),
        nullptr);
    gimbal::parameter_gradient::simulateAndSensitivity(
        inputs.model, inputs.terrain, inputs.initial, inputs.base.data(), inputs.torque.data(),
        inputs.num_steps, inputs.spec, inputs.loss, reference_parameter,
        &reference_sensitivity, nullptr);

    std::atomic<int> single_mismatch{0};
    auto single_worker = [&] {
        gimbal::control_gradient::Workspace control_workspace;
        gimbal::parameter_gradient::Workspace parameter_workspace;
        std::vector<gimbal::math::Vector4d> gradient(inputs.num_steps);
        gimbal::parameter_gradient::Gradient sensitivity;
        for (int iteration = 0; iteration < 40; ++iteration) {
            gimbal::control_gradient::simulateAndGradient(
                inputs.model, inputs.terrain, inputs.initial, inputs.base.data(),
                inputs.torque.data(), inputs.num_steps, inputs.spec, inputs.loss,
                control_workspace, gradient.data(), nullptr);
            gimbal::parameter_gradient::simulateAndSensitivity(
                inputs.model, inputs.terrain, inputs.initial, inputs.base.data(),
                inputs.torque.data(), inputs.num_steps, inputs.spec, inputs.loss,
                parameter_workspace, &sensitivity, nullptr);
            for (std::size_t k = 0; k < inputs.num_steps; ++k) {
                for (int i = 0; i < 4; ++i) {
                    if (gradient[k][i] != reference_gradient[k][i]) {
                        single_mismatch.fetch_add(1);
                    }
                }
            }
            for (int i = 0; i < gimbal::parameters::kParameterCount; ++i) {
                if (sensitivity[i] != reference_sensitivity[i]) {
                    single_mismatch.fetch_add(1);
                }
            }
        }
    };
    std::vector<std::thread> single_threads;
    for (int thread = 0; thread < 3; ++thread) {
        single_threads.emplace_back(single_worker);
    }
    for (std::thread& thread : single_threads) {
        thread.join();
    }

    const std::size_t batch = 8;
    std::vector<gimbal::simulation::State> initial_batch(batch, inputs.initial);
    std::vector<gimbal::BaseState> base_batch(batch * inputs.num_steps);
    std::vector<gimbal::math::Vector4d> torque_batch(batch * inputs.num_steps);
    for (std::size_t sample = 0; sample < batch; ++sample) {
        for (std::size_t k = 0; k < inputs.num_steps; ++k) {
            base_batch[sample * inputs.num_steps + k] = inputs.base[k];
            torque_batch[sample * inputs.num_steps + k] = inputs.torque[k];
        }
    }
    std::vector<gimbal::simulation::LossSpec> losses(batch, inputs.loss);
    std::atomic<int> batch_mismatch{0};
    auto batch_worker = [&] {
        gimbal::control_gradient::BatchWorkspace workspace;
        std::vector<gimbal::math::Vector4d> gradient(batch * inputs.num_steps);
        std::vector<gimbal::simulation::State> final_state(batch);
        for (int iteration = 0; iteration < 8; ++iteration) {
            gimbal::control_gradient::simulateAndGradientBatch(
                inputs.model, inputs.terrain, initial_batch.data(), base_batch.data(),
                torque_batch.data(), batch, inputs.num_steps, inputs.spec, losses.data(),
                workspace, gradient.data(), final_state.data());
            for (std::size_t sample = 0; sample < batch; ++sample) {
                for (std::size_t k = 0; k < inputs.num_steps; ++k) {
                    for (int i = 0; i < 4; ++i) {
                        if (gradient[sample * inputs.num_steps + k][i] !=
                            reference_gradient[k][i]) {
                            batch_mismatch.fetch_add(1);
                        }
                    }
                }
            }
        }
    };
    std::vector<std::thread> batch_threads;
    for (int thread = 0; thread < 2; ++thread) {
        batch_threads.emplace_back(batch_worker);
    }
    for (std::thread& thread : batch_threads) {
        thread.join();
    }

    printf("并发单样本不一致计数 %d，并发批量不一致计数 %d\n", single_mismatch.load(),
           batch_mismatch.load());
    if (single_mismatch.load() != 0 || batch_mismatch.load() != 0) {
        printf("并发检查失败\n");
        return 1;
    }
    printf("并发检查通过\n");
    return 0;
}
