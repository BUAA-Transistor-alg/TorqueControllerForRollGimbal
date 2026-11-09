// 线程数环境变量检查：GIMBAL_PIPELINE_THREADS=1 与 GIMBAL_BATCH_THREADS=1
// 表示只用调用线程；单样本与批量结果必须逐位一致且不挂起。
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "gimbal_dynamics/control_gradient.hpp"
#include "gimbal_dynamics/parameter_gradient.hpp"
#include "gimbal_dynamics/simulation.hpp"

int main() {
    setenv("GIMBAL_PIPELINE_THREADS", "1", 1);
    setenv("GIMBAL_BATCH_THREADS", "1", 1);

    using namespace gimbal;
    const std::size_t num_steps = 12;
    const std::size_t batch = 4;
    ModelParameters model = ModelParameters::demo();
    Terrain terrain;
    terrain.slope_angle = 0.05;
    terrain.slope_direction = 0.7;
    simulation::TrajectorySpec spec{0.01, 2};
    simulation::State initial;
    initial.angle = math::Vector4d::from(0.20, -0.10, 0.15, 0.05);
    initial.velocity = math::Vector4d::from(0.30, -0.20, 0.10, -0.05);
    std::vector<BaseState> base(batch * num_steps);
    std::vector<math::Vector4d> torque(batch * num_steps);
    std::vector<math::Vector4d> target_angle(num_steps);
    std::vector<math::Vector4d> target_velocity(num_steps);
    for (std::size_t k = 0; k < num_steps; ++k) {
        const Real t = static_cast<Real>(k) * spec.step;
        for (int i = 0; i < 4; ++i) {
            target_angle[k][i] = 0.10 * std::sin(0.5 * t + 0.7 * i);
            target_velocity[k][i] = 0.20 * std::cos(0.3 * t + 0.4 * i);
        }
        for (std::size_t sample = 0; sample < batch; ++sample) {
            BaseState& b = base[sample * num_steps + k];
            b.yaw = 0.2 + 0.4 * t;
            b.yaw_rate = 0.4;
            b.yaw_acceleration = 0.3;
            b.acceleration[0] = 0.2;
            b.acceleration[1] = 0.15;
            b.acceleration[2] = 0.05;
            torque[sample * num_steps + k] =
                math::Vector4d::from(0.20, -0.12, 0.06, -0.04);
        }
    }
    simulation::LossSpec loss;
    loss.angle_weights = math::Vector4d::from(1.0, 2.0, 0.8, 1.5);
    loss.velocity_weights = math::Vector4d::from(0.3, 0.5, 0.2, 0.4);
    loss.torque_weights = math::Vector4d::from(0.05, 0.05, 0.03, 0.03);
    loss.target_angle = target_angle.data();
    loss.target_velocity = target_velocity.data();
    std::vector<simulation::LossSpec> losses(batch, loss);
    std::vector<simulation::State> initial_batch(batch, initial);

    control_gradient::Workspace control_workspace;
    std::vector<math::Vector4d> single_gradient(num_steps);
    control_gradient::simulateAndGradient(model, terrain, initial, base.data(), torque.data(),
                                          num_steps, spec, loss, control_workspace,
                                          single_gradient.data(), nullptr);
    parameter_gradient::Workspace parameter_workspace;
    parameter_gradient::Gradient single_sensitivity;
    parameter_gradient::simulateAndSensitivity(model, terrain, initial, base.data(),
                                               torque.data(), num_steps, spec, loss,
                                               parameter_workspace, &single_sensitivity,
                                               nullptr);

    control_gradient::BatchWorkspace batch_workspace;
    std::vector<math::Vector4d> batch_gradient(batch * num_steps);
    control_gradient::simulateAndGradientBatch(
        model, terrain, initial_batch.data(), base.data(), torque.data(), batch, num_steps, spec,
        losses.data(), batch_workspace, batch_gradient.data(), nullptr);
    parameter_gradient::BatchWorkspace parameter_batch;
    std::vector<parameter_gradient::Gradient> batch_sensitivity(batch);
    parameter_gradient::simulateAndSensitivityBatch(
        model, terrain, initial_batch.data(), base.data(), torque.data(), batch, num_steps, spec,
        losses.data(), parameter_batch, batch_sensitivity.data(), nullptr);

    bool ok = true;
    for (std::size_t sample = 0; sample < batch; ++sample) {
        for (std::size_t k = 0; k < num_steps; ++k) {
            for (int i = 0; i < 4; ++i) {
                if (batch_gradient[sample * num_steps + k][i] != single_gradient[k][i]) {
                    ok = false;
                }
            }
        }
        for (int i = 0; i < parameters::kParameterCount; ++i) {
            if (batch_sensitivity[sample][i] != single_sensitivity[i]) {
                ok = false;
            }
        }
    }
    printf("%s\n", ok ? "线程数环境变量检查通过" : "线程数环境变量检查失败");
    return ok ? 0 : 1;
}
