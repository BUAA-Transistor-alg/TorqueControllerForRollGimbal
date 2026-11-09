// 异常路径检查：非法输入与非正定模型抛异常后，线程池仍可继续使用；
// 批量非法配置在调用线程抛出；批量与单样本的阶段轨迹一致。
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

#include "gimbal_dynamics/control_gradient.hpp"
#include "gimbal_dynamics/parameter_gradient.hpp"
#include "gimbal_dynamics/simulation.hpp"

namespace {

using gimbal::Real;

struct Problem {
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

Problem make_problem() {
    Problem problem;
    problem.model = gimbal::ModelParameters::demo();
    problem.terrain.slope_angle = 0.05;
    problem.terrain.slope_direction = 0.7;
    problem.initial.angle = gimbal::math::Vector4d::from(0.20, -0.10, 0.15, 0.05);
    problem.initial.velocity = gimbal::math::Vector4d::from(0.30, -0.20, 0.10, -0.05);
    problem.spec.step = 0.01;
    problem.spec.refinement = 2;
    problem.base.resize(problem.num_steps);
    problem.torque.resize(problem.num_steps);
    problem.target_angle.resize(problem.num_steps);
    problem.target_velocity.resize(problem.num_steps);
    for (std::size_t k = 0; k < problem.num_steps; ++k) {
        const Real t = static_cast<Real>(k) * problem.spec.step;
        gimbal::BaseState& base = problem.base[k];
        base.yaw = 0.2 + 0.4 * t;
        base.yaw_rate = 0.4;
        base.yaw_acceleration = 0.3;
        base.acceleration[0] = 0.2;
        base.acceleration[1] = 0.15;
        base.acceleration[2] = 0.05;
        problem.torque[k] = gimbal::math::Vector4d::from(0.20, -0.12, 0.06, -0.04);
        for (int i = 0; i < 4; ++i) {
            problem.target_angle[k][i] = 0.10 * std::sin(0.5 * t + 0.7 * i);
            problem.target_velocity[k][i] = 0.20 * std::cos(0.3 * t + 0.4 * i);
        }
    }
    problem.loss.angle_weights = gimbal::math::Vector4d::from(1.0, 2.0, 0.8, 1.5);
    problem.loss.velocity_weights = gimbal::math::Vector4d::from(0.3, 0.5, 0.2, 0.4);
    problem.loss.torque_weights = gimbal::math::Vector4d::from(0.05, 0.05, 0.03, 0.03);
    problem.loss.target_angle = problem.target_angle.data();
    problem.loss.target_velocity = problem.target_velocity.data();
    return problem;
}

bool expect_throw(const char* label, const std::function<void()>& function) {
    try {
        function();
    } catch (const std::exception& error) {
        printf("%s：抛出 %s\n", label, error.what());
        return true;
    }
    printf("%s：未抛出异常\n", label);
    return false;
}

bool run_normal(const Problem& problem) {
    gimbal::control_gradient::Workspace control_workspace;
    std::vector<gimbal::math::Vector4d> gradient(problem.num_steps);
    gimbal::control_gradient::simulateAndGradient(
        problem.model, problem.terrain, problem.initial, problem.base.data(),
        problem.torque.data(), problem.num_steps, problem.spec, problem.loss, control_workspace,
        gradient.data(), nullptr);
    gimbal::parameter_gradient::Workspace parameter_workspace;
    gimbal::parameter_gradient::Gradient sensitivity;
    gimbal::parameter_gradient::simulateAndSensitivity(
        problem.model, problem.terrain, problem.initial, problem.base.data(),
        problem.torque.data(), problem.num_steps, problem.spec, problem.loss,
        parameter_workspace, &sensitivity, nullptr);
    return true;
}

}  // namespace

int main() {
    Problem problem = make_problem();
    bool ok = true;
    ok = run_normal(problem) && ok;

    gimbal::control_gradient::Workspace control_workspace;
    std::vector<gimbal::math::Vector4d> gradient(problem.num_steps);
    ok = expect_throw("单样本 base 空指针", [&] {
             gimbal::control_gradient::simulateAndGradient(
                 problem.model, problem.terrain, problem.initial, nullptr, problem.torque.data(),
                 problem.num_steps, problem.spec, problem.loss, control_workspace,
                 gradient.data(), nullptr);
         }) && ok;
    ok = run_normal(problem) && ok;

    gimbal::ModelParameters bad = problem.model;
    bad.bodies[0].mass = -1.0e10;
    ok = expect_throw("单样本非正定模型", [&] {
             gimbal::control_gradient::simulateAndGradient(
                 bad, problem.terrain, problem.initial, problem.base.data(),
                 problem.torque.data(), problem.num_steps, problem.spec, problem.loss,
                 control_workspace, gradient.data(), nullptr);
         }) && ok;
    ok = run_normal(problem) && ok;

    std::vector<gimbal::simulation::LossSpec> losses(4, problem.loss);
    std::vector<gimbal::simulation::State> initial_batch(4, problem.initial);
    std::vector<gimbal::BaseState> base_batch(4 * problem.num_steps);
    std::vector<gimbal::math::Vector4d> torque_batch(4 * problem.num_steps);
    for (std::size_t sample = 0; sample < 4; ++sample) {
        for (std::size_t k = 0; k < problem.num_steps; ++k) {
            base_batch[sample * problem.num_steps + k] = problem.base[k];
            torque_batch[sample * problem.num_steps + k] = problem.torque[k];
        }
    }
    gimbal::control_gradient::BatchWorkspace batch_workspace;
    std::vector<gimbal::math::Vector4d> batch_gradient(4 * problem.num_steps);
    gimbal::simulation::TrajectorySpec bad_spec = problem.spec;
    bad_spec.refinement = 0;
    ok = expect_throw("批量非法细化倍数", [&] {
             gimbal::control_gradient::simulateAndGradientBatch(
                 problem.model, problem.terrain, initial_batch.data(), base_batch.data(),
                 torque_batch.data(), 4, problem.num_steps, bad_spec, losses.data(),
                 batch_workspace, batch_gradient.data(), nullptr);
         }) && ok;
    ok = expect_throw("批量非正定模型", [&] {
             gimbal::control_gradient::simulateAndGradientBatch(
                 bad, problem.terrain, initial_batch.data(), base_batch.data(),
                 torque_batch.data(), 4, problem.num_steps, problem.spec, losses.data(),
                 batch_workspace, batch_gradient.data(), nullptr);
         }) && ok;
    ok = run_normal(problem) && ok;

    // batch=0 直接返回 0
    const Real zero = gimbal::control_gradient::simulateAndGradientBatch(
        problem.model, problem.terrain, initial_batch.data(), base_batch.data(),
        torque_batch.data(), 0, problem.num_steps, problem.spec, losses.data(), batch_workspace,
        batch_gradient.data(), nullptr);
    if (zero != Real(0)) {
        printf("batch=0 返回值非 0\n");
        ok = false;
    }

    // 批量与单样本的阶段轨迹一致
    gimbal::simulation::Trace single_trace;
    gimbal::simulation::simulate(problem.model, problem.terrain, problem.initial,
                                 problem.base.data(), problem.torque.data(), problem.num_steps,
                                 problem.spec, problem.loss, single_trace);
    gimbal::control_gradient::Workspace stage_workspace;
    gimbal::control_gradient::simulateAndGradient(
        problem.model, problem.terrain, problem.initial, problem.base.data(),
        problem.torque.data(), problem.num_steps, problem.spec, problem.loss, stage_workspace,
        nullptr, nullptr);
    for (std::size_t index = 0; index < stage_workspace.trace.stage_state.size(); ++index) {
        for (int i = 0; i < 4; ++i) {
            if (stage_workspace.trace.stage_state[index].angle[i] !=
                    single_trace.stage_state[index].angle[i] ||
                stage_workspace.trace.stage_state[index].velocity[i] !=
                    single_trace.stage_state[index].velocity[i]) {
                printf("单样本阶段轨迹与参考不一致\n");
                ok = false;
                index = stage_workspace.trace.stage_state.size();
                break;
            }
        }
    }

    printf("%s\n", ok ? "异常与恢复检查通过" : "异常与恢复检查失败");
    return ok ? 0 : 1;
}
