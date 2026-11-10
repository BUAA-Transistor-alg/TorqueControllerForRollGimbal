// 校验：单步闭式解与仿真内核（逆动力学递推）在单步 RK4 上给出相同结果。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>

#include "gimbal_dynamics/handlers.hpp"

using namespace tcrg::gimbal_dynamics;

namespace {

template <typename Real>
Real run_check() {
    std::mt19937 rng(20261108);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    SingleStepDynamics<Real> single(1);
    ForwardSimulation<Real> forward(1);

    Real max_relative = Real(0);
    for (int trial = 0; trial < 200; ++trial) {
        ModelParametersT<Real> model = ModelParametersT<Real>::demo();
        for (int i = 0; i < 4; ++i) {
            model.bodies[i].mass *= Real(1.0 + 0.3 * unit(rng));
            for (int c = 0; c < 3; ++c) {
                model.bodies[i].com[c] += Real(0.01 * unit(rng));
                model.joint_offsets[i][c] += Real(0.005 * unit(rng));
            }
        }
        TerrainT<Real> terrain;
        terrain.slope_angle = Real(0.3 * unit(rng));
        terrain.slope_direction = Real(3.0 * unit(rng));
        BaseStateT<Real> base;
        base.yaw = Real(3.0 * unit(rng));
        base.yaw_rate = Real(2.0 * unit(rng));
        base.yaw_acceleration = Real(5.0 * unit(rng));
        for (int c = 0; c < 3; ++c) {
            base.acceleration[c] = Real(3.0 * unit(rng));
        }
        simulation::StateT<Real> initial;
        math::Vector<Real, 4> torque;
        for (int i = 0; i < 4; ++i) {
            initial.angle[i] = Real(2.0 * unit(rng));
            initial.velocity[i] = Real(4.0 * unit(rng));
            torque[i] = Real(2.0 * unit(rng));
        }
        const Real h = Real(0.01);

        auto base_at = [&](Real time) {
            BaseStateT<Real> result = base;
            result.yaw = base.yaw + base.yaw_rate * time +
                         Real(0.5) * base.yaw_acceleration * time * time;
            result.yaw_rate = base.yaw_rate + base.yaw_acceleration * time;
            return result;
        };
        auto acceleration = [&](const simulation::StateT<Real>& state, Real time) {
            SingleStepInputT<Real> input;
            input.model = model;
            input.terrain = terrain;
            input.base = base_at(time);
            input.angle = state.angle;
            input.velocity = state.velocity;
            input.torque = torque;
            SingleStepOutputT<Real> output;
            single.run(input, output);
            return output.acceleration;
        };
        auto derivative = [&](const simulation::StateT<Real>& state, Real time) {
            simulation::StateT<Real> result;
            result.angle = state.velocity;
            result.velocity = acceleration(state, time);
            return result;
        };
        auto combine = [&](const simulation::StateT<Real>& state, Real weight_a,
                          const simulation::StateT<Real>& a, Real weight_b,
                          const simulation::StateT<Real>& b) {
            simulation::StateT<Real> result;
            for (int i = 0; i < 4; ++i) {
                result.angle[i] = state.angle[i] + h * (weight_a * a.angle[i] + weight_b * b.angle[i]);
                result.velocity[i] =
                    state.velocity[i] + h * (weight_a * a.velocity[i] + weight_b * b.velocity[i]);
            }
            return result;
        };
        const simulation::StateT<Real> k1 = derivative(initial, Real(0));
        simulation::StateT<Real> k2 =
            derivative(combine(initial, Real(0.5), k1, Real(0), k1), h * Real(0.5));
        simulation::StateT<Real> k3 =
            derivative(combine(initial, Real(0), k2, Real(0.5), k2), h * Real(0.5));
        simulation::StateT<Real> k4 =
            derivative(combine(initial, Real(0), k3, Real(1.0), k3), h);
        simulation::StateT<Real> manual;
        for (int i = 0; i < 4; ++i) {
            manual.angle[i] =
                initial.angle[i] + h / Real(6) *
                                       (k1.angle[i] + Real(2) * k2.angle[i] +
                                        Real(2) * k3.angle[i] + k4.angle[i]);
            manual.velocity[i] =
                initial.velocity[i] + h / Real(6) *
                                          (k1.velocity[i] + Real(2) * k2.velocity[i] +
                                           Real(2) * k3.velocity[i] + k4.velocity[i]);
        }

        ForwardInputT<Real> input;
        input.model = model;
        input.terrain = terrain;
        input.initial = initial;
        input.base = &base;
        input.torque = &torque;
        input.num_steps = 1;
        input.spec.step = h;
        input.spec.refinement = 1;
        input.loss = simulation::LossSpecT<Real>{};
        ForwardOutputT<Real> output;
        forward.run(input, output);
        const simulation::StateT<Real>& kernel = output.step_end[0];

        for (int i = 0; i < 4; ++i) {
            max_relative =
                std::max(max_relative, std::fabs(manual.angle[i] - kernel.angle[i]) /
                                           std::max(Real(1), std::fabs(kernel.angle[i])));
            max_relative =
                std::max(max_relative, std::fabs(manual.velocity[i] - kernel.velocity[i]) /
                                           std::max(Real(1), std::fabs(kernel.velocity[i])));
        }
    }
    return max_relative;
}

}  // namespace

int main() {
    const double error_double = static_cast<double>(run_check<double>());
    const double error_float = static_cast<double>(run_check<float>());
    printf("double 闭式单步与仿真内核最大相对偏差 = %.3e\n", error_double);
    printf("float  闭式单步与仿真内核最大相对偏差 = %.3e\n", error_float);
    if (!(error_double < 1e-12) || !(error_float < 2e-4)) {
        printf("校验失败\n");
        return 1;
    }
    printf("校验通过\n");
    return 0;
}
