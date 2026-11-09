#pragma once

// 测试共用的算例构造与处理器封装

#include <cmath>
#include <cstddef>
#include <vector>

#include "gimbal_dynamics/handlers.hpp"

namespace support {

template <typename Real>
struct Problem {
    gimbal::ModelParametersT<Real> model;
    gimbal::TerrainT<Real> terrain;
    gimbal::simulation::StateT<Real> initial;
    gimbal::simulation::TrajectorySpecT<Real> spec;
    std::vector<gimbal::BaseStateT<Real>> base;
    std::vector<gimbal::math::Vector<Real, 4>> torque;
    std::vector<gimbal::math::Vector<Real, 4>> target_angle;
    std::vector<gimbal::math::Vector<Real, 4>> target_velocity;
    gimbal::simulation::LossSpecT<Real> loss;
    std::size_t num_steps = 20;

    void build() {
        base.resize(num_steps);
        torque.resize(num_steps);
        target_angle.resize(num_steps);
        target_velocity.resize(num_steps);
        for (std::size_t k = 0; k < num_steps; ++k) {
            const Real t = static_cast<Real>(k) * spec.step;
            gimbal::BaseStateT<Real>& b = base[k];
            b.yaw = Real(0.2) + Real(0.4) * t;
            b.yaw_rate = Real(0.4);
            b.yaw_acceleration = Real(0.3);
            b.acceleration[0] = Real(0.2);
            b.acceleration[1] = Real(0.15);
            b.acceleration[2] = Real(0.05);
            torque[k] = gimbal::math::Vector<Real, 4>::from(Real(0.20), Real(-0.12), Real(0.06),
                                                            Real(-0.04));
            for (int i = 0; i < 4; ++i) {
                target_angle[k][i] = Real(0.10) * std::sin(Real(0.5) * t + Real(0.7) * Real(i));
                target_velocity[k][i] =
                    Real(0.20) * std::cos(Real(0.3) * t + Real(0.4) * Real(i));
            }
        }
        loss.angle_weights =
            gimbal::math::Vector<Real, 4>::from(Real(1.0), Real(2.0), Real(0.8), Real(1.5));
        loss.velocity_weights =
            gimbal::math::Vector<Real, 4>::from(Real(0.3), Real(0.5), Real(0.2), Real(0.4));
        loss.torque_weights =
            gimbal::math::Vector<Real, 4>::from(Real(0.05), Real(0.05), Real(0.03), Real(0.03));
        loss.target_angle = target_angle.data();
        loss.target_velocity = target_velocity.data();
    }
};

template <typename Real>
Problem<Real> make_problem(std::size_t num_steps = 20) {
    Problem<Real> problem;
    problem.model = gimbal::ModelParametersT<Real>::demo();
    problem.terrain.slope_angle = Real(0.05);
    problem.terrain.slope_direction = Real(0.7);
    problem.initial.angle =
        gimbal::math::Vector<Real, 4>::from(Real(0.20), Real(-0.10), Real(0.15), Real(0.05));
    problem.initial.velocity =
        gimbal::math::Vector<Real, 4>::from(Real(0.30), Real(-0.20), Real(0.10), Real(-0.05));
    problem.spec.step = Real(0.01);
    problem.spec.refinement = 2;
    problem.num_steps = num_steps;
    problem.build();
    return problem;
}

template <typename Real>
gimbal::ControlGradientInputT<Real> control_input(const Problem<Real>& problem) {
    gimbal::ControlGradientInputT<Real> input;
    input.model = problem.model;
    input.terrain = problem.terrain;
    input.initial = problem.initial;
    input.base = problem.base.data();
    input.torque = problem.torque.data();
    input.num_steps = problem.num_steps;
    input.spec = problem.spec;
    input.loss = problem.loss;
    return input;
}

template <typename Real>
gimbal::ParameterSensitivityInputT<Real> parameter_input(const Problem<Real>& problem) {
    gimbal::ParameterSensitivityInputT<Real> input;
    input.model = problem.model;
    input.terrain = problem.terrain;
    input.initial = problem.initial;
    input.base = problem.base.data();
    input.torque = problem.torque.data();
    input.num_steps = problem.num_steps;
    input.spec = problem.spec;
    input.loss = problem.loss;
    return input;
}

template <typename Real>
gimbal::ForwardInputT<Real> forward_input(const Problem<Real>& problem) {
    gimbal::ForwardInputT<Real> input;
    input.model = problem.model;
    input.terrain = problem.terrain;
    input.initial = problem.initial;
    input.base = problem.base.data();
    input.torque = problem.torque.data();
    input.num_steps = problem.num_steps;
    input.spec = problem.spec;
    input.loss = problem.loss;
    return input;
}

}  // namespace support
